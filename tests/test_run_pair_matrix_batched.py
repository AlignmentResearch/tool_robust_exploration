"""Unit tests for run_pair_matrix_batched.

Exercises the lockstep orchestrator with a fake Strategy that returns canned
completions per request, so we can verify cell-result demultiplexing,
disqualifier short-circuit, attacker-history advancement, and Turn /
PairResult assembly without hitting any network.
"""

from __future__ import annotations

import asyncio
import json

from fllmingo import (
    CompletedRequest,
    CompletionRequest,
    CompletionResult,
    ModelSession,
    Support,
)

from tool_robust_poc.attack_opt import (
    JudgeTask,
    PairCell,
    ScoreResult,
    run_pair_matrix_batched,
)
from tool_robust_poc.core_types import ConditionBlueprint


# ── Fake strategy that maps cache_tag → canned text ───────────────────


class _ScriptedStrategy:
    """Strategy that resolves each request from a (cache_tag → text) script.

    Lets us test the orchestrator without spinning up real clients.
    """

    def __init__(self, scripted: dict[str, str]):
        self.scripted = scripted
        self.calls: list[list[CompletionRequest]] = []

    def check_support(self, *, session, requests):
        return Support(supported=True)

    async def run(self, *, session, requests, indices, on_complete):
        self.calls.append(list(requests))
        for idx, req in zip(indices, requests):
            text = self.scripted.get(req.cache_tag, "")
            completion = CompletionResult(
                text=text, finish_reason="stop", tool_calls=None,
            )
            await on_complete(idx, CompletedRequest(
                completion=completion, metadata=req.metadata,
            ))


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_blueprint() -> ConditionBlueprint:
    return ConditionBlueprint(
        system_text=None,
        user_text="Q: {question}\nCANDIDATE: {candidate}\nVerdict?",
    )


def _make_task() -> JudgeTask:
    """A tiny task: 'CORRECT' verdict means attack succeeded.

    item is a (item_id, question) tuple; ctx echoed unchanged.  The
    disqualifier rejects attacks that start with "DISQ:".
    """
    def make_payload(item, attack_string):
        item_id, question = item
        return {"question": question, "candidate": attack_string}, {"item_id": item_id}

    def score_fn(raw, item, ctx):
        # Match INCORRECT first to avoid the "CORRECT" substring trap.
        if "INCORRECT" in raw:
            verdict = "INCORRECT"
        elif "CORRECT" in raw:
            verdict = "CORRECT"
        else:
            verdict = None
        return ScoreResult(
            attack_succeeded=(verdict == "CORRECT"),
            parse_ok=(verdict is not None),
            verdict=verdict,
        )

    def item_id_fn(item):
        return item[0]

    def disqualify_fn(attack_string, item):
        if attack_string.startswith("DISQ:"):
            return f"attack starts with DISQ: marker (item={item[0]})"
        return None

    return JudgeTask(
        make_payload=make_payload,
        score_fn=score_fn,
        item_id_fn=item_id_fn,
        attacker_system_prompt="you are a tester",
        disqualify_fn=disqualify_fn,
    )


def _make_cell(cell_id: str, items: list[tuple[str, str]]) -> PairCell:
    return PairCell(
        cell_id=cell_id,
        condition_name=cell_id,
        blueprint=_make_blueprint(),
        task=_make_task(),
        items=items,
        cache_tag_prefix=f"unit:{cell_id}",
    )


def _attack_json(prompt: str) -> str:
    return json.dumps({"improvement": "tweak", "prompt": prompt})


def _make_session(model: str) -> ModelSession:
    return ModelSession(client=None, provider="anthropic", model=model)


# ── Tests ─────────────────────────────────────────────────────────────


def test_two_cells_one_turn_routes_results_correctly() -> None:
    cell_a = _make_cell("A", [("a1", "what is 2+2?"), ("a2", "what is 3+3?")])
    cell_b = _make_cell("B", [("b1", "name the capital of France"), ("b2", "what is 5*5?")])

    attacker_strategy = _ScriptedStrategy({
        "unit:A:attacker:turn1": _attack_json("ATTACK_A"),
        "unit:B:attacker:turn1": _attack_json("ATTACK_B"),
    })
    victim_strategy = _ScriptedStrategy({
        "unit:A:victim:turn1:a1": "VERDICT: CORRECT",
        "unit:A:victim:turn1:a2": "VERDICT: CORRECT",
        "unit:B:victim:turn1:b1": "VERDICT: INCORRECT",
        "unit:B:victim:turn1:b2": "VERDICT: CORRECT",
    })

    results = asyncio.run(run_pair_matrix_batched(
        [cell_a, cell_b],
        victim_session=_make_session("victim-x"),
        attacker_session=_make_session("attacker-y"),
        n_turns=1,
        victim_strategy=victim_strategy,
        attacker_strategy=attacker_strategy,
    ))

    # One attacker call (containing both cells) and one victim call (4 items).
    assert len(attacker_strategy.calls) == 1
    assert len(attacker_strategy.calls[0]) == 2
    assert len(victim_strategy.calls) == 1
    assert len(victim_strategy.calls[0]) == 4, "expected all victim requests in one batch"

    assert len(results) == 2
    res_a, res_b = results
    assert res_a.condition_name == "A"
    assert res_b.condition_name == "B"
    assert res_a.best_score == 1.0
    assert res_b.best_score == 0.5
    assert res_a.turns[0].attack_string == "ATTACK_A"
    assert res_b.turns[0].attack_string == "ATTACK_B"

    a_per_item = {p.item_id: p.attack_succeeded for p in res_a.turns[0].per_item}
    assert a_per_item == {"a1": True, "a2": True}
    b_per_item = {p.item_id: p.attack_succeeded for p in res_b.turns[0].per_item}
    assert b_per_item == {"b1": False, "b2": True}


def test_disqualified_items_skip_victim_batch() -> None:
    cell = _make_cell("A", [("a1", "q1"), ("a2", "q2")])
    attacker_strategy = _ScriptedStrategy({
        "unit:A:attacker:turn1": _attack_json("DISQ:cheat"),
    })
    victim_strategy = _ScriptedStrategy({})

    results = asyncio.run(run_pair_matrix_batched(
        [cell],
        victim_session=_make_session("victim-x"),
        attacker_session=_make_session("attacker-y"),
        n_turns=1,
        victim_strategy=victim_strategy,
        attacker_strategy=attacker_strategy,
    ))

    assert victim_strategy.calls == [], (
        "disqualified items should not reach victim"
    )
    turn = results[0].turns[0]
    assert all(p.disqualified for p in turn.per_item)
    assert turn.score == 0.0


def test_partial_disqualification_only_sends_eligible_items() -> None:
    base_task = _make_task()

    def selective_disq(attack_string, item):
        return "blacklisted" if item[0] == "skip_me" else None

    task_obj = JudgeTask(
        make_payload=base_task.make_payload,
        score_fn=base_task.score_fn,
        item_id_fn=base_task.item_id_fn,
        attacker_system_prompt=base_task.attacker_system_prompt,
        disqualify_fn=selective_disq,
    )
    cell = PairCell(
        cell_id="A",
        condition_name="A",
        blueprint=_make_blueprint(),
        task=task_obj,
        items=[("keep_me", "q1"), ("skip_me", "q2"), ("keep_me_2", "q3")],
        cache_tag_prefix="unit:A",
    )

    attacker_strategy = _ScriptedStrategy({
        "unit:A:attacker:turn1": _attack_json("ANY"),
    })
    victim_strategy = _ScriptedStrategy({
        "unit:A:victim:turn1:keep_me": "VERDICT: CORRECT",
        "unit:A:victim:turn1:keep_me_2": "VERDICT: INCORRECT",
    })

    results = asyncio.run(run_pair_matrix_batched(
        [cell],
        victim_session=_make_session("victim-x"),
        attacker_session=_make_session("attacker-y"),
        n_turns=1,
        victim_strategy=victim_strategy,
        attacker_strategy=attacker_strategy,
    ))

    assert len(victim_strategy.calls) == 1
    assert len(victim_strategy.calls[0]) == 2  # only the two non-disqualified items

    turn = results[0].turns[0]
    by_id = {p.item_id: p for p in turn.per_item}
    assert by_id["skip_me"].disqualified is True
    assert by_id["skip_me"].disqualified_reason == "blacklisted"
    assert by_id["keep_me"].attack_succeeded is True
    assert by_id["keep_me_2"].attack_succeeded is False
    # Original item order preserved.
    assert [p.item_id for p in turn.per_item] == ["keep_me", "skip_me", "keep_me_2"]


def test_two_turns_history_advances_and_uses_feedback() -> None:
    cell = _make_cell("A", [("a1", "q1")])
    attacker_strategy = _ScriptedStrategy({
        "unit:A:attacker:turn1": _attack_json("FIRST"),
        "unit:A:attacker:turn2": _attack_json("SECOND"),
    })
    victim_strategy = _ScriptedStrategy({
        "unit:A:victim:turn1:a1": "VERDICT: INCORRECT",
        "unit:A:victim:turn2:a1": "VERDICT: CORRECT",
    })

    results = asyncio.run(run_pair_matrix_batched(
        [cell],
        victim_session=_make_session("victim-x"),
        attacker_session=_make_session("attacker-y"),
        n_turns=2,
        victim_strategy=victim_strategy,
        attacker_strategy=attacker_strategy,
    ))
    assert len(attacker_strategy.calls) == 2
    turn2_messages = attacker_strategy.calls[1][0].messages
    assert len(turn2_messages) > len(attacker_strategy.calls[0][0].messages)
    assert any(
        "Previous attempt" in (m.get("content") or "")
        for m in turn2_messages
    ), "expected feedback user message on turn 2"

    res = results[0]
    assert [t.score for t in res.turns] == [0.0, 1.0]
    assert res.best_turn_index == 2
    assert res.best_attack == "SECOND"


def test_per_item_api_error_does_not_kill_cell() -> None:
    class _MixedStrategy:
        def check_support(self, *, session, requests):
            return Support(supported=True)
        async def run(self, *, session, requests, indices, on_complete):
            for idx, req in zip(indices, requests):
                if "victim:turn1:a2" in req.cache_tag:
                    await on_complete(idx, CompletedRequest(
                        metadata=req.metadata, error=RuntimeError("simulated 5xx"),
                    ))
                else:
                    text = (
                        "VERDICT: CORRECT"
                        if "victim:turn1:a1" in req.cache_tag
                        else _attack_json("X")
                    )
                    await on_complete(idx, CompletedRequest(
                        completion=CompletionResult(
                            text=text, finish_reason="stop", tool_calls=None,
                        ),
                        metadata=req.metadata,
                    ))

    cell = _make_cell("A", [("a1", "q1"), ("a2", "q2")])
    results = asyncio.run(run_pair_matrix_batched(
        [cell],
        victim_session=_make_session("victim-x"),
        attacker_session=_make_session("attacker-y"),
        n_turns=1,
        victim_strategy=_MixedStrategy(),
        attacker_strategy=_MixedStrategy(),
    ))
    turn = results[0].turns[0]
    by_id = {p.item_id: p for p in turn.per_item}
    assert by_id["a1"].attack_succeeded is True
    assert by_id["a2"].parse_ok is False
    assert by_id["a2"].attack_succeeded is False
