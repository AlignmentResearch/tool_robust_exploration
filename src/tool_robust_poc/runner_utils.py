"""Async evaluation utilities for judge robustness experiments.

Handles concurrency, progress tracking, and result collection. Task-agnostic:
callers provide pre-built messages and metadata.

NOTE: This interface is provisional. As we add more experiment types (ClearHarm,
MT-Bench, etc.) the signatures here may evolve. Don't treat this as a stable API.

The orchestration primitives (`ProgressTracker`, `AdaptiveRequestController`,
`run_job_queue`, plus default-concurrency constants) now live in
`fllmingo.runner` and are re-exported here for backward compatibility.  New
code should import from `fllmingo` directly and ideally use
`fllmingo.run_many` / `ModelSession` instead of hand-rolling the concurrent
fan-out.
"""

from __future__ import annotations

import asyncio
from typing import Callable

from tool_robust_poc.core_types import PairwiseVerdict

from fllmingo import (
    CostTracker,
    ResponseCache,
    cached_chat_completion,
)
from fllmingo.runner import (
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_CONCURRENCY,
    VLLM_DEFAULT_CONCURRENCY,
    VLLM_DEFAULT_MAX_CONCURRENCY,
    AdaptiveRequestController,
    ProgressTracker,
    run_job_queue,
)

__all__ = [
    "DEFAULT_CONCURRENCY",
    "DEFAULT_MAX_CONCURRENCY",
    "VLLM_DEFAULT_CONCURRENCY",
    "VLLM_DEFAULT_MAX_CONCURRENCY",
    "AdaptiveRequestController",
    "ProgressTracker",
    "run_job_queue",
    "eval_one",
    "eval_one_scalar",
    "eval_one_pairwise",
]


async def _cached_chat_completion_one(
    client,
    model: str,
    messages: list[dict],
    *,
    cache: ResponseCache,
    tracker: CostTracker,
    cache_tag: str,
    provider: str | None = None,
    sem: asyncio.Semaphore | None = None,
    extra_api_kwargs: dict | None = None,
    request_controller: AdaptiveRequestController | None = None,
    event_sink: asyncio.Queue | None = None,
):
    """Run one cached chat completion with optional concurrency gating."""
    if sem is not None:
        async with sem:
            return await cached_chat_completion(
                client,
                model=model,
                messages=messages,
                cache=cache,
                tracker=tracker,
                cache_tag=cache_tag,
                provider=provider,
                request_controller=request_controller,
                event_sink=event_sink,
                **(extra_api_kwargs or {}),
            )
    return await cached_chat_completion(
        client,
        model=model,
        messages=messages,
        cache=cache,
        tracker=tracker,
        cache_tag=cache_tag,
        provider=provider,
        request_controller=request_controller,
        event_sink=event_sink,
        **(extra_api_kwargs or {}),
    )


async def eval_one(
    client,
    model: str,
    messages: list[dict],
    *,
    cache: ResponseCache,
    tracker: CostTracker,
    cache_tag: str,
    parse_fn: Callable[[str], str | None],
    expected: str,
    positive_verdict: str = "CORRECT",
    provider: str | None = None,
    metadata: dict | None = None,
    progress: ProgressTracker | None = None,
    sem: asyncio.Semaphore | None = None,
    extra_api_kwargs: dict | None = None,
    request_controller: AdaptiveRequestController | None = None,
    event_sink: asyncio.Queue | None = None,
) -> dict:
    """Run a single evaluation call and return a structured result.

    Args:
        client: OpenAI-compatible async client.
        model: Model identifier.
        messages: Pre-built message list.
        cache: Response cache instance.
        tracker: Cost tracker instance.
        cache_tag: Unique tag for caching this call.
        parse_fn: Function to extract verdict from raw output.
        expected: Expected verdict string (e.g. "CORRECT" or "INCORRECT").
        positive_verdict: The verdict string that means "attack succeeded"
            when it shouldn't have. For GSM8K this is "CORRECT" (judge fooled
            into accepting wrong answer). For ClearHarm this would be "SAFE"
            (judge fooled into accepting harmful content).
        metadata: Extra fields merged into the result dict (e.g. q_index, condition).
        progress: Optional progress tracker.
        sem: Concurrency semaphore.
        extra_api_kwargs: Extra kwargs passed to the API call (e.g. tools).

    Returns:
        Dict with verdict, expected, ok, attack_success, raw, plus any metadata.
    """
    raw_text = ""
    completion = None
    result: dict
    try:
        completion = await _cached_chat_completion_one(
            client,
            model,
            messages,
            cache=cache,
            tracker=tracker,
            cache_tag=cache_tag,
            provider=provider,
            sem=sem,
            extra_api_kwargs=extra_api_kwargs,
            request_controller=request_controller,
            event_sink=event_sink,
        )

        raw_text = completion.text
        finish_reason = completion.finish_reason
        tool_calls = completion.tool_calls

        # Detect when the model responded with tool calls instead of text
        if tool_calls and not raw_text.strip():
            result = {
                "status": "model_tool_call",
                "expected": expected,
                "verdict": None,
                "ok": None,
                "attack_success": None,
                "raw": raw_text.strip(),
                "finish_reason": finish_reason,
                "tool_calls": tool_calls,
                "error_type": "ModelToolCall",
                "error_message": (
                    "Model responded with tool call(s) instead of text verdict."
                ),
            }
        elif not raw_text.strip() and finish_reason == "length":
            result = {
                "status": "token_limit",
                "expected": expected,
                "verdict": None,
                "ok": None,
                "attack_success": None,
                "raw": raw_text.strip(),
                "finish_reason": finish_reason,
                "tool_calls": None,
                "error_type": "TokenLimit",
                "error_message": (
                    "Empty response with finish_reason=length — likely exhausted"
                    " token budget in reasoning."
                ),
            }
        else:
            verdict = parse_fn(raw_text)
            if verdict is None:
                result = {
                    "status": "parse_error",
                    "expected": expected,
                    "verdict": None,
                    "ok": None,
                    "attack_success": None,
                    "raw": raw_text.strip(),
                    "finish_reason": finish_reason,
                    "tool_calls": tool_calls,
                    "error_type": "ParseError",
                    "error_message": "Could not parse verdict from model response.",
                }
            else:
                expected_negative = expected != positive_verdict
                result = {
                    "status": "ok",
                    "expected": expected,
                    "verdict": verdict,
                    "ok": verdict == expected,
                    # attack_success: the judge gave the positive verdict when it shouldn't have.
                    # (FP for math grading, FN for safety eval — same concept, different framing.)
                    "attack_success": expected_negative and verdict == positive_verdict,
                    "raw": raw_text.strip(),
                    "finish_reason": finish_reason,
                    "tool_calls": tool_calls,
                    "error_type": None,
                    "error_message": None,
                }
    except Exception as exc:
        result = {
            "status": "api_error",
            "expected": expected,
            "verdict": None,
            "ok": None,
            "attack_success": None,
            "raw": raw_text.strip(),
            "finish_reason": getattr(completion, "finish_reason", None) if completion else None,
            "tool_calls": None,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    if metadata:
        result = {**metadata, **result}

    if progress:
        await progress.increment()

    return result


async def eval_one_scalar(
    client,
    model: str,
    messages: list[dict],
    *,
    cache: ResponseCache,
    tracker: CostTracker,
    cache_tag: str,
    parse_score: Callable[[str], int | float | None],
    provider: str | None = None,
    metadata: dict | None = None,
    progress: ProgressTracker | None = None,
    sem: asyncio.Semaphore | None = None,
    extra_api_kwargs: dict | None = None,
    request_controller: AdaptiveRequestController | None = None,
    event_sink: asyncio.Queue | None = None,
) -> dict:
    """Run a single scalar evaluation call and return a structured result."""
    raw_text = ""
    completion = None
    result: dict
    try:
        completion = await _cached_chat_completion_one(
            client,
            model,
            messages,
            cache=cache,
            tracker=tracker,
            cache_tag=cache_tag,
            provider=provider,
            sem=sem,
            extra_api_kwargs=extra_api_kwargs,
            request_controller=request_controller,
            event_sink=event_sink,
        )

        raw_text = completion.text
        finish_reason = completion.finish_reason
        tool_calls = completion.tool_calls

        if tool_calls and not raw_text.strip():
            result = {
                "status": "model_tool_call",
                "score": None,
                "parsed_ok": False,
                "raw": raw_text.strip(),
                "finish_reason": finish_reason,
                "tool_calls": tool_calls,
                "error_type": "ModelToolCall",
                "error_message": (
                    "Model responded with tool call(s) instead of text score output."
                ),
            }
        elif not raw_text.strip() and finish_reason == "length":
            result = {
                "status": "token_limit",
                "score": None,
                "parsed_ok": False,
                "raw": raw_text.strip(),
                "finish_reason": finish_reason,
                "tool_calls": None,
                "error_type": "TokenLimit",
                "error_message": (
                    "Empty response with finish_reason=length — likely exhausted"
                    " token budget in reasoning."
                ),
            }
        else:
            score = parse_score(raw_text)
            if score is None:
                result = {
                    "status": "parse_error",
                    "score": None,
                    "parsed_ok": False,
                    "raw": raw_text.strip(),
                    "finish_reason": finish_reason,
                    "tool_calls": tool_calls,
                    "error_type": "ParseError",
                    "error_message": "Could not parse scalar score from model response.",
                }
            else:
                result = {
                    "status": "ok",
                    "score": score,
                    "parsed_ok": True,
                    "raw": raw_text.strip(),
                    "finish_reason": finish_reason,
                    "tool_calls": tool_calls,
                    "error_type": None,
                    "error_message": None,
                }
    except Exception as exc:
        result = {
            "status": "api_error",
            "score": None,
            "parsed_ok": False,
            "raw": raw_text.strip(),
            "finish_reason": getattr(completion, "finish_reason", None) if completion else None,
            "tool_calls": None,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    if metadata:
        result = {**metadata, **result}

    if progress:
        await progress.increment()

    return result


async def eval_one_pairwise(
    client,
    model: str,
    messages: list[dict],
    *,
    cache: ResponseCache,
    tracker: CostTracker,
    cache_tag: str,
    parse_preference: Callable[[str], PairwiseVerdict | None],
    provider: str | None = None,
    metadata: dict | None = None,
    progress: ProgressTracker | None = None,
    sem: asyncio.Semaphore | None = None,
    extra_api_kwargs: dict | None = None,
    request_controller: AdaptiveRequestController | None = None,
    event_sink: asyncio.Queue | None = None,
) -> dict:
    """Run a single pairwise evaluation call and return a structured result."""
    raw_text = ""
    completion = None
    result: dict
    try:
        completion = await _cached_chat_completion_one(
            client,
            model,
            messages,
            cache=cache,
            tracker=tracker,
            cache_tag=cache_tag,
            provider=provider,
            sem=sem,
            extra_api_kwargs=extra_api_kwargs,
            request_controller=request_controller,
            event_sink=event_sink,
        )

        raw_text = completion.text
        finish_reason = completion.finish_reason
        tool_calls = completion.tool_calls

        if tool_calls and not raw_text.strip():
            result = {
                "status": "model_tool_call",
                "direction": None,
                "strength": None,
                "raw_label": None,
                "parsed_ok": False,
                "raw": raw_text.strip(),
                "finish_reason": finish_reason,
                "tool_calls": tool_calls,
                "error_type": "ModelToolCall",
                "error_message": (
                    "Model responded with tool call(s) instead of text preference."
                ),
            }
        elif not raw_text.strip() and finish_reason == "length":
            result = {
                "status": "token_limit",
                "direction": None,
                "strength": None,
                "raw_label": None,
                "parsed_ok": False,
                "raw": raw_text.strip(),
                "finish_reason": finish_reason,
                "tool_calls": None,
                "error_type": "TokenLimit",
                "error_message": (
                    "Empty response with finish_reason=length — likely exhausted"
                    " token budget in reasoning."
                ),
            }
        else:
            verdict = parse_preference(raw_text)
            if verdict is None:
                result = {
                    "status": "parse_error",
                    "direction": None,
                    "strength": None,
                    "raw_label": None,
                    "parsed_ok": False,
                    "raw": raw_text.strip(),
                    "finish_reason": finish_reason,
                    "tool_calls": tool_calls,
                    "error_type": "ParseError",
                    "error_message": "Could not parse pairwise preference from model response.",
                }
            else:
                result = {
                    "status": "ok",
                    "direction": verdict.direction,
                    "strength": verdict.strength,
                    "raw_label": verdict.raw_label,
                    "parsed_ok": True,
                    "raw": raw_text.strip(),
                    "finish_reason": finish_reason,
                    "tool_calls": tool_calls,
                    "error_type": None,
                    "error_message": None,
                }
    except Exception as exc:
        result = {
            "status": "api_error",
            "direction": None,
            "strength": None,
            "raw_label": None,
            "parsed_ok": False,
            "raw": raw_text.strip(),
            "finish_reason": getattr(completion, "finish_reason", None) if completion else None,
            "tool_calls": None,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    if metadata:
        result = {**metadata, **result}

    if progress:
        await progress.increment()

    return result
