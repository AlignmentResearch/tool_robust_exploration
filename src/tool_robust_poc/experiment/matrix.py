"""Generic cross-model matrix driver.

Runs the same experiment (binary / pairwise / scalar) across multiple models
and aggregates per-model summaries into one matrix_summary.json.gz.

Task-specific details (spec construction, experiment-type function, summary
overview printer) are injected by callers; everything else - model loop,
output-file naming, matrix-summary aggregation, continue-on-error - lives here.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from tool_robust_poc.experiment.common import CommonExperimentConfig, write_json
from tool_robust_poc.sampling import stable_digest


def parse_model_spec(spec: str, default_provider: str) -> tuple[str, str]:
    """Split 'provider:model' or bare 'model' into (provider, model)."""
    if ":" in spec:
        provider, model = spec.split(":", 1)
        return provider, model
    return default_provider, spec


def model_tag(provider: str, model: str) -> str:
    """Filesystem-safe identifier for (provider, model)."""
    clean_model = model.replace("/", "_").replace(".", "").replace(":", "_")
    return f"{provider}_{clean_model}"


def sample_tag(*, order: str, seed: str, limit: int | None) -> str:
    """Filename prefix describing the sampling regime (e.g. 'n20_hash_86f61cff')."""
    seed_tag = "fileorder" if order == "file" else f"hash_{stable_digest(seed, length=8)}"
    limit_tag = f"n{limit}" if limit is not None else "nall"
    return f"{limit_tag}_{seed_tag}"


def build_matrix_summary(summaries: list[dict], *, run_metadata: dict) -> dict:
    return {"run_metadata": run_metadata, "models": summaries}


async def run_matrix(
    *,
    task_name: str,
    model_specs: list[str],
    default_provider: str,
    spec_factory: Callable[[], Any],
    run_experiment: Callable[..., Awaitable[Any]],
    common_config_kwargs: dict[str, Any],
    sample_limit: int | None,
    sample_order: str,
    sample_seed: str,
    output_dir: Path,
    matrix_summary_path: Path | None = None,
    run_metadata_extras: dict[str, Any] | None = None,
    continue_on_error: bool = False,
    overview_printer: Callable[[dict], None] | None = None,
) -> int:
    """Run `run_experiment` for each model_spec and aggregate results.

    Args:
        task_name: used in run_metadata["task"] and must be filesystem-safe.
        model_specs: list like ["openrouter:gpt-5.4", "openrouter:claude-haiku-4.5"].
        default_provider: fallback provider for bare model names.
        spec_factory: callable returning the experiment spec; invoked once per
            model so any stateful spec setup runs cleanly.
        run_experiment: one of run_binary_experiment / run_pairwise_experiment /
            run_scalar_experiment; called as `run_experiment(spec=..., config=...)`.
        common_config_kwargs: everything that goes into CommonExperimentConfig
            EXCEPT `model`, `provider`, `output`, `summary_output`, `sample_limit`,
            `sample_order`, and `sample_seed` (those are injected per-model).
        sample_limit, sample_order, sample_seed: deterministic sampling config.
        output_dir: where per-model raw/summary files and matrix_summary go.
        matrix_summary_path: defaults to `{output_dir}/matrix_summary.json.gz`.
        run_metadata_extras: merged into matrix_summary's run_metadata.
        continue_on_error: if True, log and skip failed per-model runs.
        overview_printer: optional final callback receiving the matrix_summary dict.

    Returns 0 on success, 1 on first unrecovered failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    run_tag = sample_tag(order=sample_order, seed=sample_seed, limit=sample_limit)

    for spec_str in model_specs:
        provider, model = parse_model_spec(spec_str, default_provider)
        tag = model_tag(provider, model)
        raw_output = output_dir / f"{run_tag}_{tag}.json.gz"
        summary_output = output_dir / f"{run_tag}_{tag}_summary.json.gz"

        print()
        print(f"=== Running {provider}:{model} ===")
        config = CommonExperimentConfig(
            model=model,
            provider=provider,
            output=raw_output,
            summary_output=summary_output,
            sample_limit=sample_limit,
            sample_order=sample_order,
            sample_seed=sample_seed,
            **common_config_kwargs,
        )
        try:
            artifacts = await run_experiment(spec=spec_factory(), config=config)
        except Exception as exc:
            if not continue_on_error:
                print(
                    f"Run failed for {provider}:{model}: {type(exc).__name__}: {exc}"
                )
                return 1
            print(
                f"Skipping failed run for {provider}:{model}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        summaries.append(
            {
                "provider": provider,
                "model": model,
                "raw_output": str(raw_output),
                "summary_output": str(summary_output),
                "summary": artifacts.summary,
            }
        )

    run_metadata: dict[str, Any] = {
        "task": task_name,
        "sample_limit": sample_limit,
        "sample_order": sample_order,
        "sample_seed": sample_seed,
        "models": model_specs,
    }
    if run_metadata_extras:
        run_metadata.update(run_metadata_extras)
    matrix_summary = build_matrix_summary(summaries, run_metadata=run_metadata)

    path = matrix_summary_path or (output_dir / "matrix_summary.json.gz")
    write_json(path, matrix_summary)
    print()
    print(f"Matrix summary saved to {path}")
    if overview_printer is not None:
        overview_printer(matrix_summary)
    return 0
