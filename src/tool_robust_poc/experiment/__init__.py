"""Package-level experiment runners and shared execution helpers."""

from tool_robust_poc.experiment.binary import (
    BinaryExperimentArtifacts,
    BinaryExperimentSpec,
    run_binary_experiment,
)
from tool_robust_poc.experiment.common import (
    CommonExperimentConfig,
    ExperimentRuntime,
    POC_ROOT,
    RESULTS_DIR,
    SampledItem,
    build_case_metadata,
    build_run_metadata,
    build_runtime,
    default_results_path,
    sample_items,
    sample_tag,
    shutdown_runtime,
    summary_path,
)
from tool_robust_poc.experiment.gsm8k import (
    DEFAULT_DATA as GSM8K_DEFAULT_DATA,
    build_gsm8k_spec,
    expected_gsm8k_verdict,
)
from tool_robust_poc.experiment.matrix import run_matrix
from tool_robust_poc.experiment.pairwise import (
    PairwiseExperimentArtifacts,
    PairwiseExperimentSpec,
    run_pairwise_experiment,
)
from tool_robust_poc.experiment.scalar import (
    ScalarExperimentArtifacts,
    ScalarExperimentSpec,
    run_scalar_experiment,
)

__all__ = [
    "BinaryExperimentArtifacts",
    "BinaryExperimentSpec",
    "CommonExperimentConfig",
    "ExperimentRuntime",
    "GSM8K_DEFAULT_DATA",
    "PairwiseExperimentArtifacts",
    "PairwiseExperimentSpec",
    "POC_ROOT",
    "RESULTS_DIR",
    "SampledItem",
    "ScalarExperimentArtifacts",
    "ScalarExperimentSpec",
    "build_case_metadata",
    "build_gsm8k_spec",
    "build_run_metadata",
    "build_runtime",
    "default_results_path",
    "expected_gsm8k_verdict",
    "run_binary_experiment",
    "run_matrix",
    "run_pairwise_experiment",
    "run_scalar_experiment",
    "sample_items",
    "sample_tag",
    "shutdown_runtime",
    "summary_path",
]
