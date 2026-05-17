from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[2]
BUNDLE_ROOT = BASE_DIR / "07_revision_bundle" / "agenteval_gov_external_generalization_bundle"

DATA_FILES = [
    "01_raw_data/external_validation/atbench_mapping_coverage.csv",
    "01_raw_data/external_validation/atbench_normalized_episodes.jsonl",
    "01_raw_data/external_validation/atbench_ambiguity_subset.csv",
    "01_raw_data/external_validation/atbench_external_judgements.csv",
    "01_raw_data/external_validation/agent_web3_archived_coverage.csv",
    "01_raw_data/external_validation/agent_web3_archived_adapter_summary.json",
    "01_raw_data/external_validation/atbench_adapter_summary.json",
    "01_raw_data/external_validation/appeal_ambiguity_subset.csv",
    "01_raw_data/external_validation/human_system_divergence_cases.csv",
]

RESULT_FILES = [
    "02_experiment_results/external_validation/19_cross_benchmark_frozen_transfer.json",
    "02_experiment_results/external_validation/19_cross_benchmark_frozen_transfer_scores.csv",
    "02_experiment_results/external_validation/20_cross_source_archived_trace_replication.json",
    "02_experiment_results/external_validation/20_cross_source_archived_trace_replication_scores.csv",
    "02_experiment_results/external_validation/21_independent_evaluator_transfer.json",
    "02_experiment_results/external_validation/21_independent_evaluator_transfer_scores.csv",
    "02_experiment_results/external_validation/22_atbench_ambiguity_stress_test.json",
    "02_experiment_results/external_validation/22_atbench_ambiguity_stress_test_scores.csv",
    "02_experiment_results/external_validation/23_atbench_independent_relabel_transfer.json",
    "02_experiment_results/external_validation/23_atbench_independent_relabel_transfer_scores.csv",
    "02_experiment_results/external_validation/23_atbench_independent_relabel_transfer_leave_one_evaluator_out.csv",
    "02_experiment_results/external_validation/24_appeal_ambiguity_stress_test.json",
    "02_experiment_results/external_validation/24_appeal_ambiguity_stress_test_scores.csv",
    "02_experiment_results/external_validation/25_human_reliability_recalculation.json",
    "02_experiment_results/external_validation/25_human_reliability_recalculation_scores.csv",
    "02_experiment_results/external_validation/26_domain_holdout_transfer.json",
    "02_experiment_results/external_validation/26_domain_holdout_transfer_scores.csv",
    "02_experiment_results/external_validation/27_human_system_divergence_taxonomy.json",
    "02_experiment_results/external_validation/27_human_system_divergence_taxonomy_scores.csv",
]

SCRIPT_FILES = [
    "04_scripts/external_validation/run_exp19_cross_benchmark_transfer.py",
    "04_scripts/external_validation/run_exp20_archived_trace_replication.py",
    "04_scripts/external_validation/run_exp21_independent_evaluator_transfer.py",
    "04_scripts/external_validation/run_exp22_atbench_ambiguity_stress_test.py",
    "04_scripts/external_validation/run_exp23_atbench_independent_relabel_transfer.py",
    "04_scripts/external_validation/run_exp24_appeal_ambiguity_stress_test.py",
    "04_scripts/external_validation/run_exp25_human_reliability_recalculation.py",
    "04_scripts/external_validation/run_exp26_domain_holdout_transfer.py",
    "04_scripts/external_validation/run_exp27_human_system_divergence_taxonomy.py",
    "04_scripts/external_validation/collect_atbench_external_judgements.py",
    "04_scripts/external_validation/build_external_generalization_bundle.py",
]

FIGURE_FILES: list[str] = []

RESULT_JSONS = {
    "Exp19": "02_experiment_results/external_validation/19_cross_benchmark_frozen_transfer.json",
    "Exp20": "02_experiment_results/external_validation/20_cross_source_archived_trace_replication.json",
    "Exp21": "02_experiment_results/external_validation/21_independent_evaluator_transfer.json",
    "Exp22": "02_experiment_results/external_validation/22_atbench_ambiguity_stress_test.json",
    "Exp23": "02_experiment_results/external_validation/23_atbench_independent_relabel_transfer.json",
    "Exp24": "02_experiment_results/external_validation/24_appeal_ambiguity_stress_test.json",
    "Exp25": "02_experiment_results/external_validation/25_human_reliability_recalculation.json",
    "Exp26": "02_experiment_results/external_validation/26_domain_holdout_transfer.json",
    "Exp27": "02_experiment_results/external_validation/27_human_system_divergence_taxonomy.json",
}

EXPERIMENT_DESCRIPTIONS = {
    "Exp19": "external benchmark transfer under frozen mechanical mapping",
    "Exp20": "external archived-source executability and boundary evidence",
    "Exp21": "independent evaluator transfer on package episodes",
    "Exp22": "ambiguity-bearing ATBench hard-slice stress test",
    "Exp23": "ATBench ambiguity subset with fresh independent evaluator consensus",
    "Exp24": "appeal ambiguity stress test on mixed-evidence governance cases",
    "Exp25": "human reliability recalculation with corrected bootstrap intervals",
    "Exp26": "domain-held-out external-consensus transfer readout",
    "Exp27": "human-system divergence taxonomy over mismatch cases",
}


def copy_files(relative_paths: list[str], target_dir_name: str) -> list[str]:
    copied: list[str] = []
    target_dir = BUNDLE_ROOT / target_dir_name
    target_dir.mkdir(parents=True, exist_ok=True)
    for rel in relative_paths:
        source = BASE_DIR / rel
        if not source.exists():
            continue
        destination = target_dir / Path(rel).name
        shutil.copy2(source, destination)
        copied.append(str(destination))
    return copied


def load_json(relative_path: str) -> dict[str, Any] | None:
    source = BASE_DIR / relative_path
    if not source.exists():
        return None
    with source.open() as handle:
        return json.load(handle)


def build_summary_markdown(results: dict[str, dict[str, Any] | None]) -> str:
    exp25 = results.get("Exp25") or {}
    exp26 = results.get("Exp26") or {}
    exp27 = results.get("Exp27") or {}

    exp25_kappa_ci  = exp25.get("cohens_kappa_ci", {})
    exp26_best      = exp26.get("best_baseline_overall", {})
    exp26_auc_range = exp26.get("agenteval_gov_domain_auroc_range", {})
    exp27_mismatch  = exp27.get("mismatch_direction", {})
    exp27_taxonomy  = exp27.get("taxonomy_counts", {})

    return f"""# AgentEval-Gov: External Generalization Bundle Summary

## Scope
This bundle packages the external-validation layer (Exp19--27) for the AgentEval-Gov submission.
It provides benchmark transfer, archived-source replication, independent evaluator transfer,
ambiguity stress tests, corrected human reliability statistics, per-domain readout, and
a structured human-system divergence taxonomy.

## Experiment summaries

**Exp19** — cross-benchmark frozen transfer on 1,000 mechanically mapped ATBench trajectories.

**Exp20** — archived independent-source executability over agent-web3 traces.

**Exp21** — independent external-consensus transfer on 450 episode-dimension units.

**Exp22** — ambiguity-bearing ATBench hard-slice stress test.

**Exp23** — fresh three-evaluator relabel transfer on the ATBench ambiguity subset.

**Exp24** — appeal ambiguity stress test on mixed-evidence governance cases.

**Exp25** — human reliability recalculation with corrected bootstrap confidence intervals.
Pairwise binary agreement = {exp25.get('pairwise_binary_agreement', 'N/A')},
Cohen's kappa = {exp25.get('cohens_kappa', 'N/A')}
(95% CI {exp25_kappa_ci.get('lower', 'N/A')}--{exp25_kappa_ci.get('upper', 'N/A')}),
full unanimity rate = {exp25.get('full_unanimity_rate', 'N/A')}.

**Exp26** — domain-held-out external-consensus transfer readout.
Overall AUROC = {exp26.get('agenteval_gov_overall_auroc', 'N/A')},
per-domain AUROC range = {exp26_auc_range.get('min', 'N/A')}--{exp26_auc_range.get('max', 'N/A')}.
Best baseline: {exp26_best.get('name', 'N/A')} (AUROC = {exp26_best.get('auroc', 'N/A')}).

**Exp27** — human-system divergence taxonomy over {exp27.get('n_divergence_cases', 'N/A')} divergence cases.
Dominant direction: system-fail / human-pass
({exp27_mismatch.get('system_fail_human_pass', 'N/A')} cases vs.
{exp27_mismatch.get('system_pass_human_fail', 'N/A')} in the reverse direction).
Leading taxonomy categories: environment/execution uncertainty
({exp27_taxonomy.get('environment_or_execution_uncertainty', 'N/A')}) and
threshold/uncertainty mismatch ({exp27_taxonomy.get('threshold_or_uncertainty_mismatch', 'N/A')}).

## Notes
- Canonical source paths remain under `01_raw_data/external_validation/`,
  `02_experiment_results/external_validation/`, and `04_scripts/external_validation/`.
- ATBench is an independently constructed public dataset (Apache-2.0 license,
  HuggingFace: AI45Research/ATBench). It was not built by the authors of this paper.
"""


def main() -> None:
    for child in ["data", "results", "scripts", "figures", "logs"]:
        (BUNDLE_ROOT / child).mkdir(parents=True, exist_ok=True)

    copied_data    = copy_files(DATA_FILES,    "data")
    copied_results = copy_files(RESULT_FILES,  "results")
    copied_scripts = copy_files(SCRIPT_FILES,  "scripts")
    copied_figures = copy_files(FIGURE_FILES,  "figures")

    result_payloads = {name: load_json(path) for name, path in RESULT_JSONS.items()}

    matrix = {
        "bundle": "agenteval_gov_external_generalization_bundle",
        "canonical_source_root": str(BASE_DIR),
        "experiments": EXPERIMENT_DESCRIPTIONS,
        "copied": {
            "data":    copied_data,
            "results": copied_results,
            "scripts": copied_scripts,
            "figures": copied_figures,
        },
        "missing_expected_results": [
            rel for rel in RESULT_FILES if not (BASE_DIR / rel).exists()
        ],
    }

    with (BUNDLE_ROOT / "results" / "external_generalization_matrix.json").open("w") as handle:
        json.dump(matrix, handle, indent=2, ensure_ascii=False)

    readme = """AgentEval-Gov external generalization bundle

This standalone folder mirrors the external-validation layer from the main submission package.

Canonical source paths remain under:
- 01_raw_data/external_validation/
- 02_experiment_results/external_validation/
- 04_scripts/external_validation/

Mirrored experiments in this bundle:
- Exp19: external benchmark transfer under frozen mechanical mapping
- Exp20: archived-source executability / boundary evidence
- Exp21: independent evaluator transfer on package episodes
- Exp22: ambiguity-bearing ATBench hard-slice stress test
- Exp23: ATBench ambiguity subset with fresh independent evaluator consensus
- Exp24: appeal ambiguity stress test on mixed-evidence governance cases
- Exp25: human reliability recalculation with corrected bootstrap intervals
- Exp26: domain-held-out external-consensus transfer readout
- Exp27: human-system divergence taxonomy over mismatch cases

Key synthesis files:
- results/external_generalization_matrix.json
- EXTERNAL_GENERALIZATION_SUMMARY.md
"""
    (BUNDLE_ROOT / "README.txt").write_text(readme)
    (BUNDLE_ROOT / "EXTERNAL_GENERALIZATION_SUMMARY.md").write_text(
        build_summary_markdown(result_payloads)
    )

    print(json.dumps(matrix, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
