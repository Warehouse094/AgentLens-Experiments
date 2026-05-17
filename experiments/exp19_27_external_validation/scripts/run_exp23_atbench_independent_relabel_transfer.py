from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common import (
    EXTERNAL_VALIDATION_RAW_DIR,
    EXTERNAL_VALIDATION_RESULTS_DIR,
    auc_or_default,
    bootstrap_auc_ci,
    brier_score,
    ensure_external_dirs,
    expected_calibration_error,
    threshold_metrics,
    write_json,
)

BASE_DIR = Path(__file__).resolve().parents[2]
JUDGEMENTS_CSV = EXTERNAL_VALIDATION_RAW_DIR / "atbench_external_judgements.csv"
SUBSET_CSV = EXTERNAL_VALIDATION_RAW_DIR / "atbench_ambiguity_subset.csv"
EXP22_SCORES_CSV = EXTERNAL_VALIDATION_RESULTS_DIR / "22_atbench_ambiguity_stress_test_scores.csv"
OUTPUT_JSON = EXTERNAL_VALIDATION_RESULTS_DIR / "23_atbench_independent_relabel_transfer.json"
OUTPUT_CSV = EXTERNAL_VALIDATION_RESULTS_DIR / "23_atbench_independent_relabel_transfer_scores.csv"
OUTPUT_LOO_CSV = EXTERNAL_VALIDATION_RESULTS_DIR / "23_atbench_independent_relabel_transfer_leave_one_evaluator_out.csv"

MERGE_KEYS = ["episode_id", "agent_variant", "domain", "dimension"]
FROZEN_THRESHOLD = 60.0
METHOD_COLUMNS = {
    "the framework_full": "the framework_full_score",
    "latest_score_only": "baseline_latest_score_only",
    "mean_historical": "baseline_mean_historical",
    "llm_as_judge_only": "baseline_llm_as_judge_only",
    "no_verification": "baseline_no_verification",
    "stacked_logistic_baseline": "stacked_logistic_baseline",
}


def build_consensus_frame(external: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    external = external.copy()
    external["external_pass_fail"] = external["external_pass_fail"].astype(str).str.strip().str.lower()
    external["external_fail"] = external["external_pass_fail"].eq("fail").astype(int)
    external["external_score"] = pd.to_numeric(external["external_score"], errors="coerce")

    consensus = (
        external.groupby(MERGE_KEYS)
        .agg(
            n_external_agents=("external_agent", "nunique"),
            external_agents=("external_agent", lambda s: ";".join(sorted(map(str, s.dropna().unique())))),
            consensus_n_pass=("external_pass_fail", lambda s: int((s == "pass").sum())),
            consensus_n_fail=("external_pass_fail", lambda s: int((s == "fail").sum())),
            mean_external_score=("external_score", "mean"),
            std_external_score=("external_score", lambda s: float(np.std(pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float), ddof=0)) if pd.to_numeric(s, errors="coerce").notna().any() else 0.0),
        )
        .reset_index()
    )
    consensus["external_consensus_pass_fail"] = np.where(
        consensus["consensus_n_fail"] > consensus["consensus_n_pass"],
        "fail",
        "pass",
    )
    consensus["external_consensus_fail"] = consensus["external_consensus_pass_fail"].eq("fail").astype(int)
    consensus["consensus_agreement_rate"] = (
        consensus[["consensus_n_pass", "consensus_n_fail"]].max(axis=1) / consensus["n_external_agents"]
    ).round(4)
    return consensus.merge(baselines, on=MERGE_KEYS, how="left")


def summarize_method(scores_0_100: pd.Series, fail_indicator: pd.Series, *, score_col_name: str | None = None) -> dict[str, object]:
    score_name = score_col_name or getattr(scores_0_100, "name", "")
    risk = scores_0_100.to_numpy(dtype=float) if score_name == "stacked_logistic_baseline" else 100 - scores_0_100.to_numpy(dtype=float)
    y_true = fail_indicator.astype(int).to_numpy()
    prob = np.clip(risk / 100.0, 0.0, 1.0)
    threshold_summary = threshold_metrics(scores_0_100, fail_indicator, FROZEN_THRESHOLD)
    agreement = float(scores_0_100.ge(FROZEN_THRESHOLD).eq(fail_indicator.eq(0)).mean())
    return {
        "auroc": round(auc_or_default(y_true, risk), 4),
        "auroc_ci": bootstrap_auc_ci(y_true, risk),
        "threshold_metrics": threshold_summary,
        "binary_agreement": round(agreement, 4),
        "brier_score": brier_score(y_true, prob),
        "expected_calibration_error": expected_calibration_error(y_true, prob),
        "mean_score": round(float(scores_0_100.mean()), 2),
    }


def slice_metrics(df: pd.DataFrame, group_col: str, score_col: str, target_col: str) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key, group in df.groupby(group_col):
        scores = group[score_col]
        fail_indicator = group[target_col].astype(int)
        risk = scores.to_numpy(dtype=float) if score_col == "stacked_logistic_baseline" else 100 - scores.to_numpy(dtype=float)
        summary[str(key)] = {
            "n": int(len(group)),
            "binary_agreement": round(float(scores.ge(FROZEN_THRESHOLD).eq(fail_indicator.eq(0)).mean()), 4),
            "auroc": round(auc_or_default(fail_indicator.to_numpy(), risk), 4),
            "mean_score": round(float(scores.mean()), 2),
            "fail_rate": round(float(fail_indicator.mean()), 4),
        }
    return summary


def evaluate_methods(df: pd.DataFrame, target_col: str) -> dict[str, dict[str, object]]:
    return {
        method: summarize_method(df[column], df[target_col], score_col_name=column)
        for method, column in METHOD_COLUMNS.items()
    }


def leave_one_evaluator_out(consensus_df: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    evaluators = sorted(map(str, external["external_agent"].dropna().unique()))
    for evaluator in evaluators:
        remaining = external[external["external_agent"].astype(str) != evaluator].copy()
        if remaining.empty or remaining["external_agent"].nunique() < 2:
            continue
        loo_consensus = build_consensus_frame(remaining, consensus_df.drop(columns=[
            "n_external_agents",
            "external_agents",
            "consensus_n_pass",
            "consensus_n_fail",
            "mean_external_score",
            "std_external_score",
            "external_consensus_pass_fail",
            "external_consensus_fail",
            "consensus_agreement_rate",
        ], errors="ignore").drop_duplicates(MERGE_KEYS))
        for method, column in METHOD_COLUMNS.items():
            metrics = summarize_method(loo_consensus[column], loo_consensus["external_consensus_fail"], score_col_name=column)
            rows.append({
                "held_out_evaluator": evaluator,
                "remaining_evaluators": int(remaining["external_agent"].nunique()),
                "n_units": int(len(loo_consensus)),
                "method": method,
                "binary_agreement": metrics["binary_agreement"],
                "auroc": metrics["auroc"],
                "mean_score": metrics["mean_score"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    ensure_external_dirs()
    if not JUDGEMENTS_CSV.exists():
        summary = {
            "experiment": "23_atbench_independent_relabel_transfer",
            "status": "BLOCKED",
            "reason": f"External judgement file not found at {JUDGEMENTS_CSV}",
            "required_input": str(JUDGEMENTS_CSV),
            "conclusion": "BLOCKED",
        }
        write_json(OUTPUT_JSON, summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    external = pd.read_csv(JUDGEMENTS_CSV)
    if external.empty:
        summary = {
            "experiment": "23_atbench_independent_relabel_transfer",
            "status": "BLOCKED",
            "reason": "The ATBench external judgement file exists but contains no judgement rows.",
            "required_input": str(JUDGEMENTS_CSV),
            "template_only": True,
            "conclusion": "BLOCKED",
        }
        write_json(OUTPUT_JSON, summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    subset = pd.read_csv(SUBSET_CSV)
    exp22_scores = pd.read_csv(EXP22_SCORES_CSV)

    missing_external_cols = [col for col in MERGE_KEYS + ["external_agent", "external_score", "external_pass_fail"] if col not in external.columns]
    if missing_external_cols:
        raise KeyError(f"Missing required columns in {JUDGEMENTS_CSV.name}: {missing_external_cols}")

    base_cols = list(dict.fromkeys(MERGE_KEYS + [
        "selection_bucket",
        "selection_reason",
        "is_hard_case",
        "native_fail",
        "native_label_pass_fail",
        "risk_source",
        "real_world_harm",
        *METHOD_COLUMNS.values(),
    ]))
    missing_base_cols = [col for col in base_cols if col not in exp22_scores.columns]
    if missing_base_cols:
        raise KeyError(f"Missing baseline columns in Exp22 scores: {missing_base_cols}")

    baselines = exp22_scores[base_cols].drop_duplicates(MERGE_KEYS)
    consensus_df = build_consensus_frame(external, baselines)
    if consensus_df.empty:
        summary = {
            "experiment": "23_atbench_independent_relabel_transfer",
            "status": "BLOCKED",
            "reason": "No mergeable external judgements matched the Exp22 ATBench subset.",
            "required_input": str(JUDGEMENTS_CSV),
            "subset_csv": str(SUBSET_CSV),
            "conclusion": "BLOCKED",
        }
        write_json(OUTPUT_JSON, summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    missing_method_cols = [column for column in METHOD_COLUMNS.values() if column not in consensus_df.columns]
    if missing_method_cols:
        raise KeyError(f"Missing method columns after merge for Exp23: {missing_method_cols}")

    method_metrics = evaluate_methods(consensus_df, "external_consensus_fail")
    per_domain = {method: slice_metrics(consensus_df, "domain", column, "external_consensus_fail") for method, column in METHOD_COLUMNS.items()}
    per_selection_bucket = {method: slice_metrics(consensus_df, "selection_bucket", column, "external_consensus_fail") for method, column in METHOD_COLUMNS.items()}

    hard_slice = consensus_df[consensus_df["is_hard_case"].astype(int) == 1].copy()
    hard_slice_methods = evaluate_methods(hard_slice, "external_consensus_fail") if not hard_slice.empty else {}

    loo_df = leave_one_evaluator_out(consensus_df, external)
    if not loo_df.empty:
        loo_df.to_csv(OUTPUT_LOO_CSV, index=False)

    best_baseline_name, best_baseline_agreement = max(
        ((method, payload["binary_agreement"]) for method, payload in method_metrics.items() if method != "the framework_full"),
        key=lambda item: item[1],
    )
    best_baseline_auc_name, best_baseline_auc = max(
        ((method, payload["auroc"]) for method, payload in method_metrics.items() if method != "the framework_full"),
        key=lambda item: item[1],
    )

    consensus_df.to_csv(OUTPUT_CSV, index=False)

    summary = {
        "experiment": "23_atbench_independent_relabel_transfer",
        "source": "ATBench ambiguity subset with fresh independent external evaluator judgements",
        "target_definition": "Three-evaluator external consensus on the Exp22 ATBench subset; this evaluates external benchmark episodes against a fresh evaluator target rather than only the benchmark-native label.",
        "n_episode_dimension_units": int(len(consensus_df)),
        "n_external_judgements": int(len(external)),
        "n_external_agents": int(external["external_agent"].nunique()),
        "n_domains": int(consensus_df["domain"].nunique()),
        "frozen_threshold": FROZEN_THRESHOLD,
        "subset_profile": {
            "subset_csv": str(SUBSET_CSV),
            "n_subset_rows": int(len(subset)),
            "n_matched_rows": int(len(consensus_df)),
            "selection_bucket_counts": {str(k): int(v) for k, v in consensus_df["selection_bucket"].value_counts().to_dict().items()},
            "native_label_counts": {
                "pass": int((consensus_df["native_fail"].astype(int) == 0).sum()),
                "fail": int((consensus_df["native_fail"].astype(int) == 1).sum()),
            },
        },
        "consensus_profile": {
            "consensus_label_counts": {
                "pass": int(consensus_df["external_consensus_pass_fail"].eq("pass").sum()),
                "fail": int(consensus_df["external_consensus_pass_fail"].eq("fail").sum()),
            },
            "agreement_rate": {
                "mean": round(float(consensus_df["consensus_agreement_rate"].mean()), 4),
                "full_unanimity_rate": round(float(consensus_df["consensus_agreement_rate"].eq(1.0).mean()), 4),
                "min": round(float(consensus_df["consensus_agreement_rate"].min()), 4),
            },
            "mean_external_score": round(float(consensus_df["mean_external_score"].mean()), 4),
            "mean_external_score_dispersion": round(float(consensus_df["std_external_score"].mean()), 4),
        },
        "methods": method_metrics,
        "per_domain": per_domain,
        "per_selection_bucket": per_selection_bucket,
        "hard_slice_methods": hard_slice_methods,
        "leave_one_evaluator_out": {
            "csv": str(OUTPUT_LOO_CSV),
            "n_rows": int(len(loo_df)),
        },
        "best_baseline_by_agreement": {
            "name": best_baseline_name,
            "binary_agreement": best_baseline_agreement,
            "delta_vs_the framework": round(method_metrics["the framework_full"]["binary_agreement"] - best_baseline_agreement, 4),
        },
        "best_baseline_by_auroc": {
            "name": best_baseline_auc_name,
            "auroc": best_baseline_auc,
            "delta_vs_the framework": round(method_metrics["the framework_full"]["auroc"] - best_baseline_auc, 4),
        },
        "method_note": "This experiment is stronger than Exp19, Exp21, or Exp22 alone because it combines an external benchmark source, an ambiguity-bearing subset, and fresh independent evaluator consensus. It still remains bounded: the subset is finite and curated, the evaluation is not live deployment telemetry, and the result does not establish open-world robustness or a guarantee of external generalization.",
        "conclusion": "PASS" if method_metrics["the framework_full"]["auroc"] >= 0.7 else "MARGINAL",
    }
    write_json(OUTPUT_JSON, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
