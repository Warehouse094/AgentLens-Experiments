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
EXTERNAL_CSV = BASE_DIR / "01_raw_data" / "external_agent_results_final.csv"
BASELINES_CSV = BASE_DIR / "01_raw_data" / "baseline_comparison_results.csv"
OUTPUT_JSON = EXTERNAL_VALIDATION_RESULTS_DIR / "21_independent_evaluator_transfer.json"
OUTPUT_CSV = EXTERNAL_VALIDATION_RESULTS_DIR / "21_independent_evaluator_transfer_scores.csv"

MERGE_KEYS = ["episode_id", "agent_variant", "domain", "dimension"]
FROZEN_THRESHOLD = 60.0
METHOD_COLUMNS = {
    "the framework_full": "the framework_full_score",
    "latest_score_only": "baseline_latest_score_only",
    "mean_historical": "baseline_mean_historical",
    "ema_score": "baseline_ema_score",
    "llm_as_judge_only": "baseline_llm_as_judge_only",
    "no_verification": "baseline_no_verification",
}


def build_consensus_frame(external: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    external = external.copy()
    external["external_pass_fail"] = external["external_pass_fail"].astype(str).str.strip().str.lower()
    external["external_fail"] = external["external_pass_fail"].eq("fail").astype(int)

    consensus = (
        external.groupby(MERGE_KEYS)
        .agg(
            n_external_agents=("external_agent", "nunique"),
            external_agents=("external_agent", lambda s: ";".join(sorted(map(str, s.unique())))),
            consensus_n_pass=("external_pass_fail", lambda s: int((s == "pass").sum())),
            consensus_n_fail=("external_pass_fail", lambda s: int((s == "fail").sum())),
            mean_external_score=("external_score", "mean"),
            std_external_score=("external_score", lambda s: float(np.std(s.to_numpy(dtype=float), ddof=0))),
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

    merged = consensus.merge(baselines, on=MERGE_KEYS, how="left")
    return merged


def summarize_method(scores_0_100: pd.Series, fail_indicator: pd.Series) -> dict[str, object]:
    risk = 100 - scores_0_100.to_numpy(dtype=float)
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
        risk = 100 - scores.to_numpy(dtype=float)
        summary[str(key)] = {
            "n": int(len(group)),
            "binary_agreement": round(float(scores.ge(FROZEN_THRESHOLD).eq(fail_indicator.eq(0)).mean()), 4),
            "auroc": round(auc_or_default(fail_indicator.to_numpy(), risk), 4),
            "mean_score": round(float(scores.mean()), 2),
            "fail_rate": round(float(fail_indicator.mean()), 4),
        }
    return summary


def main() -> None:
    ensure_external_dirs()
    external = pd.read_csv(EXTERNAL_CSV)
    baselines = pd.read_csv(BASELINES_CSV)

    consensus_df = build_consensus_frame(external, baselines)
    if consensus_df.empty:
        summary = {
            "experiment": "21_independent_evaluator_transfer",
            "status": "BLOCKED",
            "reason": "No mergeable external evaluator rows were available.",
            "conclusion": "BLOCKED",
        }
        write_json(OUTPUT_JSON, summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    missing_columns = [column for column in METHOD_COLUMNS.values() if column not in consensus_df.columns]
    if missing_columns:
        raise KeyError(f"Missing baseline columns for Exp21: {missing_columns}")

    method_metrics = {
        method: summarize_method(consensus_df[column], consensus_df["external_consensus_fail"])
        for method, column in METHOD_COLUMNS.items()
    }

    external_with_scores = external.merge(
        baselines[MERGE_KEYS + list(METHOD_COLUMNS.values())],
        on=MERGE_KEYS,
        how="left",
    )
    external_with_scores["external_fail"] = external_with_scores["external_pass_fail"].astype(str).str.strip().str.lower().eq("fail").astype(int)

    per_external_agent = {
        method: slice_metrics(external_with_scores, "external_agent", column, "external_fail")
        for method, column in METHOD_COLUMNS.items()
    }
    per_domain = {
        method: slice_metrics(consensus_df, "domain", column, "external_consensus_fail")
        for method, column in METHOD_COLUMNS.items()
    }

    best_baseline_name, best_baseline_agreement = max(
        (
            (method, payload["binary_agreement"])
            for method, payload in method_metrics.items()
            if method != "the framework_full"
        ),
        key=lambda item: item[1],
    )

    best_baseline_auc_name, best_baseline_auc = max(
        (
            (method, payload["auroc"])
            for method, payload in method_metrics.items()
            if method != "the framework_full"
        ),
        key=lambda item: item[1],
    )

    the framework_agent_agreements = [payload["binary_agreement"] for payload in per_external_agent["the framework_full"].values()]
    the framework_domain_agreements = [payload["binary_agreement"] for payload in per_domain["the framework_full"].values()]

    summary = {
        "experiment": "21_independent_evaluator_transfer",
        "source": "package external evaluators with frozen held-out transfer framing",
        "n_episode_dimension_units": int(len(consensus_df)),
        "n_external_judgements": int(len(external_with_scores)),
        "n_external_agents": int(external_with_scores["external_agent"].nunique()),
        "n_domains": int(consensus_df["domain"].nunique()),
        "frozen_threshold": FROZEN_THRESHOLD,
        "target_definition": "Three-agent external consensus at the episode-dimension level; secondary slices reported by individual held-out evaluator and by domain.",
        "consensus_label_counts": {
            "pass": int(consensus_df["external_consensus_pass_fail"].eq("pass").sum()),
            "fail": int(consensus_df["external_consensus_pass_fail"].eq("fail").sum()),
        },
        "consensus_agreement_rate": {
            "mean": round(float(consensus_df["consensus_agreement_rate"].mean()), 4),
            "full_unanimity_rate": round(float(consensus_df["consensus_agreement_rate"].eq(1.0).mean()), 4),
            "min": round(float(consensus_df["consensus_agreement_rate"].min()), 4),
        },
        "methods": method_metrics,
        "per_external_agent": per_external_agent,
        "per_domain": per_domain,
        "the framework_slice_stability": {
            "mean_external_agent_agreement": round(float(np.mean(the framework_agent_agreements)), 4),
            "worst_external_agent_agreement": round(float(np.min(the framework_agent_agreements)), 4),
            "mean_domain_agreement": round(float(np.mean(the framework_domain_agreements)), 4),
            "worst_domain_agreement": round(float(np.min(the framework_domain_agreements)), 4),
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
        "method_note": "Independent evaluator transfer only; this is stronger than static within-package agreement summaries but does not establish open-world benchmark validation.",
        "conclusion": "PASS" if method_metrics["the framework_full"]["binary_agreement"] >= 0.7 else "MARGINAL",
    }

    consensus_df.to_csv(OUTPUT_CSV, index=False)
    write_json(OUTPUT_JSON, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
