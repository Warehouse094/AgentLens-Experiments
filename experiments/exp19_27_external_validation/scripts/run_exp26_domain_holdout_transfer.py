from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common import EXTERNAL_VALIDATION_RESULTS_DIR, bootstrap_auc_ci, ensure_external_dirs, threshold_metrics, write_json

EXP21_CSV = EXTERNAL_VALIDATION_RESULTS_DIR / "21_independent_evaluator_transfer_scores.csv"
OUTPUT_JSON = EXTERNAL_VALIDATION_RESULTS_DIR / "26_domain_holdout_transfer.json"
OUTPUT_CSV = EXTERNAL_VALIDATION_RESULTS_DIR / "26_domain_holdout_transfer_scores.csv"
FROZEN_THRESHOLD = 60.0

METHOD_COLUMNS = {
    "the framework_full": "the framework_full_score",
    "latest_score_only": "baseline_latest_score_only",
    "mean_historical": "baseline_mean_historical",
    "llm_as_judge_only": "baseline_llm_as_judge_only",
    "no_verification": "baseline_no_verification",
    "human_only_review": "baseline_human_only_review",
}


def safe_auc(y_true: np.ndarray, scores_0_100: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    risk = 100 - scores_0_100
    return float(roc_auc_score(y_true, risk))


def summarize_holdout(group: pd.DataFrame) -> dict[str, dict[str, object]]:
    y_true = group["external_consensus_fail"].astype(int).to_numpy()
    summary: dict[str, dict[str, object]] = {}
    for method, column in METHOD_COLUMNS.items():
        scores = group[column].to_numpy(dtype=float)
        summary[method] = {
            "n": int(len(group)),
            "fail_rate": round(float(np.mean(y_true)), 4),
            "binary_agreement": round(float(group[column].ge(FROZEN_THRESHOLD).eq(group["external_consensus_fail"].eq(0)).mean()), 4),
            "auroc": round(safe_auc(y_true, scores), 4),
            "auroc_ci": bootstrap_auc_ci(y_true, 100 - scores),
            "threshold_metrics": threshold_metrics(group[column], group["external_consensus_fail"], FROZEN_THRESHOLD),
            "mean_score": round(float(np.mean(scores)), 4),
        }
    return summary


def main() -> None:
    ensure_external_dirs()
    if not EXP21_CSV.exists():
        summary = {
            "experiment": "26_domain_holdout_transfer",
            "status": "BLOCKED",
            "reason": f"Exp21 score table not found at {EXP21_CSV}",
            "conclusion": "BLOCKED",
        }
        write_json(OUTPUT_JSON, summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    df = pd.read_csv(EXP21_CSV)
    if df.empty:
        summary = {
            "experiment": "26_domain_holdout_transfer",
            "status": "BLOCKED",
            "reason": "Exp21 score table exists but is empty.",
            "conclusion": "BLOCKED",
        }
        write_json(OUTPUT_JSON, summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    df.to_csv(OUTPUT_CSV, index=False)
    per_domain = {}
    degradation_rows = []
    overall = summarize_holdout(df)
    overall_the framework_auc = overall["the framework_full"]["auroc"]

    for domain, group in df.groupby("domain"):
        domain_summary = summarize_holdout(group)
        per_domain[str(domain)] = domain_summary
        best_baseline_name, best_baseline_auc = max(
            ((method, payload["auroc"]) for method, payload in domain_summary.items() if method != "the framework_full"),
            key=lambda item: item[1],
        )
        degradation_rows.append(
            {
                "held_out_domain": str(domain),
                "n": int(len(group)),
                "the framework_full_auroc": domain_summary["the framework_full"]["auroc"],
                "overall_the framework_auroc": overall_the framework_auc,
                "delta_from_overall": round(domain_summary["the framework_full"]["auroc"] - overall_the framework_auc, 4),
                "best_baseline": best_baseline_name,
                "best_baseline_auroc": best_baseline_auc,
                "delta_vs_best_baseline": round(domain_summary["the framework_full"]["auroc"] - best_baseline_auc, 4),
                "the framework_binary_agreement": domain_summary["the framework_full"]["binary_agreement"],
            }
        )

    degradation_df = pd.DataFrame(degradation_rows)
    best_baseline_name, best_baseline_auc = max(
        ((method, payload["auroc"]) for method, payload in overall.items() if method != "the framework_full"),
        key=lambda item: item[1],
    )

    summary = {
        "experiment": "26_domain_holdout_transfer",
        "source": "Domain-held-out readout over the independent external-consensus transfer table from Exp21",
        "target_definition": "Measure whether external-consensus transfer degrades gracefully or collapses when read one domain at a time rather than only in aggregate.",
        "n_episode_dimension_units": int(len(df)),
        "n_domains": int(df["domain"].nunique()),
        "frozen_threshold": FROZEN_THRESHOLD,
        "overall_methods": overall,
        "per_domain": per_domain,
        "domain_holdout_table": degradation_rows,
        "the framework_overall_auroc": overall_the framework_auc,
        "the framework_domain_auroc_range": {
            "min": round(float(degradation_df["the framework_full_auroc"].min()), 4),
            "max": round(float(degradation_df["the framework_full_auroc"].max()), 4),
            "mean": round(float(degradation_df["the framework_full_auroc"].mean()), 4),
        },
        "the framework_delta_from_overall_range": {
            "min": round(float(degradation_df["delta_from_overall"].min()), 4),
            "max": round(float(degradation_df["delta_from_overall"].max()), 4),
            "mean": round(float(degradation_df["delta_from_overall"].mean()), 4),
        },
        "best_baseline_overall": {
            "name": best_baseline_name,
            "auroc": best_baseline_auc,
            "delta_vs_the framework": round(overall_the framework_auc - best_baseline_auc, 4),
        },
        "domains_where_the framework_beats_best_baseline": int((degradation_df["delta_vs_best_baseline"] > 0).sum()),
        "method_note": "This experiment improves the domain-scope story because it shows how external-consensus transfer behaves per domain rather than only in aggregate. But it is still a within-package holdout analysis over five Web3 governance domains, not proof of open-world cross-domain robustness.",
        "conclusion": "PASS" if float(degradation_df["the framework_full_auroc"].min()) >= 0.55 else "MARGINAL",
    }
    write_json(OUTPUT_JSON, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
