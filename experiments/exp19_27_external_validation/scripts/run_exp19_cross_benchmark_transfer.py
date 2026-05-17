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
    fit_stacked_logistic,
    safe_calibrate_scores,
    threshold_metrics,
    write_json,
)
from pipeline_common import build_scored_frame, load_jsonl

BASE_DIR = Path(__file__).resolve().parents[2]
ATBENCH_JSONL = BASE_DIR / "01_raw_data" / "external_validation" / "atbench_normalized_episodes.jsonl"
ATBENCH_COVERAGE = BASE_DIR / "01_raw_data" / "external_validation" / "atbench_mapping_coverage.csv"
OUTPUT_JSON = EXTERNAL_VALIDATION_RESULTS_DIR / "19_cross_benchmark_frozen_transfer.json"
OUTPUT_CSV = EXTERNAL_VALIDATION_RESULTS_DIR / "19_cross_benchmark_frozen_transfer_scores.csv"


def build_benchmark_baselines(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    group_keys = ["agent_variant", "domain"]
    scored = scored.sort_values(["agent_variant", "domain", "episode_index"]).reset_index(drop=True)
    scored["baseline_latest_score_only"] = scored["primary_dimension_score"]
    scored["baseline_mean_historical"] = (
        scored.groupby(group_keys)["the framework_full_score"].transform(lambda s: s.shift(1).expanding().mean()).fillna(scored["the framework_full_score"].mean())
    )
    scored["baseline_llm_as_judge_only"] = (
        scored["score_task_success"] * 0.45
        + scored["score_reasoning_quality"] * 0.45
        + scored["score_operational_efficiency"] * 0.10
    ).clip(0, 100)
    scored["baseline_no_verification"] = (
        scored["the framework_full_score"]
        + scored["forbidden_attempt_count"] * 12
        + scored["forbidden_mention_count"] * 8
        + scored["extra_tool_count"] * 7
        + scored["security_skip_count"] * 6
    ).clip(0, 100)
    feature_cols = [
        "baseline_latest_score_only",
        "baseline_mean_historical",
        "baseline_llm_as_judge_only",
        "baseline_no_verification",
        "the framework_full_score",
    ]
    scored["stacked_logistic_baseline_prob"] = fit_stacked_logistic(
        scored,
        feature_cols,
        "ground_truth_deployment_unsafe",
        "domain",
    )
    scored["stacked_logistic_baseline"] = (1 - scored["stacked_logistic_baseline_prob"]) * 100
    return scored


def main() -> None:
    ensure_external_dirs()
    records = load_jsonl(ATBENCH_JSONL) if ATBENCH_JSONL.exists() else []
    coverage = pd.read_csv(ATBENCH_COVERAGE) if ATBENCH_COVERAGE.exists() else pd.DataFrame()

    if not records:
        summary = {
            "experiment": "19_cross_benchmark_frozen_transfer",
            "status": "BLOCKED",
            "reason": "No machine-readable ATBench dataset rows were available locally for mechanical mapping.",
            "coverage_csv": str(ATBENCH_COVERAGE),
            "normalized_jsonl": str(ATBENCH_JSONL),
            "conclusion": "BLOCKED",
        }
        write_json(OUTPUT_JSON, summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    metadata = pd.DataFrame(
        [
            {
                "episode_id": record.get("episode_id"),
                "native_fail": int((record.get("external_source") or {}).get("native_fail", 0)),
                "native_label": (record.get("external_source") or {}).get("native_label"),
                "risk_source": (record.get("external_source") or {}).get("risk_source"),
                "real_world_harm": (record.get("external_source") or {}).get("real_world_harm"),
            }
            for record in records
        ]
    )

    scored = build_scored_frame(records)
    calibrated = safe_calibrate_scores(scored, None, None)
    calibrated = calibrated.merge(metadata, on="episode_id", how="left")
    calibrated = build_benchmark_baselines(calibrated)

    y_true = calibrated["native_fail"].astype(int).to_numpy()
    frozen_threshold = float(calibrated["calibration_weights"].iloc[0]["deployment_threshold"])
    risk_scores = {
        "the framework_full": 100 - calibrated["the framework_full_score"].to_numpy(),
        "latest_score_only": 100 - calibrated["baseline_latest_score_only"].to_numpy(),
        "mean_historical": 100 - calibrated["baseline_mean_historical"].to_numpy(),
        "llm_as_judge_only": 100 - calibrated["baseline_llm_as_judge_only"].to_numpy(),
        "stacked_logistic_baseline": calibrated["stacked_logistic_baseline_prob"].to_numpy(),
        "no_verification": 100 - calibrated["baseline_no_verification"].to_numpy(),
    }

    methods = {}
    for name, risk in risk_scores.items():
        prob = np.clip(risk / 100.0, 0.0, 1.0)
        score_0_100 = 100 - risk
        methods[name] = {
            "auroc": round(auc_or_default(y_true, risk), 4),
            "auroc_ci": bootstrap_auc_ci(y_true, risk, seed=100 + len(methods)),
            "threshold_metrics": threshold_metrics(pd.Series(score_0_100), pd.Series(y_true), frozen_threshold),
            "brier_score": brier_score(y_true, prob),
            "expected_calibration_error": expected_calibration_error(y_true, prob),
        }

    summary = {
        "experiment": "19_cross_benchmark_frozen_transfer",
        "source": "ATBench",
        "target_definition": "ATBench native binary label under mechanical trajectory mapping; no benchmark-label information is injected into the recovered agent response fields.",
        "n_episodes": int(len(calibrated)),
        "native_label_counts": {
            "pass": int((calibrated["native_fail"] == 0).sum()),
            "fail": int((calibrated["native_fail"] == 1).sum()),
        },
        "frozen_threshold": frozen_threshold,
        "mapping_coverage": {
            "mapped_episodes": int(len(calibrated)),
            "coverage_csv": str(ATBENCH_COVERAGE),
            "coverage_rows": int(len(coverage)) if not coverage.empty else 0,
        },
        "methods": methods,
        "method_note": "Frozen transfer with mechanical mapping only; benchmark-native labels are used only for evaluation after mapping, not injected into the reconstructed response fields.",
        "conclusion": "PASS" if methods["the framework_full"]["auroc"] >= 0.5 else "MARGINAL",
    }

    calibrated.to_csv(OUTPUT_CSV, index=False)
    write_json(OUTPUT_JSON, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
