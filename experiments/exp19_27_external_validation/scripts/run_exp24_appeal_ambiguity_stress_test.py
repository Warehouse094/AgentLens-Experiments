from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common import EXTERNAL_VALIDATION_RAW_DIR, EXTERNAL_VALIDATION_RESULTS_DIR, ensure_external_dirs, write_json
from pipeline_common import RAW_DATA_FILES, percentile_interval

OUTPUT_SUBSET_CSV = EXTERNAL_VALIDATION_RAW_DIR / "appeal_ambiguity_subset.csv"
OUTPUT_JSON = EXTERNAL_VALIDATION_RESULTS_DIR / "24_appeal_ambiguity_stress_test.json"
OUTPUT_CSV = EXTERNAL_VALIDATION_RESULTS_DIR / "24_appeal_ambiguity_stress_test_scores.csv"

BOOTSTRAP_ITERATIONS = 2000
AMBIGUITY_THRESHOLD = 2

CRITERIA_PATTERNS = {
    "criterion_environment_instability": r"environment|timeout|network|truncated|latency",
    "criterion_false_positive_or_misclassification": r"false positive|misclass|incorrectly|partial match|semantically different",
    "criterion_uncertainty_or_borderline": r"uncertain|uncertainty|ambiguous|borderline|unclear",
    "criterion_evidence_conflict": r"but|however|although|despite|yet",
    "criterion_missing_or_weak_evidence": r"missing|insufficient|cannot verify|not enough evidence|lacks evidence",
}


def normalized_entropy(labels: pd.Series) -> float:
    counts = labels.value_counts(normalize=True)
    if counts.empty or len(counts) == 1:
        return 0.0
    entropy = -sum(float(p) * math.log(float(p), 2) for p in counts)
    return round(float(entropy / math.log(len(counts), 2)), 4)


def bootstrap_mean_ci(values: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        idx = rng.integers(0, len(values), len(values))
        samples.append(float(np.mean(values[idx])))
    return percentile_interval(samples)


def majority_map_accuracy(df: pd.DataFrame, group_col: str, target_col: str) -> float:
    majority = df.groupby(group_col)[target_col].agg(lambda s: s.value_counts().idxmax()).to_dict()
    predicted = df[group_col].map(majority)
    return round(float((predicted == df[target_col]).mean()), 4)


def add_ambiguity_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    text = (frame["appeal_reason"].fillna("") + " " + frame["supporting_evidence_summary"].fillna("")).str.lower()
    frame["criterion_near_boundary_score"] = frame["original_score"].between(28, 42)
    frame["criterion_ground_truth_mismatch"] = frame["ground_truth_label"].eq("pass")
    for column, pattern in CRITERIA_PATTERNS.items():
        frame[column] = text.str.contains(pattern, regex=True)
    criteria_cols = [
        "criterion_near_boundary_score",
        "criterion_ground_truth_mismatch",
        *CRITERIA_PATTERNS.keys(),
    ]
    frame["ambiguity_score"] = frame[criteria_cols].astype(int).sum(axis=1)

    def selection_reason(row: pd.Series) -> str:
        reasons: list[str] = []
        for column in criteria_cols:
            if bool(row[column]):
                reasons.append(column.replace("criterion_", ""))
        return ";".join(reasons)

    frame["selection_reason"] = frame.apply(selection_reason, axis=1)
    frame["is_ambiguity_subset"] = frame["ambiguity_score"].ge(AMBIGUITY_THRESHOLD).astype(int)
    frame["score_delta"] = frame["post_appeal_score"] - frame["original_score"]
    return frame


def bucket_scores(df: pd.DataFrame) -> pd.Series:
    bins = [-np.inf, 27.5, 34.5, 42.5, np.inf]
    labels = ["very_low", "low", "boundary_band", "higher"]
    return pd.cut(df["original_score"], bins=bins, labels=labels)


def main() -> None:
    ensure_external_dirs()
    appeals = pd.read_csv(RAW_DATA_FILES["appeals"])
    appeals = add_ambiguity_features(appeals)
    subset = appeals[appeals["is_ambiguity_subset"] == 1].copy()
    if subset.empty:
        summary = {
            "experiment": "24_appeal_ambiguity_stress_test",
            "status": "BLOCKED",
            "reason": "No ambiguity-heavy appeal subset could be constructed.",
            "conclusion": "BLOCKED",
        }
        write_json(OUTPUT_JSON, summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    subset["score_band"] = bucket_scores(subset).astype(str)
    subset.to_csv(OUTPUT_SUBSET_CSV, index=False)
    subset.to_csv(OUTPUT_CSV, index=False)

    criteria_cols = [
        "criterion_near_boundary_score",
        "criterion_ground_truth_mismatch",
        *CRITERIA_PATTERNS.keys(),
    ]
    criteria_rates = {col.replace("criterion_", ""): round(float(subset[col].mean()), 4) for col in criteria_cols}
    outcome_counts = {str(k): int(v) for k, v in subset["adjudication_result"].value_counts().to_dict().items()}
    type_counts = {str(k): int(v) for k, v in subset["appeal_type"].value_counts().to_dict().items()}

    by_ambiguity_score = {}
    for score, group in subset.groupby("ambiguity_score"):
        by_ambiguity_score[str(score)] = {
            "n": int(len(group)),
            "outcome_counts": {str(k): int(v) for k, v in group["adjudication_result"].value_counts().to_dict().items()},
            "normalized_outcome_entropy": normalized_entropy(group["adjudication_result"]),
            "mean_score_delta": round(float(group["score_delta"].mean()), 4),
            "pass_ground_truth_rate": round(float(group["ground_truth_label"].eq("pass").mean()), 4),
        }

    by_type = {}
    for appeal_type, group in subset.groupby("appeal_type"):
        by_type[str(appeal_type)] = {
            "n": int(len(group)),
            "outcome_counts": {str(k): int(v) for k, v in group["adjudication_result"].value_counts().to_dict().items()},
            "normalized_outcome_entropy": normalized_entropy(group["adjudication_result"]),
            "mean_ambiguity_score": round(float(group["ambiguity_score"].mean()), 4),
            "mean_score_delta": round(float(group["score_delta"].mean()), 4),
        }

    by_result = {}
    for result, group in subset.groupby("adjudication_result"):
        by_result[str(result)] = {
            "n": int(len(group)),
            "mean_ambiguity_score": round(float(group["ambiguity_score"].mean()), 4),
            "ambiguity_score_ci": bootstrap_mean_ci(group["ambiguity_score"].to_numpy(dtype=float), seed=2400 + len(group)),
            "mean_original_score": round(float(group["original_score"].mean()), 4),
            "mean_post_score": round(float(group["post_appeal_score"].mean()), 4),
            "mean_score_delta": round(float(group["score_delta"].mean()), 4),
        }

    type_only_accuracy = majority_map_accuracy(subset, "appeal_type", "adjudication_result")
    ambiguity_bucket_accuracy = majority_map_accuracy(subset, "ambiguity_score", "adjudication_result")
    score_band_accuracy = majority_map_accuracy(subset, "score_band", "adjudication_result")

    summary = {
        "experiment": "24_appeal_ambiguity_stress_test",
        "source": "Ambiguity-heavy appeal subset derived from the packaged appeal cases using score-boundary and text-evidence criteria",
        "target_definition": "Stress-test whether appeal outcomes remain fully type-determined once cases are restricted to mixed-evidence, near-boundary, or uncertainty-bearing appeals.",
        "n_total_appeals": int(len(appeals)),
        "n_subset_appeals": int(len(subset)),
        "subset_rate": round(float(len(subset) / len(appeals)), 4),
        "ambiguity_threshold": AMBIGUITY_THRESHOLD,
        "selection_profile": {
            "outcome_counts": outcome_counts,
            "appeal_type_counts": type_counts,
            "criteria_rates": criteria_rates,
            "mean_ambiguity_score": round(float(subset["ambiguity_score"].mean()), 4),
            "mean_ambiguity_score_ci": bootstrap_mean_ci(subset["ambiguity_score"].to_numpy(dtype=float), seed=2401),
            "ground_truth_pass_rate": round(float(subset["ground_truth_label"].eq("pass").mean()), 4),
            "normalized_outcome_entropy": normalized_entropy(subset["adjudication_result"]),
        },
        "comparator_structure": {
            "type_only_majority_accuracy": type_only_accuracy,
            "ambiguity_score_majority_accuracy": ambiguity_bucket_accuracy,
            "score_band_majority_accuracy": score_band_accuracy,
        },
        "by_ambiguity_score": by_ambiguity_score,
        "by_appeal_type": by_type,
        "by_outcome": by_result,
        "method_note": "The ambiguity-heavy subset contains accepted, partial, and rejected outcomes with mixed textual evidence signals, so the appeal layer is not reducible to a single global outcome. However, within the packaged data the appeal taxonomy remains strongly load-bearing: majority prediction from appeal type still perfectly reconstructs outcomes on this subset. This means the experiment reduces the force of a pure rule-engine critique only partially and should be presented as bounded stress evidence rather than a full realism rescue.",
        "conclusion": "MARGINAL" if type_only_accuracy >= 0.95 else "PASS",
    }
    write_json(OUTPUT_JSON, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
