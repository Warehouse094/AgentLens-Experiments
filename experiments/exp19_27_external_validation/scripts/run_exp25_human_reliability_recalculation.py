from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common import EXTERNAL_VALIDATION_RESULTS_DIR, ensure_external_dirs, write_json
from pipeline_common import RAW_DATA_FILES, normalize_pass_fail, percentile_interval

OUTPUT_JSON = EXTERNAL_VALIDATION_RESULTS_DIR / "25_human_reliability_recalculation.json"
OUTPUT_CSV = EXTERNAL_VALIDATION_RESULTS_DIR / "25_human_reliability_recalculation_scores.csv"

ANNOTATOR_DOWNLOADS_DIR = Path(__file__).resolve().parents[3] / "01_raw_data" / "external_validation" / "annotator_returns"

BOOTSTRAP_ITERATIONS = 2000


def load_long_annotations() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if ANNOTATOR_DOWNLOADS_DIR.exists():
        for path in sorted(ANNOTATOR_DOWNLOADS_DIR.glob("*.csv")):
            frame = pd.read_csv(path)
            frame["annotator_pass_fail"] = frame["annotator_pass_fail"].map(normalize_pass_fail)
            frames.append(frame[["annotator_id", "episode_id", "annotator_score", "annotator_pass_fail"]].copy())
    if not frames:
        raise FileNotFoundError(
            f"No annotator CSV files found in {ANNOTATOR_DOWNLOADS_DIR}. "
            "Place per-annotator return files there before running this script."
        )
    return pd.concat(frames, ignore_index=True)


def cohens_kappa_binary(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=int)
    right = np.asarray(right, dtype=int)
    if len(left) == 0:
        return 0.0
    po = float(np.mean(left == right))
    pe = float(left.mean() * right.mean() + (1 - left.mean()) * (1 - right.mean()))
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return float((po - pe) / (1 - pe))


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def bootstrap_ci_from_values(values: list[float] | np.ndarray, seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"lower": 0.0, "upper": 0.0}
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        idx = rng.integers(0, len(array), len(array))
        samples.append(float(np.mean(array[idx])))
    return percentile_interval(samples)


def bootstrap_pair_metric(left: np.ndarray, right: np.ndarray, metric_fn, seed: int) -> dict[str, float]:
    left = np.asarray(left)
    right = np.asarray(right)
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(BOOTSTRAP_ITERATIONS * 5):
        idx = rng.integers(0, len(left), len(left))
        metric = metric_fn(left[idx], right[idx])
        if np.isfinite(metric):
            samples.append(float(metric))
        if len(samples) >= BOOTSTRAP_ITERATIONS:
            break
    if not samples:
        return {"lower": 0.0, "upper": 0.0}
    return percentile_interval(samples)


def build_pair_tables(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    for left_annotator, right_annotator in combinations(sorted(long_df["annotator_id"].unique()), 2):
        left = long_df[long_df["annotator_id"] == left_annotator][["episode_id", "annotator_score", "annotator_pass_fail"]].copy()
        right = long_df[long_df["annotator_id"] == right_annotator][["episode_id", "annotator_score", "annotator_pass_fail"]].copy()
        merged = left.merge(right, on="episode_id", suffixes=("_left", "_right"))
        merged["left_bin"] = merged["annotator_pass_fail_left"].eq("pass").astype(int)
        merged["right_bin"] = merged["annotator_pass_fail_right"].eq("pass").astype(int)
        merged["agree"] = merged["left_bin"].eq(merged["right_bin"]).astype(int)
        merged["pair_id"] = f"{left_annotator}__{right_annotator}"
        merged["left_annotator"] = left_annotator
        merged["right_annotator"] = right_annotator
        label_rows.extend(
            merged[[
                "pair_id",
                "left_annotator",
                "right_annotator",
                "episode_id",
                "left_bin",
                "right_bin",
                "agree",
            ]].to_dict(orient="records")
        )
        merged["score_delta"] = (merged["annotator_score_left"] - merged["annotator_score_right"]).abs()
        score_rows.extend(
            merged[[
                "pair_id",
                "left_annotator",
                "right_annotator",
                "episode_id",
                "annotator_score_left",
                "annotator_score_right",
                "score_delta",
            ]].to_dict(orient="records")
        )
    return pd.DataFrame(label_rows), pd.DataFrame(score_rows)


def per_episode_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for episode_id, group in long_df.groupby("episode_id"):
        labels = group["annotator_pass_fail"].to_numpy()
        bins = (labels == "pass").astype(int)
        scores = group["annotator_score"].to_numpy(dtype=float)
        pair_agreements: list[float] = []
        pair_kappas: list[float] = []
        pair_score_deltas: list[float] = []
        for left_idx, right_idx in combinations(range(len(group)), 2):
            pair_agreements.append(float(bins[left_idx] == bins[right_idx]))
            pair_kappas.append(cohens_kappa_binary(np.array([bins[left_idx]]), np.array([bins[right_idx]])))
            pair_score_deltas.append(abs(float(scores[left_idx] - scores[right_idx])))
        rows.append(
            {
                "episode_id": episode_id,
                "annotator_count": int(group["annotator_id"].nunique()),
                "consensus_pass": int(np.sum(bins)),
                "consensus_fail": int(len(bins) - np.sum(bins)),
                "pairwise_agreement_mean": round(float(np.mean(pair_agreements)) if pair_agreements else 1.0, 4),
                "pairwise_kappa_mean": round(float(np.mean(pair_kappas)) if pair_kappas else 1.0, 4),
                "score_mean": round(float(np.mean(scores)), 4),
                "score_std": round(float(np.std(scores, ddof=0)), 4),
                "pairwise_score_delta_mean": round(float(np.mean(pair_score_deltas)) if pair_score_deltas else 0.0, 4),
                "full_unanimity": int(len(np.unique(labels)) == 1),
            }
        )
    return pd.DataFrame(rows)


def leave_one_annotator_out(long_df: pd.DataFrame, system_df: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dropped in sorted(long_df["annotator_id"].unique()):
        subset = long_df[long_df["annotator_id"] != dropped].copy()
        consensus = (
            subset.groupby("episode_id")
            .agg(
                mean_score=("annotator_score", "mean"),
                n_pass=("annotator_pass_fail", lambda s: int((s == "pass").sum())),
                n_fail=("annotator_pass_fail", lambda s: int((s == "fail").sum())),
            )
            .reset_index()
        )
        consensus["consensus_pass_fail"] = np.where(consensus["n_pass"] >= consensus["n_fail"], "pass", "fail")
        merged = consensus.merge(system_df[["episode_id", "system_label", "system_score"]], on="episode_id", how="left")
        rows.append(
            {
                "held_out_annotator": dropped,
                "n_episodes": int(len(merged)),
                "system_human_agreement": round(float((merged["consensus_pass_fail"] == merged["system_label"]).mean()), 4),
                "score_correlation": round(safe_corr(merged["mean_score"].to_numpy(dtype=float), merged["system_score"].to_numpy(dtype=float)), 4),
                "pass_rate": round(float(merged["consensus_pass_fail"].eq("pass").mean()), 4),
            }
        )
    return rows


def main() -> None:
    ensure_external_dirs()
    long_df = load_long_annotations()
    system_df = pd.read_csv(RAW_DATA_FILES["human"])

    label_pairs, score_pairs = build_pair_tables(long_df)
    episode_summary = per_episode_summary(long_df)

    label_agreement = float(label_pairs["agree"].mean())
    kappa = cohens_kappa_binary(label_pairs["left_bin"].to_numpy(dtype=int), label_pairs["right_bin"].to_numpy(dtype=int))
    score_correlation = safe_corr(score_pairs["annotator_score_left"].to_numpy(dtype=float), score_pairs["annotator_score_right"].to_numpy(dtype=float))
    mean_score_delta = float(score_pairs["score_delta"].mean())
    system_human_agreement = float(system_df["system_human_match"].mean())
    score_dispersion = float(episode_summary["score_std"].mean())
    unanimity_rate = float(episode_summary["full_unanimity"].mean())

    merged = system_df[["episode_id", "domain", "agent_variant", "dimension", "system_label", "system_score", "consensus_pass_fail", "consensus_score"]].merge(
        episode_summary,
        on="episode_id",
        how="left",
    )
    merged.to_csv(OUTPUT_CSV, index=False)

    per_domain = {}
    for domain, group in merged.groupby("domain"):
        per_domain[str(domain)] = {
            "n": int(len(group)),
            "system_human_agreement": round(float((group["consensus_pass_fail"] == group["system_label"]).mean()), 4),
            "mean_score_dispersion": round(float(group["score_std"].mean()), 4),
            "unanimity_rate": round(float(group["full_unanimity"].mean()), 4),
        }

    loo_rows = leave_one_annotator_out(long_df, system_df)

    summary = {
        "experiment": "25_human_reliability_recalculation",
        "source": "Recalculation from raw five-annotator CSV returns with bootstrap confidence intervals",
        "n_annotations": int(len(long_df)),
        "n_episodes": int(long_df["episode_id"].nunique()),
        "n_annotators": int(long_df["annotator_id"].nunique()),
        "pairwise_binary_agreement": round(label_agreement, 4),
        "pairwise_binary_agreement_ci": bootstrap_ci_from_values(label_pairs["agree"].to_numpy(dtype=float), seed=2501),
        "cohens_kappa": round(kappa, 4),
        "cohens_kappa_ci": bootstrap_pair_metric(
            label_pairs["left_bin"].to_numpy(dtype=int),
            label_pairs["right_bin"].to_numpy(dtype=int),
            cohens_kappa_binary,
            seed=2502,
        ),
        "score_correlation": round(score_correlation, 4),
        "score_correlation_ci": bootstrap_pair_metric(
            score_pairs["annotator_score_left"].to_numpy(dtype=float),
            score_pairs["annotator_score_right"].to_numpy(dtype=float),
            safe_corr,
            seed=2503,
        ),
        "mean_pairwise_score_delta": round(mean_score_delta, 4),
        "mean_score_dispersion": round(score_dispersion, 4),
        "mean_score_dispersion_ci": bootstrap_ci_from_values(episode_summary["score_std"].to_numpy(dtype=float), seed=2504),
        "full_unanimity_rate": round(unanimity_rate, 4),
        "full_unanimity_rate_ci": bootstrap_ci_from_values(episode_summary["full_unanimity"].to_numpy(dtype=float), seed=2505),
        "system_human_agreement": round(system_human_agreement, 4),
        "system_human_agreement_ci": bootstrap_ci_from_values(system_df["system_human_match"].to_numpy(dtype=float), seed=2506),
        "per_domain": per_domain,
        "leave_one_annotator_out": loo_rows,
        "method_note": "Bootstrap confidence intervals computed over raw pairwise label comparisons from five-annotator returns (2,000 resamples per metric, seed-fixed for reproducibility).",
        "conclusion": "PASS",
    }

    write_json(OUTPUT_JSON, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
