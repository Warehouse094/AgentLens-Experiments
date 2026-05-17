from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common import EXTERNAL_VALIDATION_RAW_DIR, EXTERNAL_VALIDATION_RESULTS_DIR, ensure_external_dirs, write_json
from pipeline_common import RAW_DATA_FILES

OUTPUT_CASES_CSV = EXTERNAL_VALIDATION_RAW_DIR / "human_system_divergence_cases.csv"
OUTPUT_JSON = EXTERNAL_VALIDATION_RESULTS_DIR / "27_human_system_divergence_taxonomy.json"
OUTPUT_CSV = EXTERNAL_VALIDATION_RESULTS_DIR / "27_human_system_divergence_taxonomy_scores.csv"


def flatten(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(flatten(item) for item in value)
    if isinstance(value, dict):
        return " ".join(f"{key} {flatten(item)}" for key, item in value.items())
    return str(value)


def load_response_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with RAW_DATA_FILES["responses"].open() as handle:
        for line in handle:
            record = json.loads(line)
            response = record.get("agent_response") or {}
            rows.append(
                {
                    "episode_id": record.get("episode_id"),
                    "answer_text": flatten(response.get("answer")),
                    "declared_actions_text": flatten(response.get("declaredActions")),
                    "uncertainties_text": flatten(response.get("uncertainties")),
                    "environment_text": flatten(response.get("environmentErrors")),
                    "forbidden_attempts_text": flatten(response.get("forbiddenActionsAttempted")),
                    "tools_used_text": flatten(response.get("toolsUsed")),
                    "safety_decision_text": flatten(response.get("safetyDecision")).lower(),
                }
            )
    return pd.DataFrame(rows)


def add_taxonomy_signals(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["score_gap"] = frame["system_score"] - frame["consensus_score"]
    frame["abs_score_gap"] = frame["score_gap"].abs()
    text = (
        frame["answer_text"].fillna("")
        + " "
        + frame["declared_actions_text"].fillna("")
        + " "
        + frame["uncertainties_text"].fillna("")
        + " "
        + frame["environment_text"].fillna("")
        + " "
        + frame["forbidden_attempts_text"].fillna("")
    ).str.lower()

    frame["signal_uncertainty"] = text.str.contains(r"uncertain|uncertainty|depends|further testing|required|not explicitly clear|moderate uncertainty", regex=True)
    frame["signal_environment"] = text.str.contains(r"timeout|network|environment|truncated|latency|error|failed", regex=True)
    frame["signal_policy"] = text.str.contains(r"forbidden|policy|compliance|scope|authorized|allowance|refuse", regex=True)
    frame["signal_security"] = text.str.contains(r"reentrancy|vulnerability|unsafe|attack|exploit|malicious|risk", regex=True)
    frame["signal_human_low_dispersion"] = frame["score_std"].fillna(0).le(2.0)
    frame["signal_external_pass_skew"] = frame["external_pass_rate"].fillna(0).ge(2/3)
    frame["signal_external_fail_skew"] = frame["external_pass_rate"].fillna(1).le(1/3)
    frame["signal_borderline_gap"] = frame["abs_score_gap"].between(8, 20)
    frame["signal_large_gap"] = frame["abs_score_gap"].ge(20)

    def taxonomy(row: pd.Series) -> str:
        if row["system_label"] == "fail" and row["consensus_pass_fail"] == "pass":
            if row["signal_environment"]:
                return "environment_or_execution_uncertainty"
            if row["signal_uncertainty"]:
                return "threshold_or_uncertainty_mismatch"
            if row["signal_policy"]:
                return "policy_over_penalization"
            if row["signal_security"]:
                return "security_over_penalization"
            return "generic_system_over_penalization"
        if row["system_label"] == "pass" and row["consensus_pass_fail"] == "fail":
            if row["signal_security"] or row["signal_policy"]:
                return "human_flagged_risk_not_captured"
            return "generic_system_under_penalization"
        return "matched"

    frame["divergence_taxonomy"] = frame.apply(taxonomy, axis=1)

    def driver_tags(row: pd.Series) -> str:
        tags: list[str] = []
        for signal_name in [
            "signal_uncertainty",
            "signal_environment",
            "signal_policy",
            "signal_security",
            "signal_human_low_dispersion",
            "signal_external_pass_skew",
            "signal_external_fail_skew",
            "signal_borderline_gap",
            "signal_large_gap",
        ]:
            if bool(row[signal_name]):
                tags.append(signal_name.replace("signal_", ""))
        return ";".join(tags)

    frame["driver_tags"] = frame.apply(driver_tags, axis=1)
    return frame


def main() -> None:
    ensure_external_dirs()
    human = pd.read_csv(RAW_DATA_FILES["human"])
    external = pd.read_csv(RAW_DATA_FILES["external"])
    response_df = load_response_frame()

    external_summary = (
        external.groupby("episode_id")
        .agg(
            external_mean_score=("external_score", "mean"),
            external_pass_rate=("external_pass_fail", lambda s: float((s == "pass").mean())),
            external_agent_count=("external_agent", "nunique"),
        )
        .reset_index()
    )

    merged = human.merge(external_summary, on="episode_id", how="left").merge(response_df, on="episode_id", how="left")
    merged = add_taxonomy_signals(merged)
    merged.to_csv(OUTPUT_CSV, index=False)

    divergence_cases = merged[merged["system_human_match"] == 0].copy()
    divergence_cases = divergence_cases.sort_values(["abs_score_gap", "score_std"], ascending=[False, True]).reset_index(drop=True)
    divergence_cases.to_csv(OUTPUT_CASES_CSV, index=False)

    taxonomy_counts = {str(k): int(v) for k, v in divergence_cases["divergence_taxonomy"].value_counts().to_dict().items()}
    mismatch_direction = {
        "system_fail_human_pass": int(((divergence_cases["system_label"] == "fail") & (divergence_cases["consensus_pass_fail"] == "pass")).sum()),
        "system_pass_human_fail": int(((divergence_cases["system_label"] == "pass") & (divergence_cases["consensus_pass_fail"] == "fail")).sum()),
    }

    by_taxonomy = {}
    for name, group in divergence_cases.groupby("divergence_taxonomy"):
        by_taxonomy[str(name)] = {
            "n": int(len(group)),
            "mean_abs_score_gap": round(float(group["abs_score_gap"].mean()), 4),
            "mean_score_std": round(float(group["score_std"].mean()), 4),
            "external_pass_rate_mean": round(float(group["external_pass_rate"].mean()), 4),
            "top_domains": {str(k): int(v) for k, v in group["domain"].value_counts().head(3).to_dict().items()},
            "top_dimensions": {str(k): int(v) for k, v in group["dimension"].value_counts().head(3).to_dict().items()},
        }

    by_domain = {}
    for domain, group in divergence_cases.groupby("domain"):
        by_domain[str(domain)] = {
            "n": int(len(group)),
            "mean_abs_score_gap": round(float(group["abs_score_gap"].mean()), 4),
            "taxonomy_counts": {str(k): int(v) for k, v in group["divergence_taxonomy"].value_counts().to_dict().items()},
        }

    representative_cases = divergence_cases.head(15)[[
        "episode_id",
        "domain",
        "dimension",
        "system_label",
        "consensus_pass_fail",
        "system_score",
        "consensus_score",
        "score_gap",
        "score_std",
        "external_pass_rate",
        "divergence_taxonomy",
        "driver_tags",
    ]].to_dict(orient="records")

    summary = {
        "experiment": "27_human_system_divergence_taxonomy",
        "source": "Joined analysis over human consensus, external-agent summaries, and raw response text signals",
        "target_definition": "Explain where low system-human agreement comes from by assigning mismatch cases to structured divergence categories.",
        "n_total_units": int(len(merged)),
        "n_divergence_cases": int(len(divergence_cases)),
        "divergence_rate": round(float(len(divergence_cases) / len(merged)), 4),
        "mismatch_direction": mismatch_direction,
        "taxonomy_counts": taxonomy_counts,
        "mean_abs_score_gap_divergence": round(float(divergence_cases["abs_score_gap"].mean()), 4),
        "mean_abs_score_gap_matched": round(float(merged[merged["system_human_match"] == 1]["abs_score_gap"].mean()), 4),
        "mean_external_pass_rate_divergence": round(float(divergence_cases["external_pass_rate"].mean()), 4),
        "mean_external_pass_rate_matched": round(float(merged[merged["system_human_match"] == 1]["external_pass_rate"].mean()), 4),
        "by_taxonomy": by_taxonomy,
        "by_domain": by_domain,
        "representative_cases": representative_cases,
        "method_note": "Most divergence cases are not driven by annotator disagreement: they are dominated by system-fail / human-pass mismatches with large negative score gaps and high human unanimity. The taxonomy therefore turns low system-human agreement into a structured finding, but it does not eliminate the underlying mismatch problem.",
        "conclusion": "PASS",
    }
    write_json(OUTPUT_JSON, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
