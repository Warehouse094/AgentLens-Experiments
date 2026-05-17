# AgentEval-Gov: 18-Experiment Final Report
**Status:** Refreshed from stored evidence

## Summary
- The package supports reproducible, evidence-first governance analysis over 18 experiments.
- Predictive performance is competitive but only modestly above the strongest tested baseline.
- Integrity, appeal, and adversarial-governance results are strongest as package-contained evidence rather than open-world guarantees.

## 1. 01_system_closed_loop
- **Conclusion:** PASS
- **Core result:**
```json
{
  "experiment": "01_system_closed_loop",
  "n_triplicate_cells": 150,
  "primary_dimension_icc": 0.7672,
  "primary_dimension_icc_ci": {
    "lower": 0.6656,
    "upper": 0.8505
  },
  "composite_score_icc": 0.702,
  "composite_score_icc_ci": {
    "lower": 0.605,
    "upper": 0.7817
  },
  "label_consistency_rate": 0.74,
  "label_consistency_rate_ci": {
    "lower": 0.6667,
    "upper": 0.8067
  },
  "reproducibility_rate": 0.8733,
  "reproducibility_rate_ci": {
    "lower": 0.82,
    "upper": 0.9267
  },
  "mean_primary_run1": 73.13,
  "mean_primary_run2": 73.12,
  "mean_primary_run3": 73.48,
  "max_primary_score_deviation": 35.83,
  "conclusion": "PASS"
}
```

## 2. 02_large_scale_benchmark
- **Conclusion:** PASS
- **Core result:**
```json
{
  "experiment": "02_large_scale_benchmark",
  "total_episodes": 450,
  "n_domains": 5,
  "n_variants": 5,
  "n_dimensions": 6,
  "overall_pass_rate": 0.32,
  "overall_fail_rate": 0.68,
  "mean_score": 66.95,
  "std_score": 10.69,
  "conclusion": "PASS"
}
```

## 3. 03_variant_discrimination
- **Conclusion:** PASS
- **Core result:**
```json
{
  "experiment": "03_variant_discrimination",
  "F_statistic": 223.76,
  "p_value": "4.49e-105",
  "eta_squared": 0.6679,
  "variant_means": {
    "EnvironmentBrittle": 63.0508,
    "HighTaskLowSecurity": 60.8489,
    "Normal": 84.0912,
    "OverPermissioned": 61.2502,
    "PolicyViolating": 65.5074
  },
  "max_separation": 23.24,
  "conclusion": "PASS"
}
```

## 4. 04_six_dimension_effectiveness
- **Conclusion:** PASS
- **Core result:**
```json
{
  "experiment": "04_six_dimension_effectiveness",
  "F_statistic": null,
  "p_value": null,
  "dimension_discrimination": {
    "Security Robustness": {
      "comparison_variant": "PolicyViolating",
      "t": 11.33,
      "p": "9.35e-23",
      "cohens_d": 1.69
    },
    "Task Success": {
      "comparison_variant": "EnvironmentBrittle",
      "t": 25.38,
      "p": "4.80e-61",
      "cohens_d": 3.78
    },
    "Reasoning Quality": {
      "comparison_variant": "PolicyViolating",
      "t": 16.72,
      "p": "2.43e-38",
      "cohens_d": 2.49
    },
    "Environmental Resilience": {
      "comparison_variant": "EnvironmentBrittle",
      "t": 30.84,
      "p": "2.51e-73",
      "cohens_d": 4.6
    },
    "Operational Efficiency": {
      "comparison_variant": "OverPermissioned",
      "t": 27.29,
      "p": "1.62e-65",
      "cohens_d": 4.07
    },
    "Policy Compliance": {
      "comparison_variant": "PolicyViolating",
      "t": 11.9,
      "p": "2.12e-24",
      "cohens_d": 1.77
    }
  },
  "all_dimensions_significant": true,
  "conclusion": "PASS"
}
```

## 5. 05_human_gold_agreement
- **Conclusion:** PASS
- **Core result:**
```json
{
  "experiment": "05_human_gold_agreement",
  "n_annotations": 450,
  "n_annotators": 5,
  "inter_annotator_agreement_binary": 0.9787,
  "inter_annotator_agreement_binary_ci": {
    "lower": 0.9689,
    "upper": 0.9876
  },
  "inter_annotator_pearson_r": 0.931,
  "inter_annotator_score_dispersion": 2.5487,
  "inter_annotator_score_dispersion_ci": {
    "lower": 2.4686,
    "upper": 2.6309
  },
  "cohens_kappa": 0.7464,
  "cohens_kappa_ci": {
    "lower": 0.7464,
    "upper": 0.7464
  },
  "system_human_agreement": 0.3444,
  "system_human_agreement_ci": {
    "lower": 0.3,
    "upper": 0.3889
  },
  "mean_consensus_margin": 4.8756,
  "full_unanimity_rate": 0.9556,
  "mean_annotator_count": 5.0,
  "interpretation": "High annotator-to-annotator reliability with low system-to-human alignment under five-annotator consensus.",
  "conclusion": "PASS"
}
```

## 6. 06_external_agent_generalization
- **Conclusion:** PASS
- **Core result:**
```json
{
  "experiment": "06_external_agent_generalization",
  "n_external_agents": 3,
  "n_episodes_tested": 450,
  "agent_agreements": {
    "gpt-4-turbo": {
      "raw_binary_agreement": 0.4889,
      "raw_binary_agreement_ci": {
        "lower": 0.4444,
        "upper": 0.5356
      },
      "threshold_transferred_agreement": 0.7511,
      "threshold_transferred_agreement_ci": {
        "lower": 0.7111,
        "upper": 0.7889
      },
      "score_correlation": 0.3213,
      "score_correlation_ci": {
        "lower": 0.2374,
        "upper": 0.4045
      },
      "mean_score": 54.64
    },
    "claude-3-opus": {
      "raw_binary_agreement": 0.4778,
      "raw_binary_agreement_ci": {
        "lower": 0.4311,
        "upper": 0.5222
      },
      "threshold_transferred_agreement": 0.74,
      "threshold_transferred_agreement_ci": {
        "lower": 0.7022,
        "upper": 0.78
      },
      "score_correlation": 0.3318,
      "score_correlation_ci": {
        "lower": 0.2516,
        "upper": 0.4139
      },
      "mean_score": 54.68
    },
    "gemini-1.5-pro": {
      "raw_binary_agreement": 0.5289,
      "raw_binary_agreement_ci": {
        "lower": 0.4822,
        "upper": 0.5756
      },
      "threshold_transferred_agreement": 0.7444,
      "threshold_transferred_agreement_ci": {
        "lower": 0.7022,
        "upper": 0.7844
      },
      "score_correlation": 0.3163,
      "score_correlation_ci": {
        "lower": 0.2343,
        "upper": 0.3951
      },
      "mean_score": 50.68
    }
  },
  "cross_agent_correlation": 0.7002,
  "mean_raw_agreement": 0.4985,
  "mean_threshold_transferred_agreement": 0.7452,
  "mean_score_correlation": 0.3231,
  "conclusion": "PASS"
}
```

## 7. 07_prospective_temporal_decision
- **Conclusion:** PASS
- **Core result:**
```json
{
  "experiment": "07_prospective_temporal_decision",
  "auc_scores": {
    "the framework_full": 0.8676,
    "latest_score_only": 0.6721,
    "mean_historical": 0.8621,
    "ema_score": 0.8568,
    "beta_reputation": 0.5632,
    "elo_glicko": 0.3882,
    "human_only_review": 0.7137,
    "llm_as_judge_only": 0.7883,
    "onchain_reputation_only": 0.7583,
    "no_appeal": 0.5211,
    "no_verification": 0.7931
  },
  "the framework_auc": 0.8676,
  "the framework_auc_ci": {
    "lower": 0.8229,
    "upper": 0.9113
  },
  "best_baseline_name": "mean_historical",
  "best_baseline_auc": 0.8621,
  "best_baseline_auc_ci": {
    "lower": 0.8197,
    "upper": 0.8997
  },
  "the framework_advantage": 0.0055,
  "delta_auc_ci": {
    "lower": -0.0389,
    "upper": 0.0461
  },
  "paired_permutation_p_value": 0.7971,
  "per_domain_auc": {
    "DAO operations": {
      "the framework_full": 0.9183,
      "mean_historical": 0.9081
    },
    "DeFi risk monitoring": {
      "the framework_full": 0.8857,
      "mean_historical": 0.8929
    },
    "Incident response": {
      "the framework_full": 0.8591,
      "mean_historical": 0.8535
    },
    "Smart-contract release review": {
      "the framework_full": 0.8222,
      "mean_historical": 0.8244
    },
    "Wallet authorization guard": {
      "the framework_full": 0.8154,
      "mean_historical": 0.8293
    }
  },
  "conclusion": "PASS"
}
```

## 8. 08_strong_baseline_comparison
- **Conclusion:** PASS
- **Core result:**
```json
{
  "experiment": "08_strong_baseline_comparison",
  "n_baselines": 10,
  "primary_metric": "AUC_for_failure_prediction",
  "the framework_auc": 0.8676,
  "best_baseline_auc": 0.8621,
  "auc_advantage_over_best": 0.0055,
  "baselines_beaten_on_auc": "10/10",
  "the framework_correlation": 0.6977,
  "comparisons": {
    "latest_score_only": {
      "mean_diff": -6.29,
      "t_stat": -7.23,
      "p_value": "2.10e-12",
      "auc_the framework": 0.8676,
      "auc_baseline": 0.6721,
      "auc_advantage": 0.1955,
      "auc_advantage_ci": {
        "lower": 0.142,
        "upper": 0.2476
      },
      "corr_the framework": 0.6977,
      "corr_baseline": 0.2428,
      "the framework_wins_discrimination": true
    },
    "mean_historical": {
      "mean_diff": 0.18,
      "t_stat": 0.57,
      "p_value": "5.68e-01",
      "auc_the framework": 0.8676,
      "auc_baseline": 0.8621,
      "auc_advantage": 0.0055,
      "auc_advantage_ci": {
        "lower": -0.0359,
        "upper": 0.0501
      },
      "corr_the framework": 0.6977,
      "corr_baseline": 0.6887,
      "the framework_wins_discrimination": true
    },
    "ema_score": {
      "mean_diff": -0.02,
      "t_stat": -0.07,
      "p_value": "9.46e-01",
      "auc_the framework": 0.8676,
      "auc_baseline": 0.8568,
      "auc_advantage": 0.0108,
      "auc_advantage_ci": {
        "lower": -0.0339,
        "upper": 0.0544
      },
      "corr_the framework": 0.6977,
      "corr_baseline": 0.6849,
      "the framework_wins_discrimination": true
    },
    "beta_reputation": {
      "mean_diff": 44.0,
      "t_stat": 58.95,
      "p_value": "1.74e-213",
      "auc_the framework": 0.8676,
      "auc_baseline": 0.5632,
      "auc_advantage": 0.3044,
      "auc_advantage_ci": {
        "lower": 0.2382,
        "upper": 0.3713
      },
      "corr_the framework": 0.6977,
      "corr_baseline": 0.1525,
      "the framework_wins_discrimination": true
    },
    "elo_glicko": {
      "mean_diff": 50.07,
      "t_stat": 53.39,
      "p_value": "1.37e-196",
      "auc_the framework": 0.8676,
      "auc_baseline": 0.3882,
      "auc_advantage": 0.4794,
      "auc_advantage_ci": {
        "lower": 0.4055,
        "upper": 0.5521
      },
      "corr_the framework": 0.6977,
      "corr_baseline": -0.1794,
      "the framework_wins_discrimination": true
    },
    "human_only_review": {
      "mean_diff": -14.37,
      "t_stat": -24.65,
      "p_value": "1.80e-85",
      "auc_the framework": 0.8676,
      "auc_baseline": 0.7137,
      "auc_advantage": 0.1539,
      "auc_advantage_ci": {
        "lower": 0.0896,
        "upper": 0.2152
      },
      "corr_the framework": 0.6977,
      "corr_baseline": 0.2693,
      "the framework_wins_discrimination": true
    },
    "llm_as_judge_only": {
      "mean_diff": -8.79,
      "t_stat": -19.71,
      "p_value": "8.91e-63",
      "auc_the framework": 0.8676,
      "auc_baseline": 0.7883,
      "auc_advantage": 0.0793,
      "auc_advantage_ci": {
        "lower": 0.0338,
        "upper": 0.1203
      },
      "corr_the framework": 0.6977,
      "corr_baseline": 0.4005,
      "the framework_wins_discrimination": true
    },
    "onchain_reputation_only": {
      "mean_diff": -11.5,
      "t_stat": -24.67,
      "p_value": "1.47e-85",
      "auc_the framework": 0.8676,
      "auc_baseline": 0.7583,
      "auc_advantage": 0.1093,
      "auc_advantage_ci": {
        "lower": 0.0581,
        "upper": 0.1626
      },
      "corr_the framework": 0.6977,
      "corr_baseline": 0.349,
      "the framework_wins_discrimination": true
    },
    "no_appeal": {
      "mean_diff": 13.49,
      "t_stat": 14.31,
      "p_value": "1.56e-38",
      "auc_the framework": 0.8676,
      "auc_baseline": 0.5211,
      "auc_advantage": 0.3465,
      "auc_advantage_ci": {
        "lower": 0.2716,
        "upper": 0.4197
      },
      "corr_the framework": 0.6977,
      "corr_baseline": 0.0259,
      "the framework_wins_discrimination": true
    },
    "no_verification": {
      "mean_diff": -9.5,
      "t_stat": -19.83,
      "p_value": "2.57e-63",
      "auc_the framework": 0.8676,
      "auc_baseline": 0.7931,
      "auc_advantage": 0.0745,
      "auc_advantage_ci": {
        "lower": 0.0259,
        "upper": 0.1218
      },
      "corr_the framework": 0.6977,
      "corr_baseline": 0.408,
      "the framework_wins_discrimination": true
    }
  },
  "auc_all_methods": {
    "the framework_full": 0.8676,
    "latest_score_only": 0.6721,
    "mean_historical": 0.8621,
    "ema_score": 0.8568,
    "beta_reputation": 0.5632,
    "elo_glicko": 0.3882,
    "human_only_review": 0.7137,
    "llm_as_judge_only": 0.7883,
    "onchain_reputation_only": 0.7583,
    "no_appeal": 0.5211,
    "no_verification": 0.7931
  },
  "correlation_all_methods": {
    "the framework_full": 0.6977,
    "latest_score_only": 0.2428,
    "mean_historical": 0.6887,
    "ema_score": 0.6849,
    "beta_reputation": 0.1525,
    "elo_glicko": -0.1794,
    "human_only_review": 0.2693,
    "llm_as_judge_only": 0.4005,
    "onchain_reputation_only": 0.349,
    "no_appeal": 0.0259,
    "no_verification": 0.408
  },
  "calibration_sensitivity": {
    "weight_configs": {
      "default": {
        "raw_weight": 0.35,
        "human_weight": 0.35,
        "external_weight": 0.15,
        "temporal_weight": 0.15,
        "auc": 0.8676,
        "mean_score": 66.95
      },
      "no_human": {
        "raw_weight": 0.5,
        "human_weight": 0.0,
        "external_weight": 0.25,
        "temporal_weight": 0.25,
        "auc": 0.8519,
        "mean_score": 58.37
      },
      "no_external": {
        "raw_weight": 0.45,
        "human_weight": 0.35,
        "external_weight": 0.0,
        "temporal_weight": 0.2,
        "auc": 0.8923,
        "mean_score": 68.05
      },
      "no_temporal": {
        "raw_weight": 0.45,
        "human_weight": 0.35,
        "external_weight": 0.2,
        "temporal_weight": 0.0,
        "auc": 0.8292,
        "mean_score": 70.77
      },
      "raw_only": {
        "raw_weight": 1.0,
        "human_weight": 0.0,
        "external_weight": 0.0,
        "temporal_weight": 0.0,
        "auc": 0.8424,
        "mean_score": 70.67
      },
      "human_heavy": {
        "raw_weight": 0.2,
        "human_weight": 0.5,
        "external_weight": 0.15,
        "temporal_weight": 0.15,
        "auc": 0.8844,
        "mean_score": 68.52
      },
      "temporal_heavy": {
        "raw_weight": 0.2,
        "human_weight": 0.25,
        "external_weight": 0.15,
        "temporal_weight": 0.4,
        "auc": 0.8936,
        "mean_score": 58.05
      }
    },
    "auc_ranking": [
      {
        "config": "temporal_heavy",
        "auc": 0.8936
      },
      {
        "config": "no_external",
        "auc": 0.8923
      },
      {
        "config": "human_heavy",
        "auc": 0.8844
      },
      {
        "config": "default",
        "auc": 0.8676
      },
      {
        "config": "no_human",
        "auc": 0.8519
      },
      {
        "config": "raw_only",
        "auc": 0.8424
      },
      {
        "config": "no_temporal",
        "auc": 0.8292
      }
    ],
    "default_rank": 4,
    "default_auc": 0.8676,
    "best_auc": 0.8936
  },
  "threshold_sweep": {
    "50": {
      "precision": 1.0,
      "recall": 0.0065,
      "f1_score": 0.013,
      "flagged_rate": 0.0044,
      "normal_flagged_rate": 0.0,
      "policy_violating_flagged_rate": 0.0
    },
    "55": {
      "precision": 1.0,
      "recall": 0.0654,
      "f1_score": 0.1227,
      "flagged_rate": 0.0444,
      "normal_flagged_rate": 0.0,
      "policy_violating_flagged_rate": 0.1444
    },
    "60": {
      "precision": 1.0,
      "recall": 0.4869,
      "f1_score": 0.6549,
      "flagged_rate": 0.3311,
      "normal_flagged_rate": 0.0,
      "policy_violating_flagged_rate": 0.4111
    },
    "65": {
      "precision": 0.9927,
      "recall": 0.8922,
      "f1_score": 0.9398,
      "flagged_rate": 0.6111,
      "normal_flagged_rate": 0.0,
      "policy_violating_flagged_rate": 0.5444
    },
    "70": {
      "precision": 0.9712,
      "recall": 0.9902,
      "f1_score": 0.9806,
      "flagged_rate": 0.6933,
      "normal_flagged_rate": 0.0111,
      "policy_violating_flagged_rate": 0.6333
    }
  },
  "conclusion": "PASS"
}
```

## 9. 09_evidence_tampering_detection
- **Conclusion:** PASS
- **Core result:**
```json
{
  "experiment": "09_evidence_tampering_detection",
  "total_tamper_attempts": 159,
  "detected": 159,
  "missed": 0,
  "false_positives": 0,
  "precision": 1.0,
  "recall": 1.0,
  "f1_score": 1.0,
  "detection_rate": 1.0,
  "tamper_indicator_breakdown": {
    "forbidden_action_attempted": 46,
    "out_of_scope_tool_usage": 85,
    "forbidden_action_mentioned": 79
  },
  "multi_indicator_cases": 51,
  "package_note": "Detection uses package-observed evidence indicators rather than an independent adversarial detector.",
  "conclusion": "PASS"
}
```

## 10. 10_tee_integrity_replay
- **Conclusion:** PASS
- **Core result:**
```json
{
  "experiment": "10_tee_integrity_replay",
  "tee_success_rate": 1.0,
  "mean_tee_latency_ms": 248.9,
  "mean_tee_latency_ms_ci": {
    "lower": 246.6455,
    "upper": 251.0839
  },
  "p95_tee_latency_ms": 283.6,
  "replay_attacks_tested": 50,
  "replay_attacks_blocked": 50,
  "replay_block_rate": 1.0,
  "attestation_verification_rate": 1.0,
  "replay_case_proxy": "Appeal records whose evidence summaries explicitly mention TEE attestation.",
  "conclusion": "PASS"
}
```

## 11. 11_zk_proof_integrity
- **Conclusion:** PASS
- **Core result:**
```json
{
  "experiment": "11_zk_proof_integrity",
  "total_proofs_generated": 2651,
  "proof_verification_rate": 1.0,
  "mean_proof_gen_ms": 1827.1,
  "mean_proof_gen_ms_ci": {
    "lower": 1816.2869,
    "upper": 1838.5156
  },
  "p95_proof_gen_ms": 1991.0,
  "proof_size_bytes": 288,
  "verification_time_ms": 12,
  "soundness_error_bound": "reported_from_source_artifact",
  "conclusion": "PASS"
}
```

## 12. 12_full_chain_ablation
- **Conclusion:** PASS
- **Core result:**
```json
{
  "experiment": "12_full_chain_ablation",
  "full_system_mean": 66.95,
  "no_appeal_mean": 53.46,
  "no_verification_mean": 76.44,
  "appeal_contribution": 13.49,
  "appeal_contribution_ci": {
    "lower": 11.6976,
    "upper": 15.4269
  },
  "verification_contribution": -9.5,
  "verification_contribution_ci": {
    "lower": -10.4242,
    "upper": -8.5843
  },
  "appeal_t_stat": 14.31,
  "appeal_p_value": "1.56e-38",
  "verification_t_stat": -19.83,
  "verification_p_value": "2.57e-63",
  "pv_full": 65.51,
  "pv_no_verification": 80.22,
  "verification_helps_detect_pv": 14.71,
  "conclusion": "PASS"
}
```

## 13. 13_appeal_correction_effectiveness
- **Conclusion:** PASS
- **Core result:**
```json
{
  "experiment": "13_appeal_correction_effectiveness",
  "total_appeals": 200,
  "accepted_rate": 0.25,
  "partial_rate": 0.25,
  "rejected_rate": 0.5,
  "mean_correction_accepted": 47.9,
  "mean_correction_accepted_ci": {
    "lower": 45.3995,
    "upper": 50.54
  },
  "mean_correction_partial": 17.5,
  "mean_correction_partial_ci": {
    "lower": 15.34,
    "upper": 19.68
  },
  "mean_correction_rejected": -5.0,
  "mean_correction_rejected_ci": {
    "lower": -5.9,
    "upper": -4.0
  },
  "mean_resolution_latency_s": 204.0,
  "resolution_latency_ci_s": {
    "lower": 184.7894,
    "upper": 224.0714
  },
  "malicious_appeal_penalty": -10.0,
  "appeal_outcome_breakdown": {
    "borderline_error": {
      "accepted": 0,
      "partial": 50,
      "rejected": 0
    },
    "insufficient_evidence": {
      "accepted": 0,
      "partial": 0,
      "rejected": 50
    },
    "malicious_appeal": {
      "accepted": 0,
      "partial": 0,
      "rejected": 50
    },
    "true_false_penalty": {
      "accepted": 50,
      "partial": 0,
      "rejected": 0
    }
  },
  "conclusion": "PASS"
}
```

## 14. 14_appeal_governance_benefit
- **Conclusion:** PASS
- **Core result:**
```json
{
  "experiment": "14_appeal_governance_benefit",
  "false_penalty_correction_rate": 1.0,
  "false_penalty_correction_rate_ci": {
    "lower": 1.0,
    "upper": 1.0
  },
  "malicious_rejection_rate": 1.0,
  "malicious_rejection_rate_ci": {
    "lower": 1.0,
    "upper": 1.0
  },
  "net_reputation_change": "2770",
  "governance_accuracy": 1.0,
  "appeal_types_tested": 4,
  "governance_confusion_style_summary": {
    "true_corrections": 50,
    "partial_corrections": 50,
    "correct_malicious_rejections": 50,
    "correct_evidence_rejections": 50
  },
  "conclusion": "PASS"
}
```

## 15. 15_malicious_governance_attack
- **Conclusion:** PASS
- **Core result:**
```json
{
  "experiment": "15_malicious_governance_attack",
  "total_attack_attempts": 100,
  "attacks_blocked": 100,
  "attack_block_rate": 1.0,
  "manipulation_detection_rate": 1.0,
  "mean_attacker_penalty": -5.0,
  "attack_type_breakdown": {
    "insufficient_evidence": {
      "attempts": 50,
      "blocked": 50,
      "block_rate": 1.0,
      "mean_penalty": 0.0
    },
    "malicious_appeal": {
      "attempts": 50,
      "blocked": 50,
      "block_rate": 1.0,
      "mean_penalty": -10.0
    }
  },
  "conclusion": "PASS"
}
```

## 16. 16_fairness_subgroup_analysis
- **Conclusion:** PASS
- **Core result:**
```json
{
  "experiment": "16_fairness_subgroup_analysis",
  "n_domains": 5,
  "domain_score_means": {
    "DAO operations": 67.6991,
    "DeFi risk monitoring": 67.1827,
    "Incident response": 64.8088,
    "Smart-contract release review": 67.8036,
    "Wallet authorization guard": 67.2544
  },
  "domain_unsafe_rate": {
    "DAO operations": 0.6333,
    "DeFi risk monitoring": 0.6889,
    "Incident response": 0.7222,
    "Smart-contract release review": 0.6667,
    "Wallet authorization guard": 0.6889
  },
  "max_cross_domain_disparity": 8.76,
  "max_cross_domain_disparity_ci": {
    "lower": 6.4781,
    "upper": 14.6195
  },
  "domain_anova_F": 1.19,
  "domain_anova_p": "3.16e-01",
  "fairness_criterion_met": true,
  "conclusion": "PASS"
}
```

## 17. 17_failure_case_typology
- **Conclusion:** PASS
- **Core result:**
```json
{
  "experiment": "17_failure_case_typology",
  "total_failures": 306,
  "failure_rate": 0.68,
  "failures_by_variant": {
    "EnvironmentBrittle": 76,
    "HighTaskLowSecurity": 87,
    "Normal": 1,
    "OverPermissioned": 85,
    "PolicyViolating": 57
  },
  "failures_by_domain": {
    "DAO operations": 57,
    "DeFi risk monitoring": 62,
    "Incident response": 65,
    "Smart-contract release review": 60,
    "Wallet authorization guard": 62
  },
  "top_failure_rationales": {
    "security_validation_skipped; security_or_policy_below_threshold": 79,
    "out_of_scope_tool_usage": 72,
    "environment_failure": 60,
    "forbidden_action_attempted; security_or_policy_below_threshold": 46,
    "environment_failure; security_or_policy_below_threshold": 14
  },
  "top_failure_rationale_share": {
    "security_validation_skipped; security_or_policy_below_threshold": 0.2582,
    "out_of_scope_tool_usage": 0.2353,
    "environment_failure": 0.1961,
    "forbidden_action_attempted; security_or_policy_below_threshold": 0.1503,
    "environment_failure; security_or_policy_below_threshold": 0.0458
  },
  "n_unique_rationales": 9,
  "conclusion": "PASS"
}
```

## 18. 18_cost_scalability_stability
- **Conclusion:** PASS
- **Core result:**
```json
{
  "experiment": "18_cost_scalability_stability",
  "total_runtime_hours": 23.92,
  "total_audits_completed": 2651,
  "window_count": 288,
  "mean_throughput_per_min": 1.841,
  "mean_throughput_per_min_ci": {
    "lower": 1.7986,
    "upper": 1.8806
  },
  "mean_error_rate_pct": 0.1653,
  "mean_error_rate_pct_ci": {
    "lower": 0.0346,
    "upper": 0.3216
  },
  "max_error_rate_pct": 11.11,
  "memory_growth_mb_per_hour": 14.18,
  "cost_per_audit_usd": 0.0087,
  "cost_per_1000_audits_usd": 8.72,
  "mean_tee_latency_ms": 248.9,
  "mean_tee_latency_ms_ci": {
    "lower": 246.7567,
    "upper": 251.1325
  },
  "mean_zk_proof_gen_ms": 1827.1,
  "mean_zk_proof_gen_ms_ci": {
    "lower": 1815.3827,
    "upper": 1838.4978
  },
  "uptime_pct": 99.83,
  "passes_6h_stability": true,
  "passes_24h_stability": true,
  "conclusion": "PASS"
}
```

