# AgentLens: Experimental Supplementary Materials

This repository contains the complete experimental data, analysis scripts, and result files for the paper:

> **AgentLens: A Governance-Oriented Evaluation Framework for AI Agents in Web3 Environments**
> Submitted to IEEE Transactions on Dependable and Secure Computing (TDSC)

All 27 experiments reported in the paper are reproducible from the materials in this repository.

---

## Repository Structure

```
AgentLens-Experiments/
├── data/                              # Raw benchmark data
│   ├── agent_responses_real.jsonl     # 450-episode benchmark (5 domains × 5 variants × 18 episodes)
│   ├── human_gold_annotation_final.csv  # Expert consensus annotations (Exp05)
│   ├── appeal_cases_final.csv         # Appeal case records (Exp13–15)
│   ├── baseline_comparison_results.csv  # Baseline method scores (Exp07–08, R1–R2)
│   ├── external_agent_results_final.csv # External agent evaluation results (Exp06, Exp21)
│   └── long_term_stability_metrics.csv  # Longitudinal stability data (Exp16–18)
│
├── experiments/
│   ├── pipeline/                      # Core experiment pipeline
│   │   ├── pipeline_common.py         # Shared utilities: scoring, calibration, ICC, AUC
│   │   └── generate_figures.py        # Figure generation for all 7 paper figures
│   │
│   ├── exp01_18_results/              # Results for Experiments 01–18
│   │   ├── exp01_reproducibility.json
│   │   ├── exp02_benchmark_scores.json
│   │   ├── exp03_variant_discrimination.json
│   │   ├── exp04_dimension_discrimination.json
│   │   ├── exp05_human_agreement.json
│   │   ├── exp06_external_generalization.json
│   │   ├── exp07_temporal_prediction.json
│   │   ├── exp08_calibration_sensitivity.json
│   │   ├── exp09_tampering_detection.json
│   │   ├── exp10_tee_integrity_replay.json
│   │   ├── exp11_zk_proof_integrity.json
│   │   ├── exp12_ablation_study.json
│   │   ├── exp13_appeal_correction.json
│   │   ├── exp14_appeal_governance_benefit.json
│   │   ├── exp15_malicious_governance_attack.json
│   │   ├── exp16_fairness_cross_domain.json
│   │   ├── exp17_operational_cost.json
│   │   └── exp18_long_term_stability.json
│   │
│   ├── exp19_27_external_validation/  # Revision experiments (Exp19–27)
│   │   ├── scripts/                   # Reproduction scripts for each experiment
│   │   │   ├── run_exp19_cross_benchmark_transfer.py
│   │   │   ├── run_exp20_archived_trace_replication.py
│   │   │   ├── run_exp21_independent_evaluator_transfer.py
│   │   │   ├── run_exp22_atbench_ambiguity_stress_test.py
│   │   │   ├── run_exp23_atbench_independent_relabel_transfer.py
│   │   │   ├── run_exp24_appeal_ambiguity_stress_test.py
│   │   │   ├── run_exp25_human_reliability_recalculation.py
│   │   │   ├── run_exp26_domain_holdout_transfer.py
│   │   │   ├── run_exp27_human_system_divergence_taxonomy.py
│   │   │   ├── collect_atbench_external_judgements.py
│   │   │   └── build_external_generalization_bundle.py
│   │   └── results/                   # JSON and CSV results for Exp19–27
│   │
│   ├── integrity_supplement/          # Security testing supplement (R3–R7)
│   │   ├── README.txt
│   │   └── logs/                      # Detailed logs for each security test
│   │       ├── tee_negative_regression_summary.txt
│   │       ├── sgx_policy_matrix_summary.txt
│   │       ├── sgx_replay_rebinding_summary.txt
│   │       ├── http_attestation_guard_summary.txt
│   │       ├── dcap_authenticity_boundary_summary.txt
│   │       ├── nested_calibration_summary.txt
│   │       ├── expanded_baseline_summary.txt
│   │       └── tee_positive_state_fallback.json
│   │
│   └── final_experiment_report.md     # Consolidated report of all 27 experiments
│
├── scripts/                           # Figure generation and analysis scripts
│   ├── style.py                       # Shared matplotlib style settings
│   ├── make_ablation_study.py         # Fig. 5: Ablation study
│   ├── make_external_generalization.py  # Fig. 7: External generalization
│   ├── make_failure_taxonomy.py       # Failure taxonomy analysis
│   ├── make_integrity_performance.py  # Fig. 4: Integrity layer performance
│   └── analyze_exp27_full.py          # Exp27: Human-system divergence analysis
│
└── figures/                           # All paper figures (PDF + PNG)
    ├── fig1_mechanism_workflow.*      # Fig. 1: AgentLens six-stage pipeline
    ├── fig2_audit_validity_composite.* # Fig. 2: Audit validity (RQ1)
    ├── fig3_prediction_composite.*    # Fig. 3: Failure prediction (RQ2)
    ├── fig4_integrity_governance_composite.* # Fig. 4: Integrity layer (RQ3)
    ├── fig_ablation_study.*           # Fig. 5: Ablation study (RQ4)
    ├── fig_failure_taxonomy.*         # Fig. 6: Failure taxonomy (RQ5)
    └── fig_external_generalization.*  # Fig. 7: External generalization (RQ6)
```

---

## Experiment Overview

The 27 experiments are organized around six Research Questions (RQs):

| RQ | Scope | Experiments |
|----|-------|-------------|
| RQ1 | Reproducibility & Discrimination | Exp01–04, R1 |
| RQ2 | Prediction & Calibration | Exp05–08, R2 |
| RQ3 | Integrity & Security | Exp09–11, R3–R7 |
| RQ4 | Contestability & Appeals | Exp12–15 |
| RQ5 | Fairness & Operations | Exp16–18 |
| RQ6 | External Generalization | Exp19–27 |

---

## Reproduction

### Requirements

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

### Running experiments

Each script in `experiments/exp19_27_external_validation/scripts/` is self-contained and reads from the `data/` directory. For example:

```bash
python experiments/exp19_27_external_validation/scripts/run_exp19_cross_benchmark_transfer.py
```

### Regenerating figures

```bash
python experiments/pipeline/generate_figures.py
```

---

## Dataset

The benchmark dataset (`data/agent_responses_real.jsonl`) contains 450 AI agent execution episodes across:

- **5 domains**: DAO governance operations, DeFi risk monitoring, smart-contract release review, incident response, wallet authorization guard
- **5 behavioral variants**: Normal, EnvironmentBrittle, HighTaskLowSecurity, OverPermissioned, PolicyViolating
- **Overall pass rate**: 0.32 (intentionally failure-rich for governance evaluation)

External generalization experiments (Exp19–27) additionally use the [ATBench](https://huggingface.co/datasets/AI45Research/ATBench) dataset (Apache-2.0).

---

## License

Code: AGPL-3.0  
Data: CC BY 4.0
