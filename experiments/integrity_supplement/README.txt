the framework revision bundle (integrity-oriented supplements)

This folder keeps optional low-level supplementary artifacts.
Canonical scripts/results/figures for revised experiments are stored at the package root in:
- 04_scripts/revision_additions/
- 02_experiment_results/revision_additions/
- 03_figures/revision_additions/

Supplement items and canonical pointers:

1) Nested temporal calibration
- Script: ../../../04_scripts/revision_additions/run_nested_calibration_experiment.py
- Results: ../../../02_experiment_results/revision_additions/nested_calibration_results.json, nested_calibration_per_fold.csv, nested_calibration_threshold_choices.csv
- Figures: ../../../03_figures/revision_additions/nested_calibration_thresholds.png, nested_calibration_reliability.png

2) Expanded prospective baselines
- Script: ../../../04_scripts/revision_additions/run_expanded_baseline_experiment.py
- Results: ../../../02_experiment_results/revision_additions/expanded_baseline_results.json, expanded_baseline_leaderboard.csv
- Figures: ../../../03_figures/revision_additions/expanded_baseline_leaderboard.png, expanded_baseline_delta_auc_forest.png

3) TEE negative regression
- Script: ../../../04_scripts/revision_additions/run_tee_negative_regression.py
- Results: ../../../02_experiment_results/revision_additions/tee_negative_regression_results.json, tee_positive_local_summary.json
- Optional low-level artifacts: tee/positive_local_e2e and tee/mutated_cases/*

4) SGX/DCAP policy matrix
- Script: ../../../04_scripts/revision_additions/run_sgx_policy_matrix_experiment.py
- Results: ../../../02_experiment_results/revision_additions/sgx_policy_matrix_results.json, sgx_policy_matrix_cases.csv
- Optional low-level artifacts: tee/sgx_policy_matrix/*

5) SGX replay and rebinding regression
- Script: ../../../04_scripts/revision_additions/run_sgx_replay_rebinding_experiment.py
- Results: ../../../02_experiment_results/revision_additions/sgx_replay_rebinding_results.json, sgx_replay_rebinding_cases.csv
- Optional low-level artifacts: tee/sgx_replay_cases/*

6) Request-time HTTP attestation guard regression
- Script: ../../../04_scripts/revision_additions/run_http_attestation_guard_experiment.py
- Results: ../../../02_experiment_results/revision_additions/http_attestation_guard_results.json, http_attestation_guard_cases.csv

7) DCAP authenticity-boundary supplement
- Script: ../../../04_scripts/revision_additions/run_dcap_authenticity_boundary_experiment.py
- Results: ../../../02_experiment_results/revision_additions/dcap_authenticity_boundary_results.json, dcap_authenticity_boundary_cases.csv, dcap_authenticity_boundary_matrix.csv
- Optional low-level artifacts: tee/dcap_authenticity_boundary/*

Important boundary:
These supplements strengthen local boundary characterization, but they do not establish open-world robustness or full end-to-end DCAP-chain assurance.
