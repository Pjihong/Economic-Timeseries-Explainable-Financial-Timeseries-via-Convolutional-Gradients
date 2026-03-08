
# Module map

## src/vix_xai/config.py
- `Config`

## src/vix_xai/utils.py
- `ensure_dir`
- `set_seed`
- `get_device`
- `count_parameters`

## src/vix_xai/data.py
- `load_frame`
- `split_by_time`
- `transform_for_model`
- `SequenceDataset`
- `build_dataloaders`
- `collect_test_windows`
- `inverse_all_X_windows`

## src/vix_xai/models.py
- `RevIN`
- `Chomp1d`
- `TemporalBlock`
- `SingleTCN`
- `SingleCNN`
- `TCNEnsemble`
- `CNNEnsemble`
- `build_model`
- `build_model_from_snapshot`

## src/vix_xai/training.py
- `EarlyStopping`
- `val_loss`
- `train_model`
- `search_cnn_config_under_budget`

## src/vix_xai/eval.py
- `evaluate_level_rmse`
- `compute_baselines`
- `evaluate_cpd_performance`
- `plot_losses`
- `plot_predictions`
- `plot_revin_params`

## src/vix_xai/xai.py
- `find_last_conv_per_branch`
- `TimeSeriesGradCAMRegression`
- `extract_multivariate_embeddings`

## src/vix_xai/experiments.py
- `save_model_bundle`
- `load_model_bundle`
- `run_experiment_suite`

## src/vix_xai/event_warping.py
- `DTWResult`
- `wasserstein_1d`
- `quantile_l2_1d`
- `energy_distance_1d`
- `mmd_rbf`
- `compute_cost_matrix_dtdw_1d`
- `compute_cost_matrix_embedding`
- `dtw_from_cost_matrix`
- `apply_event_weighting`
- `dtdw_1d`
- `wdtdw_1d`
- `dtdw_embedding`
- `wdtdw_embedding`

## src/vix_xai/posthoc.py
- `benjamini_hochberg`
- `define_events_from_level`
- `build_event_definition_df`
- `subsample_nonoverlap`
- `match_and_report`
- `cam_stats`
- `paired_permutation_test`
- `cluster_permutation_test_1d`
- `deletion_test_both`
- `extract_raw_stats`
- `get_model_embeddings`
- `GradCAMEngine`
- `plot_mean_cam`
- `plot_distribution_comparison`
- `plot_deletion_comparison`
- `plot_matching_balance`
- `run_post_hoc_analysis`

## src/vix_xai/metrics.py
- `define_events_from_level`
- `auc_ci`
- `precision_recall_at_k`
- `select_references`
- `extract_timewise_embeddings_batch`
- `compute_cam_all`
- `cost_matrix_raw_l1`
- `run_metrics_over_time`

## src/vix_xai/concepts.py
- `norm_colname`
- `resolve_col`
- `benjamini_hochberg`
- `paired_perm`
- `bootstrap_ci`
- `create_single_concept_labels`
- `TCAVExtractorCV`
- `cost_l1_band`
- `compute_cdew_distance`
- `run_cdew_analysis`
- `create_concept_generic`
- `run_concept_dashboard`
