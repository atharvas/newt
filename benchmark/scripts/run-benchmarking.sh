#!/bin/bash -i
# This file runs benchmarking code only for the newt dataset.
# After this file executes, run scripts/build-classifiers.sh
# Make sure this file is executed from newt/benchmark directory.
conda activate newtenv
set -x
{
CUDA_VISIBLE_DEVICES=0 TF_CPP_MIN_LOG_LEVEL=3 python tf_extract_features.py --newt_feature_dir newt_features --fg_feature_dir fg_features --batch_size 64 --x4_batch_size 16 --overwrite --metadata_is_df
CUDA_VISIBLE_DEVICES=0 python pt_extract_features.py --newt_feature_dir newt_features --fg_feature_dir fg_features --batch_size 64
python evaluate_linear_models.py --feature_dir newt_features --result_dir newt_results_linearsvc_1000_standardize_noramlize_grid_search --model linearsvc --max_iter 1000 --standardize --normalize --grid_search --overwrite
} &>> logs/run-benchmarking.log