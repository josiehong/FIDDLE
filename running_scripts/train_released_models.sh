#!/usr/bin/env bash
# Retrain TCN + rescore models for Orbitrap and Q-TOF (031826 data).
#
# Run from the FIDDLE root directory:
#   bash running_scripts/retrain_031826.sh
#
# Adjust --device arguments to match available GPUs before running.

set -e

# ===========================================================================
# Orbitrap
# ===========================================================================

# 1. Regenerate pkl (includes GNPS FTMS spectra)
python prepare_msms.py \
    --dataset nist20 nist23 mona gnps \
    --instrument_type orbitrap \
    --config_path ./config/fiddle_tcn_orbitrap.yml \
    --pkl_dir ./data/cl_pkl_031826/ \
    --maxmin_pick \
    > prepare_msms_031826_orbitrap.out

# 2. Train TCN
python train_tcn_gpus_cl.py \
    --train_data ./data/cl_pkl_031826/orbitrap_maxmin_train.pkl \
    --test_data  ./data/cl_pkl_031826/orbitrap_maxmin_test.pkl \
    --config_path ./config/fiddle_tcn_orbitrap.yml \
    --checkpoint_path ./check_point/fiddle_tcn_orbitrap_031826.pt \
    --device 0 1 \
    > train_tcn_031826_orbitrap.out

# 3. Prepare and augment rescore training data
python prepare_augment_rescore.py \
    --train_data ./data/cl_pkl_031826/orbitrap_maxmin_train.pkl \
    --test_data  ./data/cl_pkl_031826/orbitrap_maxmin_test.pkl \
    --config_path ./config/fiddle_tcn_orbitrap.yml \
    --resume_path ./check_point/fiddle_tcn_orbitrap_031826.pt \
    --rescore_dir ./data/cl_pkl_031826 \
    --pos_cap 10 --neg_per_pos 8 --tolerance 50 \
    --num_workers 8 \
    --device 0

# 4. Train rescore model
python train_rescore.py \
    --train_data ./data/cl_pkl_031826/orbitrap_maxmin_rescore_train.pkl \
    --test_data  ./data/cl_pkl_031826/orbitrap_maxmin_rescore_test.pkl \
    --config_path ./config/fiddle_tcn_orbitrap.yml \
    --resume_path ./check_point/fiddle_tcn_orbitrap_031826.pt \
    --checkpoint_path ./check_point/fiddle_rescore_orbitrap_031826.pt \
    --device 0 \
    > train_rescore_031826_orbitrap.out

# ===========================================================================
# Q-TOF
# ===========================================================================

# 1. Regenerate pkl (includes GNPS FTMS spectra)
python prepare_msms.py \
    --dataset nist20 nist23 mona gnps \
    --instrument_type qtof \
    --config_path ./config/fiddle_tcn_qtof.yml \
    --pkl_dir ./data/cl_pkl_031826/ \
    --maxmin_pick \
    > prepare_msms_031826_qtof.out

# 2. Train TCN
python train_tcn_gpus_cl.py \
    --train_data ./data/cl_pkl_031826/qtof_maxmin_train.pkl \
    --test_data  ./data/cl_pkl_031826/qtof_maxmin_test.pkl \
    --config_path ./config/fiddle_tcn_qtof.yml \
    --checkpoint_path ./check_point/fiddle_tcn_qtof_031826.pt \
    --device 2 3 \
    > train_tcn_031826_qtof.out

# 3. Prepare and augment rescore training data
python prepare_augment_rescore.py \
    --train_data ./data/cl_pkl_031826/qtof_maxmin_train.pkl \
    --test_data  ./data/cl_pkl_031826/qtof_maxmin_test.pkl \
    --config_path ./config/fiddle_tcn_qtof.yml \
    --resume_path ./check_point/fiddle_tcn_qtof_031826.pt \
    --rescore_dir ./data/cl_pkl_031826 \
    --pos_cap 10 --neg_per_pos 8 --tolerance 50 \
    --num_workers 8 \
    --device 2

# 4. Train rescore model
python train_rescore.py \
    --train_data ./data/cl_pkl_031826/qtof_maxmin_rescore_train.pkl \
    --test_data  ./data/cl_pkl_031826/qtof_maxmin_rescore_test.pkl \
    --config_path ./config/fiddle_tcn_qtof.yml \
    --resume_path ./check_point/fiddle_tcn_qtof_031826.pt \
    --checkpoint_path ./check_point/fiddle_rescore_qtof_031826.pt \
    --device 1 \
    > train_rescore_031826_qtof.out

echo "All done."
