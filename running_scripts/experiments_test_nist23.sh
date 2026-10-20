#!/usr/bin/env bash
# NIST23 hold-out: compounds new in NIST23 form the test set; train the TCN and
# rescore models on the rest and evaluate on it. Run from the repository root;
# raw data as for train_release_v2.3.sh. GPU indices are for one machine.
# The test sets are also run with BUDDY and SIRIUS.
set -e
source running_scripts/compare.sh

DATA=./data/nist23_pkl

# As in train_release_v2.3.sh: Orbitrap trains with AMP, rescore pairs use a
# 50 ppm window
for spec in "qtof agilent nist20 nist23 mona waters gnps" "orbitrap nist20 nist23 mona gnps"; do
    set -- $spec; INST=$1; shift
    AMP=""
    [ $INST = orbitrap ] && AMP=--amp
    CONFIG=./msfiddle/config/fiddle_tcn_$INST.yml
    TCN=./check_point/fiddle_tcn_${INST}_nist23.pt
    RESCORE=./check_point/fiddle_rescore_${INST}_nist23.pt

    python scripts/prepare_msms_nist23.py --dataset "$@" --instrument_type $INST \
        --config_path $CONFIG --pkl_dir $DATA/

    python -u scripts/train_tcn_gpus_cl.py \
        --train_data $DATA/${INST}_train.pkl --val_data $DATA/${INST}_val.pkl \
        --config_path $CONFIG --checkpoint_path $TCN $AMP --device 0 1

    python scripts/prepare_augment_rescore.py \
        --train_data $DATA/${INST}_train.pkl --val_data $DATA/${INST}_val.pkl \
        --config_path $CONFIG --resume_path $TCN --rescore_dir $DATA/ \
        --pos_cap 10 --neg_per_pos 8 --tolerance 50 --device 0
    python scripts/train_rescore.py \
        --train_data $DATA/${INST}_rescore_train.pkl \
        --val_data $DATA/${INST}_rescore_val.pkl \
        --config_path $CONFIG --resume_path $TCN --checkpoint_path $RESCORE --device 0

    compare $DATA/${INST}_test.mgf $INST ${INST}_nist23 \
        --config_path $CONFIG --resume_path $TCN --rescore_resume_path $RESCORE
done
