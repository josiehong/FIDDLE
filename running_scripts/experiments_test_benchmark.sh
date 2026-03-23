# ----------------------------------------
# Experiments on external test datasets: 
# CASMI 2016, CASMI 2017, and, EMBL MCF 2.0
# ID: 092724
# ----------------------------------------
# I. Data preprocessing
# ----------------------------------------
# the structure of the benchmark dataset folder (`./data/benchmark`) should be:
# |- data
#   |- benchmark
#     |- casmi2016
#       |- MoNA-export-CASMI_2016.sdf
#     |- casmi2017
#       |- CASMI-solutions.csv
#       |- Chal1to45Summary.csv
#       |- challenges-001-045-msms-mgf-20170908 (unzip challenges-001-045-msms-mgf-20170908.zip)
#     |- embl
#       |- MoNA-export-EMBL-MCF_2.0_HRMS_Library.sdf
# ----------------------------------------

# 1. QTOF --------------------------------
python prepare_msms.py \
--dataset agilent nist20 nist23 mona waters gnps \
--instrument_type qtof \
--config_path ./config/fiddle_tcn_qtof.yml \
--pkl_dir ./data/cl_pkl_031826/ \
--test_title_list ./data/qtof_test_title_list_031826.txt \
--maxmin_pick

# 2. Orbitrap -----------------------------
python prepare_msms.py \
--dataset nist20 nist23 mona gnps \
--instrument_type orbitrap \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--pkl_dir ./data/cl_pkl_031826/ \
--test_title_list ./data/orbitrap_test_title_list_031826.txt \
--maxmin_pick

# 3. CASMI 2016, 2017 ---------------------
python casmi2mgf.py --data_config_path ./config/fiddle_tcn_casmi.yml

# 4. EMBL MCF 2.0 -------------------------
python embl2mgf.py --raw_path ./data/benchmark/embl/MoNA-export-EMBL-MCF_2.0_HRMS_Library.sdf \
                --mgf_path ./data/embl_mcf_2.0.mgf \
                --data_config_path ./config/fiddle_tcn_embl.yml



# --------------------------
# II. Train on QTOF
# --------------------------

# FIDDLE
nohup python -u train_tcn_gpus_cl.py \
--train_data ./data/cl_pkl_031826/qtof_maxmin_train.pkl \
--test_data ./data/cl_pkl_031826/qtof_maxmin_test.pkl \
--config_path ./config/fiddle_tcn_qtof.yml \
--checkpoint_path ./check_point/fiddle_tcn_qtof_031826.pt \
--resume_path ./check_point/fiddle_tcn_qtof_031826.pt \
--device 6 7 >> fiddle_tcn_qtof_031826.out 

# FIDDLES (fdr model)
python prepare_rescore.py \
--train_data ./data/cl_pkl_031826/qtof_maxmin_train.pkl \
--test_data ./data/cl_pkl_031826/qtof_maxmin_test.pkl \
--config_path ./config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_qtof_031826.pt \
--rescore_dir ./data/cl_pkl_031826/ \
--device 4 5
python train_rescore.py \
--train_data ./data/cl_pkl_031826/qtof_maxmin_rescore_train.pkl \
--test_data ./data/cl_pkl_031826/qtof_maxmin_rescore_test.pkl \
--config_path ./config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_qtof_031826.pt \
--transfer \
--checkpoint_path ./check_point/fiddle_rescore_qtof_031826.pt \
--device 4 5 



# --------------------------
# III. Train on Qrbitrap
# --------------------------
# FIDDLE
python -u train_tcn_gpus_cl.py \
--train_data ./data/cl_pkl_031826/orbitrap_maxmin_train.pkl \
--test_data ./data/cl_pkl_031826/orbitrap_maxmin_test.pkl \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--checkpoint_path ./check_point/fiddle_tcn_orbitrap_031826.pt \
--resume_path ./check_point/fiddle_tcn_orbitrap_031826.pt \
--device 4 5 

# FIDDLES (fdr model)
python prepare_rescore.py \
--train_data ./data/cl_pkl_031826/orbitrap_maxmin_train.pkl \
--test_data ./data/cl_pkl_031826/orbitrap_maxmin_test.pkl \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--resume_path ./check_point/fiddle_tcn_orbitrap_031826.pt \
--rescore_dir ./data/cl_pkl_031826/ \
--device 4 5 
python train_rescore.py \
--train_data ./data/cl_pkl_031826/orbitrap_maxmin_rescore_train.pkl \
--test_data ./data/cl_pkl_031826/orbitrap_maxmin_rescore_test.pkl \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--resume_path ./check_point/fiddle_tcn_orbitrap_031826.pt \
--transfer \
--checkpoint_path ./check_point/fiddle_rescore_orbitrap_031826.pt \
--device 4 5



# ----------------------------------------
# IV. test on CASMI
# ----------------------------------------
# FIDDLE
python run_fiddle.py --test_data ./data/casmi2016.mgf \
                --config_path ./config/fiddle_tcn_casmi.yml \
                --resume_path ./check_point/fiddle_tcn_qtof_031826.pt \
                --rescore_resume_path ./check_point/fiddle_rescore_qtof_031826.pt \
                --result_path ./result/fiddle_casmi16.csv 

python run_fiddle.py --test_data ./data/casmi2017.mgf \
                --config_path ./config/fiddle_tcn_casmi.yml \
                --resume_path ./check_point/fiddle_tcn_qtof_031826.pt \
                --rescore_resume_path ./check_point/fiddle_rescore_qtof_031826.pt \
                --result_path ./result/fiddle_casmi17.csv 

# FIDDLE + BUDDY
python run_fiddle.py --test_data ./data/casmi2016.mgf \
                    --config_path ./config/fiddle_tcn_qtof.yml \
                    --resume_path ./check_point/fiddle_tcn_qtof_031826.pt \
                    --rescore_resume_path ./check_point/fiddle_rescore_qtof_031826.pt \
                    --buddy_path ./run_buddy/buddy_casmi2016.csv \
                    --result_path ./result/two_casmi16.csv 

python run_fiddle.py --test_data ./data/casmi2017.mgf \
                    --config_path ./config/fiddle_tcn_qtof.yml \
                    --resume_path ./check_point/fiddle_tcn_qtof_031826.pt \
                    --rescore_resume_path ./check_point/fiddle_rescore_qtof_031826.pt \
                    --buddy_path ./run_buddy/buddy_casmi2017.csv \
                    --result_path ./result/two_casmi17.csv 



# ----------------------------------------
# V. test on EMBL MCF 2.0
# ----------------------------------------
# FIDDLE
python run_fiddle.py --test_data ./data/embl_mcf_2.0.mgf \
                --config_path ./config/fiddle_tcn_embl.yml \
                --resume_path ./check_point/fiddle_tcn_qtof_031826.pt \
                --rescore_resume_path ./check_point/fiddle_rescore_qtof_031826.pt \
                --result_path ./result/fiddle_embl.csv 

# FIDDLE + BUDDY
python run_fiddle.py --test_data ./data/embl_mcf_2.0.mgf \
                --config_path ./config/fiddle_tcn_embl.yml \
                --resume_path ./check_point/fiddle_tcn_qtof_031826.pt \
                --rescore_resume_path ./check_point/fiddle_rescore_qtof_031826.pt \
                --buddy_path ./run_buddy/buddy_embl.csv \
                --result_path ./result/two_embl.csv 

