# ----------------------------------------
# Experiments on internal test datasets: 
# unique compounds in NIST23
# ID: 100724
# ----------------------------------------
# I. Data preprocessing
# ----------------------------------------

# 1. QTOF --------------------------------
python prepare_msms_nist23.py \
--dataset agilent nist20 nist23 mona waters gnps \
--instrument_type qtof \
--config_path ./config/fiddle_tcn_qtof.yml \
--pkl_dir ./data/cl_pkl_1007/

# 2. Orbitrap -----------------------------
python prepare_msms_nist23.py \
--dataset nist20 nist23 mona gnps \
--instrument_type orbitrap \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--pkl_dir ./data/cl_pkl_1007/



# --------------------------
# II. Train on QTOF
# --------------------------
# FIDDLE
python -u train_tcn_gpus_cl.py \
--train_data ./data/cl_pkl_1007/qtof_train.pkl \
--test_data ./data/cl_pkl_1007/qtof_test.pkl \
--additional_f_data ./data/additional_formula.pkl \
--config_path ./config/fiddle_tcn_qtof.yml \
--checkpoint_path ./check_point/fiddle_tcn_qtof_100724.pt \
--result_path ./result/fiddle_tcn_qtof_100724.csv --device 4 5 

# FIDDLES (fdr model)
python prepare_rescore.py \
--train_data ./data/cl_pkl_1007/qtof_train.pkl \
--test_data ./data/cl_pkl_1007/qtof_test.pkl \
--config_path ./config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
--rescore_dir ./data/cl_pkl_1007/ \
--device 4 5
python train_rescore.py \
--train_data ./data/cl_pkl_1007/qtof_rescore_train.pkl \
--test_data ./data/cl_pkl_1007/qtof_rescore_test.pkl \
--config_path ./config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
--transfer \
--checkpoint_path ./check_point/fiddle_fdr_qtof_100724.pt \
--device 4 5 



# --------------------------
# III. Train on Qrbitrap
# --------------------------
# FIDDLE
python -u train_tcn_gpus_cl.py \
--train_data ./data/cl_pkl_1007/orbitrap_train.pkl \
--test_data ./data/cl_pkl_1007/orbitrap_test.pkl \
--additional_f_data ./data/additional_formula.pkl \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--checkpoint_path ./check_point/fiddle_tcn_orbitrap_100724.pt \
--result_path ./result/fiddle_tcn_orbitrap_100724.csv --device 4 7 

# FIDDLES (fdr model)
python prepare_rescore.py \
--train_data ./data/cl_pkl_1007/orbitrap_train.pkl \
--test_data ./data/cl_pkl_1007/orbitrap_test.pkl \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--resume_path ./check_point/fiddle_tcn_orbitrap_100724.pt \
--rescore_dir ./data/cl_pkl_1007/ \
--device 4 5 
python train_rescore.py --train_data ./data/cl_pkl_1007/orbitrap_rescore_train.pkl \
--test_data ./data/cl_pkl_1007/orbitrap_rescore_test.pkl \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--resume_path ./check_point/fiddle_tcn_orbitrap_100724.pt \
--transfer \
--checkpoint_path ./check_point/fiddle_fdr_orbitrap_100724.pt \
--device 4 5 



# --------------------------
# IV. Test on QTOF
# --------------------------
# FIDDLES
python run_fiddle.py --test_data ./data/cl_pkl_1007/qtof_test.mgf \
--config_path ./config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
--rescore_resume_path ./check_point/fiddle_fdr_qtof_100724.pt \
--result_path ./result/fiddle_qtof_100724.csv --device 5

# FIDDLE + BUDDY
python run_fiddle.py --test_data ./data/cl_pkl_1007/qtof_test.mgf \
--config_path ./config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
--rescore_resume_path ./check_point/fiddle_fdr_qtof_100724.pt \
--buddy_path ./run_buddy_1007/buddy_qtof_test_1007.csv \
--result_path ./result/two_qtof_test_100724.csv --device 5



# --------------------------
# V. Test on Orbitrap
# --------------------------
# FIDDLES
python run_fiddle.py --test_data ./data/cl_pkl_1007/orbitrap_test.mgf \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--resume_path ./check_point/fiddle_tcn_orbitrap_100724.pt \
--rescore_resume_path ./check_point/fiddle_fdr_orbitrap_100724.pt \
--result_path ./result/fiddle_orbitrap_100724.csv --device 4

# FIDDLE + BUDDY
python run_fiddle.py --test_data ./data/cl_pkl_1007/orbitrap_test.mgf \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--resume_path ./check_point/fiddle_tcn_orbitrap_100724.pt \
--rescore_resume_path ./check_point/fiddle_fdr_orbitrap_100724.pt \
--buddy_path ./run_buddy_1007/buddy_orbitrap_test_1007.csv \
--result_path ./result/two_orbitrap_test_100724.csv --device 4



# --------------------------
# VI. Test on CASMI and EMBL
# --------------------------
python run_fiddle.py --test_data ./data/casmi2016.mgf \
                --config_path ./config/fiddle_tcn_casmi.yml \
                --resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
                --rescore_resume_path ./check_point/fiddle_fdr_qtof_100724.pt \
                --result_path ./result/fiddle_casmi16_exnist23.csv 
python run_fiddle.py --test_data ./data/casmi2016.mgf \
                --config_path ./config/fiddle_tcn_casmi.yml \
                --resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
                --rescore_resume_path ./check_point/fiddle_fdr_qtof_100724.pt \
                --buddy_path ./run_buddy/buddy_casmi2016.csv \
                --result_path ./result/two_casmi16_exnist23.csv 
python run_fiddle.py --test_data ./data/casmi2016.mgf \
                --config_path ./config/fiddle_tcn_casmi.yml \
                --resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
                --rescore_resume_path ./check_point/fiddle_fdr_qtof_100724.pt \
                --buddy_path ./run_buddy/buddy_casmi2016.csv \
                --sirius_path ./run_sirius/sirius_casmi2016.csv \
                --result_path ./result/all_casmi16_exnist23.csv 

python run_fiddle.py --test_data ./data/casmi2017.mgf \
                --config_path ./config/fiddle_tcn_casmi.yml \
                --resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
                --rescore_resume_path ./check_point/fiddle_fdr_qtof_100724.pt \
                --result_path ./result/fiddle_casmi17_exnist23.csv 
python run_fiddle.py --test_data ./data/casmi2017.mgf \
                --config_path ./config/fiddle_tcn_casmi.yml \
                --resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
                --rescore_resume_path ./check_point/fiddle_fdr_qtof_100724.pt \
                --buddy_path ./run_buddy/buddy_casmi2017.csv \
                --result_path ./result/two_casmi17_exnist23.csv 
python run_fiddle.py --test_data ./data/casmi2017.mgf \
                --config_path ./config/fiddle_tcn_casmi.yml \
                --resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
                --rescore_resume_path ./check_point/fiddle_fdr_qtof_100724.pt \
                --buddy_path ./run_buddy/buddy_casmi2017.csv \
                --sirius_path ./run_sirius/sirius_casmi2017.csv \
                --result_path ./result/all_casmi17_exnist23.csv 

python run_fiddle.py --test_data ./data/embl_mcf_2.0.mgf \
                --config_path ./config/fiddle_tcn_embl.yml \
                --resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
                --rescore_resume_path ./check_point/fiddle_fdr_qtof_100724.pt \
                --result_path ./result/fiddle_embl_exnist23.csv 
python run_fiddle.py --test_data ./data/embl_mcf_2.0.mgf \
                --config_path ./config/fiddle_tcn_embl.yml \
                --resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
                --rescore_resume_path ./check_point/fiddle_fdr_qtof_100724.pt \
                --buddy_path ./run_buddy/buddy_embl.csv \
                --result_path ./result/two_embl_exnist23.csv 
python run_fiddle.py --test_data ./data/embl_mcf_2.0.mgf \
                --config_path ./config/fiddle_tcn_embl.yml \
                --resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
                --rescore_resume_path ./check_point/fiddle_fdr_qtof_100724.pt \
                --buddy_path ./run_buddy/buddy_embl.csv \
                --sirius_path ./run_sirius/sirius_embl.csv \
                --result_path ./result/all_embl_exnist23.csv 