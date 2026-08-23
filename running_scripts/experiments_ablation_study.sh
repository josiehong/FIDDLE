# --------------------------
# Data preprocessing for 
# all noise times and resolution of 0.2
# --------------------------
python scripts/prepare_msms_ablation.py \
--pkl_dir ./data/ablation/ \
--train_ratio 0.9 \
--config_qtof_path ./msfiddle/config/fiddle_tcn_qtof.yml \
--config_orbitrap_path ./msfiddle/config/fiddle_tcn_orbitrap.yml \
--random_seed 42

# -------------------------------
# Ablation Study of Noise Times
# -------------------------------
# QTOF (abn1)
nohup python -u train_tcn_gpus_cl.py \
--train_data ./data/ablation/ablation_qtof_train.pkl \
--test_data ./data/ablation/ablation_qtof_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_qtof_abn1.yml \
--checkpoint_path ./check_point/fiddle_tcn_qtof_ab_n1.pt \
--result_path ./result/fiddle_tcn_qtof_ab_n1.csv --device 4 5 > fiddle_tcn_qtof_ab_n1.out & 

# QTOF (abn3)
nohup python -u train_tcn_gpus_cl.py \
--train_data ./data/ablation/ablation_qtof_train.pkl \
--test_data ./data/ablation/ablation_qtof_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_qtof_abn3.yml \
--checkpoint_path ./check_point/fiddle_tcn_qtof_ab_n3.pt \
--result_path ./result/fiddle_tcn_qtof_ab_n3.csv --device 2 3 > fiddle_tcn_qtof_ab_n3.out & 

# QTOF (abn5)
nohup python -u train_tcn_gpus_cl.py \
--train_data ./data/ablation/ablation_qtof_train.pkl \
--test_data ./data/ablation/ablation_qtof_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_qtof_abn5.yml \
--checkpoint_path ./check_point/fiddle_tcn_qtof_ab_n5.pt \
--result_path ./result/fiddle_tcn_qtof_ab_n5.csv --device 2 3 > fiddle_tcn_qtof_ab_n5.out & 

# Orbitrap (abn1)
nohup python -u train_tcn_gpus_cl.py \
--train_data ./data/ablation/ablation_orbitrap_train.pkl \
--test_data ./data/ablation/ablation_orbitrap_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_orbitrap_abn1.yml \
--checkpoint_path ./check_point/fiddle_tcn_orbitrap_ab_n1.pt \
--result_path ./result/fiddle_tcn_orbitrap_ab_n1.csv --device 4 5 > fiddle_tcn_orbitrap_ab_n1.out & 

# Orbitrap (abn3)
nohup python -u train_tcn_gpus_cl.py \
--train_data ./data/ablation/ablation_orbitrap_train.pkl \
--test_data ./data/ablation/ablation_orbitrap_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_orbitrap_abn3.yml \
--checkpoint_path ./check_point/fiddle_tcn_orbitrap_ab_n3.pt \
--result_path ./result/fiddle_tcn_orbitrap_ab_n3.csv --device 0 1 > fiddle_tcn_orbitrap_ab_n3.out & 

# Orbitrap (abn5)
nohup python -u train_tcn_gpus_cl.py \
--train_data ./data/ablation/ablation_orbitrap_train.pkl \
--test_data ./data/ablation/ablation_orbitrap_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_orbitrap_abn5.yml \
--checkpoint_path ./check_point/fiddle_tcn_orbitrap_ab_n5.pt \
--result_path ./result/fiddle_tcn_orbitrap_ab_n5.csv --device 0 1 > fiddle_tcn_orbitrap_ab_n5.out & 

# -------------------------------
# Ablation Study of Resolution
# -------------------------------
# QTOF (resolution_1)
python scripts/prepare_msms_ablation.py \
--pkl_dir ./data/ablation_res_1/ \
--train_ratio 0.9 \
--config_qtof_path ./msfiddle/config/fiddle_tcn_qtof_resolution_1.yml \
--config_orbitrap_path ./msfiddle/config/fiddle_tcn_orbitrap_resolution_1.yml \
--random_seed 42
nohup python -u train_tcn_gpus_cl.py \
--train_data ./data/ablation_res_1/ablation_qtof_train.pkl \
--test_data ./data/ablation_res_1/ablation_qtof_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_qtof_resolution_1.yml \
--checkpoint_path ./check_point/fiddle_tcn_qtof_ab_res_1.pt \
--result_path ./result/fiddle_tcn_qtof_ab_res_1.csv --device 4 5 > fiddle_tcn_qtof_ab_res_1.out &

# QTOF (resolution_p1)
python scripts/prepare_msms_ablation.py \
--pkl_dir ./data/ablation_res_1/ \
--train_ratio 0.9 \
--config_qtof_path ./msfiddle/config/fiddle_tcn_qtof_resolution_p1.yml \
--config_orbitrap_path ./msfiddle/config/fiddle_tcn_orbitrap_resolution_p1.yml \
--random_seed 42
nohup python -u train_tcn_gpus_cl.py \
--train_data ./data/ablation/ablation_qtof_train.pkl \
--test_data ./data/ablation/ablation_qtof_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_qtof_resolution_p1.yml \
--checkpoint_path ./check_point/fiddle_tcn_qtof_ab_res_p1.pt \
--result_path ./result/fiddle_tcn_qtof_ab_res_p1.csv --device 4 5 > fiddle_tcn_qtof_ab_res_p1.out &

# Orbitrap (resolution_1)
nohup python -u train_tcn_gpus_cl.py \
--train_data ./data/ablation/ablation_orbitrap_train.pkl \
--test_data ./data/ablation/ablation_orbitrap_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_orbitrap_resolution_1.yml \
--checkpoint_path ./check_point/fiddle_tcn_orbitrap_ab_res_1.pt \
--result_path ./result/fiddle_tcn_orbitrap_ab_res_1.csv --device 4 5 > fiddle_tcn_orbitrap_ab_res_1.out &

# Orbitrap (resolution_p1)
nohup python -u train_tcn_gpus_cl.py \
--train_data ./data/ablation/ablation_orbitrap_train.pkl \
--test_data ./data/ablation/ablation_orbitrap_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_orbitrap_resolution_p1.yml \
--checkpoint_path ./check_point/fiddle_tcn_orbitrap_ab_res_p1.pt \
--result_path ./result/fiddle_tcn_orbitrap_ab_res_p1.csv --device 4 5 > fiddle_tcn_orbitrap_ab_res_p1.out &

# -------------------------------
# Ablation Study of Instrument Types
# -------------------------------
python scripts/prepare_msms_ablation_ins.py \
--pkl_dir ./data/ablation_ins/ \
--train_ratio 0.9 \
--config_qtof_path ./msfiddle/config/fiddle_tcn_qtof.yml \
--config_orbitrap_path ./msfiddle/config/fiddle_tcn_orbitrap.yml \
--random_seed 42

# QTOF (ab_ins)
nohup python -u train_tcn_gpus_cl.py \
--train_data ./data/ablation_ins/ablation_qtof_train.pkl \
--test_data ./data/ablation_ins/ablation_qtof_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_qtof.yml \
--checkpoint_path ./check_point/fiddle_tcn_qtof_ab_ins.pt \
--result_path ./result/fiddle_tcn_qtof_ab_ins.csv --device 4 5 > fiddle_tcn_qtof_ab_ins.out &

# Orbitrap (ab_ins)
nohup python -u train_tcn_gpus_cl.py \
--train_data ./data/ablation_ins/ablation_orbitrap_train.pkl \
--test_data ./data/ablation_ins/ablation_orbitrap_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_orbitrap.yml \
--checkpoint_path ./check_point/fiddle_tcn_orbitrap_ab_ins.pt \
--result_path ./result/fiddle_tcn_orbitrap_ab_ins.csv --device 4 5 > fiddle_tcn_orbitrap_ab_ins.out &

# Cross instrument test
python -u train_tcn_gpus_cl.py \
--train_data ./data/ablation_ins/ablation_qtof_train.pkl \
--test_data ./data/ablation_ins/ablation_qtof_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_orbitrap_testonly.yml \
--resume_path ./check_point/fiddle_tcn_orbitrap_ab_ins.pt \
--checkpoint_path ./check_point/fiddle_tcn_orbitrap_ab_ins.pt \
--result_path ./result/fiddle_tcn_orbitrap_ab_crossins.csv --device 4 5
python -u train_tcn_gpus_cl.py \
--train_data ./data/ablation_ins/ablation_orbitrap_train.pkl \
--test_data ./data/ablation_ins/ablation_orbitrap_test.pkl \
--config_path ./msfiddle/config/fiddle_tcn_qtof_testonly.yml \
--resume_path ./check_point/fiddle_tcn_qtof_ab_ins.pt \
--checkpoint_path ./check_point/fiddle_tcn_qtof_ab_ins.pt \
--result_path ./result/fiddle_tcn_qtof_ab_crossins.csv --device 4 5
