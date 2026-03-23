# --------------------------
# Data preprocessing
# --------------------------
python prepare_msms_all.py --dataset mona --instrument_type qtof --config_path ./config/fiddle_tcn_demo.yml --pkl_dir ./data/demo_cl_pkl/

# --------------------------
# Train & Predict on DEMO
# --------------------------
nohup python -u train_tcn_gpus.py \
--train_data ./data/demo_cl_pkl/qtof_random_train.pkl \
--test_data ./data/demo_cl_pkl/qtof_random_test.pkl \
--config_path ./config/fiddle_tcn_demo.yml \
--checkpoint_path ./check_point/fiddle_tcn_demo_qtof.pt \
--result_path ./result/fiddle_tcn_demo_qtof.csv --device 6 7 > fiddle_tcn_demo_qtof.out

nohup python -u train_tcn_gpus_cl.py \
--train_data ./data/demo_cl_pkl/qtof_random_train.pkl \
--test_data ./data/demo_cl_pkl/qtof_random_test.pkl \
--config_path ./config/fiddle_tcn_demo_wo_da.yml \
--checkpoint_path ./check_point/fiddle_tcn_demo_qtof_cl_wo_da.pt \
--result_path ./result/fiddle_tcn_demo_qtof_cl_wo_da.csv --device 6 7 > fiddle_tcn_demo_qtof_cl_wo_da.out

nohup python -u train_tcn_gpus_cl.py \
--train_data ./data/demo_cl_pkl/qtof_random_train.pkl \
--test_data ./data/demo_cl_pkl/qtof_random_test.pkl \
--config_path ./config/fiddle_tcn_demo.yml \
--checkpoint_path ./check_point/fiddle_tcn_demo_qtof_cl.pt \
--result_path ./result/fiddle_tcn_demo_qtof_cl.csv --device 6 7 > fiddle_tcn_demo_qtof_cl.out

# FIDDLES (fdr model)
python prepare_rescore.py \
--train_data ./data/demo_cl_pkl/qtof_random_train.pkl \
--test_data ./data/demo_cl_pkl/qtof_random_test.pkl \
--config_path ./config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_demo_qtof_cl.pt \
--rescore_dir ./data/demo_cl_pkl/ \
--device 4 5
nohup python train_rescore.py --train_data ./data/demo_cl_pkl/qtof_random_rescore_train.pkl \
--test_data ./data/demo_cl_pkl/qtof_random_rescore_test.pkl \
--config_path ./config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_demo_qtof_cl.pt \
--transfer \
--checkpoint_path ./check_point/fiddle_rescore_demo_qtof_cl.pt \
--device 4 5 > fiddle_rescore_demo_qtof_cl.out

# FIDDLES (test)
python run_fiddle.py --test_data ./data/demo_cl_pkl/qtof_random_test.mgf \
--config_path ./config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_demo_qtof_cl.pt \
--rescore_resume_path ./check_point/fiddle_rescore_demo_qtof_cl.pt \
--result_path ./result/fiddle_demo_qtof_cl.csv --device 4 5

# FIDDLE (test with interpretation)
python run_fiddle_w_interp.py \
--test_data ./data/demo_cl_pkl/qtof_random_test.mgf \
--config_path ./config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_demo_qtof_cl.pt \
--rescore_resume_path ./check_point/fiddle_rescore_demo_qtof_cl.pt \
--result_path ./result/fiddle_demo_qtof_cl_w_interp.csv \
--enable_shap \
--shap_save_path ./shap_results \
--device 4 5