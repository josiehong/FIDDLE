# ----------------------------------------
# Experiments on chimeric spectra
# using trained models 100724
# ----------------------------------------
# I. Data preprocessing
# ----------------------------------------
python prepare_msms_chimeric.py --mgf_data ./data/cl_pkl_1007/qtof_test.mgf \
--config_path ./config/fiddle_tcn_qtof.yml \
--out_dir ./data/chimeric/ --mix_ratios 90 70 50 --num_pairs 1000 --mz_tol 0.005
# BUDDY and SIRIUS QTOF
python mgf_instances.py --input_path ./data/chimeric/qtof_test_chimeric90.mgf \
--output_dir ./data_instances/qtof_chimeric90/ \
--log ./data_instances/qtof_log_chimeric90.csv
python mgf_instances.py --input_path ./data/chimeric/qtof_test_chimeric70.mgf \
--output_dir ./data_instances/qtof_chimeric70/ \
--log ./data_instances/qtof_log_chimeric70.csv
python mgf_instances.py --input_path ./data/chimeric/qtof_test_chimeric50.mgf \
--output_dir ./data_instances/qtof_chimeric50/ \
--log ./data_instances/qtof_log_chimeric50.csv

python prepare_msms_chimeric.py --mgf_data ./data/cl_pkl_1007/orbitrap_test.mgf \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--out_dir ./data/chimeric/ --mix_ratios 90 70 50 --num_pairs 1000 --mz_tol 0.005
# BUDDY and SIRIUS Orbitrap
python mgf_instances.py --input_path ./data/chimeric/orbitrap_test_chimeric90.mgf \
--output_dir ./data_instances/orbitrap_chimeric90/ \
--log ./data_instances/orbitrap_log_chimeric90.csv
python mgf_instances.py --input_path ./data/chimeric/orbitrap_test_chimeric70.mgf \
--output_dir ./data_instances/orbitrap_chimeric70/ \
--log ./data_instances/orbitrap_log_chimeric70.csv
python mgf_instances.py --input_path ./data/chimeric/orbitrap_test_chimeric50.mgf \
--output_dir ./data_instances/orbitrap_chimeric50/ \
--log ./data_instances/orbitrap_log_chimeric50.csv

# --------------------------
# II. Test on chimeric spectra (FIDDLE)
# --------------------------
# For QTOF
python run_fiddle.py --test_data ./data/chimeric/qtof_test_chimeric90.mgf \
--config_path ./config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
--rescore_resume_path ./check_point/fiddle_fdr_qtof_100724.pt \
--result_path ./result/fiddle_qtof_chimeric90_100724.csv --device 6 

python run_fiddle.py --test_data ./data/chimeric/qtof_test_chimeric70.mgf \
--config_path ./config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
--rescore_resume_path ./check_point/fiddle_fdr_qtof_100724.pt \
--result_path ./result/fiddle_qtof_chimeric70_100724.csv --device 6

python run_fiddle.py --test_data ./data/chimeric/qtof_test_chimeric50.mgf \
--config_path ./config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_qtof_100724.pt \
--rescore_resume_path ./check_point/fiddle_fdr_qtof_100724.pt \
--result_path ./result/fiddle_qtof_chimeric50_100724.csv --device 6

# For Orbitrap
python run_fiddle.py --test_data ./data/chimeric/orbitrap_test_chimeric90.mgf \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--resume_path ./check_point/fiddle_tcn_orbitrap_100724.pt \
--rescore_resume_path ./check_point/fiddle_fdr_orbitrap_100724.pt \
--result_path ./result/fiddle_orbitrap_chimeric90_100724.csv --device 6

python run_fiddle.py --test_data ./data/chimeric/orbitrap_test_chimeric70.mgf \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--resume_path ./check_point/fiddle_tcn_orbitrap_100724.pt \
--rescore_resume_path ./check_point/fiddle_fdr_orbitrap_100724.pt \
--result_path ./result/fiddle_orbitrap_chimeric70_100724.csv --device 6

python run_fiddle.py --test_data ./data/chimeric/orbitrap_test_chimeric50.mgf \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--resume_path ./check_point/fiddle_tcn_orbitrap_100724.pt \
--rescore_resume_path ./check_point/fiddle_fdr_orbitrap_100724.pt \
--result_path ./result/fiddle_orbitrap_chimeric50_100724.csv --device 6

# --------------------------
# III. Test on chimeric spectra (BUDDY)
# --------------------------
# For QTOF
python run_buddy.py --instrument_type qtof --top_k 5 \
--input_dir ./data_instances/qtof_chimeric90/ \
--result_path ./run_buddy_1007/buddy_qtof_chimeric90_1007.csv

python run_buddy.py --instrument_type qtof --top_k 5 \
--input_dir ./data_instances/qtof_chimeric70/ \
--result_path ./run_buddy_1007/buddy_qtof_chimeric70_1007.csv

python run_buddy.py --instrument_type qtof --top_k 5 \
--input_dir ./data_instances/qtof_chimeric50/ \
--result_path ./run_buddy_1007/buddy_qtof_chimeric50_1007.csv

# For Orbitrap
python run_buddy.py --instrument_type orbitrap --top_k 5 \
--input_dir ./data_instances/orbitrap_chimeric90/ \
--result_path ./run_buddy_1007/buddy_orbitrap_chimeric90_1007.csv

python run_buddy.py --instrument_type orbitrap --top_k 5 \
--input_dir ./data_instances/orbitrap_chimeric70/ \
--result_path ./run_buddy_1007/buddy_orbitrap_chimeric70_1007.csv

python run_buddy.py --instrument_type orbitrap --top_k 5 \
--input_dir ./data_instances/orbitrap_chimeric50/ \
--result_path ./run_buddy_1007/buddy_orbitrap_chimeric50_1007.csv

# --------------------------
# IV. Test on chimeric spectra (SIRIUS)
# --------------------------
# QTOF
python -u run_sirius.py --instrument_type qtof \
--input_dir ./data_instances/qtof_chimeric90/ \
--output_dir ./run_sirius_1007/qtof_chimeric90_sirius_output/ \
--summary_dir ./run_sirius_1007/qtof_chimeric90_sirius_summary/ \
--output_log_dir ./run_sirius_1007/qtof_chimeric90_sirius_log/ \
--input_log ./data_instances/qtof_log_chimeric90.csv \
--output_log ./run_sirius_1007/sirius_qtof_chimeric90_test_1007.csv

python -u run_sirius.py --instrument_type qtof \
--input_dir ./data_instances/qtof_chimeric70/ \
--output_dir ./run_sirius_1007/qtof_chimeric70_sirius_output/ \
--summary_dir ./run_sirius_1007/qtof_chimeric70_sirius_summary/ \
--output_log_dir ./run_sirius_1007/qtof_chimeric70_sirius_log/ \
--input_log ./data_instances/qtof_log_chimeric70.csv \
--output_log ./run_sirius_1007/sirius_qtof_chimeric70_test_1007.csv

python -u run_sirius.py --instrument_type qtof \
--input_dir ./data_instances/qtof_chimeric50/ \
--output_dir ./run_sirius_1007/qtof_chimeric50_sirius_output/ \
--summary_dir ./run_sirius_1007/qtof_chimeric50_sirius_summary/ \
--output_log_dir ./run_sirius_1007/qtof_chimeric50_sirius_log/ \
--input_log ./data_instances/qtof_log_chimeric50.csv \
--output_log ./run_sirius_1007/sirius_qtof_chimeric50_test_1007.csv

# Orbitrap
python -u run_sirius.py --instrument_type orbitrap \
--input_dir ./data_instances/orbitrap_chimeric90/ \
--output_dir ./run_sirius_1007/orbitrap_chimeric90_sirius_output/ \
--summary_dir ./run_sirius_1007/orbitrap_chimeric90_sirius_summary/ \
--output_log_dir ./run_sirius_1007/orbitrap_chimeric90_sirius_log/ \
--input_log ./data_instances/orbitrap_log_chimeric90.csv \
--output_log ./run_sirius_1007/sirius_orbitrap_chimeric90_test_1007.csv

python -u run_sirius.py --instrument_type orbitrap \
--input_dir ./data_instances/orbitrap_chimeric70/ \
--output_dir ./run_sirius_1007/orbitrap_chimeric70_sirius_output/ \
--summary_dir ./run_sirius_1007/orbitrap_chimeric70_sirius_summary/ \
--output_log_dir ./run_sirius_1007/orbitrap_chimeric70_sirius_log/ \
--input_log ./data_instances/orbitrap_log_chimeric70.csv \
--output_log ./run_sirius_1007/sirius_orbitrap_chimeric70_test_1007.csv

python -u run_sirius.py --instrument_type orbitrap \
--input_dir ./data_instances/orbitrap_chimeric50/ \
--output_dir ./run_sirius_1007/orbitrap_chimeric50_sirius_output/ \
--summary_dir ./run_sirius_1007/orbitrap_chimeric50_sirius_summary/ \
--output_log_dir ./run_sirius_1007/orbitrap_chimeric50_sirius_log/ \
--input_log ./data_instances/orbitrap_log_chimeric50.csv \
--output_log ./run_sirius_1007/sirius_orbitrap_chimeric50_test_1007.csv