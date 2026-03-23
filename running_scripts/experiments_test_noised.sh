# ----------------------------------------
# Experiments on chimeric spectra
# using trained models 100724
# ----------------------------------------
# I. Data preprocessing
# ----------------------------------------
# QTOF
python prepare_msms_noised.py --mgf_data ./data/cl_pkl_031826/qtof_test.mgf \
--config_path ./config/fiddle_tcn_qtof.yml \
--out_path ./data/noised/noised_qtof_test.mgf --noise_times 1 2 3 4 5 --num_spectra 1000

# BUDDY and SIRIUS QTOF
python mgf_instances.py --input_path ./data/noised/noised_qtof_test.mgf \
--output_dir ./data_instances/qtof_noised/ \
--log ./data_instances/qtof_log_noised.csv

# Orbitrap
python prepare_msms_noised.py --mgf_data ./data/cl_pkl_031826/orbitrap_test.mgf \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--out_path ./data/noised/noised_orbitrap_test.mgf --noise_times 1 2 3 4 5 --num_spectra 1000

# BUDDY and SIRIUS Orbitrap
python mgf_instances.py --input_path ./data/noised/noised_orbitrap_test.mgf \
--output_dir ./data_instances/orbitrap_noised/ \
--log ./data_instances/orbitrap_log_noised.csv

# ----------------------------------------
# II. Test on noised spectra (FIDDLE)
# ----------------------------------------
# For QTOF
python run_fiddle.py --test_data ./data/noised/noised_qtof_test.mgf \
--config_path ./config/fiddle_tcn_qtof.yml \
--resume_path ./check_point/fiddle_tcn_qtof_031826.pt \
--rescore_resume_path ./check_point/fiddle_rescore_qtof_031826.pt \
--result_path ./result/fiddle_qtof_noised_100724.csv --device 6

# For Orbitrap
python run_fiddle.py --test_data ./data/noised/noised_orbitrap_test.mgf \
--config_path ./config/fiddle_tcn_orbitrap.yml \
--resume_path ./check_point/fiddle_tcn_orbitrap_031826.pt \
--rescore_resume_path ./check_point/fiddle_rescore_orbitrap_031826.pt \
--result_path ./result/fiddle_orbitrap_noised_100724.csv --device 7

# ----------------------------------------
# III. Test on noised spectra (BUDDY)
# ----------------------------------------
python run_buddy.py --instrument_type qtof --top_k 5 \
--input_dir ./data_instances/qtof_noised/ \
--result_path ./run_buddy_031826/buddy_qtof_noised_031826.csv

python run_buddy.py --instrument_type orbitrap --top_k 5 \
--input_dir ./data_instances/orbitrap_noised/ \
--result_path ./run_buddy_031826/buddy_orbitrap_noised_031826.csv

# ----------------------------------------
# IV. Test on noised spectra (SIRIUS)
# ----------------------------------------
# For QTOF
python -u run_sirius.py --instrument_type qtof \
--input_dir ./data_instances/qtof_noised/ \
--output_dir ./run_sirius_031826/qtof_noised_sirius_output/ \
--summary_dir ./run_sirius_031826/qtof_noised_sirius_summary/ \
--output_log_dir ./run_sirius_031826/qtof_noised_sirius_log/ \
--input_log ./data_instances/qtof_log_noised.csv \
--output_log ./run_sirius_031826/sirius_qtof_noised_test_031826.csv

# For Orbitrap
python -u run_sirius.py --instrument_type orbitrap \
--input_dir ./data_instances/orbitrap_noised/ \
--output_dir ./run_sirius_031826/orbitrap_noised_sirius_output/ \
--summary_dir ./run_sirius_031826/orbitrap_noised_sirius_summary/ \
--output_log_dir ./run_sirius_031826/orbitrap_noised_sirius_log/ \
--input_log ./data_instances/orbitrap_log_noised.csv \
--output_log ./run_sirius_031826/sirius_orbitrap_noised_test_031826.csv
