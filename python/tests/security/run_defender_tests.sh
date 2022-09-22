set -e

cd defense
python test_bulyan.py
python test_cclip.py
python test_coordinate_median.py
python test_coordinate_wise_trimmed_mean.py
python test_foolsgold_defense.py
python test_geometric_median.py
python test_krum.py
python test_norm_diff_clipping.py
python test_robust_learning_rate_defense.py
python test_slsgd_defense.py
#python test_soteria.py
python test_wbc.py
python test_weak_dp.py
python test_crfl.py
python test_rfa.py
python test_residual_based_reweighting.py

cd ..