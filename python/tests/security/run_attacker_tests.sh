set -e

cd attack
echo "python test_backdoor.py"
python test_backdoor.py

echo "python test_byzantine_attack.py"
python test_byzantine_attack.py

#echo "python test_dlg.py"
#python test_dlg.py

echo "python test_edge_case_backdoor_attack.py"
python test_edge_case_backdoor_attack.py

#echo "python test_invertgradient.py"
#python test_invertgradient.py

echo "python test_label_flipping_attack.py"
python test_label_flipping_attack.py

echo "python test_model_replacement_backdoor_attack.py"
python test_model_replacement_backdoor_attack.py