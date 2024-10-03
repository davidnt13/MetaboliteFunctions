# Reading options from json file:
python ChemblMetaboliteTests.py --json input_tests.json

# Reading options from the command line:
python ChemblMetaboliteTests.py --target_name "Alpha-2b" \
				--fileName Alpha-2b_adrenergic_receptor_CHEMBL1942_defined_splits.csv \
				--model "RF" \
				--desc "RDKit" \
				--trainData "train-nonNP"

# Test MGK model:
python ChemblMetaboliteTests.py --target_name "Alpha-2b" \
				--fileName Alpha-2b_adrenergic_receptor_CHEMBL1942_defined_splits.csv \
				--use_mgk \
				--model "MGK" \
				--desc "RDKit" \
				--trainData "train-NP"
