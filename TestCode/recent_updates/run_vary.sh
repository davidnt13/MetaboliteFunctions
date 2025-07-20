#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --partition=common

#SBATCH -o %j.out
#SBATCH -e %j.err

#SBATCH --export=ALL

#SBATCH --job-name=MetaboliteTrialOne


module load Anaconda3
source activate myenv

export PYTHONPATH="${PYTHONPATH}:${HOME}/Programs/:${HOME}/Programs/MetaboliteFunctions"

# python -u ChemblMetaboliteTests.py --json bert_tests.json -d knn_gridsearch_bert_tests

# python -u Vary_NonNP_Metabolite_Tests.py --fileName "RepDatasets/Alpha-2b_adrenergic_receptor_CHEMBL1942_defined_splits_reps.csv" \
# 	--model "KNN" --desc "RDKit" --trainData "train-stratifiedNP" --title "alpha2b_knn_rdkit"

# python -u Vary_NonNP_Metabolite_Tests.py --fileName "./ClassifyDatasets/Cyto_classification_mean_split.csv" \
# 	--model "RF" --desc "Both" --trainData "train-stratifiedNP" --title "Mean_Cyto_rf_both_2" \
# 	--classify -d Classification_Vary_Prop

python -u Vary_NonNP_Metabolite_Tests.py --json classify_tests.json -d Regression_Vary_Prop --title "Mean_Split"
