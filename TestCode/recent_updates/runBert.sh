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

python -u ChemblMetaboliteTests.py --json bert_tests.json -d knn_gridsearch_bert_tests

