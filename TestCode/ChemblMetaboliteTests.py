#!/usr/bin/env python

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--use_mgk',
                    action = 'store_true',
                    help="For running metabolite models with MGK")
parser.add_argument('--input_test',
                    help = "For inputting the desired Test Set")
parser.add_argument('--input_train',
                    help = "For inputting the desired Train Set")
parser.add_argument('--target_name',
                    help = "For inputting the target name")
parser.add_argument('--fileName',
                    help = "Making the name of the file")
parser.add_argument('--model',
                    help = "Choosing which Model to Utilize")
args = parser.parse_args()


if args.use_mgk:
    os.environ["USE_MGK"] = "TRUE"
else:
    os.environ["USE_MGK"] = "FALSE"

from MetFunctionsUpdated import modelCVSetTest, makeModel, makeModelCVAvg, makeModelCVAvg2

fileDict = json.load(open("FileNames.json"))

targetName = args.target_name

#testData = 'Acetylcholinesterase_CHEMBL220_nonNP.csv'
#trainData = 'Acetylcholinesterase_CHEMBL220_NP.csv'

trainData = fileDict["dataset"][targetName][args.input_train]
testData = fileDict["dataset"][targetName][args.input_test]

#descriptors = ["RDKit", "Morgan", "Both"]
descriptors = ["Coati"]
#descriptors = ["RDKit"]
trainSet = [[testData, trainData], [trainData, testData]]
#trainSet = [[trainData, trainData]]
#model = ["RF", "XGBR", "SVR"]
#model = ['MGK']
model = ['torchNN']
#model = ['chemprop']
#model = ['MGKSVR']
#model = ['RF']
#All_Models = pd.DataFrame(data = [], columns = ['Fold', 'Number of Molecules', 'r2', 'rmsd', 'bias', 'sdep', 'Model', 'Descriptor', 'Index', 'Train Set'])
#All_Models = pd.DataFrame(data = [], columns = ['Descriptors', 'Model', 'Train','Test', 'R2', 'RMSD', 'Bias', 'SDEP', 'Index'])
All_Models = avgResults = pd.DataFrame(data= [], columns=['Fold', 'Number of Molecules', 'r2', 'rmsd', 'bias', 'sdep'])

# Examining Train/Test Sets
for train, test in trainSet:
    if train == testData:
       trainName = "NonNP"
       testName = "NP"
    elif train == trainData:
       trainName = "NP"
       testName = "NonNP"
    for descr in descriptors:
       for mod in model:
            df_ret =  modelCVSetTest(train, descr, mod, f"{targetName}-Train-{trainName}-Test-{testName}", trainName)
            #df_ret = makeModelCVAvg2(train, test, mod, f"Train-{trainName}-Test-{testName}", f"{targetName}-{trainName}")
            All_Models = pd.concat([All_Models, df_ret])
All_Models.to_csv(f"{args.fileName}.csv", index = True)
