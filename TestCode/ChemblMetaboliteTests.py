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
# = JC: Only need one input filename or target_name, since training and test 
# data are now combined and the filename is read from json file anyway.
#parser.add_argument('--input_test',
#                    help = "For inputting the desired Test Set")
#parser.add_argument('--input_train',
#                    help = "For inputting the desired Train Set")
parser.add_argument('--target_name',
                    help = "For inputting the target name")
parser.add_argument('--fileName',
                    help = "Making the name of the file")
parser.add_argument('--model',
                    help = "Choosing which Model to Utilize")
# = JC: Add command line argument to select which descriptors to use.
parser.add_argument('--desc',
                    help="Choose which descriptors to use")
# = JC: Add command line argument to select which split to use.
parser.add_argument('--trainData',
                    help="Choose which split to use (train-NP, "\
                           "train-nonNP or train-stratifiedNP")
# = JC: Add command line argument to read all options from a json file.
parser.add_argument('--json',
                    help="Read options from a json file")

args = parser.parse_args()


if args.use_mgk:
    os.environ["USE_MGK"] = "TRUE"
else:
    os.environ["USE_MGK"] = "FALSE"

#from MetFunctionsUpdated import modelCVSetTest, makeModel, makeModelCVAvg, makeModelCVAvg2
from MetFunctionsUpdated import loopedKfoldCV

#if args.json is not None:
#    fileDict = json.load(open(args.json))
#else:
#    fileDict = {"datasets" : {args.target : {"filename" : args.fileName : }


#fileDict = json.load(open("FileNames.json"))

#targetName = args.target_name

#testData = 'Acetylcholinesterase_CHEMBL220_nonNP.csv'
#trainData = 'Acetylcholinesterase_CHEMBL220_NP.csv'

#trainData = fileDict["dataset"][targetName][args.input_train]
#testData = fileDict["dataset"][targetName][args.input_test]

#descriptors = ["RDKit", "Morgan", "Both"]
#descriptors = ["Coati"]
#descriptors = ["RDKit"]
# = JC: Train method now read from command line --split_type method.
#trainSet = [[testData, trainData], [trainData, testData]]
#trainSet = [[trainData, trainData]]
#model = ["RF", "XGBR", "SVR"]
#model = ['MGK']
#model = ['torchNN']
#model = ['chemprop']
#model = ['MGKSVR']
#model = ['RF']
#All_Models = pd.DataFrame(data = [], columns = ['Fold', 'Number of Molecules', 'r2', 'rmsd', 'bias', 'sdep', 'Model', 'Descriptor', 'Index', 'Train Set'])
#All_Models = pd.DataFrame(data = [], columns = ['Descriptors', 'Model', 'Train','Test', 'R2', 'RMSD', 'Bias', 'SDEP', 'Index'])
#All_Models = avgResults = pd.DataFrame(data= [], columns=['Fold', 'Number of Molecules', 'r2', 'rmsd', 'bias', 'sdep'])

# Examining Train/Test Sets
#for train, test in trainSet:
#    if train == testData:
#       trainName = "NonNP"
#       testName = "NP"
#    elif train == trainData:
#       trainName = "NP"
#       testName = "NonNP"

if args.json is not None:
    # If a json file has been given, read all options from this file:
    fileDict = json.load(open(args.json))
    for dataset in fileDict["datasets"].keys():
        dataset_file = fileDict["datasets"][dataset]["filename"]
        for trainData in fileDict["datasets"][dataset]["splits"]:
            for modelType in fileDict["models"].keys():
                for desc in fileDict["models"][modelType]["descriptors"]:
                    print('Currently training model...')
                    print(f'\tDataset: {dataset}')
                    print(f'\tTraining data split: {trainData}')
                    print(f'\tModel: {modelType}, descriptors: {desc}')
                    myPreds, predictionStats = \
                    loopedKfoldCV(modelType=modelType,
                                  desc=desc,
                                  dataset_file=dataset_file,
                                  #title=title,
                                  split_method='predefined',
                                  split_columns=[
                                      '{}_CVfold-{}'.format(trainData, i) for i in range(5)],
                                           )
                    output_filename = f'{dataset}_{trainData}_{modelType}_{desc}'.strip('_')
                    myPreds.to_csv(output_filename+'_preds.csv')
                    predictionStats.to_csv(output_filename+'_stats.csv')
                    #All_Models = pd.concat([All_Models, df_ret])
                    #All_Models.to_csv(f"{args.fileName}.csv", index = True)
                    print(f'\tResults saved to: {output_filename}_preds.csv,')
                    print(f'\t                  {output_filename}_stats.csv')
                    print(f'\tAverage model performance metrics:')
                    print(pd.read_csv(f'{output_filename}_stats.csv', 
                                      header=[0, 1], index_col=0)\
                            .rename(columns={'Number of Molecules' : 'n_mols'})\
                            .loc[['avg']].round(2).to_string(index=False))
                    print('Done')

else:
    # If a json file has not been given, read options from the command line:
    print('Currently training model...')
    myPreds, predictionStats = \
    loopedKfoldCV(modelType=args.model,
                  desc=args.desc,
                  dataset_file=args.fileName,
                  #title=args.title,
                  split_method='predefined',
                  split_columns=[
                      '{}_CVfold-{}'.format(args.trainData, i) for i in range(5)],
                 )
    output_filename = f'{args.target_name}_{args.trainData}_{args.model}_{args.desc}'.strip('_')
    myPreds.to_csv(output_filename+'_preds.csv')
    predictionStats.to_csv(output_filename+'_stats.csv')
    print(f'\tResults saved to: {output_filename}_preds.csv,')
    print(f'\t                  {output_filename}_stats.csv')
    print(f'\tAverage model performance metrics:')
    print(pd.read_csv(f'{output_filename}_stats.csv', 
                      header=[0, 1], index_col=0)\
            .rename(columns={'Number of Molecules' : 'n_mols'})\
            .loc[['avg']].round(2).to_string(index=False))
    print('Done')
