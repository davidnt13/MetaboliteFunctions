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
parser.add_argument('--directory', '-d',
                    default='',
                    help="Desired directory to save the file")
parser.add_argument('--subsamp')
parser.add_argument('--subsampProp', type=int)

args = parser.parse_args()


if args.use_mgk:
    os.environ["USE_MGK"] = "TRUE"
else:
    os.environ["USE_MGK"] = "FALSE"

from MetFunctionsUpdated import loopedKfoldCV

if args.directory is not None:
    os.makedirs(args.directory, exist_ok=True)

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
                    all_preds = pd.DataFrame()
                    for i in range(5):
                        rep_count = f'_rep-{i}'
                        if i == 0:
                            rep_count = ''
                        myPreds, predictionStats = \
                        loopedKfoldCV(modelType=modelType,
                                      desc=desc,
                                      dataset_file=dataset_file,
                                      #title=title,
                                      split_method='predefined',
                                      split_columns=[
                                          f'{trainData}_CVfold-{j}{rep_count}' for j in range(5)]     
                                            ) 
                        avg_row = predictionStats.loc['avg']
                        avg_row.name = f'avg_{i}'
                        all_preds = pd.concat([all_preds, avg_row.to_frame().T])
                    overall_avg = all_preds.mean(axis=0)
                    overall_std = all_preds.std(axis=0)
                    overall_avg.name = 'avg'
                    overall_std.name = 'std'
                    all_preds = pd.concat([all_preds, overall_avg.to_frame().T, overall_std.to_frame().T])
                    output_filename = f'{args.directory}/{dataset}_{trainData}_{modelType}_{desc}'.strip('_')
                    myPreds.to_csv(output_filename+'_preds.csv')
                    all_preds.to_csv(output_filename+'_stats.csv')
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
    output_filename = f'{args.directory}/{args.target_name}_{args.trainData}_{args.model}_{args.desc}'.strip('_')
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
