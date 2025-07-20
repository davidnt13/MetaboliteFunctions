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
parser.add_argument('--target_name',
                    help = "For inputting the target name")
parser.add_argument('--fileName',
                    help = "Making the name of the file")
parser.add_argument('--model',
                    help = "Choosing which Model to Utilize")
parser.add_argument('--desc',
                    help="Choose which descriptors to use")
parser.add_argument('--trainData',
                    help="Choose which split to use (train-NP, "\
                           "train-nonNP or train-stratifiedNP")
parser.add_argument('--json',
                    help="Read options from a json file")
parser.add_argument('--directory', '-d',
                    default='.',
                    help="Desired directory to save the file")
parser.add_argument('--title', default='')
parser.add_argument('--classify', action='store_true', default=False)

args = parser.parse_args()


if args.use_mgk:
    os.environ["USE_MGK"] = "TRUE"
else:
    os.environ["USE_MGK"] = "FALSE"

from MetFunctionsUpdated import loopedKfoldCV, vary_non_np_proportion

if args.directory != '.':
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
                    vary_nonNP_results = \
                    vary_non_np_proportion(modelType=modelType,
                                           desc=desc,
                                           traindata=trainData,
                                           title=f'{dataset}_{args.title}',
                                           dataset_file=dataset_file,
                                           classify=args.classify,
                                           directory=args.directory,
                                           split_columns=[
                                                '{}_CVfold-{}'.format(trainData, i) for i in range(5)])

                    output_filename = f'{args.directory}/{dataset}_{args.title}_{modelType}_{desc}'.strip('_')
                    vary_nonNP_results.to_csv(output_filename+'_vary_nonNP.csv')
                    
                    print(f'\tResults saved to: {output_filename}_vary_nonNP.csv,')
                    print('Done')

else:
    # If a json file has not been given, read options from the command line:
    print('Currently training model...')
    vary_nonNP_results = \
    vary_non_np_proportion(modelType=args.model,
                           desc=args.desc,
                           traindata=args.trainData,
                           title=args.title,
                           dataset_file=args.fileName,
                           classify=args.classify,
                           directory=args.directory,
                           split_columns=[
                                '{}_CVfold-{}'.format(args.trainData, i) for i in range(5)])

    # myPreds, predictionStats = \
    # loopedKfoldCV(modelType=args.model,
    #               desc=args.desc,
    #               dataset_file=args.fileName,
    #               #title=args.title,
    #               split_method='predefined',
    #               split_columns=[
    #                   '{}_CVfold-{}'.format(args.trainData, i) for i in range(5)],
    #              )
    output_filename = f'{args.directory}/{args.title}_{args.target_name}_{args.trainData}_{args.model}_{args.desc}'.strip('_')
    # myPreds.to_csv(output_filename+'_preds.csv')
    # predictionStats.to_csv(output_filename+'_stats.csv')
    vary_nonNP_results.to_csv(output_filename+'_vary_nonNP.csv')
    print(f'\tResults saved to: {output_filename}_preds.csv,')
    print(f'\t                  {output_filename}_stats.csv')
    # print(f'\tAverage model performance metrics:')
    # print(pd.read_csv(f'{output_filename}_stats.csv', 
    #                   header=[0, 1], index_col=0)\
    #         .rename(columns={'Number of Molecules' : 'n_mols'})\
    #         .loc[['avg']].round(2).to_string(index=False))
    print('Done')
