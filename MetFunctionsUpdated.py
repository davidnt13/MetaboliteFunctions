# Trying to Remove Warnings
import warnings
warnings.filterwarnings('ignore')

# Establishing New Function for Calculating RMSE
def root_mean_squared_error(y_true, y_pred):
  """Calculate RMSE."""
  return np.sqrt(mean_squared_error(y_true, y_pred))

# General Import Statements
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
import matplotlib.pyplot as plt

from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
import os

modelTypes = {}

use_mgk = os.environ.get("USE_MGK")

if use_mgk != "TRUE":

    from GenerateDescriptors import *
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.svm import SVR
    modelTypes['RF'] = RandomForestRegressor
    modelTypes['XGBR'] = XGBRegressor
    modelTypes['SVR'] = SVR

    try:
        import chemprop
        from models import SimpleNN, ChempropModel
        modelTypes['torchNN'] = SimpleNN
        modelTypes['chemprop'] = ChempropModel
    except ModuleNotFoundError:
        print('WARNING: Cannot import chemprop python module')

else:

    try:
        import mgktools
        import mgktools.data
        import mgktools.kernels.utils
        import mgktools.hyperparameters
        from mgktools.models.regression.GPRgraphdot.gpr import GPR
        from mgktools.models.regression import SVR
        modelTypes['MGK'] = GPR
        modelTypes['MGKSVR'] = SVR
    except ModuleNotFoundError:
        print('WARNING: Cannot import mgktools python modules')

# = JC: Commented out function below as no longer used in code.
# Making Train and Test, in addition to Descriptors/Fingerprints
#def makeTrainAndTestDesc(fileNameTrain, fileNameTest, target, desc):
#    dfTrain = pd.read_csv(fileNameTrain)
#    dfTest = pd.read_csv(fileNameTest)
#
#    if desc == "RDKit":
#        descTrain = CalcRDKitDescriptors(fileNameTrain)
#        descTest = CalcRDKitDescriptors(fileNameTest)
#    elif desc == "Morgan":
#        descTrain = CalcMorganFingerprints(fileNameTrain)
#        descTest = CalcMorganFingerprints(fileNameTest)
#    elif desc == "Both":
#        descTrain = calcBothDescriptors(fileNameTrain)
#        descTest = calcBothDescriptors(fileNameTest)
#    elif desc == "Coati":
#        descTrain = calcCoati(fileNameTrain)
#        descTest = calcCoati(fileNameTest)
#
#    train_X = descTrain.dropna(axis = 1)
#    train_y = dfTrain[target]
#    test_X = descTest.dropna(axis = 1)
#    test_y = dfTest[target]
#
#    common_columns = train_X.columns.intersection(test_X.columns)
#    train_X = train_X[common_columns]
#    test_X = test_X[common_columns]
#
#    return train_X, train_y, test_X, test_y

# = JC: Commented out function below as no longer used in code.
# Making Train and Test (Descriptors already in files)
#def makeTrainAndTest(fileNameTrain, fileNameTest, target):
#    dfTrain = pd.read_csv(fileNameTrain)
#    dfTest = pd.read_csv(fileNameTest)
#
#    train_X = dfTrain.dropna(axis = 1)\
#                    .drop(target)
#    train_y = dfTrain[target]
#    test_X = dfTest.dropna(axis = 1)\
#                    .drop(target)
#    test_y = dfTest[target]
#
#    common_columns = train_X.columns.intersection(test_X.columns)
#    train_X = train_X[common_columns]
#    test_X = test_X[common_columns]
#
#    return train_X, train_y, test_X, test_y

# = JC: Commented out function below as no longer used in code.
# Making Train and Test (For ChemProp, MGK)
#def makeTrainAndTestGraph(fileNameTrain, fileNameTest, target):
#    dfTrain = pd.read_csv(fileNameTrain)
#    dfTest = pd.read_csv(fileNameTest)
#    #train_X = dfTrain.dropna(axis = 0)\
#     #               .reset_index()\
#      #              .drop(target, axis = 1)\
#       #             .drop("ChEMBL_ID", axis = 1)\
#        #            .drop("natural_product", axis = 1)
#    train_X = dfTrain['SMILES']
#    train_y = dfTrain[target]
#    #test_X = dfTest.dropna(axis = 0)\
#    #                .reset_index()\
#    #                .drop(target, axis = 1)\
#    #                .drop("ChEMBL_ID", axis = 1)\
#    #                .drop("natural_product", axis = 1)
#    test_X = dfTest['SMILES']
#    test_y = dfTest[target]
#
#    return train_X, train_y, test_X, test_y

# = JC: Commented out function below as no longer used in code.
#def makeTrainAndTestGraphCV(CVName, fileNameTrain, fileNameTest, target):
#    df = pd.read_csv(fileNameTrain)
#    dfTrain = df.loc(df["natural_product"] == "TRUE")
#    dfTest = df.loc(df["natural_product"] == "FALSE")
#    return dfTrain, dfTest

# Plotting CV Results
def plotCVResults(train_y, myPreds, title = None):

    nptrain_y = train_y.to_numpy() if isinstance(train_y, pd.Series) else train_y
    npy_pred = myPreds['Prediction']

    minVal = min(nptrain_y.min(), npy_pred.min())
    maxVal = max(nptrain_y.max(), npy_pred.max())

    a, b = np.polyfit(nptrain_y, npy_pred, 1)
    xvals = np.linspace(minVal - 1, maxVal + 1, 100)
    yvals = xvals

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(xvals, yvals, '--')
    ax.scatter(nptrain_y, npy_pred)
    ax.plot(nptrain_y, a * nptrain_y + b)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.set_aspect('equal')
    ax.set_title(f'{title}: CV Model Results')
    plt.savefig(f'{title}: CV_modelResults.png')

# = JC: No longer need separate loopedKfold... functions for different splits, 
# as the loopedKfoldCV function can now handle different split methods.
# Looped Kfold CrossVal (NOT A MIXED SET)
#def loopedKfoldCrossVal(modelType, num_cv, train_X, train_y, title, distributor = None):
#
#    predictions_filename = f'{title}: CV{modelType}_predictions.csv'
#
#    predStats = {'r2_sum': 0, 'rmsd_sum': 0, 'bias_sum': 0, 'sdep_sum': 0}
#    predictionStats = pd.DataFrame(data=np.zeros((num_cv, 6)), columns=['Fold', 'Number of Molecules', 'r2', 'rmsd', 'bias', 'sdep'])
#
#    myPreds = pd.DataFrame(index=range(len(train_y)), #index=train_y.index,
#                           columns=['Prediction', 'Fold'])
#    myPreds['Prediction'] = np.nan
#    myPreds['Fold'] = np.nan
#
#    if distributor is None:
#        train_test_split = KFold(n_splits = num_cv, shuffle=True, random_state=1)
#    else:
#        train_test_split = StratifiedKFold(n_splits = num_cv, shuffle = True, random_state = 1)
#
#    if modelType == 'chemprop':
#        dataset = [chemprop.data.MoleculeDatapoint.from_smi(smi, [y])
#                   for smi, y in zip(train_X.loc[:,'SMILES'], train_y)]
#
#    elif modelType == 'MGK':
#        #if isinstance(train_y, pd.Series):
#         #   train_y = train_y.to_frame(name='Target')  # Convert Series to DataFrame
#        #else:
#         #   train_y = train_y.rename(columns={train_y.columns[0]: 'Target'})
#        dataset = mgktools.data.data.Dataset.from_df(
#                      pd.concat([train_X, train_y], axis=1).reset_index(),
#                      pure_columns=[train_X.name],
#                      #pure_columns=['Mol'],
#                      #target_columns=['Target'],
#                      target_columns=[train_y.name],
#                      n_jobs=-1
#                     )
#        kernel_config = mgktools.kernels.utils.get_kernel_config(
#                            dataset,
#                            graph_kernel_type='graph',
#                            # Arguments for marginalized graph kernel:
#                            mgk_hyperparameters_files=[
#                                mgktools.hyperparameters.product_msnorm],
#                           )
#        dataset.graph_kernel_type = 'graph'
#
#    for n, (train_idx, test_idx) in enumerate(train_test_split.split(train_X, distributor)):
#
#        y_train = train_y.iloc[train_idx]
#        y_test = train_y.iloc[test_idx]
#
#        model_opts = {}
#        model_fit_opts = {}
#
#        if modelType == 'MGK':
#            x_train = mgktools.data.split.get_data_from_index(dataset,
#                                                              train_idx).X
#            x_test = mgktools.data.split.get_data_from_index(dataset,
#                                                             test_idx).X
#            model_opts = {'kernel' : kernel_config.kernel,
#                          'optimizer' : None,
#                          'alpha' : 0.01,
#                          'normalize_y' : True}
#
#        elif modelType == 'chemprop':
#            # Split data into training and test sets:
#            train_idx, val_idx, _ = make_split_indices(train_idx, sizes=(0.9, 0.1, 0))
#            train_data, val_data, test_data = \
#            chemprop.data.split_data_by_indices(dataset,
#                                                train_indices=train_idx,
#                                                val_indices=None,
#                                                test_indices=test_idx
#                                               )
#
#            # Calculate features for molecules:
#            featurizer = chemprop.featurizers.SimpleMoleculeMolGraphFeaturizer()
#            train_dset = chemprop.data.MoleculeDataset(train_data, featurizer)
#            test_dset = chemprop.data.MoleculeDataset(test_data, featurizer)
#
#            # Scale y data based on training set:
#            scaler = train_dset.normalize_targets()
#            model_opts = {'y_scaler' : scaler}
#
#            # Set up dataloaders for feeding data into models:
#            train_loader = chemprop.data.build_dataloader(train_dset)
#            test_loader = chemprop.data.build_dataloader(test_dset, shuffle=False)
#
#            # Make name consistent with non-chemprop models:
#            x_train = train_loader
#            x_test = test_loader
#
#        else:
#            x_train = train_X.iloc[train_idx]
#            x_test = train_X.iloc[test_idx]
#
#            scaler = StandardScaler()
#            x_train = scaler.fit_transform(x_train)
#            x_test = scaler.transform(x_test)
#
#            if modelType == 'torchNN':
#                y_scaler = StandardScaler()
#                y_train = y_scaler.fit_transform(np.array(y_train).reshape(-1, 1))
#                # x_train = SimplePyTorchDataset(x_train, y_train)
#                # x_test = SimplePyTorchDataset(x_test, y_test)
#                model_opts = {'y_scaler' : y_scaler, 'input_size' : x_train.shape[1]}
#                model_fit_opts = {'X_val' : torch.tensor(x_test, dtype=torch.float32),
#                                  'y_val' : y_test
#                                 }
#
#        model = modelTypes[modelType]
#        model = model(**model_opts)
#        
#        import pickle as pk
#        pk.dump(y_train, open('y_train.pk', "wb"))
#
#        # Train model
#        model.fit(x_train, y_train.squeeze(), **model_fit_opts)
#        # model.plot_training_loss()
#
#        y_pred = model.predict(x_test)
#
#        # Metrics calculations
#        r2 = r2_score(y_test, y_pred)
#        rmsd = root_mean_squared_error(y_test, y_pred)
#        bias = np.mean(y_pred - y_test)
#        sdep = np.std(y_pred - y_test)
#
#        # Update stats
#        predStats['r2_sum'] += r2
#        predStats['rmsd_sum'] += rmsd
#        predStats['bias_sum'] += bias
#        predStats['sdep_sum'] += sdep
#
#        # Update predictions
#        myPreds.loc[test_idx, 'Prediction'] = y_pred
#        myPreds.loc[test_idx, 'Fold'] = n + 1
#
#        # Ensure correct number of values are assigned
#        predictionStats.iloc[n] = [n + 1, len(test_idx), r2, rmsd, bias, sdep]
#
#    # Calculate averages
#    r2_av = predStats['r2_sum'] / num_cv
#    rmsd_av = predStats['rmsd_sum'] / num_cv
#    bias_av = predStats['bias_sum'] / num_cv
#    sdep_av = predStats['sdep_sum'] / num_cv
#
#    # Create a DataFrame row for averages
#    avg_row = pd.DataFrame([['Average', len(train_y), r2_av, rmsd_av, bias_av, sdep_av]], columns=predictionStats.columns)
#
#    # Append average row to the DataFrame
#    predictionStats = pd.concat([predictionStats, avg_row], ignore_index=True)
#
#    return myPreds, predictionStats, avg_row

# = JC: New function to set up dataset splits for different split methods 
# (e.g. K-fold, stratified, predefined etc.), it returns a generator which 
# will generate the datapoint indices for each split.
def get_dataset_splitter(df_data,
                         split_method='k-fold',
                         n_splits=5,
                         frac_test=0.3,
                         split_columns=[],
                         strat_column=None,
                         rand_seed=None
                        ):
    """
    Function to set up different types of dataset splits and return a 
    generator to generate the train/test set indices for each split.

    :param df_data: DataFrame containing dataset to split
    :param split_method: Method to split dataset, options include random, 
        k-fold, stratified_random, stratified_k-fold and predefined
    :param n_splits (optional): Number of splits to generate (e.g. the number 
        of folds for cross-validation), used for all split methods except 
        predefined
    :param frac_test (optional): Fraction of the dataset to put into the 
        test set, for random split method only
    :param split_columns (optional): List of the column names which contain the
        predefined splits in the df_data DataFrame
    :param strat_column (optional): Name of the column in the df_data DataFrame 
        to use to stratify the data for a stratified split
    :param rand_seed (optional): Random seed to use to shuffle data before 
        splitting

    >>> df_data = pd.DataFrame(data=[[1.2, 2, 'train'], [2.1, 2, 'testset1'],
    ...                              [2.4, 3, 'train'], [3.9, 2, 'train'], 
    ...                              [0.7, 3, 'train'], [1.5, 2, 'testset2']],
    ...                        columns=['y', 'cls', 'split1'])
    >>> for train_idx, test_idx in get_dataset_splitter(df_data, n_splits=2, 
    ...                                 split_method='k-fold', rand_seed=1):
    ...     print(train_idx, test_idx)
    [0 3 5] [1 2 4]
    [1 2 4] [0 3 5]
    >>> for train_idx, test_idx in get_dataset_splitter(df_data, n_splits=2, 
    ...     split_method='stratified_random', strat_column='cls', rand_seed=1):
    ...     print(train_idx, test_idx)
    [5 0 2 3] [4 1]
    [2 1 3 5] [0 4]
    >>> for train_idx, test_idx in get_dataset_splitter(df_data, n_splits=2, 
    ...     split_method='predefined', rand_seed=1, split_columns=['split1']):
    ...     print(train_idx, test_idx)
    [0, 2, 3, 4] {'testset1': [1], 'testset2': [5]}
    """

    # Get list of dataset row numbers to use as IDs:
    data_ids = list(range(len(df_data)))

    # Random split using sklearn:
    if split_method in ['rand', 'random']:
        splitter = ShuffleSplit(n_splits=n_splits,
                                test_size=frac_test,
                                random_state=rand_seed)
        data_splits = splitter.split(data_ids)

    # K-fold split using sklearn:
    elif split_method == 'k-fold':
        splitter = KFold(n_splits=n_splits,
                         shuffle=True,
                         random_state=rand_seed)
        data_splits = splitter.split(data_ids)

    # Stratified random split using sklearn:
    elif split_method == 'stratified_random':
        c = df_data[strat_column]
        splitter = StratifiedShuffleSplit(n_splits=n_splits,
                                          test_size=frac_test,
                                          random_state=rand_seed)
        data_splits = splitter.split(data_ids, c)

    # Stratified K-fold split using sklearn:
    elif split_method == 'stratified_k-fold':
        c = df_data[strat_column]
        splitter = StratifiedKFold(n_splits=n_splits,
                                   shuffle=True,
                                   random_state=rand_seed)
        data_splits = splitter.split(data_ids, c)

    # Predefined dataset split based on column in dataset file:
    elif split_method == 'predefined':
        # Set up a generator to return train and test set indices for each 
        # split in the dataset:
        def split_generator(df_data, split_columns):
            for split_col in split_columns:
                train_idx = df_data.loc[df_data[split_col] == 'train']\
                                   .index.to_list()
                # Get names of different test sets from dataset column:
                test_set_names = \
                    [set_name 
                     for set_name in sorted(df_data[split_col].unique())
                     if set_name.startswith('test')]
                test_idx = {key : df_data.loc[df_data[split_col] == key]\
                                         .index.to_list()
                            for key in test_set_names}
                yield train_idx, test_idx
        data_splits = split_generator(df_data, split_columns)

    else:
        raise NotImplementedError('Split method {} not implemented'\
                                  .format(split_method))

    return data_splits

# = JC: Modified this function to make some of the variable names clearer and 
# enable it to handle different splitting methods.
def loopedKfoldCV(modelType, 
                  desc, 
                  dataset_file, 
                  # = JC: group argument no longer needed.
                  #group, 
                  #title, 
                  # = JC: distributor argument no longer needed, I 
                  # think this was the column for doing stratified 
                  # splits, but I've renamed this strat_column.
                  #distributor = None,
                  strat_column=None,
                  split_method='k-fold',
                  n_splits=5,
                  split_columns=[],
                  frac_test=None,
                  subsample='',
                  subsampleProportion=1):

    # = JC: Renamed dfTrain to df_all_data for clarity, to distinguish it from 
    # the training set.
    df_all_data = pd.read_csv(dataset_file)
    
    # = JC: This is no longer needed since the fold columns are given as the 
    # split_columns argument to the overall function.
    #if (group == 'NP'):
    # #   dfTrain = df.loc(df["natural_product"] == "TRUE")
    #    fold_columns = [col for col in dfTrain.columns if 'train-NP' in col]
    #elif (group == 'NonNP'):
    # #   dfTrain = df.loc(df["natural_product"] == "FALSE")
    #    fold_columns = [col for col in dfTrain.columns if 'train-nonNP' in col]
    #elif (group == 'Mix'):
    #    fold_columns = [col for col in dfTrain.columns if 'stratified' in col]

    # = JC: Renamed train_X/train_y to all_X/all_y for clarity:
    all_X = df_all_data["SMILES"]
    all_y = df_all_data["pIC50"]
    # Get IDs for data points if ID column is present and check that they are 
    # unique:
    all_ids = df_all_data.get("ID")
    if (all_ids is not None) and (all_ids.duplicated().sum() > 0):
        raise ValueError(
            '"ID" column of dataset contains duplicate values: {}'.format(
            ', '.join(all_ids.loc[all_ids.duplicated(keep=False)]\
                             .astype(str).to_list())))

    #predictions_filename = f'{title}: CV{modelType}_predictions.csv'

    # =JC: Don't need to save sums of metrics if the values for each CV-fold 
    # are also being saved.
    #predStats = {'r2_sum': 0, 'rmsd_sum': 0, 'bias_sum': 0, 'sdep_sum': 0}

    #predictionStats = {"NP": pd.DataFrame(data=np.zeros((5, 6)), columns=['Fold', 'Number of Molecules', 'r2', 'rmsd', 'bias', 'sdep']), "NonNP": pd.DataFrame(data=np.zeros((5, 6)), columns=['Fold', 'Number of Molecules', 'r2', 'rmsd', 'bias', 'sdep'])}

    # = JC: Removed number of fold from predictionStats as the DataFrame index 
    # can be used to label the fold.
    predictionStats = None
    empty_predictionStats = pd.DataFrame(data=np.zeros((n_splits, 7)), 
                                   columns=['Train Set Size', 
                                            'Number of Molecules', 
                                            'r2', 'rmsd', 'bias', 
                                            'sdep', 'Test Set stdev'], 
                                   index=range(1, n_splits+1))
    empty_predictionStats.index.name = 'Fold'

    myPreds = None
    empty_myPreds = pd.DataFrame(index=range(len(all_y)), #index=train_y.index,
                           columns=['Prediction', 'Fold'])
    empty_myPreds['Prediction'] = np.nan
    empty_myPreds['Fold'] = np.nan
    

    if modelType == 'chemprop':
        dataset = [chemprop.data.MoleculeDatapoint.from_smi(smi, [y])
                   for smi, y in zip(all_X, all_y)]

    elif modelType == 'MGK':
        dataset = mgktools.data.data.Dataset.from_df(
                      pd.concat([all_X, all_y], axis=1).reset_index(),
                      pure_columns=[all_X.name],
                      target_columns=[all_y.name],
                      n_jobs=-1
                     )
        kernel_config = mgktools.kernels.utils.get_kernel_config(
                            dataset,
                            graph_kernel_type='graph',
                            # Arguments for marginalized graph kernel:
                            mgk_hyperparameters_files=[
                                mgktools.hyperparameters.product_msnorm],
                           )
        dataset.graph_kernel_type = 'graph'

    else:
        # If not using chemprop or MGK, calculate descriptors:
        if desc == "RDKit":
            all_X = CalcRDKitDescriptors(all_X)
        elif desc == "Morgan":
            all_X = CalcMorganFingerprints(all_X)
        elif desc == "Both":
            all_X = calcBothDescriptors(all_X)
        elif desc == "Coati":
            all_X = calcCoati(all_X)

    # Generate the dataset splitter:
    splitter = get_dataset_splitter(df_all_data,
                                    split_method=split_method,
                                    split_columns=split_columns,
                                    n_splits=n_splits,
                                    frac_test=frac_test)

    # Loop over dataset splits:
    for fold_number, [train_idx, test_idx] in enumerate(splitter):
        
        if subsample == 'random':
            train_idx = np.random.choice(train_idx, size = int(len(train_idx)*subsampleProportion), replace = False)
        elif subsample == 'stratified':
            stratCol = df_all_data['natural_product']
            newtrain_idx = []
            for val in stratCol.unique():
                newtrain_idx += np.random.choice(train_idx[stratCol == val], size = int(len(train_idx[stratCol==val])*subsampleProportion), replace = False)
            train_idx = newtrain_idx

        # On the first loop, set up myPreds and predictionStats for each test 
        # set:
        if myPreds is None:
            myPreds = {key: empty_myPreds.copy() for key in test_idx.keys()}
        if predictionStats is None:
            predictionStats = {key: empty_predictionStats.copy() 
                               for key in test_idx.keys()}

        y_train = all_y.iloc[train_idx]
        y_test = {}
        for key in test_idx.keys():
            y_test[key] = all_y.iloc[test_idx[key]] 

        model_opts = {}
        model_fit_opts = {}

        if modelType == 'MGK':
            x_train = mgktools.data.split.get_data_from_index(dataset,
                                                              train_idx).X
            x_test = {}
            for key in test_idx.keys():
                x_test[key] = mgktools.data.split.get_data_from_index(
                                  dataset, test_idx[key]).X

            model_opts = {'kernel' : kernel_config.kernel,
                          'optimizer' : None,
                          'alpha' : 0.01,
                          'normalize_y' : True}

        elif modelType == 'chemprop':
            # Split data into training and test sets:
            train_data, _, _ = \
            chemprop.data.split_data_by_indices(dataset,
                                                train_indices=train_idx,
                                               )
            test_data = {}
            for key in test_idx.keys():
                _, _, test_data[key] = chemprop.data.split_data_by_indices(
                                           dataset, test_indices=test_idx[key])

            # Calculate features for molecules:
            featurizer = chemprop.featurizers.SimpleMoleculeMolGraphFeaturizer()
            train_dset = chemprop.data.MoleculeDataset(train_data, featurizer)

            test_dset = {}
            for key in test_idx.keys():
                test_dset[key] = chemprop.data.MoleculeDataset(test_data[key], 
                                                               featurizer)

            # Scale y data based on training set:
            scaler = train_dset.normalize_targets()
            model_opts = {'y_scaler' : scaler}

            # Set up dataloaders for feeding data into models:
            train_loader = chemprop.data.build_dataloader(train_dset)
            test_loader = {}
            for key in test_idx.keys():
                test_loader[key] = chemprop.data.build_dataloader(
                                       test_dset[key], shuffle=False)

            # Make name consistent with non-chemprop models:
            x_train = train_loader
            x_test = {}
            for key in test_idx.keys():
                x_test[key] = test_loader[key]

        # All descriptor based models:
        else:
            x_train = all_X.iloc[train_idx]
            x_test = {}
            for key in test_idx.keys():
                x_test[key] = all_X.iloc[test_idx[key]]

            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            for key in test_idx.keys():
                x_test[key] = scaler.transform(x_test[key])

            if modelType == 'torchNN':
                y_scaler = StandardScaler()
                y_train = y_scaler.fit_transform(np.array(y_train).reshape(-1, 1))
                # x_train = SimplePyTorchDataset(x_train, y_train)
                # x_test = SimplePyTorchDataset(x_test, y_test)
                model_opts = {'y_scaler' : y_scaler, 'input_size' : x_train.shape[1]}
                #model_fit_opts = {'X_val' : torch.tensor(x_test, dtype=torch.float32),
                #                  'y_val' : y_test
                #                 }

        model = modelTypes[modelType]
        model = model(**model_opts)

        # Train model:
        model.fit(x_train, y_train, **model_fit_opts)
        # model.plot_training_loss()
        
        sizeTrain = len(x_train)

        # Analyse performance on each test set:
        for key in test_idx.keys():
             y_pred = model.predict(x_test[key])

             # Metrics calculations
             r2 = r2_score(y_test[key], y_pred)
             rmsd = root_mean_squared_error(y_test[key], y_pred)
             bias = np.mean(y_pred - y_test[key])
             sdep = np.std(y_pred - y_test[key])
             
             stdev = np.std(y_test[key])
             # = JC: Don't need to store these sums anymore if the metrics for 
             # each fold are also being stored.
             # Update stats
             #predStats['r2_sum'] += r2
             #predStats['rmsd_sum'] += rmsd
             #predStats['bias_sum'] += bias
             #predStats['sdep_sum'] += sdep

             # Update predictions
             myPreds[key].loc[test_idx[key], 'Prediction'] = y_pred
             myPreds[key].loc[test_idx[key], 'Fold'] = fold_number + 1

             # Save prediction stats for this CV fold:
             predictionStats[key].loc[fold_number+1] = [sizeTrain,
                                                        len(test_idx[key]), r2, 
                                                        rmsd, bias, sdep, stdev]

    # = JC: As above, sums of metrics are no longer needed.
    # Calculate averages
    #r2_av = predStats['r2_sum'] / 5
    #rmsd_av = predStats['rmsd_sum'] / 5
    #bias_av = predStats['bias_sum'] / 5
    #sdep_av = predStats['sdep_sum'] / 5

    # Create a DataFrame row for averages
    predictionStats = pd.concat(predictionStats, axis=1)
    avg_row = predictionStats.mean(axis=0)
    stdev_row = predictionStats.std(axis=0, ddof=0)
    avg_row.name = 'avg'
    stdev_row.name= 'stdev'

    predictionStats = pd.concat([predictionStats, avg_row.to_frame().T, 
                                 stdev_row.to_frame().T], axis=0)
    #predictionStats = predictionStats.append(avg_row)
    #for key in test_idx.keys():
        #predictionStats[key].mean(access=0)
    
    # Append average row to the DataFrame
    #predictionStats = pd.concat([predictionStats, avg_row], ignore_index=True)
    
   #  predictionStats = pd.concat([predictionStats["NP"], predictionStats["NonNP"]], ignore_index=True, keys = ["NP", "NonNP"])

    # Concatenate the dataframes containing the predictions for different test 
    # sets, the final dataframe will have a 2 layer header with the test set 
    # name as the first level:
    myPreds = pd.concat(myPreds, axis=1)
    # Add original y data to the beginning of the DataFrame:
    myPreds.insert(0, 'Original Y', all_y)
    #myPreds['Original Y'] = all_y
    if all_ids is not None:
        myPreds.insert(0, 'ID', all_ids)

    return myPreds, predictionStats

# Downloading CV Stats if desired
def downloadCVStats(myPreds, predictionStats, title = None):
    predictions_filename = f'{title}: CV_predictions.csv'
    myPreds.to_csv(predictions_filename, index=True)
    predictionStats.to_csv(f'{title}: CV_stats.csv', index=False)

# = JC: No longer need separate loopedKfold... functions for different splits, 
# as the loopedKfoldCVSetSplits function can now handle different split 
# methods.
# Looped Kfold CrossVal (MIXED SET)
#def loopedKfoldCrossValMix(modelType, num_cv, train_X, train_y, title, distributor = None):
#    predictions_filename = f'{title}: CV{modelType}_predictions.csv'
#
#    predStats = {'r2_sum': 0, 'rmsd_sum': 0, 'bias_sum': 0, 'sdep_sum': 0}
#    predictionStats = pd.DataFrame(data=np.zeros((num_cv, 6)), columns=['Fold', 'Number of Molecules', 'r2', 'rmsd', 'bias', 'sdep'])
#
#    myPreds = pd.DataFrame(index=range(len(train_y)), #index=train_y.index,
#                           columns=['Prediction', 'Fold'])
#    myPreds['Prediction'] = np.nan
#    myPreds['Fold'] = np.nan
#
#    if distributor is None:
#        train_test_split = KFold(n_splits = num_cv, shuffle=True, random_state=1)
#    else:
#        train_test_split = GroupKFold(n_splits = num_cv)
#
#    for n, (train_idx, test_idx) in enumerate(train_test_split.split(train_X, train_y, distributor)):
#        x_train = train_X.iloc[train_idx]
#        x_test = train_X.iloc[test_idx]
#        y_train = train_y.iloc[train_idx]
#        y_test = train_y.iloc[test_idx]
#
#        model = modelTypes[modelType]
#
#        # Train model
#        model.fit(x_train, y_train)
#
#        y_pred = model.predict(x_test)
#
#        # Metrics calculations
#        r2 = r2_score(y_test, y_pred)
#        rmsd = mean_squared_error(y_test, y_pred, squared=False)
#        bias = np.mean(y_pred - y_test)
#        sdep = np.std(y_pred - y_test)
#
#        # Update stats
#        predStats['r2_sum'] += r2
#        predStats['rmsd_sum'] += rmsd
#        predStats['bias_sum'] += bias
#        predStats['sdep_sum'] += sdep
#
#        # Update predictions
#        myPreds.loc[test_idx, 'Prediction'] = y_pred
#        myPreds.loc[test_idx, 'Fold'] = n + 1
#
#        # Ensure correct number of values are assigned
#        predictionStats.iloc[n] = [n + 1, len(test_idx), r2, rmsd, bias, sdep]
#
#    # Calculate averages
#    r2_av = predStats['r2_sum'] / num_cv
#    rmsd_av = predStats['rmsd_sum'] / num_cv
#    bias_av = predStats['bias_sum'] / num_cv
#    sdep_av = predStats['sdep_sum'] / num_cv
#
#    # Create a DataFrame row for averages
#    avg_row = pd.DataFrame([['Average', len(train_y), r2_av, rmsd_av, bias_av, sdep_av]], columns=predictionStats.columns)
#
#    # Append average row to the DataFrame
#    predictionStats = pd.concat([predictionStats, avg_row], ignore_index=True)
#
#    return myPreds, predictionStats, avg_row

# = JC: Don't think that this function is used anymore.
# Mixed Set Cross Validation
#def mixedCV(fileName, descr, model):
#
#    mixDf = pd.read_csv(fileName)
#
#    if descr == "RDKit":
#        df2Mix = CalcRDKitDescriptors(fileName)
#    elif descr == "Morgan":
#        df2Mix = CalcMorganFingerprints(fileName)
#    elif descr == "Both":
#        df2Mix = calcBothDescriptors(fileName)
#
#    allMetabolites = mixDf["natural_product"].tolist()
#    df2Mix["natural_product"] = allMetabolites
#    train_X = df2Mix.dropna(axis = 1)
#    train_y = mixDf.pIC50
#    metabolites = mixDf.natural_product
#    train_X = train_X.drop("natural_product", axis = 1)
#
#    for index in range(1, 4):
#        loopedKfoldCrossVal(model, 10, train_X, train_y, f"Mixture_{model}_{descr}_{index}", metabolites)

# = JC: Don't think that this function is used anymore.
# Mixed Set Cross Validation (And Saving Results)
#def mixedCVSaveAvg(fileName, descr, model):
#
#    mixDf = pd.read_csv(fileName)
#
#    if descr == "RDKit":
#        df2Mix = CalcRDKitDescriptors(fileName)
#    elif descr == "Morgan":
#        df2Mix = CalcMorganFingerprints(fileName)
#    elif descr == "Both":
#        df2Mix = calcBothDescriptors(fileName)
#
#    allMetabolites = mixDf["natural_product"].tolist()
#    df2Mix["natural_product"] = allMetabolites
#    train_X = df2Mix.dropna(axis = 1)
#    train_y = mixDf.pIC50
#    metabolites = mixDf.natural_product
#    train_X = train_X.drop("natural_product", axis = 1)
#
#    avgResults = pd.DataFrame(data= [], columns=['Fold', 'Number of Molecules', 'r2', 'rmsd', 'bias', 'sdep', 'Model', 'Descriptor', 'Index'])
#
#    for index in range(1, 4):
#        myPreds,_, avgVals = loopedKfoldCrossVal(model, 10, train_X, train_y, f"Mixture_{model}_{descr}_{index}", metabolites)
#        avgVals['Model'] = model
#        avgVals['Descriptor'] = descr
#        avgVals['Index'] = index
#        avgResults = pd.concat([avgResults, avgVals])
#    plotCVResults(train_y, myPreds, f"Mixture_{model}_{descr}_{index}")
#
#    return avgResults

# = JC: Don't think that this function is used anymore.
# Mixed Scaffold Cross Validation (And Saving Results)
#def mixedCVScaffSaveAvg(fileName, descr, model):
#
#    mixDf = pd.read_csv(fileName)
#
#    if descr == "RDKit":
#        df2Mix = CalcRDKitDescriptors(fileName)
#    elif descr == "Morgan":
#        df2Mix = CalcMorganFingerprints(fileName)
#    elif descr == "Both":
#        df2Mix = calcBothDescriptors(fileName)
#
#    mixDf['scaffold'] = mixDf['SMILES'].apply(MurckoScaffoldSmiles)
#
#    allScaff = mixDf["scaffold"].tolist()
#    df2Mix["scaffold"] = allScaff
#    train_X = df2Mix.dropna(axis = 1)
#    train_y = mixDf.pIC50
#    scaffs = mixDf.scaffold
#    train_X = train_X.drop("scaffold", axis = 1)
#
#    avgResults = pd.DataFrame(data= [], columns=['Fold', 'Number of Molecules', 'r2', 'rmsd', 'bias', 'sdep', 'Model', 'Descriptor', 'Index'])
#
#    for index in range(1, 4):
#        _,_, avgVals = loopedKfoldCrossValMix(model, 10, train_X, train_y, f"Mixture + {model} + {descr} + {index}", scaffs)
#        avgVals['Model'] = model
#        avgVals['Descriptor'] = descr
#        avgVals['Index'] = index
#        avgResults = pd.concat([avgResults, avgVals])
#
#    return avgResults

# Creating Bar Chart for CV Splits
def createSplitsBarChart(predictionStats, title):

    columns_to_plot = ['r2', 'rmsd', 'bias', 'sdep']
    df = predictionStats.iloc[:-1]  # Exclude the last row

    num_rows = 5
    num_cols = int(df.shape[0] / num_rows) + (df.shape[0] % num_rows > 0)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, num_rows * 4), constrained_layout=True)
    axes = axes.flatten()  # Reshape to 1D array even for single row

    for idx, row in df.iterrows():
        row_to_plot = row[columns_to_plot]
        axes[idx].bar(columns_to_plot, row_to_plot)
        axes[idx].set_title(f'Fold {idx + 1}')

    plt.savefig(f'{title}: StatisticsPerFold.png')

# Creating Bar Chart for CV Average
def createAvgBarChart(predictionStats, title):
    df = predictionStats.iloc[:-1]
    cols = ['r2', 'rmsd', 'bias', 'sdep']

    means, stds = df[cols].mean(), df[cols].std()

    plt.bar(cols, means, yerr=stds, capsize=7)
    plt.xlabel('Statistic')
    plt.ylabel('Value (Mean ± Standard Deviation)')
    plt.title(f'{title}: Average Prediction Statistics')
    plt.savefig(f'{title}: AverageStatsCV.png')

# Helper Method for Calculating a Model's Stats
def modelStats(test_y, y_pred):
    # Coefficient of determination
    r2 = r2_score(test_y, y_pred)
    # Root mean squared error
    rmsd = root_mean_squared_error(test_y, y_pred)
    # Bias
    bias = np.mean(y_pred - test_y)
    # Standard deviation of the error of prediction
    sdep = np.mean(((y_pred - test_y) - np.mean(y_pred - test_y))**2)**0.5
    return r2, rmsd, bias, sdep

# Plotting a Model's Results
def plotter(modelType, test_y, y_pred, title):

    r2, rmsd, bias, sdep = modelStats(test_y, y_pred)
    statisticValues = f"r2: {round(r2, 3)}\nrmsd: {round(rmsd, 3)}\nbias: {round(bias, 3)}\nsdep: {round(sdep, 3)}"

    nptest_y = test_y.to_numpy() if isinstance(test_y, pd.Series) else test_y
    npy_pred = y_pred

    minVal = min(nptest_y.min(), npy_pred.min())
    maxVal = max(nptest_y.max(), npy_pred.max())

    a, b = np.polyfit(test_y, y_pred, 1)
    xvals = np.linspace(minVal - 1, maxVal + 1, 100)
    yvals = xvals

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(xvals, yvals, '--')
    ax.scatter(nptest_y, npy_pred)
    ax.plot(nptest_y, a * nptest_y + b)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.set_aspect('equal')
    ax.set_title(f'{title}: {modelType} Model')
    ax.text(0.01, 0.99, statisticValues, transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
    plt.savefig(f'{title}: {modelType}_model.png')

# Helper Method for Returning Average Stats of a Model Run
def listAvg(df, index, model_vars, test_y, y_pred):
    r2, rmsd, bias, sdep = modelStats(test_y, y_pred)
    stats = [r2, rmsd, bias, sdep, index]
    combined_vars = model_vars + stats
    df_new = df.copy()
    df_new.loc[len(df_new)] = combined_vars
    return df_new

# Method for Plotting a Model's Results
def plotModel(modelType, train_X, train_y, test_X, test_y, title):

    model_opts = {}
    model_fit_opts = {}

    if modelType == 'chemprop':
        train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size = 0.1)
        train_dataset = [chemprop.data.MoleculeDatapoint.from_smi(smi, [y])
                   for smi, y in zip(train_X, train_y)]
        test_dataset = [chemprop.data.MoleculeDatapoint.from_smi(smi, [y])
                   for smi, y in zip(test_X, test_y)]
        val_dataset = [chemprop.data.MoleculeDatapoint.from_smi(smi, [y])
                   for smi, y in zip(val_X, val_y)] 
        featurizer = chemprop.featurizers.SimpleMoleculeMolGraphFeaturizer()
        train_dset = chemprop.data.MoleculeDataset(train_dataset, featurizer)
        test_dset = chemprop.data.MoleculeDataset(test_dataset, featurizer)
        val_dset = chemprop.data.MoleculeDataset(val_dataset, featurizer)
        scaler = train_dset.normalize_targets()
        model_opts = {'y_scaler' : scaler}
        train_loader = chemprop.data.build_dataloader(train_dset)
        test_loader = chemprop.data.build_dataloader(test_dset, shuffle=False)
        val_loader = chemprop.data.build_dataloader(val_dset, shuffle = False)
        train_X = train_loader
        test_X = test_loader
        val_X = val_loader

    elif modelType == 'torchNN':
        scaler = StandardScaler()
        train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size = 0.1)
        train_X = scaler.fit_transform(train_X)
        val_X = scaler.transform(val_X)
        test_X = scaler.transform(test_X)
        y_scaler = StandardScaler()
        train_y = y_scaler.fit_transform(np.array(train_y).reshape(-1, 1))
        # x_train = SimplePyTorchDataset(x_train, y_train)
        # x_test = SimplePyTorchDataset(x_test, y_test)
        model_opts = {'y_scaler' : y_scaler, 'input_size' : train_X.shape[1]}
        model_fit_opts = {'X_val' : torch.tensor(val_X, dtype=torch.float32),
                          'y_val' : val_y
                        }
    
    elif modelType == 'MGK':
        dataset_train = mgktools.data.data.Dataset.from_df(
                      pd.concat([train_X, train_y], axis=1).reset_index(),
                      pure_columns=[train_X.name],
                      #pure_columns=['Mol'],
                      #target_columns=['Target'],
                      target_columns=[train_y.name],
                      n_jobs=-1
                     )
        kernel_config = mgktools.kernels.utils.get_kernel_config(
                            dataset_train,
                            graph_kernel_type='graph',
                            # Arguments for marginalized graph kernel:
                            mgk_hyperparameters_files=[
                                mgktools.hyperparameters.product_msnorm],
                           )
        dataset_train.graph_kernel_type = 'graph'
        model_opts = {'kernel' : kernel_config.kernel,
                          'optimizer' : None,
                          'alpha' : 0.01,
                          'normalize_y' : True}
        dataset_test = mgktools.data.data.Dataset.from_df(
                      pd.concat([test_X, test_y], axis=1).reset_index(),
                      pure_columns=[test_X.name],
                      #pure_columns=['Mol'],
                      #target_columns=['Target'],
                      target_columns=[test_y.name],
                      n_jobs=-1
                     )
        dataset_test.graph_kernel_type = 'graph'
        train_X = dataset_train.X
        train_y = dataset_train.y
        test_X = dataset_test.X
        test_y = dataset_test.y

    elif modelType == 'MGKSVR':
        dataset_train = mgktools.data.data.Dataset.from_df(
                      pd.concat([train_X, train_y], axis=1).reset_index(),
                      pure_columns=[train_X.name],
                      #pure_columns=['Mol'],
                      #target_columns=['Target'],
                      target_columns=[train_y.name],
                      n_jobs=-1
                     )
        kernel_config = mgktools.kernels.utils.get_kernel_config(
                            dataset_train,
                            graph_kernel_type='graph',
                            # Arguments for marginalized graph kernel:
                            mgk_hyperparameters_files=[
                                mgktools.hyperparameters.product_msnorm],
                           )
        dataset_train.graph_kernel_type = 'graph'
        model_opts = {'kernel' : kernel_config.kernel,
                          'C' : 10.0}
        dataset_test = mgktools.data.data.Dataset.from_df(
                      pd.concat([test_X, test_y], axis=1).reset_index(),
                      pure_columns=[test_X.name],
                      #pure_columns=['Mol'],
                      #target_columns=['Target'],
                      target_columns=[test_y.name],
                      n_jobs=-1
                     )
        dataset_test.graph_kernel_type = 'graph'
        train_X = dataset_train.X
        train_y = dataset_train.y
        test_X = dataset_test.X
        test_y = dataset_test.y

    model = modelTypes[modelType]
    model = model(**model_opts)

    if (modelType == 'chemprop'):
        model.fit(train_X, val_X, **model_fit_opts)
    else:
        model.fit(train_X, train_y, **model_fit_opts)
    
    y_pred = model.predict(test_X)
    #plotter(modelType, test_y, y_pred, title)
    return y_pred

# The Full Stuff: Making a Train/Test, Running CV, and Getting Model Results
def makeModel(fileNameTrain, fileNameTest, desc, model, title, distributor = None):
    if model != 'chemprop' and  model != 'MGK' and model != 'MGKSVR':
        train_X, train_y, test_X, test_y = makeTrainAndTestDesc(fileNameTrain, fileNameTest, 'pIC50', desc)
    else:
        train_X, train_y, test_X, test_y = makeTrainAndTestGraph(fileNameTrain, fileNameTest, 'pIC50')
    df = pd.DataFrame(data = [], columns = ['Descriptors', 'Model', 'Train','Test', 'R2', 'RMSD', 'Bias', 'SDEP', 'Index'])
    modelVars = [desc, model, fileNameTrain, fileNameTest]
    for i in range(1, 4):
        #myPreds, predictionStats = loopedKfoldCrossVal(model, 10, train_X, train_y, f"{title} + {i}", distributor)
        #createSplitsBarChart(predictionStats, f"{title} + {i}")
        #createAvgBarChart(predictionStats, f"{title} + {i}")
        y_pred = plotModel(model, train_X, train_y, test_X, test_y,  f"{title} + {i}")
        df = listAvg(df, i, modelVars, test_y, y_pred)
    return df

# Making a Train/Test and Running CV
def makeModelCVAvg(fileNameTrain, fileNameTest, desc, model, title, trainName, distributor = None):
    train_X, train_y, test_X, test_y = makeTrainAndTestDesc(fileNameTrain, fileNameTest, 'pIC50', desc)
    avgResults = pd.DataFrame(data= [], columns=['Fold', 'Number of Molecules', 'r2', 'rmsd', 'bias', 'sdep', 'Model', 'Descriptor', 'Index', 'Train Set'])
    for i in range(1, 4):
        _,_, avgVals = loopedKfoldCrossVal(model, 10, train_X, train_y, f"{title}_{model}_{desc}_{i}")
        avgVals['Model'] = model
        avgVals['Descriptor'] = desc
        avgVals['Index'] = i
        avgVals['Train Set'] = trainName
        avgResults = pd.concat([avgResults, avgVals])
    return avgResults

# Making a Train/Test and Running CV (Autogenerated Desc)
def makeModelCVAvg2(fileNameTrain, fileNameTest, model, title, trainName, distributor = None):
    train_X, train_y, test_X, test_y = makeTrainAndTestGraph(fileNameTrain, fileNameTest, 'pIC50')
    avgResults = pd.DataFrame(data= [], columns=['Fold', 'Number of Molecules', 'r2', 'rmsd', 'bias', 'sdep', 'Model', 'Descriptor', 'Index', 'Train Set'])
    for i in range(1, 4):
        _,_, avgVals = loopedKfoldCrossVal(model, 10, train_X, train_y, f"{title}_{model}_{i}")
        avgVals['Model'] = model
        avgVals['Descriptor'] = 'N/A'
        avgVals['Index'] = i
        avgVals['Train Set'] = trainName
        avgResults = pd.concat([avgResults, avgVals])
    return avgResults

# = JC: This function is no longer needed, since it just runs 
# loopedKfoldCVSetSplits.
#def modelCVSetTest(fileName, desc, model, title, group, distributor = None):
#    #train_X, train_y, test_X, test_y = makeTrainAndTestGraph(fileNameTrain, fileNameTest, 'pIC50')
#    avgResults = pd.DataFrame(data= [], columns=['Fold', 'Number of Molecules', 'r2', 'rmsd', 'bias', 'sdep', 'Model', 'Descriptor', 'Index', 'Train Set'])
#   # for i in range(1, 4):
#   #     _,_, avgVals = loopedKfoldCVSetSplits(model, 5, fileNameTrain, 'NP',  f"{title}_{model}_{i}")
#   #     avgVals['Model'] = model
#   #     avgVals['Descriptor'] = 'N/A'
#   #     avgVals['Index'] = i
#   #     avgVals['Train Set'] = trainName
#   #     avgResults = pd.concat([avgResults, avgVals])
#    _, predStats, _ = loopedKfoldCVSetSplits(model, desc, fileName, group, title, distributor = None)
#    return predStats
