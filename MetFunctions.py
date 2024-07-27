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

modelTypes = {}

use_mgk = True

if not use_mgk:

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
        modelTypes['MGK'] = GPR
    except ModuleNotFoundError:
        print('WARNING: Cannot import mgktools python modules')

# Making Train and Test, in addition to Descriptors/Fingerprints
def makeTrainAndTestDesc(fileNameTrain, fileNameTest, target, desc):
    dfTrain = pd.read_csv(fileNameTrain)
    dfTest = pd.read_csv(fileNameTest)

    if desc == "RDKit":
        descTrain = CalcRDKitDescriptors(fileNameTrain)
        descTest = CalcRDKitDescriptors(fileNameTest)
    elif desc == "Morgan":
        descTrain = CalcMorganFingerprints(fileNameTrain)
        descTest = CalcMorganFingerprints(fileNameTest)
    elif desc == "Both":
        descTrain = calcBothDescriptors(fileNameTrain)
        descTest = calcBothDescriptors(fileNameTest)
    elif desc == "Coati":
        descTrain = calcCoati(fileNameTrain)
        descTest = calcCoati(fileNameTest)
    
    train_X = descTrain.dropna(axis = 1)
    train_y = dfTrain[target]
    test_X = descTest.dropna(axis = 1)
    test_y = dfTest[target]
    
    common_columns = train_X.columns.intersection(test_X.columns)
    train_X = train_X[common_columns]
    test_X = test_X[common_columns]
    
    return train_X, train_y, test_X, test_y

# Making Train and Test (Descriptors already in files)
def makeTrainAndTest(fileNameTrain, fileNameTest, target):
    dfTrain = pd.read_csv(fileNameTrain)
    dfTest = pd.read_csv(fileNameTest)

    train_X = dfTrain.dropna(axis = 1)\
                    .drop(target)
    train_y = dfTrain[target]
    test_X = dfTest.dropna(axis = 1)\
                    .drop(target)
    test_y = dfTest[target]

    common_columns = train_X.columns.intersection(test_X.columns)
    train_X = train_X[common_columns]
    test_X = test_X[common_columns]

    return train_X, train_y, test_X, test_y

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

# Looped Kfold CrossVal (NOT A MIXED SET)
def loopedKfoldCrossVal(modelType, num_cv, train_X, train_y, title, distributor = None):
    predictions_filename = f'{title}: CV{modelType}_predictions.csv'

    predStats = {'r2_sum': 0, 'rmsd_sum': 0, 'bias_sum': 0, 'sdep_sum': 0}
    predictionStats = pd.DataFrame(data=np.zeros((num_cv, 6)), columns=['Fold', 'Number of Molecules', 'r2', 'rmsd', 'bias', 'sdep'])

    myPreds = pd.DataFrame(index=range(len(train_y)), #index=train_y.index,
                           columns=['Prediction', 'Fold'])
    myPreds['Prediction'] = np.nan
    myPreds['Fold'] = np.nan

    if distributor is None:
        train_test_split = KFold(n_splits = num_cv, shuffle=True, random_state=1)
    else:
        train_test_split = StratifiedKFold(n_splits = num_cv, shuffle = True, random_state = 1)

    if modelType == 'chemprop':
        dataset = [chemprop.data.MoleculeDatapoint.from_smi(smi, [y]) 
                   for smi, y in zip(train_X, train_y)]

    elif modelType == 'MGK':
        dataset = mgktools.data.data.Dataset.from_df(
                      pd.concat([train_X, train_y], axis=1).reset_index(), 
                      pure_columns=[train_X.name],
                      target_columns=[train_y.name],
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

    for n, (train_idx, test_idx) in enumerate(train_test_split.split(train_X, distributor)):

        y_train = train_y.iloc[train_idx]
        y_test = train_y.iloc[test_idx]

        model_opts = {}
        model_fit_opts = {}

        if modelType == 'MGK':
            x_train = mgktools.data.split.get_data_from_index(dataset, 
                                                              train_idx).X
            x_test = mgktools.data.split.get_data_from_index(dataset, 
                                                             test_idx).X
            model_opts = {'kernel' : kernel_config.kernel, 
                          'optimizer' : None, 
                          'alpha' : 0.01, 
                          'normalize_y' : True}

        elif modelType == 'chemprop':
            # Split data into training and test sets:
            train_data, val_data, test_data = \
            chemprop.data.split_data_by_indices(dataset, 
                                                train_indices=train_idx, 
                                                val_indices=None, 
                                                test_indices=test_idx
                                               )

            # Calculate features for molecules:
            featurizer = chemprop.featurizers.SimpleMoleculeMolGraphFeaturizer()
            train_dset = chemprop.data.MoleculeDataset(train_data, featurizer)
            test_dset = chemprop.data.MoleculeDataset(test_data, featurizer)

            # Scale y data based on training set:
            scaler = train_dset.normalize_targets()
            model_opts = {'y_scaler' : scaler}

            # Set up dataloaders for feeding data into models:
            train_loader = chemprop.data.build_dataloader(train_dset)
            test_loader = chemprop.data.build_dataloader(test_dset, shuffle=False)

            # Make name consistent with non-chemprop models:
            x_train = train_loader
            x_test = test_loader

        else:
            x_train = train_X.iloc[train_idx]
            x_test = train_X.iloc[test_idx]

            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            if modelType == 'torchNN':
                y_scaler = StandardScaler()
                y_train = y_scaler.fit_transform(np.array(y_train).reshape(-1, 1))
                # x_train = SimplePyTorchDataset(x_train, y_train)
                # x_test = SimplePyTorchDataset(x_test, y_test)
                model_opts = {'y_scaler' : y_scaler, 'input_size' : x_train.shape[1]}
                model_fit_opts = {'X_val' : torch.tensor(x_test, dtype=torch.float32), 
                                  'y_val' : y_test
                                 }

        model = modelTypes[modelType]
        model = model(**model_opts)

        # Train model
        model.fit(x_train, y_train, **model_fit_opts)
        # model.plot_training_loss()

        y_pred = model.predict(x_test)

        # Metrics calculations
        r2 = r2_score(y_test, y_pred)
        rmsd = root_mean_squared_error(y_test, y_pred)
        bias = np.mean(y_pred - y_test)
        sdep = np.std(y_pred - y_test)

        # Update stats
        predStats['r2_sum'] += r2
        predStats['rmsd_sum'] += rmsd
        predStats['bias_sum'] += bias
        predStats['sdep_sum'] += sdep

        # Update predictions
        myPreds.loc[test_idx, 'Prediction'] = y_pred
        myPreds.loc[test_idx, 'Fold'] = n + 1

        # Ensure correct number of values are assigned
        predictionStats.iloc[n] = [n + 1, len(test_idx), r2, rmsd, bias, sdep]

    # Calculate averages
    r2_av = predStats['r2_sum'] / num_cv
    rmsd_av = predStats['rmsd_sum'] / num_cv
    bias_av = predStats['bias_sum'] / num_cv
    sdep_av = predStats['sdep_sum'] / num_cv

    # Create a DataFrame row for averages
    avg_row = pd.DataFrame([['Average', len(train_y), r2_av, rmsd_av, bias_av, sdep_av]], columns=predictionStats.columns)

    # Append average row to the DataFrame
    predictionStats = pd.concat([predictionStats, avg_row], ignore_index=True)

    return myPreds, predictionStats, avg_row

# Downloading CV Stats if desired
def downloadCVStats(myPreds, predictionStats, title = None):
    predictions_filename = f'{title}: CV_predictions.csv'
    myPreds.to_csv(predictions_filename, index=True)
    predictionStats.to_csv(f'{title}: CV_stats.csv', index=False)

# Looped Kfold CrossVal (MIXED SET)
def loopedKfoldCrossValMix(modelType, num_cv, train_X, train_y, title, distributor = None):
    predictions_filename = f'{title}: CV{modelType}_predictions.csv'

    predStats = {'r2_sum': 0, 'rmsd_sum': 0, 'bias_sum': 0, 'sdep_sum': 0}
    predictionStats = pd.DataFrame(data=np.zeros((num_cv, 6)), columns=['Fold', 'Number of Molecules', 'r2', 'rmsd', 'bias', 'sdep'])

    myPreds = pd.DataFrame(index=range(len(train_y)), #index=train_y.index,
                           columns=['Prediction', 'Fold'])
    myPreds['Prediction'] = np.nan
    myPreds['Fold'] = np.nan

    if distributor is None:
        train_test_split = KFold(n_splits = num_cv, shuffle=True, random_state=1)
    else:
        train_test_split = GroupKFold(n_splits = num_cv)

    for n, (train_idx, test_idx) in enumerate(train_test_split.split(train_X, train_y, distributor)):
        x_train = train_X.iloc[train_idx]
        x_test = train_X.iloc[test_idx]
        y_train = train_y.iloc[train_idx]
        y_test = train_y.iloc[test_idx]

        model = modelTypes[modelType]

        # Train model
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        # Metrics calculations
        r2 = r2_score(y_test, y_pred)
        rmsd = mean_squared_error(y_test, y_pred, squared=False)
        bias = np.mean(y_pred - y_test)
        sdep = np.std(y_pred - y_test)

        # Update stats
        predStats['r2_sum'] += r2
        predStats['rmsd_sum'] += rmsd
        predStats['bias_sum'] += bias
        predStats['sdep_sum'] += sdep

        # Update predictions
        myPreds.loc[test_idx, 'Prediction'] = y_pred
        myPreds.loc[test_idx, 'Fold'] = n + 1

        # Ensure correct number of values are assigned
        predictionStats.iloc[n] = [n + 1, len(test_idx), r2, rmsd, bias, sdep]

    # Calculate averages
    r2_av = predStats['r2_sum'] / num_cv
    rmsd_av = predStats['rmsd_sum'] / num_cv
    bias_av = predStats['bias_sum'] / num_cv
    sdep_av = predStats['sdep_sum'] / num_cv

    # Create a DataFrame row for averages
    avg_row = pd.DataFrame([['Average', len(train_y), r2_av, rmsd_av, bias_av, sdep_av]], columns=predictionStats.columns)

    # Append average row to the DataFrame
    predictionStats = pd.concat([predictionStats, avg_row], ignore_index=True)

    return myPreds, predictionStats, avg_row

# Mixed Set Cross Validation
def mixedCV(fileName, descr, model):

    mixDf = pd.read_csv(fileName)

    if descr == "RDKit":
        df2Mix = CalcRDKitDescriptors(fileName)
    elif descr == "Morgan":
        df2Mix = CalcMorganFingerprints(fileName)
    elif descr == "Both":
        df2Mix = calcBothDescriptors(fileName)

    allMetabolites = mixDf["natural_product"].tolist()
    df2Mix["natural_product"] = allMetabolites
    train_X = df2Mix.dropna(axis = 1)
    train_y = mixDf.pIC50
    metabolites = mixDf.natural_product
    train_X = train_X.drop("natural_product", axis = 1)

    for index in range(1, 4):
        loopedKfoldCrossVal(model, 10, train_X, train_y, f"Mixture_{model}_{descr}_{index}", metabolites)

# Mixed Set Cross Validation (And Saving Results)
def mixedCVSaveAvg(fileName, descr, model):
    
    mixDf = pd.read_csv(fileName)

    if descr == "RDKit":
        df2Mix = CalcRDKitDescriptors(fileName)
    elif descr == "Morgan":
        df2Mix = CalcMorganFingerprints(fileName)
    elif descr == "Both":
        df2Mix = calcBothDescriptors(fileName)

    allMetabolites = mixDf["natural_product"].tolist()
    df2Mix["natural_product"] = allMetabolites
    train_X = df2Mix.dropna(axis = 1)
    train_y = mixDf.pIC50
    metabolites = mixDf.natural_product
    train_X = train_X.drop("natural_product", axis = 1)

    avgResults = pd.DataFrame(data= [], columns=['Fold', 'Number of Molecules', 'r2', 'rmsd', 'bias', 'sdep', 'Model', 'Descriptor', 'Index'])

    for index in range(1, 4):
        myPreds,_, avgVals = loopedKfoldCrossVal(model, 10, train_X, train_y, f"Mixture_{model}_{descr}_{index}", metabolites)
        avgVals['Model'] = model
        avgVals['Descriptor'] = descr
        avgVals['Index'] = index
        avgResults = pd.concat([avgResults, avgVals])
    plotCVResults(train_y, myPreds, f"Mixture_{model}_{descr}_{index}")

    return avgResults

# Mixed Scaffold Cross Validation (And Saving Results)
def mixedCVScaffSaveAvg(fileName, descr, model):

    mixDf = pd.read_csv(fileName)

    if descr == "RDKit":
        df2Mix = CalcRDKitDescriptors(fileName)
    elif descr == "Morgan":
        df2Mix = CalcMorganFingerprints(fileName)
    elif descr == "Both":
        df2Mix = calcBothDescriptors(fileName)

    mixDf['scaffold'] = mixDf['SMILES'].apply(MurckoScaffoldSmiles)

    allScaff = mixDf["scaffold"].tolist()
    df2Mix["scaffold"] = allScaff
    train_X = df2Mix.dropna(axis = 1)
    train_y = mixDf.pIC50
    scaffs = mixDf.scaffold
    train_X = train_X.drop("scaffold", axis = 1)

    avgResults = pd.DataFrame(data= [], columns=['Fold', 'Number of Molecules', 'r2', 'rmsd', 'bias', 'sdep', 'Model', 'Descriptor', 'Index'])

    for index in range(1, 4):
        _,_, avgVals = loopedKfoldCrossValMix(model, 10, train_X, train_y, f"Mixture + {model} + {descr} + {index}", scaffs)
        avgVals['Model'] = model
        avgVals['Descriptor'] = descr
        avgVals['Index'] = index
        avgResults = pd.concat([avgResults, avgVals])

    return avgResults

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
    rmsd = mean_squared_error(test_y, y_pred)**0.5
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
    model = modelTypes[modelType]
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    plotter(modelType, test_y, y_pred, title)
    return y_pred

# The Full Stuff: Making a Train/Test, Running CV, and Getting Model Results
def makeModel(fileNameTrain, fileNameTest, desc, model, title, distributor = None):
    train_X, train_y, test_X, test_y = makeTrainAndTest(fileNameTrain, fileNameTest, 'pIC50', desc)
    df = pd.DataFrame(data = [], columns = ['Descriptors',	'Model', 'Train','Test', 'R2', 'RMSD', 'Bias', 'SDEP', 'Index'])
    modelVars = [desc, model, fileNameTrain, fileNameTest]
    for i in range(1, 4):
        myPreds, predictionStats = loopedKfoldCrossVal(model, 10, train_X, train_y, f"{title} + {i}", distributor)
        createSplitsBarChart(predictionStats, f"{title} + {i}")
        createAvgBarChart(predictionStats, f"{title} + {i}")
        y_pred = plotModel(model, train_X, train_y, test_X, test_y,  f"{title} + {i}")
        df = listAvg(df, i, modelVars, test_y, y_pred)
    return df

# Making a Train/Test and Running CV
def makeModelCVAvg(fileNameTrain, fileNameTest, desc, model, title, trainName, distributor = None):
    train_X, train_y, test_X, test_y = makeTrainAndTest(fileNameTrain, fileNameTest, 'pIC50', desc)
    avgResults = pd.DataFrame(data= [], columns=['Fold', 'Number of Molecules', 'r2', 'rmsd', 'bias', 'sdep', 'Model', 'Descriptor', 'Index', 'Train Set'])
    for i in range(1, 4):
        _,_, avgVals = loopedKfoldCrossVal(model, 10, train_X, train_y, f"{title}_{model}_{descr}_{i}")
        avgVals['Model'] = model
        avgVals['Descriptor'] = descr
        avgVals['Index'] = i
        avgVals['Train Set'] = trainName
        avgResults = pd.concat([avgResults, avgVals])
    return avgResults
