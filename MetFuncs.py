import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor
from sklearn.svm import SVR

modelTypes = {}
modelTypes['RF'] = RandomForestRegressor()
modelTypes['XGBR'] = XGBRegressor()
modelTypes['SVR'] = SVR()
modelTypes['SVRLinear'] = SVR(kernel = "linear")

def CalcRDKitDescriptors(fileName):
    df = pd.read_csv(fileName)
    smiles_strings = df['SMILES'].tolist()
    mySmiles = [Chem.MolFromSmiles(mol) for mol in smiles_strings]
    myDescriptors = [Descriptors.CalcMolDescriptors(mol) for mol in mySmiles]
    return pd.DataFrame(myDescriptors, index = df.index)

def morganHelper(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return list(fp)

def CalcMorganFingerprints(fileName):
    df = pd.read_csv(fileName)
    df['MorganFingerprint'] = df['SMILES'].apply(morganHelper)
    df = df.dropna(subset=['MorganFingerprint'])
    return pd.DataFrame(df['MorganFingerprint'].tolist())

def calcBothDescriptors(fileName):
    dfMorgan = CalcMorganFingerprints(fileName)
    dfDescr = CalcRDKitDescriptors(fileName)
    bothDescr = pd.concat([dfDescr, dfMorgan], axis=1)
    bothDescr.columns = bothDescr.columns.astype(str)
    return bothDescr

def makeTrainAndTest(fileNameTrain, fileNameTest, target, desc):
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
    
    train_X = descTrain.dropna(axis = 1)
    train_y = dfTrain[target]
    test_X = descTest.dropna(axis = 1)
    test_y = dfTest[target]
    
    common_columns = train_X.columns.intersection(test_X.columns)
    train_X = train_X[common_columns]
    test_X = test_X[common_columns]
    
    return train_X, train_y, test_X, test_y

def plotCVResults(modelType, train_y, myPreds, title):
    
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
    ax.set_title(f'{title}: CV {modelType} Model Results')
    plt.savefig(f'{title}: CV{modelType}_modelResults.png')
    #plt.show()

def loopedKfoldCrossVal(modelType, num_cv, train_X, train_y, title, distributor = None):
    predictions_filename = f'{title}: CV{modelType}_predictions.csv'

    predStats = {'r2_sum': 0, 'rmsd_sum': 0, 'bias_sum': 0, 'sdep_sum': 0}
    predictionStats = pd.DataFrame(data=np.zeros((num_cv, 6)), columns=['Fold', 'Number of Molecules', 'r2', 'rmsd', 'bias', 'sdep'])

    myPreds = pd.DataFrame(index=range(len(train_y)), #index=train_y.index,
                           columns=['Prediction', 'Fold'])
    myPreds['Prediction'] = np.nan
    myPreds['Fold'] = np.nan

    if distributor is None:
        train_test_split = KFold(n_splits = num_cv, shuffle=True)
    else:
        train_test_split = StratifiedKFold(n_splits = num_cv, shuffle = True)

    for n, (train_idx, test_idx) in enumerate(train_test_split.split(train_X, distributor)):
        x_train = train_X.iloc[train_idx]
        x_test = train_X.iloc[test_idx]
        y_train = train_y.iloc[train_idx]
        y_test = train_y.iloc[test_idx]

        model = modelTypes[modelType]

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

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

    #myPreds.to_csv(predictions_filename, index=True)
    #predictionStats.to_csv(f'{title}: CV{modelType}_stats.csv', index=False)

    #plotCVResults(modelType, train_y, myPreds, title)

    return myPreds, predictionStats, avg_row

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
        loopedKfoldCrossVal(model, 10, train_X, train_y, f"Mixture + {model} + {descr} + {index}", metabolites)

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
        _,_, avgVals = loopedKfoldCrossVal(model, 10, train_X, train_y, f"Mixture + {model} + {descr} + {index}", metabolites)
        avgVals['Model'] = model
        avgVals['Descriptor'] = descr
        avgVals['Index'] = index
        avgResults = pd.concat([avgResults, avgVals])

    return avgResults

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
    #plt.show()

def createAvgBarChart(predictionStats, title):
    df = predictionStats.iloc[:-1]
    cols = ['r2', 'rmsd', 'bias', 'sdep']
    
    means, stds = df[cols].mean(), df[cols].std()
    
    plt.bar(cols, means, yerr=stds, capsize=7)
    plt.xlabel('Statistic')
    plt.ylabel('Value (Mean ± Standard Deviation)')
    plt.title(f'{title}: Average Prediction Statistics')
    plt.savefig(f'{title}: AverageStatsCV.png')
    #plt.show()

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
    #plt.show()

def listAvg(df, index, model_vars, test_y, y_pred):
    
    r2, rmsd, bias, sdep = modelStats(test_y, y_pred)
    stats = [r2, rmsd, bias, sdep, index]
    
    combined_vars = model_vars + stats
    
    df_new = df.copy()
    
    df_new.loc[len(df_new)] = combined_vars
    
    return df_new

def plotModel(modelType, train_X, train_y, test_X, test_y, title):
    model = modelTypes[modelType]
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    #plotter(modelType, test_y, y_pred, title)
    return y_pred

def makeModel(fileNameTrain, fileNameTest, desc, model, title, distributor = None):
    train_X, train_y, test_X, test_y = makeTrainAndTest(fileNameTrain, fileNameTest, 'pIC50', desc)
    df = pd.DataFrame(data = [], columns = ['Descriptors','Model', 'Train','Test', 'R2', 'RMSD', 'Bias', 'SDEP', 'Index'])
    modelVars = [desc, model, fileNameTrain, fileNameTest]
    for i in range(1, 4):
        myPreds, predictionStats = loopedKfoldCrossVal(model, 10, train_X, train_y, f"{title} + {i}", distributor)
        #createSplitsBarChart(predictionStats, f"{title} + {i}")
        #createAvgBarChart(predictionStats, f"{title} + {i}")
        y_pred = plotModel(model, train_X, train_y, test_X, test_y,  f"{title} + {i}")
        df = listAvg(df, i, modelVars, test_y, y_pred)
    #df.to_csv(f"Model Results-{title}.csv", index=True)
    return df

def makeModelCVAvg(fileNameTrain, fileNameTest, desc, model, title, trainName, distributor = None):
    train_X, train_y, test_X, test_y = makeTrainAndTest(fileNameTrain, fileNameTest, 'pIC50', desc)
    avgResults = pd.DataFrame(data= [], columns=['Fold', 'Number of Molecules', 'r2', 'rmsd', 'bias', 'sdep', 'Model', 'Descriptor', 'Index', 'Train Set'])
    for i in range(1, 4):
        _,_, avgVals = loopedKfoldCrossVal(model, 10, train_X, train_y, f"{title}-{model}-{desc}-{i}")
        avgVals['Model'] = model
        avgVals['Descriptor'] = desc
        avgVals['Index'] = i
        avgVals['Train Set'] = trainName
        avgResults = pd.concat([avgResults, avgVals])
    return avgResults
