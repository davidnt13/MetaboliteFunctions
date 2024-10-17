import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import numpy as np
import matplotlib.pyplot as plt
from coati.generative.coati_purifications import embed_smiles
from rdkit import Chem

# = JC: I've modified all functions so that they take a list of SMILES rather 
# than a filename, since these functions are called when the dataset file has 
# already been read.

# Calculating RDKit Descriptors
#def CalcRDKitDescriptors(fileName):
#    df = pd.read_csv(fileName)
#    smiles_strings = df['SMILES'].tolist()
# = JC: I've renamed some of the variables in this function to make it a bit 
# more intuitive.
def CalcRDKitDescriptors(smiles_ls):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_ls]
    myDescriptors = [Descriptors.CalcMolDescriptors(mol) for mol in mols]
    if 'Ipc' in myDescriptors.columns:
        myDescriptors['Ipc'] = [Chem.GraphDescriptors.Ipc(mol, avg=True) for mol in mols]
    return pd.DataFrame(myDescriptors)

# Calculating Morgan Fingerprints
def morganHelper(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fpgen = AllChem.GetMorganGenerator(radius, fpSize=n_bits)
    fp = fpgen.GetFingerprint(mol)
    return list(fp)
#def CalcMorganFingerprints(fileName):
#    df = pd.read_csv(fileName)
def CalcMorganFingerprints(smiles_ls):
    df = pd.DataFrame(data=smiles_ls, columns=['SMILES'])
    df['MorganFingerprint'] = df['SMILES'].apply(morganHelper)
    df = df.dropna(subset=['MorganFingerprint'])
    return pd.DataFrame(df['MorganFingerprint'].tolist())

# Calculating Both RDkit Descriptors and Morgan Fingerprints
def calcBothDescriptors(smiles_ls):
    dfMorgan = CalcMorganFingerprints(smiles_ls)
    dfDescr = CalcRDKitDescriptors(smiles_ls)
    bothDescr = pd.concat([dfDescr, dfMorgan], axis=1)
    bothDescr.columns = bothDescr.columns.astype(str)
    return bothDescr

# Calculating Coati
from rdkit.Chem.SaltRemover import SaltRemover
remover = SaltRemover()
import torch
from coati.models.io.coati import load_e3gnn_smiles_clip_e2e

# Model parameters are pulled from the url and stored in a local models/ dir.
encoder, tokenizer = load_e3gnn_smiles_clip_e2e(
    freeze=True,
    device="cpu", # USE FOR CPU
    # model parameters to load.
    doc_url="s3://terray-public/models/barlow_closed.pkl",
)
def calcCoati(smiles_ls):
    #df = pd.read_csv(fileName)\
    #       .set_index('ChEMBL_ID', verify_integrity=True)

    # = JC: I put this in to catch errors with the last datasets where there 
    # were some molecules with multiple components, but hopefully these 
    # molecules should have been removed from the improved datasets.
    # Have to remove any SMILES with multiple separate molecules/components
    # (these should be removed from the final dataset anyway)
    #smiles = [Chem.MolToSmiles(
    #              Chem.MolStandardize.rdMolStandardize.FragmentParent(
    #                  remover.StripMol(
    #                      Chem.MolFromSmiles(smi),
    #                                   dontRemoveEverything=True)))
    #          for smi in smiles_ls]

    # Empty dataframe for saving COATI descriptors:
    df_coati = pd.DataFrame(data=np.zeros((len(smiles_ls), 256)),
                            columns=['coati_'+str(i) for i in range(256)],
                            index=range(len(smiles_ls)))
    df_coati.loc[:,:] = np.nan

    # Generate COATI embeddings
    df_coati[['coati_'+str(i) for i in range(256)]] = \
    [embed_smiles(smi, encoder, tokenizer).cpu().numpy() for smi in smiles_ls]

    return df_coati
