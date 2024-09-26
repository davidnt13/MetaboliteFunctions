import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import numpy as np
import matplotlib.pyplot as plt
from coati.generative.coati_purifications import embed_smiles
from rdkit import Chem

# Calculating RDKit Descriptors
def CalcRDKitDescriptors(fileName):
    df = pd.read_csv(fileName)
    smiles_strings = df['SMILES'].tolist()
    mySmiles = [Chem.MolFromSmiles(mol) for mol in smiles_strings]
    myDescriptors = [Descriptors.CalcMolDescriptors(mol) for mol in mySmiles]
    return pd.DataFrame(myDescriptors, index = df.index)

# Calculating Morgan Fingerprints
def morganHelper(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fpgen = AllChem.GetMorganGenerator(radius, fpSize=n_bits)
    fp = fpgen.GetFingerprint(mol)
    return list(fp)
def CalcMorganFingerprints(fileName):
    df = pd.read_csv(fileName)
    df['MorganFingerprint'] = df['SMILES'].apply(morganHelper)
    df = df.dropna(subset=['MorganFingerprint'])
    return pd.DataFrame(df['MorganFingerprint'].tolist())

# Calculating Both RDkit Descriptors and Morgan Fingerprints
def calcBothDescriptors(fileName):
    dfMorgan = CalcMorganFingerprints(fileName)
    dfDescr = CalcRDKitDescriptors(fileName)
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
def calcCoati(fileName):
    df = pd.read_csv(fileName)\
           .set_index('ChEMBL_ID', verify_integrity=True)

    # Have to remove any SMILES with multiple separate molecules/components
    # (these should be removed from the final dataset anyway)
    smiles = [Chem.MolToSmiles(
                  Chem.MolStandardize.rdMolStandardize.FragmentParent(
                      remover.StripMol(
                          Chem.MolFromSmiles(smi),
                                       dontRemoveEverything=True)))
              for smi in df['SMILES'].to_list()]

    # Empty dataframe for saving COATI descriptors:
    df_coati = pd.DataFrame(data=np.zeros((len(df), 256)),
                            columns=['coati_'+str(i) for i in range(256)],
                            index=df.index)
    df_coati.loc[:,:] = np.nan

    # Generate COATI embeddings
    df_coati[['coati_'+str(i) for i in range(256)]] = \
    [embed_smiles(smi, encoder, tokenizer).cpu().numpy() for smi in smiles]

    return df_coati
    #return df_coati.join(df['pIC50'])
