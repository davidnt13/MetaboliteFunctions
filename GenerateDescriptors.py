import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import numpy as np
import matplotlib.pyplot as plt
from coati.generative.coati_purifications import embed_smiles
from rdkit import Chem

# Imports for ChemBERTa
from transformers import AutoTokenizer, AutoModel
import deepchem as dc

def CalcRDKitDescriptors(smiles_ls, verbose=True):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_ls]
    myDesc = [Descriptors.CalcMolDescriptors(mol) for mol in mols]
    myDescriptors = pd.DataFrame(myDesc)
    if 'Ipc' in myDescriptors.columns:
        myDescriptors['Ipc'] = [Chem.GraphDescriptors.Ipc(mol, avg=True) 
                                if mol is not None else None for mol in mols]
    # Check for any molecules with all NaN values in the DataFrame and return 
    # an error if found:
    if myDescriptors.isna().all(axis=1).any():
        raise ValueError(
            'All RDKit descriptors are NaN for molecule(s): {}'.format(
            ', '.join([smiles_ls[i] 
                       for i, v in enumerate(myDescriptors.isna().all(axis=1)) 
                       if v])))
    # Optionally print out a message about any descriptors which have NaN 
    # values for any molecule(s):
    if verbose:
        if myDescriptors.isna().any(axis=None):
            print('RDKit descriptor(s): {} contain NaN for some or all '
                  'molecule(s): {}'.format(
                  ', '.join(myDescriptors.columns[myDescriptors.isna().any(axis=0)]),
                  ', '.join([smiles_ls[i] 
                             for i, v in enumerate(myDescriptors.isna().any(axis=1)) 
                             if v])))
    return myDescriptors.dropna(axis = 1)

def CalcRDKitChemprop(smiles_ls):
    # Convert SMILES to RDKit Mol objects
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_ls]

    # Compute descriptors for each molecule
    myDesc = []
    for mol in mols:
        if mol is not None:  # Only process valid molecules
            try:
                desc = Descriptors.CalcMolDescriptors(mol)
                # Make sure it's a list or array, if not, wrap it in a list
                if isinstance(desc, tuple):
                    desc = list(desc)  # Ensure it's a list if tuple
                myDesc.append(desc)
            except Exception as e:
                print(f"Error processing molecule: {e}")
                pass  # Skip molecules that fail descriptor computation

    if not myDesc:
        raise ValueError("No valid molecular descriptors computed.")

    # Convert list of descriptors to NumPy array
    descriptor_array = np.array(myDesc)

    # Handle cases where only one descriptor is computed (1D array case)
    if descriptor_array.ndim == 1:  # If it's a 1D array, make it 2D
        descriptor_array = descriptor_array.reshape(1, -1)
    
    descriptor_array = np.nan_to_num(descriptor_array)

    return descriptor_array

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
    if df['MorganFingerprint'].isna().any():
        raise ValueError(
            'Morgan fingerprint is NaN for molecule(s): {}'.format(
            ', '.join(df['SMILES'].loc[df['MorganFingerprint'].isna()])))
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
    problem_smiles = []
    for smi in smiles_ls:
        if '.' in smi:
            problem_smiles.append(smi)
    if len(problem_smiles) > 0:
        raise ValueError('SMILES: {} has a disconnection ("."), this '
                         'cannot be used with COATI descriptors'.format(
                         ', '.join(problem_smiles)))

    # Empty dataframe for saving COATI descriptors:
    df_coati = pd.DataFrame(data=np.zeros((len(smiles_ls), 256)),
                            columns=['coati_'+str(i) for i in range(256)],
                            index=range(len(smiles_ls)))
    df_coati.loc[:,:] = np.nan

    # Generate COATI embeddings
    df_coati[['coati_'+str(i) for i in range(256)]] = \
    [embed_smiles(smi, encoder, tokenizer).cpu().numpy() for smi in smiles_ls]

    return df_coati

# Calculating ChemBerta
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

def calcChemBert(smiles_list, batch_size=8):
    embeddings = []
    smiles_list = [str(s) for s in smiles_list if isinstance(s, str) and s.strip()]
    # Ensure there are valid SMILES
    if not smiles_list:
        raise ValueError("No valid SMILES found in input!")
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i : i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend(batch_embeddings)

    return pd.DataFrame(embeddings)
