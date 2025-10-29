import deepchem as dc
import torch
from torch_geometric.utils import dense_to_sparse
from rdkit import Chem
import pandas as pd
import numpy as np
from utils import get_MACCS
from drug_util import drug_feature_extract


def getData(cell_exp_path, drug_synergy_path):
    """
    Load and preprocess multi-omics data for drug synergy prediction.
    
    Args:
        cell_exp_path (str): Path to the cell line gene expression CSV file.
        drug_synergy_path (str): Path to the drug synergy dataset CSV file.
    Returns:
        tuple: Contains processed drug features, adjacency matrices for GO/ATC networks,
               SMILES-based fingerprints, cell line features, synergy labels, and index mappings.
    """
    # File paths for drug-related data
    drug_smiles_file = '../Data/TRAIN/drug_smiles.csv'  # Drug SMILES structures
    drug_go_file = '../Data/TRAIN/drug_go.csv'  # Drug-GO similarity network
    drug_atc_file = '../Data/TRAIN/drug_atc.csv'  # Drug-ATC similarity network
    
    # Load drug SMILES data
    drug_smiles = pd.read_csv(drug_smiles_file, sep=",", header=0)
    drug_data = pd.DataFrame()
    drug_smiles_fea = []

    # Create a featurizer to convert SMILES into molecular graph representations
    featurizer = dc.feat.ConvMolFeaturizer()
    
    # Process each drug's SMILES string
    for tup in zip(drug_smiles['drugbank_id'], drug_smiles['smiles']):
        # Convert SMILES to RDKit molecule object
        mol = Chem.MolFromSmiles(tup[1])
        # Generate graph features
        mol_f = featurizer.featurize(mol)
        # Store atom features and adjacency list using drugbank_id as key
        drug_data[str(tup[0])] = [mol_f[0].get_atom_features(), mol_f[0].get_adjacency_list()]
        # Extract MACCS fingerprint from SMILES and append to list
        drug_smiles_fea.append(get_MACCS(tup[1]))

    # Map drug IDs to indices for indexing purposes
    drug_num = len(drug_data.keys())
    d_map = dict(zip(drug_data.keys(), range(drug_num)))
    # Convert raw drug graph data into model-ready format (e.g., PyTorch Geometric Data objects)
    drug_feature = drug_feature_extract(drug_data)

    # Load drug-GO similarity matrix and convert to sparse representation
    drug_go = pd.read_csv(drug_go_file, sep=",", header=0, index_col=[0])
    drug_go_tensor = torch.from_numpy(drug_go.values).float()
    drug_go_adj_weight = dense_to_sparse(drug_go_tensor)

    # Load drug-ATC similarity matrix and convert to sparse representation
    drug_atc = pd.read_csv(drug_atc_file, sep=",", header=0, index_col=[0])
    drug_atc_tensor = torch.from_numpy(drug_atc.values).float()
    drug_atc_adj_weight = dense_to_sparse(drug_atc_tensor)
    
    # Load cell line gene expression data
    cell_exp = pd.read_csv(cell_exp_path, sep=',', header=0, index_col=[0])
    cell_num = len(cell_exp.index)
    c_map = dict(zip(cell_exp.index, range(cell_num)))
    cell_feature = np.array(cell_exp, dtype='float32')
    
    # Load synergy data and map identifiers to indices
    synergy_load = pd.read_csv(drug_synergy_path, sep=',', header=0)
    # Filter valid entries where both drugs exist in drug_data and cell line exists
    synergy = [[d_map[str(row[0])], d_map[str(row[1])], c_map[row[2]], float(row[3])] for index, row in
               synergy_load.iterrows() if (str(row[0]) in drug_data.keys() and str(row[1]) in drug_data.keys() and
                                           str(row[2]) in cell_exp.index)]
    
    return drug_feature, drug_go_adj_weight, drug_atc_adj_weight, drug_smiles_fea, cell_feature, synergy, d_map, c_map

        

