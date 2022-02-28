import json
import random
from random import shuffle
from pathlib import Path
from tqdm import tqdm
import numpy as np

import pandas as pd
import torch
from torch_scatter import scatter
from torch_geometric.data import Data, InMemoryDataset, Batch
from utils import one_of_k_encoding, random_scaffold_split, seed_torch
from torch.nn.functional import one_hot

try:
    import rdkit
    import sqlite3
    from rdkit import Chem, RDConfig
    from rdkit.Chem import ChemicalFeatures, MolFromSmiles
    from rdkit.Chem.rdchem import HybridizationType as HT
    from rdkit.Chem.rdchem import BondType as BT
    from rdkit import RDLogger

    RDLogger.DisableLog('rdApp.*')
except:
    rdkit, sqlite3, Chem, RDConfig, MolFromSmiles, ChemicalFeatures, HT, BT = 8 * [None]
    print('Please install rdkit for data processing')


def get_mol_nodes_edges(mol):
    # Read node features
    N = mol.GetNumAtoms()
    atom_type = []
    atomic_number = []
    aromatic = []
    hybridization = []
    # num_hs = []
    for atom in mol.GetAtoms():
        atom_type.append(atom.GetSymbol())
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization.append(atom.GetHybridization())

    # Read edge features
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bond.GetBondType()]
    edge_index = torch.LongTensor([row, col])
    edge_type = [one_of_k_encoding(t, [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]) for t in edge_type]
    edge_attr = torch.FloatTensor(edge_type)
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]
    row, col = edge_index

    # Concat node fetures
    hs = (torch.tensor(atomic_number, dtype=torch.long) == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=N).tolist()
    x_atom_type = [one_of_k_encoding(t, ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I']) for t in atom_type]
    x_hybridization = [one_of_k_encoding(h, [HT.SP, HT.SP2, HT.SP3]) for h in hybridization]
    x2 = torch.tensor([atomic_number, aromatic, num_hs], dtype=torch.float).t().contiguous()
    x = torch.cat([torch.FloatTensor(x_atom_type), torch.FloatTensor(x_hybridization), x2], dim=-1)

    return x, edge_index, edge_attr


class Dataset(InMemoryDataset):
    def __init__(self, root, dataset='drugbank_caster', split='random', split_seed=1234, transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.train_val_test_GLAMs = {'train': 0, 'val': 0, 'test': 0}
        self.dataset = dataset
        super(Dataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.mol_data = torch.load(self.processed_paths[1])
        trn, val, test = self.split(type=split, seed=split_seed)
        self.train, self.val, self.test = trn, val, test
        mol_GLAM = list(self.mol_data.values())[0]
        self.mol_num_node_features = mol_GLAM.x.shape[1]
        self.mol_num_edge_features = mol_GLAM.edge_attr.shape[1]

    @property
    def raw_file_names(self):
        if self.dataset == 'drugbank_caster':
            return ['drugbank_caster/ddi_total.csv']
        else:
            raise ValueError('Error dataset input!')

    @property
    def processed_file_names(self):
        return ['{}_smi_labels.pt'.format(self.dataset), '{}_mol_data.pt'.format(self.dataset)]

    def ddi2datalist(self, drug1_list: list, drug2_list: list, interaction_list: list, list_all_smiles=None):
        data_list = []
        for drug1, drug2, label in tqdm(zip(drug1_list, drug2_list, interaction_list)):
            drug1 = self.canonical(drug1)
            drug2 = self.canonical(drug2)
            if drug1 is None or drug2 is None: continue
            if list_all_smiles is not None:
                if drug1 not in list_all_smiles or drug2 not in list_all_smiles: continue
            data = Data(smi1=drug1, smi2=drug2, y=label)
            data_list.append(data)
        return data_list

    def smiles2datadict(self, smiles_list: list):
        data_dict = {}
        for smi in tqdm(set(smiles_list)):
            smi = self.canonical(smi)
            if smi is None: continue
            x, edge_index, edge_attr = get_mol_nodes_edges(MolFromSmiles(smi))
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data_dict[smi] = data
        return data_dict

    @staticmethod
    def canonical(smi):
        try:
            return Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)
        except:
            print("failed smiles: ", smi, end='\t')
            return None

    def process(self):
        if self.dataset == 'drugbank_caster':
            drugbank_total = pd.read_csv(self.raw_paths[0])
            data_list = self.ddi2datalist(drugbank_total['Drug1_SMILES'].to_list(),
                                          drugbank_total['Drug2_SMILES'].to_list(),
                                          drugbank_total['label'].to_list())
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])

            data_dict = self.smiles2datadict(
                drugbank_total['Drug1_SMILES'].to_list() + drugbank_total['Drug2_SMILES'].to_list())
            torch.save(data_dict, self.processed_paths[1])
        else:
            raise ValueError('Unknown dataset {}!'.format(self.dataset))

    @staticmethod
    def is_valid_smiles(smi):
        try:
            Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        except:
            print("not successfully processed smiles: ", smi)
            return False
        return True

    def split(self, type, seed=0):
        seed_torch(seed)
        save_path = Path(self.processed_paths[0]).parent / 'split_{}_{}_{}.ckpt'.format(self.dataset, type, seed)
        if save_path.exists():
            trn, val, test = torch.load(save_path)
            return trn, val, test
        elif type == 'random' and self.dataset in ['drugbank_caster']:
            shuffled = self.shuffle()
            train_size = int(0.7 * len(shuffled))
            valid_size = int(0.1 * len(shuffled))
            trn = shuffled[: train_size]
            val = shuffled[train_size: train_size + valid_size]
            test = shuffled[train_size + valid_size:]
            torch.save([trn, val, test], save_path)
            assert len(trn) + len(val) + len(test) == len(self)
            return trn, val, test
        else:
            self.log('Error: Unknown split type!!')


def extract_batch_data(mol_data, id_batch):
    smis1, smis2, ys = id_batch.smi1, id_batch.smi2, id_batch.y
    mol1_batch_list = [mol_data[smi] for smi in smis1]  # smi[0] for unpack smi from [['smi1'], ['smi2']...]
    mol1_batch = Batch().from_data_list(mol1_batch_list)
    mol2_batch_list = [mol_data[smi] for smi in smis2]
    mol2_batch = Batch().from_data_list(mol2_batch_list)
    return mol1_batch, mol2_batch


if __name__ == '__main__':
    dataset = Dataset('../../Dataset/GLAM-DDI', dataset='drugbank_caster')
