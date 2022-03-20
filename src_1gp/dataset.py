import json
import random
from random import shuffle
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch_scatter import scatter
from torch_geometric.data import Data, InMemoryDataset
from feature import one_of_k_encoding, toxcast_tasks
from utils import random_scaffold_split

try:
    import rdkit
    from rdkit import Chem, RDConfig
    from rdkit.Chem import ChemicalFeatures, MolFromSmiles
    from rdkit.Chem.rdchem import HybridizationType as HT
    from rdkit.Chem.rdchem import BondType as BT
    from rdkit import RDLogger

    RDLogger.DisableLog('rdApp.*')
except:
    rdkit, Chem, RDConfig, MolFromSmiles, ChemicalFeatures, HT, BT = 7 * [None]
    print('Please install rdkit for data processing')

dataset_names = {
    "r": ['esol', 'freesolv', 'lipophilicity', 'physprop_perturb'],
    "c": ['demo', 'bbbp', 'bace', 'sider', 'toxcast', 'toxcast', 'tox21'],
    "a": ['esol', 'freesolv', 'lipophilicity', 'physprop_perturb', 'demo', 'bbbp', 'bace', 'sider', 'toxcast', 'toxcast', 'tox21'],
}



def auto_dataset(args):
    from dataset import Dataset
    if args.dataset in dataset_names["c"] + ['demo']:
        _dataset = Dataset(args.dataset_root,
                           dataset=args.dataset, split_seed=args.split_seed)
        if args.loss in ['ce', 'mtce']:
            from trainer import TrainerMolBinaryClassificationNAN as Trainer
            args.out_dim = 2 * _dataset.num_tasks
        elif args.loss in ['bce', 'bcel']:
            from trainer import TrainerMolBinaryClassificationNANBCE as Trainer
            args.out_dim = 1 * _dataset.num_tasks
        else:
            raise Exception('error loss input')
    elif args.dataset in dataset_names["r"] + ['physprop_perturb']:
        from trainer import TrainerMolRegression as Trainer
        if args.dataset == 'physprop_perturb':
            from dataset import PertubationDataset as Dataset
        _dataset = Dataset(args.dataset_root,
                           dataset=args.dataset, split_seed=args.split_seed)
        args.out_dim = 1 * _dataset.num_tasks
    else:
        raise Exception('error dataset input')
    return args, _dataset, Trainer


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
    def __init__(self, root, dataset='bbbp', split='scaffold', split_seed=1234, transform=None, pre_transform=None,
                 pre_filter=None):
        self.dataset = dataset  # random / random_nan / scaffold
        self.split_seed = split_seed
        super(Dataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        trn, val, test = self.split(type=split)
        self.train, self.val, self.test = trn, val, test
        self.mol_num_node_features = self[0].x.shape[1]
        self.mol_num_edge_features = self[0].edge_attr.shape[1]
        self.num_tasks = len(self.tasks)

    @property
    def raw_file_names(self):
        return ['{}.csv'.format(self.dataset)]

    @property
    def processed_file_names(self):
        return ['dataset_{}.pt'.format(self.dataset)]

    def process(self):
        # load csv
        df = pd.read_csv(self.raw_paths[0])
        target = df[self.tasks].values
        smiles_list = df.smiles.values
        data_list = []

        for i, smi in enumerate(tqdm(smiles_list)):
            if not self.is_valid_smiles(smi): continue
            mol = MolFromSmiles(smi)
            if mol is None: return None
            x, edge_index, edge_attr = get_mol_nodes_edges(mol)
            label = target[i]
            # label[np.isnan(label)] = 6
            if self.dataset in dataset_names["r"]:
                y = torch.FloatTensor(label).unsqueeze(0)
            elif self.dataset in dataset_names["c"] :
                label[np.isnan(label)] = -1  # Fill in -1 for those NaN labels
                y = torch.LongTensor(label).unsqueeze(0)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smi=smi, y=y)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @staticmethod
    def is_valid_smiles(smi):
        try:
            Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True)
        except:
            print("not successfully processed smiles: ", smi)
            return False
        return True

    def split(self, type):
        save_path = Path(self.processed_paths[0]).parent / 'split_{}_{}_{}.ckpt'.format(self.split_seed, self.dataset,
                                                                                         type)
        if save_path.exists():
            trn, val, test = torch.load(save_path)
            return trn, val, test
        elif type == 'random':
            shuffled = self.shuffle()
            train_size = int(0.8 * len(shuffled))
            val_size = int(0.1 * len(shuffled))
            trn = shuffled[:train_size]
            val = shuffled[train_size:(train_size + val_size)]
            test = shuffled[(train_size + val_size):]
            torch.save([trn, val, test], save_path)
            return trn, val, test
        elif type == 'scaffold':
            shuffled = self.shuffle()
            trn, val, test = random_scaffold_split(dataset=shuffled, smiles_list=shuffled.data.smi,
                                                   null_value=-1, seed=self.split_seed)
            torch.save([trn, val, test], save_path)
            return trn, val, test
        else:
            self.log('Error: Unknown split type!!')

    @property
    def tasks(self):
        d = {
            'demo': ['label'],
            'muv': ["MUV-466", "MUV-548", "MUV-600", "MUV-644", "MUV-652", "MUV-689", "MUV-692", "MUV-712", "MUV-713",
                    "MUV-733", "MUV-737", "MUV-810", "MUV-832", "MUV-846", "MUV-852", "MUV-858", "MUV-859"],
            'tox21': ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
                      'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'],
            'toxcast': toxcast_tasks,
            'sider': ['SIDER1', 'SIDER2', 'SIDER3', 'SIDER4', 'SIDER5', 'SIDER6', 'SIDER7', 'SIDER8', 'SIDER9',
                      'SIDER10', 'SIDER11', 'SIDER12', 'SIDER13', 'SIDER14', 'SIDER15', 'SIDER16', 'SIDER17', 'SIDER18',
                      'SIDER19', 'SIDER20', 'SIDER21', 'SIDER22', 'SIDER23', 'SIDER24', 'SIDER25', 'SIDER26',
                      'SIDER27'],
            'clintox': ['FDA_APPROVED', 'CT_TOX'],
            'bbbp': ['BBBP'],
            'bace': ['Class'],
            'esol': ['measured log solubility in mols per litre'],
            'freesolv': ['expt'],
            'lipophilicity': ['exp'],
            'hiv': ['HIV_active'],
            'physprop_perturb': ['LogP'],
        }
        return d[self.dataset]


class PertubationDataset(Dataset):
    def __init__(self, root, dataset='physprop_perturb', split='scaffold', split_seed=1234, transform=None,
                 pre_transform=None, pre_filter=None):
        self.dataset = dataset  # random / random_nan / scaffold
        self.split_seed = split_seed
        super(PertubationDataset, self).__init__(root=root, dataset=dataset, split=None, split_seed=None)
        self.data, self.slices = torch.load(self.processed_paths[0])
        trn, val, test = self.split(type=split)
        self.train, self.val, self.test = trn, val, test
        self.mol_num_node_features = self[0].x.shape[1]
        self.mol_num_edge_features = self[0].edge_attr.shape[1]
        self.num_tasks = len(self.tasks)

    def process(self):
        # load csv
        df = pd.read_csv(self.raw_paths[0])
        target = df[self.tasks].values
        smiles_list = df.SMILES.values
        data_list = []
        for i, smi in enumerate(tqdm(smiles_list)):
            if not self.is_valid_smiles(smi): continue
            mol = MolFromSmiles(smi, sanitize=False) #  sanitize
            if mol is None: return None
            x, edge_index, edge_attr = get_mol_nodes_edges(mol)
            label = target[i]
            if self.dataset in ['physprop_perturb']:
                y = torch.FloatTensor(label).unsqueeze(0)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smi=smi, y=y)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def split(self, type):
        save_path = Path(self.processed_paths[0]).parent / 'split_{}_{}_{}.ckpt'.format(self.split_seed, self.dataset,
                                                                                         type)
        if save_path.exists():
            trn, val, test = torch.load(save_path)
            return trn, val, test
        df = pd.read_csv(self.raw_paths[0])
        train_size = len(df[df['Label'] == 'train'])
        val_size = len(df[df['Label'] == 'val'])
        trn, val, test = self[:train_size], self[train_size:train_size + val_size], self[train_size + val_size:]
        torch.save([trn, val, test], save_path)
        return trn, val, test


def preprocss(smiles_list, labels):
    data_list = []
    for smi, label in tqdm(zip(smiles_list, labels)):
        x, edge_index, edge_attr = get_mol_nodes_edges(Chem.MolFromSmiles(smi))
        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smi=smi, y=torch.tensor(label)))
    return data_list

def dataset_init(data_list): # dataset, 
    # reference to InMemoryDataset.copy and .__init__ method   
    
    class Dataset(InMemoryDataset):
        def __init__(self, root='./', transform=None, pre_transform=None, pre_filter=None):
            super(Dataset, self).__init__(root, transform, pre_transform, pre_filter)
            self.data, self.slices = self.collate(data_list)

        @property
        def processed_file_names(self):
            return ['dataset.py']  # ./dataset.py exist to cheat the InmemoryDataset.__init__
    
        # def process(self):
        #     data, slices = self.collate(data_list)
        #     return data, slices

    return Dataset()

def randomize_smiles(smiles_list):
    smiles_list = [Chem.MolToSmiles(MolFromSmiles(x), doRandom=True) for x in smiles_list]
    return smiles_list

def perturb_test(dataset, level, split=None, split_seed=None):
    labels = {1: 'SMILES_1', 2: 'SMILES_2', 3: 'SMILES_3'}
    save_root = Path('../../Dataset/GLAM-GP/')

    print('Loading perturbed dataset...')
    df = pd.read_csv(save_root / 'raw/{}.csv'.format(dataset))
    col_name = labels[level]
    test_total = df[df.Label == 'test'] # [df[col_name].notna()]
    perturbed_smiles_list = test_total[col_name].to_list()
    original_smiles_list = test_total['SMILES'].to_list()
    labels = test_total['LogP'].to_list()
    # perturbed_path = save_root / 'processed/perturbed_{}_{}_{}.ckpt'.format(dataset, split_seed, level)

    print('Processing test dataset level {}...'.format(level))
    # _, _, M = torch.load(str(save_root / 'processed/split_{}_{}_{}.ckpt'.format(split_seed, dataset, split)))
    # original_smiles_list = randomize_smiles(original_smiles_list)
    data_list1 = preprocss(original_smiles_list, labels)
    M = dataset_init(data_list1) # M, 

    print('Processing perturbed test dataset level {}...'.format(level))
    # _, _, M_prime = torch.load(str(save_root / 'processed/split_{}_{}_{}.ckpt'.format(split_seed, dataset, split)))
    # perturbed_smiles_list = randomize_smiles(perturbed_smiles_list)
    data_list2 = preprocss(perturbed_smiles_list, labels)
    M_prime = dataset_init(data_list2) # M_prime,

    print('perturbs/total:{}/{}'.format(len(test_total), len(df[df.Label == 'test'])))
    print('Perturbation done!')
    Q = test_total['LogP']
    Q_prime = test_total['LogP_{}'.format(level)]
    return M, M_prime, Q.to_numpy(), Q_prime.to_numpy()

