import json
from random import shuffle
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch_scatter import scatter
from torch_geometric.data import Data, InMemoryDataset, Batch
from sklearn.utils.class_weight import compute_class_weight
from utils import one_of_k_encoding, get_residue_features

try:
    import rdkit
    from rdkit import Chem, RDConfig
    from rdkit.Chem import ChemicalFeatures
    from rdkit.Chem.rdchem import HybridizationType as HT
    from rdkit.Chem.rdchem import BondType as BT
    from rdkit import RDLogger

    RDLogger.DisableLog('rdApp.*')
except:
    rdkit, Chem, RDConfig, ChemicalFeatures, HT, BT = 6 * [None]
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


def get_pro_nodes_edges(protein_seq, contact_map):
    # add node information
    feat = []
    for residue in protein_seq:
        residue_features = get_residue_features(residue)
        feat.append(residue_features)
    node_attr = torch.FloatTensor(feat)

    # add main_chain information
    m_index_row, m_index_col, m_edge_attr = [], [], []
    for i in range(len(protein_seq) - 1):
        m_index_row += [i, i + 1]
        m_index_col += [i + 1, i]
        m_edge_attr.append([1, 1, 0, 0, 0, 0, 0, 1])  # read the code below about edge feature extract
        m_edge_attr.append([1, 1, 0, 0, 0, 0, 0, 1])

    # read edge features from contactmap.txt
    edge_attr = []
    index_row, index_col = np.where(contact_map > 0)
    index_row, index_col = index_row.tolist(), index_col.tolist()
    for i, j in zip(index_row, index_col):
        main_chain = 0  # int(np.abs(i - j) == 1)
        prob = contact_map[i, j]
        reversed_prob = 1 - prob
        # prob level range
        l1 = int(0 <= prob < 0.3)
        l2 = int(0.3 <= prob < 0.5)
        l3 = int(0.5 <= prob < 0.7)
        l4 = int(0.5 <= prob < 0.9)
        l5 = int(0.9 <= prob <= 1)
        edge_attr.append([main_chain, prob, reversed_prob, l1, l2, l3, l4, l5])

    edge_index = torch.LongTensor([m_index_row + index_row, m_index_col + index_col])
    edge_attr = torch.FloatTensor(m_edge_attr + edge_attr)
    # print(node_attr.shape, edge_index.shape, edge_attr.shape)
    # assert edge_index.shape[1] == edge_attr.shape[0]
    return node_attr, edge_index, edge_attr


# for small interaction dataset do not need to consider storage space
def proteinmol2graph(mol, protein_seq, contact_map):
    if mol is None: return None

    # Extrat molecular and protein's features
    node_attr, edge_index, edge_attr = get_mol_nodes_edges(mol)
    pro_node_attr, pro_edge_index, pro_edge_attr = get_pro_nodes_edges(protein_seq, contact_map)

    # Build pyg data
    data = Data(
        x=node_attr, edge_index=edge_index, edge_attr=edge_attr,  # pos=pos,
        pro_x=pro_node_attr, pro_edge_index=pro_edge_index, pro_edge_attr=pro_edge_attr,
        y=None,  # None as a placeholder
        # id=None,
    )
    return data


class BindingDBProMolInteactionDataset(InMemoryDataset):
    train_val_test_GLAMs = {'train': 0, 'val': 0, 'test': 0}

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(BindingDBProMolInteactionDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_val_test_GLAMs = torch.load(self.processed_paths[1])
        self.mol_datas = torch.load(self.processed_paths[2])
        self.pro_datas = torch.load(self.processed_paths[3])
        n_trn, n_v, n_t = self.train_val_test_GLAMs.values()
        self.train, self.val, self.test = self[:n_trn], self[n_trn:n_trn + n_v], self[n_trn + n_v:n_trn + n_v + n_t]
        self.train = self.train.shuffle()
        # smi[0] for unpack smi from [['smi1'], ['smi2']...]
        self.mol_num_node_features = self.mol_datas[self[0].smi].x.shape[1]
        self.mol_num_edge_features = self.mol_datas[self[0].smi].edge_attr.shape[1]
        self.pro_num_node_features = self.pro_datas[self[0].pro].x.shape[1]
        self.pro_num_edge_features = self.pro_datas[self[0].pro].edge_attr.shape[1]

    @property
    def raw_file_names(self):
        return ['bindingdb/train.txt', 'bindingdb/dev.txt', 'bindingdb/test.txt',
                'bindingdb/pro_contact_map/protein_maps_dict.ckpt']

    @property
    def processed_file_names(self):
        return ['bindingdb_src_7.3/interaction_processed.pt', 'bindingdb_src_7.3/train_val_test_GLAMs.pt',
                'bindingdb_src_7.3/mol_processed.pt', 'bindingdb_src_7.3/pro_processed.pt', ]

    def process(self):
        # mkdir processed/pdbbind
        Path.mkdir(Path(self.processed_paths[0]).parent, exist_ok=True)

        # load all unique mol and protein
        train = pd.read_csv(self.raw_paths[0], sep=' ')
        dev = pd.read_csv(self.raw_paths[1], sep=' ')
        test = pd.read_csv(self.raw_paths[2], sep=' ')
        unique_pros = set(pd.concat([train, dev, test])['target_sequence'].to_list())
        unique_smis = set(pd.concat([train, dev, test])['compound_iso_smiles'].to_list())
        unique_smis = [Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True) for smi in unique_smis]

        # mol preprocess and save
        mol_data_dict = {}
        for i, smi in tqdm(enumerate(unique_smis)):
            mol = Chem.MolFromSmiles(smi)
            node_attr, edge_index, edge_attr = get_mol_nodes_edges(mol)
            data = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
            mol_data_dict[smi] = data
        torch.save(mol_data_dict, Path(self.processed_paths[2]))

        # pro preprocess and save
        protein_maps_dict_path = Path(self.raw_paths[3])
        protein_maps_dict = torch.load(protein_maps_dict_path)  #
        pro_data_dict = {}
        for i, pro in tqdm(enumerate(unique_pros)):
            if pro not in protein_maps_dict.keys(): continue  # skipped some removed protein
            contact_map = protein_maps_dict[pro]
            pro_node_attr, pro_edge_index, pro_edge_attr = get_pro_nodes_edges(pro, contact_map)
            data = Data(x=pro_node_attr, edge_index=pro_edge_index, edge_attr=pro_edge_attr, )
            pro_data_dict[pro] = data
        torch.save(pro_data_dict, Path(self.processed_paths[3]))

        # mol-pro inteaction save
        data_list = []
        for i, dataset_type in enumerate(self.train_val_test_GLAMs):  # 0, 1, 2 for  train, val, test
            data_path = self.raw_paths[i]
            data_pd = pd.read_csv(data_path, sep=' ')
            for i, items in tqdm(enumerate(data_pd.values)):
                smi, pro, interaction = items[0], items[1], items[2]
                smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True)
                assert smi in mol_data_dict.keys()
                data = Data(smi=smi, pro=pro, y=interaction)
                if pro in pro_data_dict.keys():
                    data_list.append(data)
                    self.train_val_test_GLAMs[dataset_type] += 1

        data, slices = self.collate(data_list)
        torch.save((data, slices), Path(self.processed_paths[0]))
        torch.save(self.train_val_test_GLAMs, Path(self.processed_paths[1]))


class LIT_PCBA(InMemoryDataset):
    def __init__(self, root, target='ADRB2', transform=None, pre_transform=None, pre_filter=None):
        self.target = target
        super(LIT_PCBA, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_val_test_GLAMs = torch.load(self.processed_paths[1])
        self.mol_datas = torch.load(self.processed_paths[2])
        self.pro_datas = torch.load(self.processed_paths[3])
        self.weight = torch.FloatTensor(
            compute_class_weight('balanced', [0, 1], self.data.y.numpy()))  # compute for loss # [1, 0]???
        n_trn, n_v, n_t = self.train_val_test_GLAMs.values()
        self.train, self.val, self.test = self[:n_trn], self[n_trn:n_trn + n_v], self[n_trn + n_v:n_trn + n_v + n_t]
        self.train = self.train.shuffle()
        # smi[0] for unpack smi from [['smi1'], ['smi2']...]
        self.mol_num_node_features = self.mol_datas[self[0].smi].x.shape[1]
        self.mol_num_edge_features = self.mol_datas[self[0].smi].edge_attr.shape[1]
        self.pro_num_node_features = self.pro_datas[self[1].pro].x.shape[1]
        self.pro_num_edge_features = self.pro_datas[self[1].pro].edge_attr.shape[1]

    @property
    def raw_file_names(self):
        return ['lit_pcba_raw/{}'.format(self.target), 'lit_pcba_raw/raptorx_pred/contact_8.5/protein_maps_dict.ckpt']

    @property
    def processed_file_names(self):
        return ['lit_pcba_8.5/{}/interaction_processed.pt'.format(self.target),
                'lit_pcba_8.5/{}/train_val_test_GLAMs.pt'.format(self.target),
                'lit_pcba_8.5/{}/mol_processed.pt'.format(self.target),
                'lit_pcba_8.5/{}/pro_processed.pt'.format(self.target), ]

    @property
    def pro_fastas(self):
        pros = {
            'ALDH1': ['MSSSGTPDLPVLLTDLKIQYTKIFINNEWHDSVSGKKFPVFNPATEEELCQVEEGDKEDVDKAVKAARQAFQIGSPWRTMDASERGRLLYKLADL'
                      'IERDRLLLATMESMNGGKLYSNAYLSDLAGCIKTLRYCAGWADKIQGRTIPIDGNFFTYTRHEPIGVCGQIIPWNFPLVMLIWKIGPALSCGNTV'
                      'VVKPAEQTPLTALHVASLIKEAGFPPGVVNIVPGYGPTAGAAISSHMDIDKVAFTGSTEVGKLIKEAAGKSNLKRVTLELGGKSPCIVLADADLD'
                      'NAVEFAHHGVFYHQGQCCIAASRIFVEESIYDEFVRRSVERAKKYILGNPLTPGVTQGPQIDKEQYDKILDLIESGKKEGAKLECGGGPWGNKGY'
                      'FVQPTVFSNVTDEMRIAKEEIFGPVQQIMKFKSLDDVIKRANNTFYGLSAGVFTKDIDKAITISSALQAGTVWVNCYGVVSAQCPFGGFKMSGNG'
                      'RELGEYGFHEYTEVKTVTVKISQKNS'],
            'ESR1_ant': ['NSLALSLTADQMVSALLDAEPPILYSEYDPTRPFSEASMMGLLTNLADRELVHMINWAKRVPGFVDLTLHDQVHLLESAWLEILMIGLVWRS'
                         'MEHPGKLLFAPNLLLDRNQGKSVEGMVEIFDMLLATSSRFRMMNLQGEEFVCLKSIILLNSGVYTFLSSTLKSLEEKDHIHRVLDKITDTLI'
                         'HLMAKAGLTLQQQHQRLAQLLLILSHIRHMSNKGMEHLYSMKSKNVVPLYDLLLEMLDAHRLHA'],
            'KAT2A': ['GSGIIEFHVIGNSLTPKANRRVLLWLVGLQNVFSHQLPRMPKEYIARLVFDPKHKTLALIKDGRVIGGICFRMFPTQGFTEIVFCAVTSNEQVKG'
                      'YGTHLMNHLKEYHIKHNILYFLTYADEYAIGYFKKQGFSKDIKVPKSRYLGYIKDYEGATLMECELNPRIPYT'],
            'MAPK1': ['GDLGSDELMAAAAAAGAGPEMVRGQVFDVGPRYTNLSYIGEGAYGMVCSAYDNVNKVRVAIKKISPFEHQTYCQRTLREIKILLRFRHENIIGIN'
                      'DIIRAPTIEQMKDVYIVQDLMETDLYKLLKTQHLSNDHICYFLYQILRGLKYIHSANVLHRDLKPSNLLLNTTCDLKICDFGLARVADPDHDHTG'
                      'FLTEYVATRWYRAPEIMLNSKGYTKSIDIWSVGCILAEMLSNRPIFPGKHYLDQLNHILGILGSPSQEDLNCIINLKARNYLLSLPHKNKVPWNR'
                      'LFPNADSKALDLLDKMLTFNPHKRIEVEQALAHPYLEQYYDPSDEPIAEAPFKFDMELDDLPKEKLKELIFEETARFQPGYRS'],
            'FEN1': ['MGIQGLAKLIADVAPSAIRENDIKSYFGRKVAIDASMSIYQFLIAVRQGGDVLQNEEGETTSHLMGMFYRTIRMMENGIKPVYVFDGKPPQLKSGE'
                     'LAKRSERRAEAEKQLQQAQAAGAEQEVEKFTKRLVKVTKQHNDECKHLLSLMGIPYLDAPSEAEASCAALVKAGKVYAAATEDMDCLTFGSPVLMR'
                     'HLTASEAKKLPIQEFHLSRILQELGLNQEQFVDLCILLGSDYCESIRGIGPKRAVDLIQKHKSIEEIVRRLDPNKYPVPENWLHKEAHQLFLEPEV'
                     'LDPESVELKWSEPNEEELIKFMCGEKQFSEERIRSGVKRLSKSRQGSTLEVLFQGPGGGHHHHHH'],
        }
        return pros[self.target]

    def process(self):
        # mkdir processed/pdbbind
        Path.mkdir(Path(self.processed_paths[0]).parent.parent, exist_ok=True)
        Path.mkdir(Path(self.processed_paths[0]).parent, exist_ok=True)

        # read all data
        # active_T, active_V, inactive_T, inactive_V = [None] * 4  # placehold
        for p in Path(self.raw_paths[0]).glob('*.smi'):
            if '_active_T.smi' in str(p): active_T = pd.read_csv(p, sep=' ', header=None).iloc[:, 0].tolist()
            if '_active_V.smi' in str(p): active_V = pd.read_csv(p, sep=' ', header=None).iloc[:, 0].tolist()
            if '_inactive_T.smi' in str(p): inactive_T = pd.read_csv(p, sep=' ', header=None).iloc[:, 0].tolist()
            if '_inactive_V.smi' in str(p): inactive_V = pd.read_csv(p, sep=' ', header=None).iloc[:, 0].tolist()
        # todo: test canonicial

        # mol preprocess and save
        mol_data_dict = {}
        for smi in tqdm(active_T + active_V + inactive_T + inactive_V):
            mol = Chem.MolFromSmiles(smi)
            node_attr, edge_index, edge_attr = get_mol_nodes_edges(mol)
            data = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
            mol_data_dict[smi] = data
        torch.save(mol_data_dict, Path(self.processed_paths[2]))

        # pro preprocess and save
        protein_maps_dict = torch.load(self.raw_paths[1])  #
        pro_data_dict = {}
        print(self.target, self.pro_fastas)  # for dataset build check
        for pro in tqdm(self.pro_fastas):
            if pro not in protein_maps_dict.keys(): continue  # skipped some removed protein
            contact_map = protein_maps_dict[pro]
            pro_node_attr, pro_edge_index, pro_edge_attr = get_pro_nodes_edges(pro, contact_map)
            data = Data(x=pro_node_attr, edge_index=pro_edge_index, edge_attr=pro_edge_attr)
            pro_data_dict[pro] = data
        torch.save(pro_data_dict, Path(self.processed_paths[3]))

        # mol-pro inteaction train val split and save
        aT_list = []
        for smi in tqdm(active_T):
            data = Data(smi=smi, pro=self.pro_fastas[0], y=1)
            aT_list.append(data)
        iT_list = []
        for smi in tqdm(inactive_T):
            data = Data(smi=smi, pro=self.pro_fastas[0], y=0)
            iT_list.append(data)
        shuffle(aT_list)
        shuffle(iT_list)
        split_rate = 0.7 if len(aT_list) > 50 else 0.5
        print(len(aT_list), len(iT_list), split_rate)
        aT_split = int(len(aT_list) * 0.70)
        iT_split = int(len(iT_list) * 0.70)
        train = aT_list[:aT_split] + iT_list[:iT_split]
        valid = aT_list[aT_split:] + iT_list[iT_split:]

        # mol-pro inteaction test save
        test = []
        for smi in tqdm(active_V):
            data = Data(smi=smi, pro=self.pro_fastas[0], y=1)
            test.append(data)
        for smi in tqdm(inactive_V):
            data = Data(smi=smi, pro=self.pro_fastas[0], y=0)
            test.append(data)
        self.train_val_test_GLAMs = {'train': len(train), 'val': len(valid), 'test': len(test)}
        print('{} mol and {} protein seqs , {} interaction saved'.format(
            len(mol_data_dict.keys()), len(pro_data_dict.keys()), self.train_val_test_GLAMs))
        torch.save(self.train_val_test_GLAMs, Path(self.processed_paths[1]))

        data, slices = self.collate(train + valid + test)
        torch.save((data, slices), Path(self.processed_paths[0]))


def extract_batch_data(dataset, id_batch):
    smis, pros, ys = id_batch.smi, id_batch.pro, id_batch.y
    mol_batch_list = [dataset.mol_datas[smi] for smi in smis]  # smi[0] for unpack smi from [['smi1'], ['smi2']...]
    mol_batch = Batch().from_data_list(mol_batch_list)
    pro_batch_list = [dataset.pro_datas[pro] for pro in pros]
    pro_batch = Batch().from_data_list(pro_batch_list)
    return mol_batch, pro_batch


if __name__ == '__main__':
    # dataset = BindingDBProMolInteactionDataset('../../Dataset/GLAM-DTI')

    for target in ['ALDH1', 'ESR1_ant', 'KAT2A', 'MAPK1']:
        dataset = LIT_PCBA('../../Dataset/GLAM-DTI', target=target)
        id_batch = Batch().from_data_list(dataset.train[:16])
        mol_batch, pro_batch = extract_batch_data(dataset, id_batch)
    torch.save((mol_batch, pro_batch), './other/test_batch.pt')
    # train, val, test, mol_datas, pro_datas = load_dataset_bindingdb('../dataset/')
