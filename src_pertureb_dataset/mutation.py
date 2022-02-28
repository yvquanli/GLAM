from numpy.random import choice
import copy
import networkx as nx
import pickle
import numpy as np
import os.path as op

try:
    from rdkit import DataStructs
    from rdkit.Chem import AllChem as Chem
    from rdkit.Chem import rdMolDescriptors

    possible_bonds = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE
    ]
except:
    DataStructs, Chem, rdMolDescriptors, possible_bonds = 4 * [None]

_fscores = None
table_of_elements = {
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F',
    16: 'S',
    17: 'Cl',
    35: 'Br',
    53: 'I',
}
vocab_nodes_encode = {
    'C': 1, 'N': 2, 'O': 3, 'S': 3, 'F': 4, 'Cl': 4, 'Br': 4, 'I': 4
}

'''Molecule's object class be an individual in population for evolutionary algorithm'''


def readFragmentScores(name='fpscores'):
    import gzip
    global _fscores
    # generate the full path filename:
    if name == "fpscores":
        name = op.join(op.dirname(__file__), name)
    _fscores = pickle.load(gzip.open('%s.pkl.gz' % name))
    outDict = {}
    for i in _fscores:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict


def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro


class Molecule(object):
    def __init__(self, smiles):
        self.smiles = smiles

        self.possible_bonds = possible_bonds
        self.table_of_elements = table_of_elements
        self.vocab_nodes_encode = vocab_nodes_encode
        self.mol = Chem.MolFromSmiles(smiles)

        self.adj = self._get_adj_mat(smiles)
        self.node_list = self._get_node_list(smiles)
        self.num_atom = len(self.node_list)
        self.expand_mat = self._get_expand_mat(self.adj, self.node_list)
        self.life_time = 0
        self.pool_life_time = 0
        self.similarity = -1
        self.prior_flag = False

    def __hash__(self):
        return hash(self.smiles)

    def __eq__(self, other):
        self_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.smiles))
        other_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(other.smiles))
        return self_smiles == other_smiles

    def _get_adj_mat(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(mol)
        G = mol2nx(mol)
        atomic_nums = nx.get_node_attributes(G, 'atomic_num')
        adj = np.zeros([len(atomic_nums), len(atomic_nums)])
        bond_list = nx.get_edge_attributes(G, 'bond_type')
        for edge in G.edges():
            first, second = edge
            adj[[first], [second]] = self.possible_bonds.index(bond_list[first, second]) + 1
            adj[[second], [first]] = self.possible_bonds.index(bond_list[first, second]) + 1
        return adj

    def _get_node_list(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        G = mol2nx(mol)
        atomic_nums = nx.get_node_attributes(G, 'atomic_num')
        node_list = []
        for i in range(len(atomic_nums)):
            try:
                node_list.append(self.table_of_elements[atomic_nums[i]])
            except KeyError:
                pass
        return node_list

    def _get_expand_mat(self, adj, node_list):
        def _get_diag_mat(node_list):
            length = len(node_list)
            diag_mat = np.zeros([length, length])
            for i in range(length):
                diag_mat[[i], [i]] = self.vocab_nodes_encode[node_list[i]]
            return diag_mat

        diag_mat = _get_diag_mat(node_list)
        return adj + diag_mat

    def _smilarity_between_two_smiles(self, smi2):
        smi1 = self.smiles
        mol1, mol2 = Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)

        vec1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 2)
        vec2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 2)

        tani = DataStructs.TanimotoSimilarity(vec1, vec2)
        return tani


class Config(object):
    def __init__(self):
        self.poplution_size = 50
        self.crossover_rate = 0.8
        self.init_poplution_file_name = './randomv2.sdf'
        self.crossover_mu = 0.5
        self.crossover_sigma = 0.1
        self.graph_size = 80
        self.full_valence = 5
        self.mutation_rate = [0.45, 0.45, 0.1]
        self.temp_elements = {
            0: 'C',
            1: 'N',
            2: 'O',
            3: 'S',
        }
        self.vocab_nodes_encode = {
            'C': 1, 'N': 2, 'O': 3, 'S': 3, 'F': 4, 'Cl': 4, 'Br': 4, 'I': 4
        }
        self.possible_bonds = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE
        ]
        self.length_elements = len(self.temp_elements)
        self.mutate_rate = 0.03


# mask the matrix to find valid row to take mutation or crossover operation
def mask(expand_adj):
    config = Config()
    node_num = np.count_nonzero(expand_adj.diagonal())
    row_sum = np.sum(expand_adj[:node_num, :node_num], axis=0)
    mask_row = np.argwhere(row_sum < config.full_valence).squeeze(axis=1).tolist()
    return mask_row


# adj2mol is to convert adjacent matrix into mol object in rdkit
def adj2mol(nodes, adj, possible_bonds):
    mol = Chem.RWMol()

    for i in range(len(nodes)):
        # print(nodes[i])
        atom = Chem.Atom(nodes[i])
        mol.AddAtom(atom)

    for i in range(len(nodes) - 1):
        for j in range(i + 1, len(nodes)):
            if adj[i, j]:
                mol.AddBond(i, j, possible_bonds[adj[i, j] - 1])

    return mol


# mol2nx is to convert mol object in rdkit into network object
def mol2nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   formal_charge=atom.GetFormalCharge(),
                   chiral_tag=atom.GetChiralTag(),
                   hybridization=atom.GetHybridization(),
                   num_explicit_hs=atom.GetNumExplicitHs(),
                   is_aromatic=atom.GetIsAromatic())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G


def diversity_scores(mols, data):
    rand_mols = np.random.choice(data.data, 100)
    fps = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048) for mol in rand_mols]

    scores = np.array(
        list(map(lambda x: __compute_diversity(x, fps) if x is not None else 0, mols)))
    scores = np.clip(remap(scores, 0.9, 0.945), 0.0, 1.0)

    return scores


def __compute_diversity(mol, fps):
    ref_fps = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
    dist = DataStructs.BulkTanimotoSimilarity(ref_fps, fps, returnDistance=True)
    score = np.mean(dist)
    return score


def remap(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)


def canonicalize_matrix(matrix, node_list):
    config = Config()
    atom_num = len(node_list)
    exp_mat = matrix

    for i in range(atom_num):
        exp_mat[i, i] = config.vocab_nodes_encode[node_list[i]]
    row_sum = np.sum(exp_mat[:atom_num, :atom_num], axis=0)
    error_row = np.argwhere(row_sum > config.full_valence).squeeze(axis=1).tolist()
    if len(error_row) > 0:
        return False
    return True


def _smilarity_between_two_mols(mol1, mol2):
    # mol1, mol2 = Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)
    vec1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 4, nBits=512)
    vec2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 4, nBits=512)

    tani = DataStructs.TanimotoSimilarity(vec1, vec2)
    return tani


class Configuration(object):
    def __init__(self,
                 population_size=50,
                 crossover_mu=0.5,
                 graph_size=800,
                 crossover_rate=1,
                 mutate_rate=0.3,
                 alpha=0,
                 num_mutation_max=1,
                 num_mutation_min=1,
                 n_layers=3,
                 replace_hp=0.01,
                 replace_rate=0.25,
                 property_name='J_score'):
        self.n_layers = n_layers
        # parameters for population
        self.population_size = population_size
        self.init_poplution_file_name = '/home/jeffzhu/aaai_ga/data/randomv2.sdf'

        # parameters for crossover
        self.crossover_mu = crossover_mu
        self.graph_size = graph_size

        # parameters for constants
        self.possible_bonds = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE
        ]
        self.table_of_elements = {
            6: 'C',
            7: 'N',
            8: 'O',
            9: 'F',
            16: 'S',
            17: 'Cl',
            35: 'Br',
            53: 'I',
        }
        self.inverse_table_of_elements = {
            'C': 6,
            'N': 7,
            'O': 8,
            'F': 9,
            'S': 16,
            'Cl': 17,
            'Br': 35,
            'I': 53
        }
        self.vocab_nodes_encode = {
            'C': 1, 'N': 2, 'O': 3, 'S': 3, 'F': 4, 'Cl': 4, 'Br': 4, 'I': 4
        }

        # parameters for mutation
        self.mutation_rate = [0.45, 0.45, 0.1]
        self.temp_elements = {
            0: 'C',
            1: 'N',
            2: 'O',
            3: 'S',
            4: 'Br',
            5: 'Cl',
            6: 'F'
        }
        self.length_elements = len(self.temp_elements)
        self.between_atom_length = 4

        # parameters for trainer
        self.crossover_rate = crossover_rate
        self.alpha = alpha
        self.crossover_sigma = 0.1
        self.full_valence = 5

        # parameters for mutation
        self.mutate_rate = mutate_rate
        self.num_mutation_max = num_mutation_max
        self.num_mutation_min = num_mutation_min
        self.replace_hp = replace_hp
        self.replace_rate = replace_rate

        self.property_name = property_name

        self.init_file_style = 'sdf'
        self.property_name = 'J_score'


class Mutation():
    def __init__(self, config):
        self.config = config

    def _add_bond(self, molecule):
        temp_mol = copy.deepcopy(molecule)

        if temp_mol.num_atom < 2:
            return molecule

        temp_expand_adj = temp_mol.expand_mat
        temp_adj = temp_mol.adj
        mask_row = mask(temp_expand_adj)

        goal_mol = None
        goal_smiles = None

        for i in mask_row:
            row = temp_adj[i]
            for j in range(len(row)):
                if row[j] > 0 and j in mask_row:
                    temp_adj[i][j] += 1
                    temp_adj[j][i] += 1
                    goal_adj = temp_adj
                    goal_node_list = temp_mol.node_list
                    goal_mol = adj2mol(goal_node_list, goal_adj.astype(int), self.config.possible_bonds)
                    goal_smiles = Chem.MolToSmiles(goal_mol)
                    break
            if goal_mol != None:
                break

        if goal_mol != None:
            return Molecule(goal_smiles)
        else:
            return molecule

    def _add_atom(self, molecule):
        temp_mol = copy.deepcopy(molecule)

        temp_node_list = copy.deepcopy(temp_mol.node_list)
        temp_adj = copy.deepcopy(temp_mol.adj)
        temp_expand_adj = copy.deepcopy(temp_mol.expand_mat)

        temp_elements = self.config.temp_elements

        atom_index = np.random.choice(self.config.length_elements, 1)[
            0]  # , p=[0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])[0]
        atom = temp_elements[atom_index]
        mask_row = mask(temp_expand_adj)
        if len(mask_row) < 1:
            return molecule
        mask_index = np.random.choice(mask_row, 1)[0]

        goal_length = len(temp_node_list) + 1
        goal_adj = np.zeros([goal_length, goal_length])
        goal_adj[:goal_length - 1, :goal_length - 1] = temp_adj
        goal_adj[goal_length - 1, mask_index] = goal_adj[mask_index, goal_length - 1] = 1

        temp_node_list.append(atom)
        goal_node_list = temp_node_list

        goal_mol = adj2mol(goal_node_list, goal_adj.astype(int), self.config.possible_bonds)
        goal_smiles = Chem.MolToSmiles(goal_mol)

        return Molecule(goal_smiles)

    def _remove_atom(self, molecule):
        mol = Chem.RWMol(molecule.mol)
        atom_to_remove = choice(mol.GetAtoms())
        mol.RemoveAtom(atom_to_remove.GetIdx())
        goal_smiles = Chem.MolToSmiles(mol)
        return Molecule(goal_smiles)

    def _add_atom_between_bond(self, molecule):
        temp_mol = copy.deepcopy(molecule)

        temp_elements = self.config.temp_elements
        atom_index = np.random.choice(4, 1)[0]  # , p=[0.7, 0.1, 0.1, 0.1])[0]
        atom = temp_elements[atom_index]

        temp_adj = temp_mol.adj

        length = temp_mol.num_atom
        insert_index1 = np.random.choice(length, 1)
        insert_row = temp_adj[insert_index1][0]

        insert_index2 = 0
        for i in range(len(insert_row)):
            if insert_row[i] > 0:
                insert_index2 = i

        temp_adj[insert_index1, insert_index2] = temp_adj[insert_index2, insert_index1] = 0

        goal_adj = np.zeros([length + 1, length + 1])
        goal_adj[:length, :length] = temp_adj
        goal_adj[length, insert_index1] = goal_adj[insert_index1, length] = 1
        goal_adj[insert_index2, length] = goal_adj[length, insert_index2] = 1

        temp_node_list = temp_mol.node_list
        temp_node_list.append(atom)
        goal_node_list = temp_node_list

        goal_mol = adj2mol(goal_node_list, goal_adj.astype(int), self.config.possible_bonds)
        goal_smiles = Chem.MolToSmiles(goal_mol)

        return Molecule(goal_smiles)

    def _change_atom(self, molecule):
        temp_mol = copy.deepcopy(molecule)
        temp_node_list = temp_mol.node_list
        temp_adj = temp_mol.adj

        length_molecule = temp_mol.num_atom

        sum_row = np.sum(temp_adj, axis=0)
        sorted_index = np.argsort(sum_row)

        flag = False
        for idx in range(int(length_molecule * 0.3)):
            if flag == True:
                break
            now_index = sorted_index[idx]
            original_atom_type = temp_node_list[now_index]
            bond_value = sum_row[now_index]

            for k in range(5):
                atom_index = np.random.randint(0, 7)
                atom_type = self.config.temp_elements[atom_index]
                if atom_type != original_atom_type and self.config.vocab_nodes_encode[atom_type] + bond_value <= 5:
                    flag = True
                    temp_node_list[now_index] = self.config.inverse_table_of_elements[atom_type]
                    break
        goal_mol = adj2mol(temp_node_list, temp_adj.astype(int), self.config.possible_bonds)
        goal_smiles = Chem.MolToSmiles(goal_mol)
        return Molecule(goal_smiles)

    def mutate(self, mol):
        molecule = Molecule(mol)
        num_mutation_max, num_mutation_min = self.config.num_mutation_max, self.config.num_mutation_min
        num_mutation = \
            np.random.choice([number for number in range(num_mutation_min, num_mutation_max + 1)], 1, replace=False)[0]

        for iteration in range(num_mutation):
            # choice = np.random.choice(5, 1, p=[0.2, 0.2, 0.1, 0.3, 0.2])[0]
            choice = np.random.choice(4, 1, p=[0.3, 0.3, 0.1, 0.3])[0]
            if choice == 0:
                molecule = self._add_atom(molecule)
            elif choice == 1:
                molecule = self._add_atom_between_bond(molecule)
            elif choice == 2:
                molecule = self._add_bond(molecule)
            elif choice == 3:
                molecule = self._change_atom(molecule)
            # elif choice == 4:
            # molecule = self._remove_atom(molecule)
        return molecule


if __name__ == '__main__':
    config = Configuration()
    mutate_op = Mutation(config)
    smiles = 'CCCCCCCCCCCCCCCCC'
    # mol = Molecule(smiles)
    print(mutate_op.mutate(smiles).smiles)
