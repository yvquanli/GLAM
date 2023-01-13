import random
import os
import time
import hashlib
import torch
import numpy as np
import random
from itertools import compress
from collections import defaultdict
try:
    from rdkit.Chem.Scaffolds import MurckoScaffold
except:
    MurckoScaffold = None
    print('Please install rdkit for data processing')


class Option(object):
    def __init__(self, d):
        self.__dict__ = d


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


def split(scaffolds_dict, labels, weights, GLAM_size, random_seed=0):
    # bbbp,bace,hiv:task=1
    count = 0
    minor_count = 0
    minor_class = np.argmax(weights[0])  # weights are inverse of the ratio
    minor_ratio = 1 / weights[0][minor_class]
    if minor_class == 0:
        minor_class = -1
    optimal_count = 0.1 * len(labels)
    while (count < optimal_count * 0.9 or count > optimal_count * 1.1) \
            or (minor_count < minor_ratio * optimal_count * 0.9 or minor_count > minor_ratio * optimal_count * 1.1):
        random_seed += 1
        random.seed(random_seed)
        scaffold = random.GLAM(list(scaffolds_dict.keys()), GLAM_size)
        count = sum([len(scaffolds_dict[s]) for s in scaffold])
        index = [index for s in scaffold for index in scaffolds_dict[s]]
        minor_count = (labels[index, 0] == minor_class).sum()
    return scaffold, index


def scaffold_split_fp(dataset, smiles_list, raw_labels, random_seed=8, null_value=-1, task_idx=None):
    """
    Adapted from Attentive FP
    """
    if task_idx != None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[:, task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    # smiles_list, raw_labels = df['smiles'].values, df['labels'].values
    labels = np.array(raw_labels)
    if labels.ndim == 1:
        labels = labels[:, np.newaxis]
    pos = (labels == 1).sum(0)
    neg = (labels == -1).sum(0)
    all_sum = pos + neg
    neg_weight = all_sum / neg
    pos_weight = all_sum / pos
    weights = [[neg_weight[i], pos_weight[i]] for i in range(len(all_sum))]
    print('weights', weights)

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds_dict = {}
    for i, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds_dict:
            all_scaffolds_dict[scaffold] = [i]
        else:
            all_scaffolds_dict[scaffold].append(i)
    print("++" * 30)
    GLAMs_size = int(len(all_scaffolds_dict.keys()) * 0.1)
    test_scaffold, test_index = split(all_scaffolds_dict, labels, weights, GLAMs_size,
                                      random_seed=random_seed)
    training_scaffolds_dict = {
        x: all_scaffolds_dict[x] for x in all_scaffolds_dict.keys() if x not in test_scaffold}
    valid_scaffold, valid_index = split(training_scaffolds_dict, labels, weights, GLAMs_size,
                                        random_seed=random_seed)

    training_scaffolds_dict = {x: training_scaffolds_dict[x] for x in training_scaffolds_dict.keys() if
                               x not in valid_scaffold}
    train_index = []
    for ele in training_scaffolds_dict.values():
        train_index += ele
    assert len(train_index) + len(valid_index) + \
        len(test_index) == len(smiles_list)
    train_dataset = dataset[torch.tensor(train_index)]
    valid_dataset = dataset[torch.tensor(valid_index)]
    test_dataset = dataset[torch.tensor(test_index)]
    return train_dataset, valid_dataset, test_dataset


def random_scaffold_split(dataset, smiles_list, task_idx=None, null_value=0,
                          frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0):
    """
    Adapted from https://github.com/pfnet-research/chainer-chemistry/blob/master/chainer_chemistry/dataset/splitters/scaffold_splitter.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
    examples with null value in specified task column of the data.y tensor
    prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in data.y to filter if
    task_idx is provided
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param seed;
    :return: train, valid, test slices of the input dataset obj
    """

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx != None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[:, task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    

    scaffolds = defaultdict(list)
    for ind, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(ind)

    # rng = np.random.RandomState(seed)
    # scaffold_sets = rng.permutation(list(scaffolds.values()))  new version of numpy will raise error
    scaffold_sets = list(scaffolds.values())
    np.random.shuffle(scaffold_sets)

    n_total_valid = int(np.floor(frac_valid * len(dataset)))
    n_total_test = int(np.floor(frac_test * len(dataset)))

    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    return train_dataset, valid_dataset, test_dataset





class GPUManager():
    # copy from https://raw.githubusercontent.com/wnm1503303791/Multi_GPU_Runner/master/src/manager_torch.py

    def __init__(self, qargs=[]):
        self.qargs = qargs
        self.gpus = self.query_gpu(qargs)
        self.gpu_num = len(self.gpus)

    def _sort_by_memory(self, gpus, by_size=False):
        if by_size:
            print('Sorted by free memory size')
            return sorted(gpus, key=lambda d: d['memory.free'], reverse=True)
        else:
            print('Sorted by free memory rate')
            return sorted(gpus, key=lambda d: float(d['memory.free']) / d['memory.total'], reverse=True)

    def auto_choice(self, thre):
        for old_infos, new_infos in zip(self.gpus, self.query_gpu(self.qargs)):
            old_infos.update(new_infos)
        print('Choosing the GPU device has largest free memory...')
        chosen_gpu = self._sort_by_memory(self.gpus, True)[0]
        if float(chosen_gpu['memory.free']) / chosen_gpu['memory.total'] < thre:
            return None

        index = chosen_gpu['index']
        print('Using GPU {i}:\n{info}'.format(i=index, info='\n'.join(
            [str(k) + ':' + str(v) for k, v in chosen_gpu.items()])))
        return int(index)

    def wait_free_gpu(self, thre=0.7):
        if not torch.cuda.is_available():
            return -1
        while self.auto_choice(thre) is None:  # waiting for free gpu
            print('Keep Looking @ {}'.format(time.asctime(time.localtime(time.time()))))
            time.sleep(30)
        return self.auto_choice(thre)

    @staticmethod
    def parse(line, qargs):
        numberic_args = ['memory.free', 'memory.total',
                         'power.draw', 'power.limit']  # keys

        # lambda function to check power management support
        def power_manage_enable(v): return (not 'Not Support' in v)

        def to_numberic(v): return float(v.upper().strip().replace(
            'MIB', '').replace('W', ''))  # remove the unit
        def process(k, v): return (
            (int(to_numberic(v)) if power_manage_enable(v) else 1) if k in numberic_args else v.strip())
        return {k: process(k, v) for k, v in zip(qargs, line.strip().split(','))}

    def query_gpu(self, qargs=[]):
        qargs = ['index', 'gpu_name', 'memory.free', 'memory.total'] + qargs
        cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(
            ','.join(qargs))
        results = os.popen(cmd).readlines()
        return [self.parse(line, qargs) for line in results]


def md5(s: str):
    return hashlib.md5(s.encode('utf-8')).hexdigest()[-5:]
