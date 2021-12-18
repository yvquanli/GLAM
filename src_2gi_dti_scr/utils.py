import random
import os
import time
import hashlib
import torch
import pandas as pd
import numpy as np
import pathlib
from torch_geometric.utils import remove_self_loops
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, precision_recall_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)


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


class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):
        # https://discuss.pytorch.org/t/focal-loss-for-imbalanced-multi-class-classification-in-pytorch/61289
        # important to add reduction='none' to keep per-batch-item loss
        # alpha=0.25, gamma=2 from https://www.cnblogs.com/qi-yuan-008/p/11992156.html
        ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


def get_loss(loss_str):
    d = {
        'mse': torch.nn.MSELoss(),
        'mae': torch.nn.L1Loss(),
        'huber': torch.nn.SmoothL1Loss(),
        'smae': torch.nn.SmoothL1Loss(),
        'bce': torch.nn.BCELoss(),
        'bcel': torch.nn.BCEWithLogitsLoss(),
        'kl': torch.nn.KLDivLoss(),
        'hinge': torch.nn.HingeEmbeddingLoss(),
        'nll': torch.nn.NLLLoss(),
        'ce': torch.nn.CrossEntropyLoss(),
        'focal': FocalLoss(alpha=0.25),
    }
    loss_will_set_in_trainer = ['wce']
    if loss_str not in d.keys():
        if loss_str in loss_will_set_in_trainer: return None
        raise ValueError('Error loss: {}!'.format(loss_str))
    else:
        return d[loss_str]


def binary_metrics(y_true, y_score, y_pred=None, threshod=0.5):
    auc = roc_auc_score(y_true, y_score)
    if y_pred is None: y_pred = (y_score >= threshod).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prauc = metrics.auc(precision_recall_curve(y_true, y_score)[1], precision_recall_curve(y_true, y_score)[0])
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    d = {'auc': auc, 'prauc': prauc, 'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1}
    return d


def cal_ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci


def regression_metrics(y_true, y_pred):
    # mae = mean_absolute_error(y_true, y_pred)
    ci = cal_ci(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)
    d = {'ci': ci, 'mse': mse, 'rmse': rmse, 'r2': r2}
    return d


def bedroc_score(y_true, y_score, decreasing=True, alpha=20.0):
    # https://github.com/lewisacidic/scikit-chem/blob/master/skchem/metrics.py
    big_n = len(y_true)
    n = sum(y_true == 1)
    if decreasing:
        order = np.argsort(-y_score)
    else:
        order = np.argsort(y_score)

    m_rank = (y_true[order] == 1).nonzero()[0] + 1
    s = np.sum(np.exp(-alpha * m_rank / big_n))
    r_a = n / big_n
    rand_sum = r_a * (1 - np.exp(-alpha)) / (np.exp(alpha / big_n) - 1)
    fac = r_a * np.sinh(alpha / 2) / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * r_a))
    cte = 1 / (1 - np.exp(alpha * (1 - r_a)))
    return s * fac / rand_sum + cte


def enrichment_factor_single(y_true, y_score, threshold=0.005):
    # https://github.com/gitter-lab/pria_lifechem/blob/1fd892505a/pria_lifechem/evaluation.py
    labels_arr, scores_arr, percentile = y_true, y_score, threshold
    non_missing_indices = np.argwhere(labels_arr != -1)[:, 0]
    labels_arr = labels_arr[non_missing_indices]
    scores_arr = scores_arr[non_missing_indices]
    GLAM_size = int(labels_arr.shape[0] * percentile)  # determine number mols in subset
    indices = np.argsort(scores_arr, axis=0)[::-1][:GLAM_size]  # get the index positions for these in library
    n_actives = np.nansum(labels_arr)  # count number of positive labels in library
    n_experimental = np.nansum(labels_arr[indices])  # count number of positive labels in subset

    if n_actives > 0.0:
        ef = float(n_experimental) / n_actives / percentile  # calc EF at percentile
    else:
        raise Exception('n actives == 0')
    # return n_actives, ef, ef_max
    return ef


def screening_metrics(y_true, y_score, y_pred=None, threshod=0.5):
    auc = roc_auc_score(y_true, y_score)
    if y_pred is None: y_pred = (y_score > threshod).astype(int)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    bedroc = bedroc_score(y_true, y_score)
    ef_001 = enrichment_factor_single(y_true, y_score, 0.001)
    ef_005 = enrichment_factor_single(y_true, y_score, 0.005)
    ef_01 = enrichment_factor_single(y_true, y_score, 0.01)
    ef_02 = enrichment_factor_single(y_true, y_score, 0.02)
    ef_05 = enrichment_factor_single(y_true, y_score, 0.05)
    d = {'auc': auc, 'acc': acc, 'precision': precision, 'recall': recall, 'bedroc': bedroc, 'ef_001': ef_001,
         'ef_005': ef_005, 'ef_01': ef_01, 'ef_02': ef_02, 'ef_05': ef_05, }
    return d


def angle(vector1, vector2):
    cos_angle = vector1.dot(vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(cos_angle)
    # angle2=angle*360/2/np.pi
    return angle  # , angle2


def area_triangle(vector1, vector2):
    trianglearea = 0.5 * np.linalg.norm(np.cross(vector1, vector2))
    return trianglearea


def area_triangle_vertex(vertex1, vertex2, vertex3):
    trianglearea = 0.5 * np.linalg.norm(np.cross(vertex2 - vertex1, vertex3 - vertex1))
    return trianglearea


def cal_angle_area(vector1, vector2):
    return angle(vector1, vector2), area_triangle(vector1, vector2)


def cal_dist(vertex1, vertex2, ord=2):
    return np.linalg.norm(vertex1 - vertex2, ord=ord)


'''
    Usages: 
    vij=np.array([ 0, 1,  1])
    vik=np.array([ 0, 2,  0])
    cal_angle_area(vij, vik)   # (0.7853981633974484, 1.0)
    vertex1 = np.array([1,2,3])
    vertex2 = np.array([4,5,6])
    cal_dist(vertex1, vertex2, ord=1), np.sum(vertex1-vertex2) # (9.0, -9)
    cal_dist(vertex1, vertex2, ord=2), np.sqrt(np.sum(np.square(vertex1-vertex2)))  # (5.19, 5.19)
    cal_dist(vertex1, vertex2, ord=3)  # 4.3267487109222245
'''


def read_probs(path, mean_prob=False):
    # fh = open(filename, 'r')
    # content = [line.strip() for line in list(fh)]
    # fh.close()
    # print(content)
    with open(path, 'r') as f:
        content = f.readlines()

    assert len(content) >= 5  # '1. the input file contains fewer than 5 lines'

    seq = ""
    infos = {}
    probs = []

    for line in content:
        # print(line)
        if 'SEQ' in line:
            seq += line.split()[-1]
            continue
        if line.startswith('PFRMAT') or line.startswith('TARGET') or line.startswith('AUTHOR') or \
                line.startswith('METHOD') or line.startswith('RMODE') or line.startswith('RMODE') or \
                line.startswith('MODEL') or line.startswith('REMARK') or line.startswith('END'):
            infos[line.split()[0]] = line.split()[1:]
            continue

        columns = line.split()

        if len(columns) >= 3:
            indices = [int(x) for x in columns[0:2]]
            prob = np.float32(columns[2])
            # if mean_prob:
            #     prob = np.mean([float(x) for x in columns[-10:-1:2]])  # todo: need to check when using

            assert 0 <= prob <= 1  # 'The contact prob shall be between 0 and 1: '
            # assert 0 < c < 20  # 'The distance shall be between 0 and 20: '
            assert indices[0] < indices[1]  # 'The first index in a residue pair shall be smaller than the 2nd one:'

            if indices[0] < 1 or indices[0] > len(seq) or indices[1] < 1 or indices[1] > len(seq):
                print('The residue index in the following line is out of range: \n', line)
                return None
            probs.append(indices + [prob])
        else:
            print('The following line in the input file has an incorrect format: ')
            print(line)
            return None
    return probs, seq, infos


def load_contactmap(path, thre=0.1):
    # 0.1  thre to keep 2988/30000 prob data of a 894 AAs protein
    # 0.05 thre to keep 4700/30000 prob data
    # 0.3  thre to keep 1505/30000 prob data
    probs, seq, infos = read_probs(path)
    contactmap = np.zeros((len(seq), len(seq)), dtype=np.float32)
    for p in probs:
        if p[2] >= thre:
            contactmap[p[0] - 1, p[1] - 1] = p[2]
            contactmap[p[1] - 1, p[0] - 1] = p[2]
    return contactmap, seq, infos


# nomarlize
def dic_normalize(dic):
    # print(dic)
    keys = list(dic.keys())
    values = np.array(list(dic.values()))
    values = (values - values.mean()) / values.std()  # norm
    new_dic = {}
    for i, key in enumerate(keys):
        new_dic[key] = values[i].tolist()
    return new_dic


def dic_normalize_2d(dic):
    # print(dic)
    keys = list(dic.keys())
    values = np.array(list(dic.values()))
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    values = (values - mean) / std  # norm
    new_dic = {}
    for i, key in enumerate(keys):
        new_dic[key] = values[i, :].tolist()
    return new_dic


# nomarlize
def dic_normalize_min_max(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        pass
        # raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


res_type_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', ]
# 'X' for unkown?
res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
res_aromatic_table = ['F', 'W', 'Y']

res_non_polar_table = ['A', 'G', 'P', 'V', 'L', 'I', 'M']
res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
res_acidic_charged_table = ['D', 'E']  # res_polar_negatively_table = ['D', 'E']
res_basic_charged_table = ['H', 'K', 'R']  # res_polar_positively_table = ['R', 'K', 'H']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}
res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}
res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}
res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}
res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}
res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}
res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}
meiler_feature_table = {
    'A': [1.28, 0.05, 1.00, 0.31, 6.11, 0.42, 0.23],
    'C': [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41],
    'D': [1.60, 0.11, 2.78, -0.77, 2.95, 0.25, 0.20],
    'E': [1.56, 0.15, 3.78, -0.64, 3.09, 0.42, 0.21],
    'F': [2.94, 0.29, 5.89, 1.79, 5.67, 0.30, 0.38],
    'G': [0.00, 0.00, 0.00, 0.00, 6.07, 0.13, 0.15],
    'H': [2.99, 0.23, 4.66, 0.13, 7.69, 0.27, 0.30],
    'I': [4.19, 0.19, 4.00, 1.80, 6.04, 0.30, 0.45],
    'K': [1.89, 0.22, 4.77, -0.99, 9.99, 0.32, 0.27],
    'L': [2.59, 0.19, 4.00, 1.70, 6.04, 0.39, 0.31],
    'M': [2.35, 0.22, 4.43, 1.23, 5.71, 0.38, 0.32],
    'N': [1.60, 0.13, 2.95, -0.60, 6.52, 0.21, 0.22],
    'P': [2.67, 0.00, 2.72, 0.72, 6.80, 0.13, 0.34],
    'Q': [1.56, 0.18, 3.95, -0.22, 5.65, 0.36, 0.25],
    'R': [2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25],
    'S': [1.31, 0.06, 1.60, -0.04, 5.70, 0.20, 0.28],
    'T': [3.03, 0.11, 2.60, 0.26, 5.60, 0.21, 0.36],
    'V': [3.67, 0.14, 3.00, 1.22, 6.02, 0.27, 0.49],
    'W': [3.21, 0.41, 8.08, 2.25, 5.94, 0.32, 0.42],
    'Y': [2.94, 0.30, 6.47, 0.96, 5.66, 0.25, 0.41],
    # 'PTR': [2.94, 0.30, 6.47, 0.96, 5.66, 0.25, 0.41],
    # 'TPO': [3.03, 0.11, 2.60, 0.26, 5.60, 0.21, 0.36],
    # 'SEP': [1.31, 0.06, 1.60, -0.04, 5.70, 0.20, 0.28],
    # 'KCX': [1.89, 0.22, 4.77, -0.99, 9.99, 0.32, 0.27],
    # 'LLP': [1.89, 0.22, 4.77, -0.99, 9.99, 0.32, 0.27],
    # 'PCA': [1.56, 0.15, 3.78, -0.64, 3.09, 0.42, 0.21],
    # 'MSE': [2.35, 0.22, 4.43, 1.23, 5.71, 0.38, 0.32],
    # 'CSO': [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41],
    # 'CAS': [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41],
    # 'CAF': [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41],
    # 'CSD': [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41],
    # 'UNKNOWN': [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
}
kidera_feature_table = {
    'A': [-1.56, -1.67, -0.97, -0.27, -0.93, -0.78, -0.2, -0.08, 0.21, -0.48],
    'C': [0.12, -0.89, 0.45, -1.05, -0.71, 2.41, 1.52, -0.69, 1.13, 1.1],
    'E': [-1.45, 0.19, -1.61, 1.17, -1.31, 0.4, 0.04, 0.38, -0.35, -0.12],
    'D': [0.58, -0.22, -1.58, 0.81, -0.92, 0.15, -1.52, 0.47, 0.76, 0.7],
    'G': [1.46, -1.96, -0.23, -0.16, 0.1, -0.11, 1.32, 2.36, -1.66, 0.46],
    'F': [-0.21, 0.98, -0.36, -1.43, 0.22, -0.81, 0.67, 1.1, 1.71, -0.44],
    'I': [-0.73, -0.16, 1.79, -0.77, -0.54, 0.03, -0.83, 0.51, 0.66, -1.78],
    'H': [-0.41, 0.52, -0.28, 0.28, 1.61, 1.01, -1.85, 0.47, 1.13, 1.63],
    'K': [-0.34, 0.82, -0.23, 1.7, 1.54, -1.62, 1.15, -0.08, -0.48, 0.6],
    'M': [-1.4, 0.18, -0.42, -0.73, 2.0, 1.52, 0.26, 0.11, -1.27, 0.27],
    'L': [-1.04, 0.0, -0.24, -1.1, -0.55, -2.05, 0.96, -0.76, 0.45, 0.93],
    'N': [1.14, -0.07, -0.12, 0.81, 0.18, 0.37, -0.09, 1.23, 1.1, -1.73],
    'Q': [-0.47, 0.24, 0.07, 1.1, 1.1, 0.59, 0.84, -0.71, -0.03, -2.33],
    'P': [2.06, -0.33, -1.15, -0.75, 0.88, -0.45, 0.3, -2.3, 0.74, -0.28],
    'S': [0.81, -1.08, 0.16, 0.42, -0.21, -0.43, -1.89, -1.15, -0.97, -0.23],
    'R': [0.22, 1.27, 1.37, 1.87, -1.7, 0.46, 0.92, -0.39, 0.23, 0.93],
    'T': [0.26, -0.7, 1.21, 0.63, -0.1, 0.21, 0.24, -1.15, -0.56, 0.19],
    'W': [0.3, 2.1, -0.72, -1.57, -1.16, 0.57, -0.48, -0.4, -2.3, -0.6],
    'V': [-0.74, -0.71, 2.04, -0.4, 0.5, -0.81, -1.07, 0.06, -0.46, 0.65],
    'Y': [1.38, 1.48, 0.8, -0.56, -0.0, -0.68, -0.31, 1.03, -0.05, 0.53],
    # 'UNKNOWN': [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
}


# res_weight_table = dic_normalize(res_weight_table)  # min-max norm???
# res_pka_table = dic_normalize(res_pka_table)
# res_pkb_table = dic_normalize(res_pkb_table)
# res_pkx_table = dic_normalize(res_pkx_table)
# res_pl_table = dic_normalize(res_pl_table)
# res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
# res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)
# meiler_feature_table = dic_normalize_2d(meiler_feature_table)
# kidera_feature_table = dic_normalize_2d(kidera_feature_table)  # kidera_feature had been normed

def get_residue_features(residue):
    res_type = one_of_k_encoding(residue, res_type_table)
    res_type = [int(x) for x in res_type]
    res_property1 = [1 if residue in res_aliphatic_table else 0, 1 if residue in res_aromatic_table else 0,
                     1 if residue in res_polar_neutral_table else 0, 1 if residue in res_acidic_charged_table else 0,
                     1 if residue in res_basic_charged_table else 0, ]
    res_property2 = [res_weight_table[residue], res_pka_table[residue],
                     res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue],
                     res_hydrophobic_ph7_table[residue], ]
    res_property3 = meiler_feature_table[residue] + kidera_feature_table[residue]
    return res_type + res_property1 + res_property2 + res_property3


def model_args(args):
    _other_args_name = ['dataset_root', 'dataset', 'seed', 'gpu', 'note', 'batch_size', 'epochs', 'loss', 'optim', 'k',
                        'lr', 'lr_reduce_rate', 'lr_reduce_patience', 'early_stop_patience', 'verbose_patience']
    model_args_dict = {}
    for k, v in args.__dict__.items():
        if k not in _other_args_name:
            model_args_dict[k] = v
    return model_args_dict


def auto_dataset(args):
    from dataset import BindingDBProMolInteactionDataset, LIT_PCBA

    if args.dataset == 'bindingdb_c':
        from trainer import TrainerBinaryClassification as Trainer
        args.out_dim = 2
        _dataset = BindingDBProMolInteactionDataset(args.dataset_root)
    elif args.dataset in ['ALDH1', 'ESR1_ant', 'KAT2A', 'MAPK1', 'FEN1']:
        from trainer import TrainerScreening as Trainer
        args.out_dim = 2
        _dataset = LIT_PCBA(args.dataset_root, target=args.dataset)
    else:
        raise Exception('error dataset input')
    return args, _dataset, Trainer


class GPUManager():
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
        if not torch.cuda.is_available(): return -1
        while self.auto_choice(thre) is None:  # waiting for free gpu
            print('Keep Looking @ {}'.format(time.asctime(time.localtime(time.time()))))
            time.sleep(30)
        return self.auto_choice(thre)

    @staticmethod
    def parse(line, qargs):
        numberic_args = ['memory.free', 'memory.total', 'power.draw', 'power.limit']  # keys
        power_manage_enable = lambda v: (not 'Not Support' in v)  # lambda function to check power management support
        to_numberic = lambda v: float(v.upper().strip().replace('MIB', '').replace('W', ''))  # remove the unit
        process = lambda k, v: (
            (int(to_numberic(v)) if power_manage_enable(v) else 1) if k in numberic_args else v.strip())
        return {k: process(k, v) for k, v in zip(qargs, line.strip().split(','))}

    def query_gpu(self, qargs=[]):
        qargs = ['index', 'gpu_name', 'memory.free', 'memory.total'] + qargs
        cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
        results = os.popen(cmd).readlines()
        return [self.parse(line, qargs) for line in results]


def md5(s: str):
    return hashlib.md5(s.encode('utf-8')).hexdigest()[-5:]


def print_red(s: str):
    print("\033[1;31m{}\033[0m".format(s))


def read_logs(logs_dir: pathlib.PosixPath):
    logs = []
    for p in logs_dir.glob('./*seed*'):
        log_path = p / 'log.txt'
        with open(log_path) as f:
            # read a single logfile to lines, and skip the log files without test
            lines = f.readlines()
            if not lines[-1].startswith('{'): continue

            # read a sigle logfile to config from -2 line, and short the words for better visual experience
            str_config_dict = lines[-2].replace('\n', '').strip().replace('mol_', 'm').replace('pro_', 'p') \
                .replace('depth', 'd').replace('graph_res', 'res').replace('batch_size', 'bs') \
                .replace('_TripletMessage', 'Trim').replace('_NNConv', 'NN').replace('_GCNConv', 'GCN') \
                .replace('_GATConv', 'GAT').replace('hid_dim_alpha', 'a').replace('message_steps', 'step') \
                .replace('Dropout(', '(').replace('Global', '').replace('_norm', 'n') \
                .replace('_LayerNorm', 'LN').replace('_BatchNorm', 'BN').replace('_PairNorm', 'PN') \
                .replace('more_epochs_run', 'mer').replace('_None', '0') \
                .replace('LeakyReLU', 'LReLU')
            config_for_print = eval(str_config_dict)
            for item in ['dataset_root', 'seed', 'gpu', 'verbose_patience', 'out_dim',
                         'early_stop_patience', 'lr_reduce_rate', 'lr_reduce_patience']:
                del config_for_print[item]

            # read a single logfile to loss, test information, valid information.
            loss_info, test_info, valid_info = lines[-1].replace('\n', '').strip().split('|')

            log = {'id': p.name}
            log.update(eval(loss_info))
            log.update(eval(test_info))
            log.update(eval(valid_info))
            log.update(config_for_print)
            log.update({'config': lines[-2]})
            logs.append(log)
    return logs


def summarize_logs(logs_dir: pathlib.PosixPath, metrics: list):
    logs = read_logs(logs_dir)
    if len(logs) >= 1:
        # group, sort, and print the logs
        logs_pd = pd.DataFrame(logs).sort_values(metrics[0], ascending=False)
        logs_summary = []
        for note, df in logs_pd.groupby('note'):
            d = {'id(note)': note, 'n_GLAM': len(df), 'dataset': df['dataset'].iloc[0],
                 'config': df['config'].iloc[0]}
            for m in metrics:
                array = df[m].astype(float)
                for opt in ['mean', 'min', 'max', 'std']:
                    d[opt + m] = eval('array.{}()'.format(opt))
            d.update({})
            logs_summary.append(d)
        logs_summary = pd.DataFrame(logs_summary).sort_values('mean' + metrics[0], ascending=False)
        save_path = str(logs_dir / 'logs_summary.csv')
        print_red('Search Result Info, more info and config can be found in {}'.format(save_path))
        print(logs_summary.drop(labels=['config'], axis=1))  # print info without config
        logs_summary.to_csv(save_path)

        # search results details in groups
        search_result = []
        groups = logs_pd.groupby('note')
        for note in logs_summary.loc[:, 'id(note)']:
            group = groups.get_group(note).sort_values(metrics[0], ascending=False)
            search_result.append(group)
        search_result = pd.concat(search_result)
        save_path = str(logs_dir / 'search_result.csv')
        print_red('Detailed Search Result Info,more info and config can be found in {}'.format(save_path))
        print(search_result.drop(labels=['config'], axis=1))  # print info without config
        search_result.to_csv(save_path)
        return logs_summary


def print_ongoing_info(logs_dir: pathlib.PosixPath):
    for p in logs_dir.glob('./*seed*'):
        with open(p / 'log.txt') as f:
            lines = f.readlines()
            if lines[-1].startswith('{'): continue
            lines.reverse()
            for i, line in enumerate(lines):
                if 'Model saved at epoch' in line:
                    print(p, '----------->', lines[i + 1], end='')
                    break


def auto_metrics(dataset: str):
    metrics = ['valauc', 'auc']
    # if dataset.startswith('lit_'): metrics = ['valbedroc', 'bedroc']
    return metrics


def auto_summarize_logs(dataset: str, ongoing=False):
    logs_dir = pathlib.Path('./log_{}/'.format(dataset))
    if not logs_dir.exists():
        return None
    print('\n\n', '#' * 30, dataset, '#' * 30)
    results = summarize_logs(logs_dir=logs_dir, metrics=auto_metrics(dataset))
    print_red('Ongoing task details')
    if ongoing: print_ongoing_info(logs_dir=logs_dir)
    return results


def config2cmd(config: dict):
    _ = ' '.join(['--' + k + ' ' + str(v) for k, v in config.items()])
    cmd = 'python3 run.py {}'.format(_)
    cmd = cmd.replace('(', '\(').replace(')', '\)')  # for shell run
    print(cmd)
    return cmd


if __name__ == '__main__':
    # dataset = 'bindingdb_c'
    # results = auto_summarize_logs(dataset, ongoing=True)
    # evaluate_best_configs(results, n_configs=3)
    # from trainer import Inferencer
    # inf = Inferencer(dataset, n_blend=3)
    # inf.blend_and_inference()

    datasets = ['bindingdb_c', 'ADRB2', 'ALDH1', 'ESR1_ago', 'ESR1_ant', 'FEN1', 'GBA', 'IDH1',
                'KAT2A', 'MAPK1', 'MTORC1', 'OPRK1', 'PKM2', 'PPARG', 'TP53', 'VDR']
    for dataset in datasets:
        results = auto_summarize_logs(dataset, ongoing=True)
