import random
import os
import time
import hashlib
import torch
import pandas as pd
import numpy as np
import pathlib
from torch_geometric.utils import remove_self_loops
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
import random
from itertools import compress
from collections import defaultdict

try:
    from rdkit.Chem.Scaffolds import MurckoScaffold
except:
    MurckoScaffold = None
    print('Please install rdkit for data processing')

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


class MultiTargetCrossEntropy(torch.nn.Module):
    def __init__(self, C_dim=2):
        super(MultiTargetCrossEntropy, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=C_dim)
        self.nll_loss = torch.nn.NLLLoss()

    def forward(self, input, target):
        # input = input.view(target.shape[0], self.C, target.shape[1])
        '''
            input: shoud be `(N, T, C)` where `C = number of classes` and `T = number of targets`
            target: should be `(N, T)` where `T = number of targets`
        '''
        assert input.shape[0] == target.shape[0]
        assert input.shape[1] == target.shape[1]
        out = self.log_softmax(input)
        out = self.nll_loss(out, target)
        return out


def get_loss(loss_str):
    d = {
        'mse': torch.nn.MSELoss(),
        'mae': torch.nn.L1Loss(),
        'huber': torch.nn.SmoothL1Loss(),
        'smae': torch.nn.SmoothL1Loss(),
        'bce': torch.nn.BCELoss(),
        'bcen': torch.nn.BCELoss(reduction="none"),
        'bcel': torch.nn.BCEWithLogitsLoss(),
        'bceln': torch.nn.BCEWithLogitsLoss(reduction="none"),
        'mtce': MultiTargetCrossEntropy(),
        'kl': torch.nn.KLDivLoss(),
        'hinge': torch.nn.HingeEmbeddingLoss(),
        'nll': torch.nn.NLLLoss(),
        'ce': torch.nn.CrossEntropyLoss(),
        'focal': FocalLoss(alpha=0.25),
    }
    if loss_str not in d.keys():
        raise ValueError('loss not found')
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


def binary_metrics_multi_target_nan(y_true, y_score, y_pred=None, nan_fill=-1, threshod=0.5):
    '''
       y_true and y_score should be `(N, T)` where N = number of GLAMs, and T = number of targets
    '''
    if y_pred is None: y_pred = (y_score >= threshod).astype(int)
    roc_list, acc_list, prc_list, rec_list = [], [], [], []
    for i in range(y_true.shape[1]):
        if (y_true[:, i] == 1).sum() == 0 or (y_true[:, i] == 0).sum() == 0:
            print('Skipped target, cause AUC is only defined when there is at least one positive data.')
            continue

        if nan_fill == -1:
            is_valid = y_true[:, i] >= 0
            y_true_st = y_true[is_valid, i]
            y_score_st = y_score[is_valid, i]
            y_pred_st = y_pred[is_valid, i]
            roc_list.append(roc_auc_score(y_true_st, y_score_st))
            acc_list.append(accuracy_score(y_true_st, y_pred_st))
            prc_list.append(precision_score(y_true_st, y_pred_st))
            rec_list.append(recall_score(y_true_st, y_pred_st))
    d = {'auc': sum(roc_list) / len(roc_list), 'acc': sum(acc_list) / len(acc_list),
         'precision': sum(prc_list) / len(prc_list), 'recall': sum(rec_list) / len(rec_list)}
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
    if minor_class == 0: minor_class = -1
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

    if labels.ndim == 1: labels = labels[:, np.newaxis]
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
    training_scaffolds_dict = {x: all_scaffolds_dict[x] for x in all_scaffolds_dict.keys() if x not in test_scaffold}
    valid_scaffold, valid_index = split(training_scaffolds_dict, labels, weights, GLAMs_size,
                                        random_seed=random_seed)

    training_scaffolds_dict = {x: training_scaffolds_dict[x] for x in training_scaffolds_dict.keys() if
                               x not in valid_scaffold}
    train_index = []
    for ele in training_scaffolds_dict.values():
        train_index += ele
    assert len(train_index) + len(valid_index) + len(test_index) == len(smiles_list)
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

    rng = np.random.RandomState(seed)

    scaffolds = defaultdict(list)
    for ind, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(ind)

    scaffold_sets = rng.perperturb(list(scaffolds.values()))

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


toxcast_tasks = ['ACEA_T47D_80hr_Negative', 'ACEA_T47D_80hr_Positive', 'APR_HepG2_CellCycleArrest_24h_dn',
                 'APR_HepG2_CellCycleArrest_24h_up', 'APR_HepG2_CellCycleArrest_72h_dn',
                 'APR_HepG2_CellLoss_24h_dn', 'APR_HepG2_CellLoss_72h_dn', 'APR_HepG2_MicrotubuleCSK_24h_dn',
                 'APR_HepG2_MicrotubuleCSK_24h_up', 'APR_HepG2_MicrotubuleCSK_72h_dn',
                 'APR_HepG2_MicrotubuleCSK_72h_up', 'APR_HepG2_MitoMass_24h_dn', 'APR_HepG2_MitoMass_24h_up',
                 'APR_HepG2_MitoMass_72h_dn', 'APR_HepG2_MitoMass_72h_up', 'APR_HepG2_MitoMembPot_1h_dn',
                 'APR_HepG2_MitoMembPot_24h_dn', 'APR_HepG2_MitoMembPot_72h_dn', 'APR_HepG2_MitoticArrest_24h_up',
                 'APR_HepG2_MitoticArrest_72h_up', 'APR_HepG2_NuclearSize_24h_dn', 'APR_HepG2_NuclearSize_72h_dn',
                 'APR_HepG2_NuclearSize_72h_up', 'APR_HepG2_OxidativeStress_24h_up',
                 'APR_HepG2_OxidativeStress_72h_up', 'APR_HepG2_StressKinase_1h_up',
                 'APR_HepG2_StressKinase_24h_up', 'APR_HepG2_StressKinase_72h_up', 'APR_HepG2_p53Act_24h_up',
                 'APR_HepG2_p53Act_72h_up', 'APR_Hepat_Apoptosis_24hr_up', 'APR_Hepat_Apoptosis_48hr_up',
                 'APR_Hepat_CellLoss_24hr_dn', 'APR_Hepat_CellLoss_48hr_dn', 'APR_Hepat_DNADamage_24hr_up',
                 'APR_Hepat_DNADamage_48hr_up', 'APR_Hepat_DNATexture_24hr_up', 'APR_Hepat_DNATexture_48hr_up',
                 'APR_Hepat_MitoFxnI_1hr_dn', 'APR_Hepat_MitoFxnI_24hr_dn', 'APR_Hepat_MitoFxnI_48hr_dn',
                 'APR_Hepat_NuclearSize_24hr_dn', 'APR_Hepat_NuclearSize_48hr_dn', 'APR_Hepat_Steatosis_24hr_up',
                 'APR_Hepat_Steatosis_48hr_up', 'ATG_AP_1_CIS_dn', 'ATG_AP_1_CIS_up', 'ATG_AP_2_CIS_dn',
                 'ATG_AP_2_CIS_up', 'ATG_AR_TRANS_dn', 'ATG_AR_TRANS_up', 'ATG_Ahr_CIS_dn', 'ATG_Ahr_CIS_up',
                 'ATG_BRE_CIS_dn', 'ATG_BRE_CIS_up', 'ATG_CAR_TRANS_dn', 'ATG_CAR_TRANS_up', 'ATG_CMV_CIS_dn',
                 'ATG_CMV_CIS_up', 'ATG_CRE_CIS_dn', 'ATG_CRE_CIS_up', 'ATG_C_EBP_CIS_dn', 'ATG_C_EBP_CIS_up',
                 'ATG_DR4_LXR_CIS_dn', 'ATG_DR4_LXR_CIS_up', 'ATG_DR5_CIS_dn', 'ATG_DR5_CIS_up', 'ATG_E2F_CIS_dn',
                 'ATG_E2F_CIS_up', 'ATG_EGR_CIS_up', 'ATG_ERE_CIS_dn', 'ATG_ERE_CIS_up', 'ATG_ERRa_TRANS_dn',
                 'ATG_ERRg_TRANS_dn', 'ATG_ERRg_TRANS_up', 'ATG_ERa_TRANS_up', 'ATG_E_Box_CIS_dn',
                 'ATG_E_Box_CIS_up', 'ATG_Ets_CIS_dn', 'ATG_Ets_CIS_up', 'ATG_FXR_TRANS_up', 'ATG_FoxA2_CIS_dn',
                 'ATG_FoxA2_CIS_up', 'ATG_FoxO_CIS_dn', 'ATG_FoxO_CIS_up', 'ATG_GAL4_TRANS_dn', 'ATG_GATA_CIS_dn',
                 'ATG_GATA_CIS_up', 'ATG_GLI_CIS_dn', 'ATG_GLI_CIS_up', 'ATG_GRE_CIS_dn', 'ATG_GRE_CIS_up',
                 'ATG_GR_TRANS_dn', 'ATG_GR_TRANS_up', 'ATG_HIF1a_CIS_dn', 'ATG_HIF1a_CIS_up',
                 'ATG_HNF4a_TRANS_dn', 'ATG_HNF4a_TRANS_up', 'ATG_HNF6_CIS_dn', 'ATG_HNF6_CIS_up',
                 'ATG_HSE_CIS_dn', 'ATG_HSE_CIS_up', 'ATG_IR1_CIS_dn', 'ATG_IR1_CIS_up', 'ATG_ISRE_CIS_dn',
                 'ATG_ISRE_CIS_up', 'ATG_LXRa_TRANS_dn', 'ATG_LXRa_TRANS_up', 'ATG_LXRb_TRANS_dn',
                 'ATG_LXRb_TRANS_up', 'ATG_MRE_CIS_up', 'ATG_M_06_TRANS_up', 'ATG_M_19_CIS_dn',
                 'ATG_M_19_TRANS_dn', 'ATG_M_19_TRANS_up', 'ATG_M_32_CIS_dn', 'ATG_M_32_CIS_up',
                 'ATG_M_32_TRANS_dn', 'ATG_M_32_TRANS_up', 'ATG_M_61_TRANS_up', 'ATG_Myb_CIS_dn', 'ATG_Myb_CIS_up',
                 'ATG_Myc_CIS_dn', 'ATG_Myc_CIS_up', 'ATG_NFI_CIS_dn', 'ATG_NFI_CIS_up', 'ATG_NF_kB_CIS_dn',
                 'ATG_NF_kB_CIS_up', 'ATG_NRF1_CIS_dn', 'ATG_NRF1_CIS_up', 'ATG_NRF2_ARE_CIS_dn',
                 'ATG_NRF2_ARE_CIS_up', 'ATG_NURR1_TRANS_dn', 'ATG_NURR1_TRANS_up', 'ATG_Oct_MLP_CIS_dn',
                 'ATG_Oct_MLP_CIS_up', 'ATG_PBREM_CIS_dn', 'ATG_PBREM_CIS_up', 'ATG_PPARa_TRANS_dn',
                 'ATG_PPARa_TRANS_up', 'ATG_PPARd_TRANS_up', 'ATG_PPARg_TRANS_up', 'ATG_PPRE_CIS_dn',
                 'ATG_PPRE_CIS_up', 'ATG_PXRE_CIS_dn', 'ATG_PXRE_CIS_up', 'ATG_PXR_TRANS_dn', 'ATG_PXR_TRANS_up',
                 'ATG_Pax6_CIS_up', 'ATG_RARa_TRANS_dn', 'ATG_RARa_TRANS_up', 'ATG_RARb_TRANS_dn',
                 'ATG_RARb_TRANS_up', 'ATG_RARg_TRANS_dn', 'ATG_RARg_TRANS_up', 'ATG_RORE_CIS_dn',
                 'ATG_RORE_CIS_up', 'ATG_RORb_TRANS_dn', 'ATG_RORg_TRANS_dn', 'ATG_RORg_TRANS_up',
                 'ATG_RXRa_TRANS_dn', 'ATG_RXRa_TRANS_up', 'ATG_RXRb_TRANS_dn', 'ATG_RXRb_TRANS_up',
                 'ATG_SREBP_CIS_dn', 'ATG_SREBP_CIS_up', 'ATG_STAT3_CIS_dn', 'ATG_STAT3_CIS_up', 'ATG_Sox_CIS_dn',
                 'ATG_Sox_CIS_up', 'ATG_Sp1_CIS_dn', 'ATG_Sp1_CIS_up', 'ATG_TAL_CIS_dn', 'ATG_TAL_CIS_up',
                 'ATG_TA_CIS_dn', 'ATG_TA_CIS_up', 'ATG_TCF_b_cat_CIS_dn', 'ATG_TCF_b_cat_CIS_up',
                 'ATG_TGFb_CIS_dn', 'ATG_TGFb_CIS_up', 'ATG_THRa1_TRANS_dn', 'ATG_THRa1_TRANS_up',
                 'ATG_VDRE_CIS_dn', 'ATG_VDRE_CIS_up', 'ATG_VDR_TRANS_dn', 'ATG_VDR_TRANS_up',
                 'ATG_XTT_Cytotoxicity_up', 'ATG_Xbp1_CIS_dn', 'ATG_Xbp1_CIS_up', 'ATG_p53_CIS_dn',
                 'ATG_p53_CIS_up', 'BSK_3C_Eselectin_down', 'BSK_3C_HLADR_down', 'BSK_3C_ICAM1_down',
                 'BSK_3C_IL8_down', 'BSK_3C_MCP1_down', 'BSK_3C_MIG_down', 'BSK_3C_Proliferation_down',
                 'BSK_3C_SRB_down', 'BSK_3C_Thrombomodulin_down', 'BSK_3C_Thrombomodulin_up',
                 'BSK_3C_TissueFactor_down', 'BSK_3C_TissueFactor_up', 'BSK_3C_VCAM1_down', 'BSK_3C_Vis_down',
                 'BSK_3C_uPAR_down', 'BSK_4H_Eotaxin3_down', 'BSK_4H_MCP1_down', 'BSK_4H_Pselectin_down',
                 'BSK_4H_Pselectin_up', 'BSK_4H_SRB_down', 'BSK_4H_VCAM1_down', 'BSK_4H_VEGFRII_down',
                 'BSK_4H_uPAR_down', 'BSK_4H_uPAR_up', 'BSK_BE3C_HLADR_down', 'BSK_BE3C_IL1a_down',
                 'BSK_BE3C_IP10_down', 'BSK_BE3C_MIG_down', 'BSK_BE3C_MMP1_down', 'BSK_BE3C_MMP1_up',
                 'BSK_BE3C_PAI1_down', 'BSK_BE3C_SRB_down', 'BSK_BE3C_TGFb1_down', 'BSK_BE3C_tPA_down',
                 'BSK_BE3C_uPAR_down', 'BSK_BE3C_uPAR_up', 'BSK_BE3C_uPA_down', 'BSK_CASM3C_HLADR_down',
                 'BSK_CASM3C_IL6_down', 'BSK_CASM3C_IL6_up', 'BSK_CASM3C_IL8_down', 'BSK_CASM3C_LDLR_down',
                 'BSK_CASM3C_LDLR_up', 'BSK_CASM3C_MCP1_down', 'BSK_CASM3C_MCP1_up', 'BSK_CASM3C_MCSF_down',
                 'BSK_CASM3C_MCSF_up', 'BSK_CASM3C_MIG_down', 'BSK_CASM3C_Proliferation_down',
                 'BSK_CASM3C_Proliferation_up', 'BSK_CASM3C_SAA_down', 'BSK_CASM3C_SAA_up', 'BSK_CASM3C_SRB_down',
                 'BSK_CASM3C_Thrombomodulin_down', 'BSK_CASM3C_Thrombomodulin_up', 'BSK_CASM3C_TissueFactor_down',
                 'BSK_CASM3C_VCAM1_down', 'BSK_CASM3C_VCAM1_up', 'BSK_CASM3C_uPAR_down', 'BSK_CASM3C_uPAR_up',
                 'BSK_KF3CT_ICAM1_down', 'BSK_KF3CT_IL1a_down', 'BSK_KF3CT_IP10_down', 'BSK_KF3CT_IP10_up',
                 'BSK_KF3CT_MCP1_down', 'BSK_KF3CT_MCP1_up', 'BSK_KF3CT_MMP9_down', 'BSK_KF3CT_SRB_down',
                 'BSK_KF3CT_TGFb1_down', 'BSK_KF3CT_TIMP2_down', 'BSK_KF3CT_uPA_down', 'BSK_LPS_CD40_down',
                 'BSK_LPS_Eselectin_down', 'BSK_LPS_Eselectin_up', 'BSK_LPS_IL1a_down', 'BSK_LPS_IL1a_up',
                 'BSK_LPS_IL8_down', 'BSK_LPS_IL8_up', 'BSK_LPS_MCP1_down', 'BSK_LPS_MCSF_down',
                 'BSK_LPS_PGE2_down', 'BSK_LPS_PGE2_up', 'BSK_LPS_SRB_down', 'BSK_LPS_TNFa_down',
                 'BSK_LPS_TNFa_up', 'BSK_LPS_TissueFactor_down', 'BSK_LPS_TissueFactor_up', 'BSK_LPS_VCAM1_down',
                 'BSK_SAg_CD38_down', 'BSK_SAg_CD40_down', 'BSK_SAg_CD69_down', 'BSK_SAg_Eselectin_down',
                 'BSK_SAg_Eselectin_up', 'BSK_SAg_IL8_down', 'BSK_SAg_IL8_up', 'BSK_SAg_MCP1_down',
                 'BSK_SAg_MIG_down', 'BSK_SAg_PBMCCytotoxicity_down', 'BSK_SAg_PBMCCytotoxicity_up',
                 'BSK_SAg_Proliferation_down', 'BSK_SAg_SRB_down', 'BSK_hDFCGF_CollagenIII_down',
                 'BSK_hDFCGF_EGFR_down', 'BSK_hDFCGF_EGFR_up', 'BSK_hDFCGF_IL8_down', 'BSK_hDFCGF_IP10_down',
                 'BSK_hDFCGF_MCSF_down', 'BSK_hDFCGF_MIG_down', 'BSK_hDFCGF_MMP1_down', 'BSK_hDFCGF_MMP1_up',
                 'BSK_hDFCGF_PAI1_down', 'BSK_hDFCGF_Proliferation_down', 'BSK_hDFCGF_SRB_down',
                 'BSK_hDFCGF_TIMP1_down', 'BSK_hDFCGF_VCAM1_down', 'CEETOX_H295R_11DCORT_dn',
                 'CEETOX_H295R_ANDR_dn', 'CEETOX_H295R_CORTISOL_dn', 'CEETOX_H295R_DOC_dn', 'CEETOX_H295R_DOC_up',
                 'CEETOX_H295R_ESTRADIOL_dn', 'CEETOX_H295R_ESTRADIOL_up', 'CEETOX_H295R_ESTRONE_dn',
                 'CEETOX_H295R_ESTRONE_up', 'CEETOX_H295R_OHPREG_up', 'CEETOX_H295R_OHPROG_dn',
                 'CEETOX_H295R_OHPROG_up', 'CEETOX_H295R_PROG_up', 'CEETOX_H295R_TESTO_dn', 'CLD_ABCB1_48hr',
                 'CLD_ABCG2_48hr', 'CLD_CYP1A1_24hr', 'CLD_CYP1A1_48hr', 'CLD_CYP1A1_6hr', 'CLD_CYP1A2_24hr',
                 'CLD_CYP1A2_48hr', 'CLD_CYP1A2_6hr', 'CLD_CYP2B6_24hr', 'CLD_CYP2B6_48hr', 'CLD_CYP2B6_6hr',
                 'CLD_CYP3A4_24hr', 'CLD_CYP3A4_48hr', 'CLD_CYP3A4_6hr', 'CLD_GSTA2_48hr', 'CLD_SULT2A_24hr',
                 'CLD_SULT2A_48hr', 'CLD_UGT1A1_24hr', 'CLD_UGT1A1_48hr', 'NCCT_HEK293T_CellTiterGLO',
                 'NCCT_QuantiLum_inhib_2_dn', 'NCCT_QuantiLum_inhib_dn', 'NCCT_TPO_AUR_dn', 'NCCT_TPO_GUA_dn',
                 'NHEERL_ZF_144hpf_TERATOSCORE_up', 'NVS_ADME_hCYP19A1', 'NVS_ADME_hCYP1A1', 'NVS_ADME_hCYP1A2',
                 'NVS_ADME_hCYP2A6', 'NVS_ADME_hCYP2B6', 'NVS_ADME_hCYP2C19', 'NVS_ADME_hCYP2C9',
                 'NVS_ADME_hCYP2D6', 'NVS_ADME_hCYP3A4', 'NVS_ADME_hCYP4F12', 'NVS_ADME_rCYP2C12', 'NVS_ENZ_hAChE',
                 'NVS_ENZ_hAMPKa1', 'NVS_ENZ_hAurA', 'NVS_ENZ_hBACE', 'NVS_ENZ_hCASP5', 'NVS_ENZ_hCK1D',
                 'NVS_ENZ_hDUSP3', 'NVS_ENZ_hES', 'NVS_ENZ_hElastase', 'NVS_ENZ_hFGFR1', 'NVS_ENZ_hGSK3b',
                 'NVS_ENZ_hMMP1', 'NVS_ENZ_hMMP13', 'NVS_ENZ_hMMP2', 'NVS_ENZ_hMMP3', 'NVS_ENZ_hMMP7',
                 'NVS_ENZ_hMMP9', 'NVS_ENZ_hPDE10', 'NVS_ENZ_hPDE4A1', 'NVS_ENZ_hPDE5', 'NVS_ENZ_hPI3Ka',
                 'NVS_ENZ_hPTEN', 'NVS_ENZ_hPTPN11', 'NVS_ENZ_hPTPN12', 'NVS_ENZ_hPTPN13', 'NVS_ENZ_hPTPN9',
                 'NVS_ENZ_hPTPRC', 'NVS_ENZ_hSIRT1', 'NVS_ENZ_hSIRT2', 'NVS_ENZ_hTrkA', 'NVS_ENZ_hVEGFR2',
                 'NVS_ENZ_oCOX1', 'NVS_ENZ_oCOX2', 'NVS_ENZ_rAChE', 'NVS_ENZ_rCNOS', 'NVS_ENZ_rMAOAC',
                 'NVS_ENZ_rMAOAP', 'NVS_ENZ_rMAOBC', 'NVS_ENZ_rMAOBP', 'NVS_ENZ_rabI2C',
                 'NVS_GPCR_bAdoR_NonSelective', 'NVS_GPCR_bDR_NonSelective', 'NVS_GPCR_g5HT4', 'NVS_GPCR_gH2',
                 'NVS_GPCR_gLTB4', 'NVS_GPCR_gLTD4', 'NVS_GPCR_gMPeripheral_NonSelective', 'NVS_GPCR_gOpiateK',
                 'NVS_GPCR_h5HT2A', 'NVS_GPCR_h5HT5A', 'NVS_GPCR_h5HT6', 'NVS_GPCR_h5HT7', 'NVS_GPCR_hAT1',
                 'NVS_GPCR_hAdoRA1', 'NVS_GPCR_hAdoRA2a', 'NVS_GPCR_hAdra2A', 'NVS_GPCR_hAdra2C',
                 'NVS_GPCR_hAdrb1', 'NVS_GPCR_hAdrb2', 'NVS_GPCR_hAdrb3', 'NVS_GPCR_hDRD1', 'NVS_GPCR_hDRD2s',
                 'NVS_GPCR_hDRD4.4', 'NVS_GPCR_hH1', 'NVS_GPCR_hLTB4_BLT1', 'NVS_GPCR_hM1', 'NVS_GPCR_hM2',
                 'NVS_GPCR_hM3', 'NVS_GPCR_hM4', 'NVS_GPCR_hNK2', 'NVS_GPCR_hOpiate_D1', 'NVS_GPCR_hOpiate_mu',
                 'NVS_GPCR_hTXA2', 'NVS_GPCR_p5HT2C', 'NVS_GPCR_r5HT1_NonSelective', 'NVS_GPCR_r5HT_NonSelective',
                 'NVS_GPCR_rAdra1B', 'NVS_GPCR_rAdra1_NonSelective', 'NVS_GPCR_rAdra2_NonSelective',
                 'NVS_GPCR_rAdrb_NonSelective', 'NVS_GPCR_rNK1', 'NVS_GPCR_rNK3', 'NVS_GPCR_rOpiate_NonSelective',
                 'NVS_GPCR_rOpiate_NonSelectiveNa', 'NVS_GPCR_rSST', 'NVS_GPCR_rTRH', 'NVS_GPCR_rV1',
                 'NVS_GPCR_rabPAF', 'NVS_GPCR_rmAdra2B', 'NVS_IC_hKhERGCh', 'NVS_IC_rCaBTZCHL',
                 'NVS_IC_rCaDHPRCh_L', 'NVS_IC_rNaCh_site2', 'NVS_LGIC_bGABARa1', 'NVS_LGIC_h5HT3',
                 'NVS_LGIC_hNNR_NBungSens', 'NVS_LGIC_rGABAR_NonSelective', 'NVS_LGIC_rNNR_BungSens',
                 'NVS_MP_hPBR', 'NVS_MP_rPBR', 'NVS_NR_bER', 'NVS_NR_bPR', 'NVS_NR_cAR', 'NVS_NR_hAR',
                 'NVS_NR_hCAR_Antagonist', 'NVS_NR_hER', 'NVS_NR_hFXR_Agonist', 'NVS_NR_hFXR_Antagonist',
                 'NVS_NR_hGR', 'NVS_NR_hPPARa', 'NVS_NR_hPPARg', 'NVS_NR_hPR', 'NVS_NR_hPXR',
                 'NVS_NR_hRAR_Antagonist', 'NVS_NR_hRARa_Agonist', 'NVS_NR_hTRa_Antagonist', 'NVS_NR_mERa',
                 'NVS_NR_rAR', 'NVS_NR_rMR', 'NVS_OR_gSIGMA_NonSelective', 'NVS_TR_gDAT', 'NVS_TR_hAdoT',
                 'NVS_TR_hDAT', 'NVS_TR_hNET', 'NVS_TR_hSERT', 'NVS_TR_rNET', 'NVS_TR_rSERT', 'NVS_TR_rVMAT2',
                 'OT_AR_ARELUC_AG_1440', 'OT_AR_ARSRC1_0480', 'OT_AR_ARSRC1_0960', 'OT_ER_ERaERa_0480',
                 'OT_ER_ERaERa_1440', 'OT_ER_ERaERb_0480', 'OT_ER_ERaERb_1440', 'OT_ER_ERbERb_0480',
                 'OT_ER_ERbERb_1440', 'OT_ERa_EREGFP_0120', 'OT_ERa_EREGFP_0480', 'OT_FXR_FXRSRC1_0480',
                 'OT_FXR_FXRSRC1_1440', 'OT_NURR1_NURR1RXRa_0480', 'OT_NURR1_NURR1RXRa_1440',
                 'TOX21_ARE_BLA_Agonist_ch1', 'TOX21_ARE_BLA_Agonist_ch2', 'TOX21_ARE_BLA_agonist_ratio',
                 'TOX21_ARE_BLA_agonist_viability', 'TOX21_AR_BLA_Agonist_ch1', 'TOX21_AR_BLA_Agonist_ch2',
                 'TOX21_AR_BLA_Agonist_ratio', 'TOX21_AR_BLA_Antagonist_ch1', 'TOX21_AR_BLA_Antagonist_ch2',
                 'TOX21_AR_BLA_Antagonist_ratio', 'TOX21_AR_BLA_Antagonist_viability',
                 'TOX21_AR_LUC_MDAKB2_Agonist', 'TOX21_AR_LUC_MDAKB2_Antagonist',
                 'TOX21_AR_LUC_MDAKB2_Antagonist2', 'TOX21_AhR_LUC_Agonist', 'TOX21_Aromatase_Inhibition',
                 'TOX21_AutoFluor_HEK293_Cell_blue', 'TOX21_AutoFluor_HEK293_Media_blue',
                 'TOX21_AutoFluor_HEPG2_Cell_blue', 'TOX21_AutoFluor_HEPG2_Cell_green',
                 'TOX21_AutoFluor_HEPG2_Media_blue', 'TOX21_AutoFluor_HEPG2_Media_green', 'TOX21_ELG1_LUC_Agonist',
                 'TOX21_ERa_BLA_Agonist_ch1', 'TOX21_ERa_BLA_Agonist_ch2', 'TOX21_ERa_BLA_Agonist_ratio',
                 'TOX21_ERa_BLA_Antagonist_ch1', 'TOX21_ERa_BLA_Antagonist_ch2', 'TOX21_ERa_BLA_Antagonist_ratio',
                 'TOX21_ERa_BLA_Antagonist_viability', 'TOX21_ERa_LUC_BG1_Agonist', 'TOX21_ERa_LUC_BG1_Antagonist',
                 'TOX21_ESRE_BLA_ch1', 'TOX21_ESRE_BLA_ch2', 'TOX21_ESRE_BLA_ratio', 'TOX21_ESRE_BLA_viability',
                 'TOX21_FXR_BLA_Antagonist_ch1', 'TOX21_FXR_BLA_Antagonist_ch2', 'TOX21_FXR_BLA_agonist_ch2',
                 'TOX21_FXR_BLA_agonist_ratio', 'TOX21_FXR_BLA_antagonist_ratio',
                 'TOX21_FXR_BLA_antagonist_viability', 'TOX21_GR_BLA_Agonist_ch1', 'TOX21_GR_BLA_Agonist_ch2',
                 'TOX21_GR_BLA_Agonist_ratio', 'TOX21_GR_BLA_Antagonist_ch2', 'TOX21_GR_BLA_Antagonist_ratio',
                 'TOX21_GR_BLA_Antagonist_viability', 'TOX21_HSE_BLA_agonist_ch1', 'TOX21_HSE_BLA_agonist_ch2',
                 'TOX21_HSE_BLA_agonist_ratio', 'TOX21_HSE_BLA_agonist_viability', 'TOX21_MMP_ratio_down',
                 'TOX21_MMP_ratio_up', 'TOX21_MMP_viability', 'TOX21_NFkB_BLA_agonist_ch1',
                 'TOX21_NFkB_BLA_agonist_ch2', 'TOX21_NFkB_BLA_agonist_ratio', 'TOX21_NFkB_BLA_agonist_viability',
                 'TOX21_PPARd_BLA_Agonist_viability', 'TOX21_PPARd_BLA_Antagonist_ch1',
                 'TOX21_PPARd_BLA_agonist_ch1', 'TOX21_PPARd_BLA_agonist_ch2', 'TOX21_PPARd_BLA_agonist_ratio',
                 'TOX21_PPARd_BLA_antagonist_ratio', 'TOX21_PPARd_BLA_antagonist_viability',
                 'TOX21_PPARg_BLA_Agonist_ch1', 'TOX21_PPARg_BLA_Agonist_ch2', 'TOX21_PPARg_BLA_Agonist_ratio',
                 'TOX21_PPARg_BLA_Antagonist_ch1', 'TOX21_PPARg_BLA_antagonist_ratio',
                 'TOX21_PPARg_BLA_antagonist_viability', 'TOX21_TR_LUC_GH3_Agonist', 'TOX21_TR_LUC_GH3_Antagonist',
                 'TOX21_VDR_BLA_Agonist_viability', 'TOX21_VDR_BLA_Antagonist_ch1', 'TOX21_VDR_BLA_agonist_ch2',
                 'TOX21_VDR_BLA_agonist_ratio', 'TOX21_VDR_BLA_antagonist_ratio',
                 'TOX21_VDR_BLA_antagonist_viability', 'TOX21_p53_BLA_p1_ch1', 'TOX21_p53_BLA_p1_ch2',
                 'TOX21_p53_BLA_p1_ratio', 'TOX21_p53_BLA_p1_viability', 'TOX21_p53_BLA_p2_ch1',
                 'TOX21_p53_BLA_p2_ch2', 'TOX21_p53_BLA_p2_ratio', 'TOX21_p53_BLA_p2_viability',
                 'TOX21_p53_BLA_p3_ch1', 'TOX21_p53_BLA_p3_ch2', 'TOX21_p53_BLA_p3_ratio',
                 'TOX21_p53_BLA_p3_viability', 'TOX21_p53_BLA_p4_ch1', 'TOX21_p53_BLA_p4_ch2',
                 'TOX21_p53_BLA_p4_ratio', 'TOX21_p53_BLA_p4_viability', 'TOX21_p53_BLA_p5_ch1',
                 'TOX21_p53_BLA_p5_ch2', 'TOX21_p53_BLA_p5_ratio', 'TOX21_p53_BLA_p5_viability',
                 'Tanguay_ZF_120hpf_AXIS_up', 'Tanguay_ZF_120hpf_ActivityScore', 'Tanguay_ZF_120hpf_BRAI_up',
                 'Tanguay_ZF_120hpf_CFIN_up', 'Tanguay_ZF_120hpf_CIRC_up', 'Tanguay_ZF_120hpf_EYE_up',
                 'Tanguay_ZF_120hpf_JAW_up', 'Tanguay_ZF_120hpf_MORT_up', 'Tanguay_ZF_120hpf_OTIC_up',
                 'Tanguay_ZF_120hpf_PE_up', 'Tanguay_ZF_120hpf_PFIN_up', 'Tanguay_ZF_120hpf_PIG_up',
                 'Tanguay_ZF_120hpf_SNOU_up', 'Tanguay_ZF_120hpf_SOMI_up', 'Tanguay_ZF_120hpf_SWIM_up',
                 'Tanguay_ZF_120hpf_TRUN_up', 'Tanguay_ZF_120hpf_TR_up', 'Tanguay_ZF_120hpf_YSE_up']


def model_args(args):
    _other_args_name = ['dataset_root', 'dataset', 'split', 'seed', 'gpu', 'note', 'batch_size', 'epochs', 'loss',
                        'optim', 'k', 'lr', 'lr_reduce_rate', 'lr_reduce_patience', 'early_stop_patience',
                        'verbose_patience', 'split_seed']
    model_args_dict = {}
    for k, v in args.__dict__.items():
        if k not in _other_args_name:
            model_args_dict[k] = v
    return model_args_dict


def auto_dataset(args):
    from dataset import Dataset
    if args.dataset in dataset_names["c"] + ['demo']:
        _dataset = Dataset(args.dataset_root, dataset=args.dataset, split_seed=args.split_seed)
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
        if args.dataset is 'physprop_perturb': from dataset import PertubationDataset as Dataset
        _dataset = Dataset(args.dataset_root, dataset=args.dataset, split_seed=args.split_seed)
        args.out_dim = 1 * _dataset.num_tasks
    else:
        raise Exception('error dataset input')
    return args, _dataset, Trainer


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
            # print(p, loss_info, test_info)  # for some inf set
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
            d = {'id(note)': note, 'n_run': len(df), 'dataset': df['dataset'].iloc[0],
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
    if dataset in dataset_names["r"] + ['physprop_perturb']: metrics = ['valr2', 'r2']
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



