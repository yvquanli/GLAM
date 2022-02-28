from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from dataset import dataset_names
import numpy as np
import torch

def auto_metrics(dataset: str):
    metrics = ['valauc', 'auc']
    if dataset in dataset_names["r"] + ['physprop_mutate']:
        metrics = ['valr2', 'r2']
    # if dataset.startswith('lit_'): metrics = ['valbedroc', 'bedroc']
    return metrics

def binary_metrics(y_true, y_score, y_pred=None, threshod=0.5):
    auc = roc_auc_score(y_true, y_score)
    if y_pred is None:
        y_pred = (y_score >= threshod).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prauc = metrics.auc(precision_recall_curve(y_true, y_score)[
                        1], precision_recall_curve(y_true, y_score)[0])
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    d = {'auc': auc, 'prauc': prauc, 'acc': acc,
         'precision': precision, 'recall': recall, 'f1': f1}
    return d


def binary_metrics_multi_target_nan(y_true, y_score, y_pred=None, nan_fill=-1, threshod=0.5):
    '''
       y_true and y_score should be `(N, T)` where N = number of GLAMs, and T = number of targets
    '''
    if y_pred is None:
        y_pred = (y_score >= threshod).astype(int)
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
    fac = r_a * np.sinh(alpha / 2) / (np.cosh(alpha / 2) -
                                      np.cosh(alpha / 2 - alpha * r_a))
    cte = 1 / (1 - np.exp(alpha * (1 - r_a)))
    return s * fac / rand_sum + cte


def enrichment_factor_single(y_true, y_score, threshold=0.005):
    # https://github.com/gitter-lab/pria_lifechem/blob/1fd892505a/pria_lifechem/evaluation.py
    labels_arr, scores_arr, percentile = y_true, y_score, threshold
    non_missing_indices = np.argwhere(labels_arr != -1)[:, 0]
    labels_arr = labels_arr[non_missing_indices]
    scores_arr = scores_arr[non_missing_indices]
    # determine number mols in subset
    GLAM_size = int(labels_arr.shape[0] * percentile)
    # get the index positions for these in library
    indices = np.argsort(scores_arr, axis=0)[::-1][:GLAM_size]
    # count number of positive labels in library
    n_actives = np.nansum(labels_arr)
    # count number of positive labels in subset
    n_experimental = np.nansum(labels_arr[indices])

    if n_actives > 0.0:
        ef = float(n_experimental) / n_actives / \
            percentile  # calc EF at percentile
    else:
        raise Exception('n actives == 0')
    # return n_actives, ef, ef_max
    return ef


def screening_metrics(y_true, y_score, y_pred=None, threshod=0.5):
    auc = roc_auc_score(y_true, y_score)
    if y_pred is None:
        y_pred = (y_score > threshod).astype(int)
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



def blend_regression(outputs: list, opt='mean', return_pred=False):
    ls, pls = [], []
    for _l, _pl in outputs:
        ls.append(_l)
        pls.append(_pl)
    blendd_l = ls[0]
    blendd_pl = torch.stack(pls, dim=1).mean(dim=1) if opt == 'mean' else None
    if return_pred is True:
        return blendd_pl
    return regression_metrics(blendd_l.numpy(), y_pred=blendd_pl.numpy())


def blend_binary_classification(outputs: list, opt='vote', metrics_fn=binary_metrics):
    ls, pls, ss = [], [], []
    for _l, _pl, _s in outputs:
        ls.append(_l)
        pls.append(_pl)
        ss.append(_s)
    blendd_l = ls[0]
    blendd_pl = torch.stack(pls, dim=1).mode(
        dim=1)[0] if opt == 'vote' else None
    blendd_ss = torch.stack(ss, dim=1).mean(dim=1)
    return metrics_fn(blendd_l.numpy(), y_score=blendd_ss.numpy(), y_pred=blendd_pl.numpy())


def blend_binary_classification_mt(outputs: list, opt='vote', metrics_fn=binary_metrics):
    ls, ss = [], []
    for _s, _l in outputs:
        ls.append(_l)
        ss.append(_s)
    blendd_l = ls[0]
    # blendd_pl = torch.stack(pls, dim=2).mode(dim=2)[0] if opt == 'vote' else None
    blendd_ss = torch.stack(ss, dim=2).mean(dim=2)
    return metrics_fn(blendd_l.numpy(), y_score=blendd_ss.numpy())

