import copy
import io
import json
import random
from os.path import isfile
import numpy as np
import pandas as pd
import torch
import pickle
from torch.utils.data import DataLoader

from ppmi_data.dataset import MLDataset

MASK_WEIGHT = 2


def print_time(end_time, start_time):
    elapsed_time = int(end_time - start_time)
    hr = elapsed_time // 3600
    mi = (elapsed_time - hr * 3600) // 60
    sec = elapsed_time - hr * 3600 - mi * 60
    print(f"training done in {hr} H {mi} M {sec} S")


def calculate_a1_a2(mdl, test_loaders, device):
    A1 = 0
    A1_num = 0
    A2 = 0
    A2_num = 0

    for client in test_loaders:
        for idx, (data_in, data_out, masks) in enumerate(client):
            data_in = data_in.to(device)
            scores = mdl(data_in).cpu()

            for score, datum, mask in zip(scores, data_out, masks):
                for i, m in enumerate(mask):
                    if i >= 1:
                        if m == 1:
                            A1_num += 1
                            A1 += (score[i] - datum[i]) ** 2
                        elif m == MASK_WEIGHT:
                            A2_num += 1
                            A2 += (score[i] - datum[i]) ** 2

    return (A1.item() / A1_num), (A2.item() / A2_num)


def initialize(config):
    torch.manual_seed(config["SEED"])
    torch.cuda.manual_seed(config["SEED"])
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    torch.backends.cudnn.benchmarks = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch.device(f'cuda:{config["gpu"]}' if torch.cuda.is_available() else 'cpu')


def impute_nan(dataset, dataset_org):
    dataset_copy = copy.deepcopy(dataset_org)

    m, n = dataset_org.shape
    for i in range(m):
        for j in range(n):
            if np.isnan(dataset_org.iloc[i, j]):
                dataset_copy.iloc[i, j] = dataset.iloc[i, j]

    return dataset_copy


def update_test_losses(config, a1, a2):
    filename = f"results/results_b{config['batch_size']}{config['demo']}/corr_ratio_{config['corr_ratio']}/clin_frac_{config['client_fractions']}/test_losses.json"

    f = open(filename, "r" if isfile(filename) else "w+")
    try:
        data = json.load(f)
    except json.decoder.JSONDecodeError or io.UnsupportedOperation:
        data = {"0.1": {}}
    f.close()

    data[str(config['miss_ratio'])][config['fed_name'] + str(config['SEED'])] = (a1, a2)

    f = open(filename, 'w')
    json.dump(data, f)
    f.close()


def update_downstream_results(config, acc_mean, acc_std, f1_mean, f1_std):
    filename = f"results/results_b{config['batch_size']}{config['demo']}/corr_ratio_{config['corr_ratio']}/clin_frac_{config['client_fractions']}/test_results.json"

    f = open(filename, "r" if isfile(filename) else "w+")
    try:
        data = json.load(f)
    except json.decoder.JSONDecodeError or io.UnsupportedOperation:
        data = {'updrs3_score': {}, 'updrs1_score': {}, 'updrs2_score': {}, 'updrs_totscore': {}}
    f.close()
    if config['imputation']:
        data[config['downstream_column']][config['fed_name'] + str(config['SEED'])] = {
            'acc_mean': acc_mean,
            'acc_std': acc_std,
            'f1_mean': f1_mean,
            'f1_std': f1_std
        }
    else:
        data[config['downstream_column']][config['na_impute'] + str(config['SEED'])] = {
            'acc_mean': acc_mean,
            'acc_std': acc_std,
            'f1_mean': f1_mean,
            'f1_std': f1_std
        }

    f = open(filename, 'w')
    json.dump(data, f)
    f.close()


def update_downstream_clf(config, fpr, tprs, tpr, aucs, auc, auc_std, pr_aucs, pr_auc, pr_auc_std, precision, recall,
                          accs, f1_scores):
    filename = f"results/results_b{config['batch_size']}{config['demo']}/corr_ratio_{config['corr_ratio']}/clin_frac_{config['client_fractions']}/downstream_clf.pkl"
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    except:
        data = {'updrs3_score': {}, 'updrs1_score': {}, 'updrs2_score': {}, 'updrs_totscore': {}}

    if config['imputation']:
        data[config['downstream_column']][config['fed_name'] + str(config['SEED'])] = {'fpr': fpr,
                                                                                       'tprs': tprs,
                                                                                       'tpr': tpr,
                                                                                       'aucs': aucs,
                                                                                       'auc': auc,
                                                                                       'auc_std': auc_std,
                                                                                       'pr_aucs': pr_aucs,
                                                                                       'pr_auc': pr_auc,
                                                                                       'pr_auc_std': pr_auc_std,
                                                                                       'precision': precision,
                                                                                       'recall': recall,
                                                                                       'accs': accs,
                                                                                       'f1_scores': f1_scores}
    else:
        data[config['downstream_column']][config['na_impute'] + str(config['SEED'])] = {'fpr': fpr,
                                                                                        'tprs': tprs,
                                                                                        'tpr': tpr,
                                                                                        'aucs': aucs,
                                                                                        'auc': auc,
                                                                                        'auc_std': auc_std,
                                                                                        'pr_aucs': pr_aucs,
                                                                                        'pr_auc': pr_auc,
                                                                                        'pr_auc_std': pr_auc_std,
                                                                                        'precision': precision,
                                                                                        'recall': recall,
                                                                                        'accs': accs,
                                                                                        'f1_scores': f1_scores}

    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def impute(config, ppmi_dataset, ppmi_cols, mdl, device):
    ppmi_dataset = primary_initialization(ppmi_dataset)
    ppmi_loader = DataLoader(MLDataset(ppmi_dataset), batch_size=config['batch_size'])

    x_imputed = torch.Tensor().to(device)
    for x in ppmi_loader:
        x_imputed = torch.cat([x_imputed, mdl(x.to(device))])

    return pd.DataFrame(x_imputed.cpu().detach().numpy(), columns=ppmi_cols)


def primary_initialization(dataset):
    dataset_copy = copy.deepcopy(dataset)
    dataset_copy[np.isnan(dataset_copy)] = 0

    means = np.mean(dataset_copy, axis=0)

    for record in dataset:
        for i in range(len(record)):
            if np.isnan(record[i]):
                record[i] = means[i]

    return dataset
