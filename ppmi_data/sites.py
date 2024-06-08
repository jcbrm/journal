import random
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from ppmi_data.dataset import SampleDataset
from utils import MASK_WEIGHT, primary_initialization


def get_sites(ppmi, miss_ratio, corr_ratio, seed):
    train_sites = [26, 36, 31, 14, 72, 23, 19, 21, 70, 28, 67, 29, 25, 37, 71, 10, 24, 61, 32,
                   63, 35, 74, 13, 34, 22, 42, 64]
    test_sites = [12, 16, 38, 40, 60, 68, 75, 43, 33, 18, 27, 69, 39, 30, 41, 44, 15, 20, 76, 11, 66, 65, 17, 78, 73]                   

    random.seed(seed)
    
    # Split train/test (80/20)
    test = ppmi[ppmi['SITE'].isin(test_sites)]
    train = ppmi[~ppmi['SITE'].isin(test_sites)]

    def get_clients(dataset, sites, merge):
        if merge:
            client_dataset = dataset[dataset['SITE'].isin(sites)].drop(['SITE', 'COHORT'], axis=1).reset_index(drop=True) 
        else:
            client_dataset = [dataset[dataset['SITE'] == site].drop(['SITE', 'COHORT'], axis=1).reset_index(drop=True) for site in sites]
        # client_datasets = []
        # for c in client_dataset:
        #     # if len(c) >= 5:
        #     client_datasets.append(c.drop(['SITE', 'COHORT'], axis=1).reset_index(drop=True))
        return client_dataset

    train_clients = get_clients(train, train_sites, False)
    test_clients = get_clients(test, test_sites, True)

    def get_loader(dataset, ratio, modality):
        return [SampleDataset(*process(ds.to_numpy(), ratio, modality)) for ds in dataset]

    def split_train_valid(datasets):
        train = []
        valid = []        
        
        samples_quantity = [math.floor(ds['PATNO'].nunique() * 0.30) for ds in datasets]
        patients = [random.sample(list(ds['PATNO'].unique()),samples_quantity[i]) for i, ds in enumerate(datasets)]

        valid = [datasets[i][datasets[i]["PATNO"].isin(p)] for i, p in enumerate(patients)]
        train = [datasets[i][~datasets[i]["PATNO"].isin(p)] for i, p in enumerate(patients)]
        
        # valid = [ds[ds['PATNO'].isin(patients[i])] for i, ds in enumerate(datasets)]
        # train = [ds[ds['PATNO'].isin(patients[i])] for i, ds in enumerate(datasets)]

        return train, valid  

    train_clients, valid_clients = split_train_valid(train_clients) 

    train_clients = [ds.drop(['PATNO'], axis=1) for ds in train_clients]
    valid_clients = [ds.drop(['PATNO'], axis=1) for ds in valid_clients]
    test_clients = [test_clients.drop(['PATNO'], axis=1)] 

    print(test_clients[0].isna().sum())

    train_datasets = get_loader(train_clients, corr_ratio, True)
    valid_clients = get_loader(valid_clients, miss_ratio, False)
    test_datasets = get_loader(test_clients, miss_ratio, False)

    return train_datasets, valid_clients, test_datasets


def remove_modalities(dataset, cor_r):
    for row in dataset:
        for i, mask in enumerate(get_modalities(cor_r)):
            if mask:
                row[i] = np.nan
    return dataset


def remove_entities(dataset, miss_r):
    for row in dataset:
        for i, mask in enumerate(row):
            if random.random() < miss_r:
                row[i] = np.nan
    return dataset


def process(dataset, ratio, modality):
    # This mask is for the initial Nan
    # import ipdb; ipdb.set_trace()
    masks = [[0 if np.isnan(score) else 1 for score in record] for record in dataset]

    if modality:
        new_dataset = remove_modalities(dataset, ratio)
    else:
        new_dataset = remove_entities(dataset, ratio)

    for record, mask in zip(new_dataset, masks):
        for i, score in enumerate(record):
            if np.isnan(score) and mask[i] != 0:
                mask[i] = MASK_WEIGHT

    # Replacing missing values with mean (column) values
    x_data = primary_initialization(new_dataset)
    y_data = primary_initialization(dataset)

    # Scaling data between 0-1 for the network
    sc = MinMaxScaler(feature_range=(0, 1))

    x_data = sc.fit_transform(x_data)
    y_data = sc.fit_transform(y_data)

    return x_data, y_data, np.array(masks)


def get_modalities(corr_r):
    modalities = []
    modalities.append([1])
    modalities.append([2])
    modalities.append([3])
    modalities.append([4, 4, 4, 4, 4, 4])
    modalities.append([5])
    modalities.append([6])
    modalities.append([7])
    modalities.append([8])
    modalities.append([9])
    modalities.append([10])
    modalities.append([11])
    modalities.append([12, 12, 12, 12, 12, 12, 12, 12, 12])
    modalities.append([13])
    modalities.append([14, 14, 14, 14, 14, 14, 14])
    modalities.append([15])
    modalities.append([16, 16, 16])
    modalities.append([17, 17, 17, 17, 17, 17, 17, 17])
    modalities.append([18])
    modalities.append([19])
    modalities.append([20])
    modalities.append([21])
    modalities.append([22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22])  # SPECT imaging
    len_modalities = [len(modal) for modal in modalities]
    modality_mask = []
    for i in range(22):
        if random.random() < corr_r:
            modality_mask.extend([True for _ in range(len_modalities[i])])
        else:
            modality_mask.extend([False for _ in range(len_modalities[i])])

    return modality_mask
