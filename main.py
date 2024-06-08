from pandas.errors import SettingWithCopyWarning
from sklearn.preprocessing import MinMaxScaler

from downstream_analysis import learn_classification
from federated_learnings.cwt_pred import cwt_pred
from federated_learnings.ditto_pred import ditto_pred
from federated_learnings.fedavg_pred import fedavg_pred
from federated_learnings.fedmiss_pred import fedmiss_pred
from federated_learnings.fedprox_pred import fedprox_pred
from federated_learnings.pw_pred import pw_pred
from federated_learnings.simpavg_pred import simpavg_pred
from federated_learnings.baseline_pred import baseline_pred
from ppmi_data.sites import get_sites
from plot import save_log, plot_a1_a2, plot_roc_pr
from test import create_df, create_new_df
from utils import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

config = {
    "SEED": 43,
    "gpu": 0,

    "NUM_FEATURES": 64,
    "IR_SIZE": 7,
    "layer_width": 10,
    "depth": 3,
    "drop_out": 0.1,

    "total_iterations": 150,
    "client_iterations": 120,
    "client_fractions": 1.0,  # [0.2, 0.5]

    "demo": "",  # ["_demo", ""]
    "miss_ratio": 0.1,  # [0.1]
    "corr_ratio": 0.6,  # [0.1, 0.3, 0.6]
    "batch_size": 16,  # [16, 32]

    "imputation": True,  # [True, False]
    "fed_name": "baseline",  # ['baseline', 'cwt', 'ditto', 'fed_avg', 'fed_miss', 'fed_prox', 'pw', 'simp_avg']
    "na_impute": "mean",  # ['zero', 'mean']

    "lr": 1e-6,
    "alpha": 0.1,  # fed_prox
    "lambda": 0.1,  # ditto

    "downstream_column": "updrs1_score"  # ['updrs1_score', 'updrs2_score', 'updrs3_score', 'updrs_totscore']
}


def run(train_datasets, valid_datasets, test_datasets, seed):
    fed_solver = None
    if config['fed_name'] == "fed_avg":
        fed_solver = fedavg_pred
    elif config['fed_name'] == "fed_prox":
        fed_solver = fedprox_pred
    elif config['fed_name'] == "cwt":
        fed_solver = cwt_pred
    elif config['fed_name'] == "fed_miss":
        fed_solver = fedmiss_pred
    elif config['fed_name'] == "ditto":
        fed_solver = ditto_pred
    elif config['fed_name'] == "pw":
        fed_solver = pw_pred
    elif config['fed_name'] == "simp_avg":
        fed_solver = simpavg_pred
    elif config['fed_name'] == "baseline":
        fed_solver = baseline_pred

    model_weights = f"weights/weights_b{config['batch_size']}{config['demo']}/corr_ratio_{config['corr_ratio']}/clin_frac_{config['client_fractions']}/{config['fed_name']}_{seed}.h5"

    if isfile(model_weights):
        print(f"{config['fed_name']} Loaded")
        server = torch.load(model_weights).to(device)
    else:
        print("Starting training since file was not found: ", model_weights)
        loss_s, server = fed_solver(train_datasets, valid_datasets, test_datasets, config).train()
        if len(loss_s) == 1:
            loss_s = [loss_s[0] for _ in range(config['total_iterations'])]
        save_log(f"results/results_b{config['batch_size']}{config['demo']}/corr_ratio_{config['corr_ratio']}/clin_frac_{config['client_fractions']}/train_losses/{config['fed_name']}_train_loss_" + str(seed) + ".txt", loss_s)
        torch.save(server, model_weights)

    a1, a2 = calculate_a1_a2(server, [DataLoader(test_dataset, config['batch_size']) for test_dataset in test_datasets], device)
    update_test_losses(config, a1, a2)

    return server


def downstream(dataset, dataset_PATNO, server=None):
    if config['imputation']:
        dataset_org = dataset
        sc = MinMaxScaler(feature_range=(0, 1))
        columns = dataset.columns
        dataset = sc.fit_transform(dataset)
        dataset = impute(config, dataset, columns, server, device)
        dataset = sc.inverse_transform(dataset)
        dataset = pd.DataFrame(dataset, columns=columns)
        dataset = impute_nan(dataset, dataset_org)

    else:
        if config['na_impute'] == "zero":
            for c in dataset.columns:
                dataset[c].fillna(0, inplace=True)
        if config['na_impute'] == "mean":
            # for c in dataset.columns:
            #     dataset[c].fillna(dataset[c].mean(skipna=True), inplace=True)
            dataset = dataset.apply(lambda col: col.fillna(col.mean()), axis=0)

    dataset['PATNO'] = dataset_PATNO
    acc_mean, acc_std, f1_mean, f1_std = learn_classification(dataset, config)

    update_downstream_results(config, acc_mean, acc_std, f1_mean, f1_std)


if __name__ == "__main__":
    # ppmi = pd.read_csv("ppmi_data/train_curated.txt", sep=',')
    # print(ppmi.shape)
    # nans_df = pd.DataFrame.from_dict(
    #     {c: [ppmi[c].isna().sum()] for c in ppmi.columns})
    # nans_df.to_csv("nans_df.csv")
    # print(nans_df)
    # exit()

    # for d in ['updrs1_score', 'updrs2_score', 'updrs3_score', 'updrs_totscore']:
    #     config['downstream_column'] = d
    #     create_new_df(config)
    #     create_df(config)

    for seed in [7, 43, 101, 123, 988]:
        for d in ['updrs1_score', 'updrs2_score', 'updrs3_score', 'updrs_totscore']:
            config['imputation'] = True
            config['SEED'] = seed
            config['downstream_column'] = d

            device = initialize(config)
            if config['demo'] == "":
                ppmi = pd.read_csv("ppmi_data/train_curated.txt", sep=',')
            else:
                ppmi = pd.read_csv("ppmi_data/train_curated_demo.txt", sep=',')
            ppmi_sites = ppmi['PATNO']

            train_datasets, valid_datasets, test_datasets = get_sites(ppmi, config["miss_ratio"], config["corr_ratio"], config['SEED'])

            ppmi = ppmi.drop(['SITE', 'PATNO', 'COHORT'], axis=1)
            ppmi = ppmi.dropna(subset=[config['downstream_column']])

            for fed_name in ['cwt', 'ditto', 'fed_avg', 'fed_prox', 'pw', 'simp_avg', 'fed_miss', 'baseline']:
                config['fed_name'] = fed_name
                print("Running Algorithm: {0}".format(config['fed_name']))
                server = run(train_datasets, valid_datasets, test_datasets, config['SEED'])
                downstream(ppmi, ppmi_sites, server)

            config['imputation'] = False
            for na_impute in ['zero', 'mean']:
                config['na_impute'] = na_impute
                downstream(ppmi, ppmi_sites)

        # plot_a1_a2(config)
        # plot_roc_pr(config)

    # device = initialize(config)
    # if config['demo'] == "":
    #     ppmi = pd.read_csv("ppmi_data/train_curated.txt", sep=',')
    # else:
    #     ppmi = pd.read_csv("ppmi_data/train_curated_demo.txt", sep=',')
    # ppmi_sites = ppmi['PATNO']
    #
    # train_datasets, valid_datasets, test_datasets = get_sites(ppmi, config["miss_ratio"], config["corr_ratio"], config['SEED'])
    #
    # ppmi = ppmi.drop(['SITE', 'PATNO', 'COHORT'], axis=1)
    # ppmi = ppmi.dropna(subset=[config['downstream_column']])
    #
    # server = run(train_datasets, valid_datasets, test_datasets, config['SEED'])
    # downstream(ppmi, ppmi_sites, server)
