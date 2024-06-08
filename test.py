import json
import pickle
from ast import literal_eval
from os.path import isfile
import pandas as pd


def create_df(config):
    filename = f"results/results_b{config['batch_size']}{config['demo']}/corr_ratio_{config['corr_ratio']}/clin_frac_{config['client_fractions']}/test_results.json"

    f = open(filename, "r" if isfile(filename) else "w+")

    data = json.load(f)

    all_models = data[config['downstream_column']]

    df = pd.DataFrame.from_dict(all_models, orient='index')
    # print(df.index)
    selected_df = df[df.index.isin(['mean', 'zero', 'baseline43', 'cwt101', 'ditto101', 'fed_avg101', 'fed_prox101', 'pw101', 'fed_miss7', 'simp_avg123'])]

    # baseline = 43
    # cwt, ditto, fed_avg, fed_prox, pw = 101
    # fed_miss = 7
    # fed_simple = 123


    selected_df.to_csv(f"results/results_b{config['batch_size']}{config['demo']}/corr_ratio_{config['corr_ratio']}/clin_frac_{config['client_fractions']}/{config['downstream_column']}_.csv")


def create_new_df(config):
    filename = f"results/results_b{config['batch_size']}{config['demo']}/corr_ratio_{config['corr_ratio']}/clin_frac_{config['client_fractions']}/downstream_clf.pkl"
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    all_models = data[config['downstream_column']]

    df = pd.DataFrame.from_dict(all_models, orient='index')
    # print(df.index)
    df = df[['aucs', 'pr_aucs', 'accs', 'f1_scores']]
    # df = df[df.index.isin(['mean', 'zero', 'baseline43', 'cwt101', 'ditto101', 'fed_avg101', 'fed_prox101', 'pw101', 'fed_miss7', 'simp_avg123'])][['aucs', 'pr_aucs', 'accs']]
    # Convert the string lists into actual lists
    df['aucs'] = df['aucs'].apply(lambda x: [float(z)for z in x])
    df['pr_aucs'] = df['pr_aucs'].apply(lambda x: [float(z)for z in x])
    df['accs'] = df['accs'].apply(lambda x: [float(z)for z in x])
    # df['tprs'] = df['tprs'].apply(lambda x: [float(z)for z in x])
    df['f1_scores'] = df['f1_scores'].apply(lambda x: [float(z)for z in x])


    # Explode the 'aucs' column into multiple rows
    df = df.explode(['aucs', 'pr_aucs', 'accs', 'f1_scores'])
    # baseline = 43
    # cwt, ditto, fed_avg, fed_prox, pw = 101
    # fed_miss = 7
    # fed_simple = 123


    df.to_csv(f"results/results_b{config['batch_size']}{config['demo']}/corr_ratio_{config['corr_ratio']}/clin_frac_{config['client_fractions']}/{config['downstream_column']}_dict_.csv")
