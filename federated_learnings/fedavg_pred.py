import random
import torch
import time
from torch.utils.data import DataLoader

from federated_learnings.Client import Client
from federated_learnings.Server import Server
from utils import print_time, calculate_a1_a2


class fedavg_pred:
    def __init__(self, train_datasets, valid_datasets, test_datasets, config):
        self.clients = [Client(train_datasets[i], valid_datasets[i], config) for i in range(len(train_datasets))]
        self.test_loader = [DataLoader(test_dataset, config['batch_size'], shuffle=False) for test_dataset in test_datasets]
        self.device = torch.device(f'cuda:{config["gpu"]}' if torch.cuda.is_available() else 'cpu')
        self.server = Server(config, self.device)
        self.client_iters = config['client_iterations']
        self.total_iters = config['total_iterations']
        self.n_clients = len(train_datasets)
        self.sample_cli = int(config['client_fractions'] * self.n_clients)
        self.seed = int(config['SEED'])

    def train(self):
        losses = []
        t_start_time = time.perf_counter()
        random.seed(self.seed)

        # Initialize early stopping parameters
        patience = 10
        early_stopping_counter = 0
        best_validation_loss = float('inf')
        best_model = None

        for idx in range(self.total_iters):
            i_start_time = time.perf_counter()

            print(f"iteration [{idx + 1}/{self.total_iters}]")

            clients_selected = random.sample([i for i in range(self.n_clients)], self.sample_cli)

            for jdx in clients_selected:
                print(f"############## client {jdx} ##############")
                self.clients[jdx].replace_mdl(self.server.mdl)
                self.clients[jdx].train_client(self.client_iters, self.device)

            print("############## server ##############")
            self.server.aggregate_models([self.clients[i].mdl for i in clients_selected],
                                         [len(self.clients[i].train_loader.dataset) for i in clients_selected])

            server_A1, server_A2 = calculate_a1_a2(self.server.mdl, self.test_loader, self.device)
            print("A1", round(server_A1, 3), "A2", round(server_A2, 3))
            losses.append((server_A1, server_A2))

            i_end_time = time.perf_counter()
            print_time(i_end_time, i_start_time)

            # Check if the validation loss has improved
            if server_A2 < best_validation_loss:
                best_validation_loss = server_A2
                best_model = self.server.mdl
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            # Check if we should early stop
            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                break

        t_end_time = time.perf_counter()
        print_time(t_end_time, t_start_time)

        return losses, best_model
