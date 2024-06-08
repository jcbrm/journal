import random
import torch
import time
from collections import OrderedDict
from torch.utils.data import DataLoader

from federated_learnings.Client import Client
from model import auto_encoder_model
from utils import print_time, calculate_a1_a2


class Server:
    def __init__(self, config, device):
        self.mdl = auto_encoder_model(
            number_features=config['NUM_FEATURES'],
            layer_width=config['layer_width'],
            depth=config['depth'],
            IR_size=config['IR_SIZE'],
            dropout_rate=config['drop_out']
        ).to(device)

    def aggregate_models(self, clients_model):
        update_state = OrderedDict()

        n_clients = len(clients_model)

        for k, client in enumerate(clients_model):
            local_state = client.state_dict()
            for key in self.mdl.state_dict().keys():
                if k == 0:
                    update_state[key] = local_state[key] / n_clients
                else:
                    update_state[key] += local_state[key] / n_clients

        self.mdl.load_state_dict(update_state)


class simpavg_pred:
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
            self.server.aggregate_models([self.clients[i].mdl for i in clients_selected])

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
