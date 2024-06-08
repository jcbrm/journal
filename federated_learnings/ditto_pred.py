import random
import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from federated_learnings.Server import Server
from model import auto_encoder_model
from utils import print_time, calculate_a1_a2


class Client:
    def __init__(self, train_dataset, valid_dataset, config):
        self.lr = config['lr']
        # train_dataset, validation_dataset = train_test_split(dataset, test_size=0.2, random_state=config['SEED'])
        self.train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=False)
        self.validation_loader = DataLoader(valid_dataset, config['batch_size'], shuffle=False)
        self.mdl = auto_encoder_model(
            number_features=config['NUM_FEATURES'],
            layer_width=config['layer_width'],
            depth=config['depth'],
            IR_size=config['IR_SIZE'],
            dropout_rate=config['drop_out'])
        self.loss_fn = nn.MSELoss()
        self.landa = config['lambda']

    def train_client(self, best_mdl, n_epochs, device):
        self.mdl = self.mdl.to(device)
        opt = optim.Adam(self.mdl.parameters(), lr=self.lr)

        # Initialize early stopping parameters
        patience = 10
        early_stopping_counter = 0
        best_validation_loss = float('inf')

        # train_loss = 0
        for epochs in range(n_epochs):
            self.mdl.train()
            train_loss = 0
            for data_in, data_out, mask in self.train_loader:
                data_in = data_in.to(device)
                data_out = data_out.to(device)
                mask = mask.to(device)

                opt.zero_grad()

                scores = self.mdl(data_in)
                scores = scores * mask
                data_out = data_out * mask

                loss = self.loss_fn(scores, data_out) + self.ditto(best_mdl)
                cur_loss = self.loss_fn(scores, data_out)
                train_loss += cur_loss.item() / len(self.train_loader)
                loss.backward()

                opt.step()

            self.mdl.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data_in, data_out, mask in self.validation_loader:
                    data_in = data_in.to(device)
                    data_out = data_out.to(device)
                    mask = mask.to(device)
                    
                    output = self.mdl(data_in)
                    val_loss += self.loss_fn(output, data_out)#.item()
                # server_A1, server_A2 = calculate_a1_a2(self.mdl, [self.validation_loader], device)
                # print(f"epochs: [{epochs + 1}/{n_epochs}] train_loss: {train_loss:.3f}",
                #     "A1", round(server_A1, 3), "A2", round(server_A2, 3))
            val_loss /= len(self.validation_loader)
            # Check if the validation loss has improved
            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            # Check if we should early stop
            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                break

        return self.mdl, train_loss

    def replace_mdl(self, server_mdl):
        self.mdl = copy.deepcopy(server_mdl)

    def ditto(self, best_mdl):
        params1 = dict(self.mdl.named_parameters())
        params2 = dict(best_mdl.named_parameters())

        loss_val = 0

        for name, params in params1.items():
            norm_val = torch.norm(params1[name] - params2[name]) ** 2
            loss_val += self.landa * 0.5 * norm_val

        return loss_val


class ditto_pred:
    def __init__(self, train_datasets, valid_datasets, test_datasets, config):
        self.clients = [Client(train_datasets[i], valid_datasets[i], config) for i in range(len(train_datasets))]
        self.test_loader = [DataLoader(test_dataset, config['batch_size'], shuffle=False) for test_dataset in test_datasets]
        self.device = torch.device(f'cuda:{config["gpu"]}' if torch.cuda.is_available() else 'cpu')
        self.server = Server(config, self.device)
        self.client_iters = config['client_iterations']
        self.total_iters = config['total_iterations']
        self.n_clients = len(train_datasets)
        self.sample_cli = int(config['client_fractions'] * self.n_clients)
        self.best_mdl = auto_encoder_model(
            number_features=config['NUM_FEATURES'],
            layer_width=config['layer_width'],
            depth=config['depth'],
            IR_size=config['IR_SIZE'],
            dropout_rate=config['drop_out']).to(self.device)
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

            for pdx in clients_selected:
                self.clients[pdx].replace_mdl(self.server.mdl)

            models = {}
            for jdx in clients_selected:
                print(f"############## client {jdx} ##############")
                mdl, loss = self.clients[jdx].train_client(self.best_mdl, self.client_iters, self.device)
                models[mdl] = loss
            self.best_mdl = copy.deepcopy(min(models, key=models.get))

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
