import random
import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from model import auto_encoder_model
from utils import print_time, calculate_a1_a2


class Client:
    def __init__(self, train_dataset, valid_dataset, config):
        # train_dataset, validation_dataset = train_test_split(dataset, test_size=0.2, random_state=config['SEED'])
        self.train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=False)
        self.validation_loader = DataLoader(valid_dataset, config['batch_size'], shuffle=False)
        self.lr = config['lr']
        self.loss_fn = nn.MSELoss()

    def train_client(self, c_mdl, n_epochs, device):
        # self.mdl = self.mdl.to(device)
        opt = optim.Adam(c_mdl.parameters(), lr=self.lr)

        # Initialize early stopping parameters
        patience = 10
        early_stopping_counter = 0
        best_validation_loss = float('inf')

        for epochs in range(n_epochs):
            c_mdl.train()
            # train_loss = 0
            for idx, (data_in, data_out, mask) in enumerate(self.train_loader):
                data_in = data_in.to(device)
                data_out = data_out.to(device)
                mask = mask.to(device)

                opt.zero_grad()

                scores = c_mdl(data_in)
                scores = scores * mask
                data_out = data_out * mask

                loss = self.loss_fn(scores, data_out)
                # train_loss += loss.item() / len(self.train_loader)
                loss.backward()

                opt.step()

            c_mdl.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data_in, data_out, mask in self.validation_loader:
                    data_in = data_in.to(device)
                    data_out = data_out.to(device)
                    mask = mask.to(device)
                    
                    output = c_mdl(data_in)
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

        return c_mdl


class cwt_pred:
    def __init__(self, train_datasets, valid_datasets, test_datasets, config):
        self.clients = [Client(train_datasets[i], valid_datasets[i], config) for i in range(len(train_datasets))]
        self.test_loader = [DataLoader(test_dataset, config['batch_size']) for test_dataset in test_datasets]
        self.device = torch.device(f'cuda:{config["gpu"]}' if torch.cuda.is_available() else 'cpu')
        self.client_iters = config['client_iterations']
        self.total_iters = config['total_iterations']
        self.center_mdl = auto_encoder_model(
            number_features=config['NUM_FEATURES'],
            layer_width=config['layer_width'],
            depth=config['depth'],
            IR_size=config['IR_SIZE'],
            dropout_rate=config['drop_out']
        ).to(self.device)
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
                mdl = self.clients[jdx].train_client(self.center_mdl, self.client_iters, self.device)
                self.center_mdl = copy.deepcopy(mdl)

            print("############## server ##############")
            server_A1, server_A2 = calculate_a1_a2(self.center_mdl, self.test_loader, self.device)
            print("A1", round(server_A1, 3), "A2", round(server_A2, 3))
            losses.append((server_A1, server_A2))

            i_end_time = time.perf_counter()
            print_time(i_end_time, i_start_time)

            # Check if the validation loss has improved
            if server_A2 < best_validation_loss:
                best_validation_loss = server_A2
                best_model = self.center_mdl
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
