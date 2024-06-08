import random
import torch
import time
import copy
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torch.utils.data import DataLoader

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

    def train_client(self, n_epochs, device):
        eps = 1e-07
        self.mdl = self.mdl.to(device)
        opt = optim.Adam(self.mdl.parameters(), lr=self.lr)

        # Initialize early stopping parameters
        patience = 10
        early_stopping_counter = 0
        best_validation_loss = float('inf')

        for epochs in range(n_epochs):
            self.mdl.train()
            # train_loss = 0
            for data_in, data_out, mask in self.train_loader:
                data_in = data_in.to(device)
                data_out = data_out.to(device)
                mask = mask.to(device)

                opt.zero_grad()

                scores = self.mdl(data_in)
                scores = scores * mask
                data_out = data_out * mask

                loss = self.loss_fn(scores, data_out)
                # train_loss += loss.item() / len(self.train_loader)
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

        second_moments = {}
        for group in opt.param_groups:
            for param in group['params']:
                state = opt.state[param]
                if 'exp_avg_sq' in state:
                    second_moments[param] = state['exp_avg_sq'] + eps

        return second_moments

    def replace_mdl(self, server_mdl):
        self.mdl = copy.deepcopy(server_mdl)


class Server:
    def __init__(self, config, device):
        self.mdl = auto_encoder_model(
            number_features=config['NUM_FEATURES'],
            layer_width=config['layer_width'],
            depth=config['depth'],
            IR_size=config['IR_SIZE'],
            dropout_rate=config['drop_out']
        ).to(device)

    def aggregate_models(self, clients_model, training_samples, optimizers):
        update_state = OrderedDict()
        sum_v = OrderedDict()
        sum_weighted_v = OrderedDict()

        k = 0 # this index is used to loop through each v estimate in the optimizers
        for _, key in enumerate(self.mdl.state_dict().keys()):
            weights = [c.state_dict()[key] for c in clients_model]
            if key.endswith("weight") or key.endswith("bias"):
                variances = [list(opt.values())[k] for opt in optimizers ] 
                
                sum_v[key] = torch.sum(torch.stack([variances[i] * training_samples[i] for i in range(len(training_samples))],dim=0),dim=0)
                sum_weighted_v[key] = torch.sum(torch.stack([variances[i] * training_samples[i] * weights[i] for i in range(len(training_samples))], dim=0), dim=0)

                if ".bn" in key:
                    # skip aggregation of batch norm layers
                    # update_state[key] = self.mdl.state_dict()[key]
                    update_state[key] = clients_model[0].state_dict()[key]
                else:
                    update_state[key] = torch.divide(sum_weighted_v[key], sum_v[key])
                k = k + 1
            else:
                print("\n* Warning: There are no variances for layer '{0}'. Applying FedAvg for this layer.\n".format(key))
                # Note: we can measure and compare different ways to aggregate these layers
                # Option 1: replace with [untrained] initial weights from global model   
                # update_state[key] = self.mdl.state_dict()[key]
                # Option 2: replace with trained waits from one of the clients. Here the questions is which client to choose? We could try the client with the lowest variability among its layers.
                # update_state[layer_w] = c_mdl.state_dict()[layer_w]
                # Option 3: aggregate them with FedAvg
                update_state[key] = torch.sum(torch.stack([(training_samples[i] * weights[i]) / sum(training_samples) for i in range(len(weights))], dim=0), dim=0)
                # Option 4: can you think of anything else?


        self.mdl.load_state_dict(update_state)


class pw_pred:
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

            optimizers = []
            for pdx in clients_selected:
                print(f"############## client {pdx} ##############")
                self.clients[pdx].replace_mdl(self.server.mdl)
                optimizers.append(self.clients[pdx].train_client(self.client_iters, self.device))

            print("############## server ##############")
            self.server.aggregate_models([self.clients[i].mdl for i in clients_selected],
                                         [len(self.clients[i].train_loader.dataset) for i in clients_selected],
                                         optimizers)

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
