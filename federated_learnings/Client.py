import torch.nn as nn
import torch.optim as optim
import copy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch

from model import auto_encoder_model
from utils import calculate_a1_a2


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

        return opt.state

    def replace_mdl(self, server_mdl):
        self.mdl = copy.deepcopy(server_mdl)
