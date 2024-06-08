import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader

from model import auto_encoder_model
from utils import print_time, calculate_a1_a2


class baseline_pred:
    def __init__(self, train_datasets, valid_datasets, test_datasets, config):
        self.device = torch.device(f'cuda:{config["gpu"]}' if torch.cuda.is_available() else 'cpu')
        self.client_iters = config['client_iterations']
        self.train_loaders = [DataLoader(train_dataset, config['batch_size'], shuffle=False) for train_dataset in train_datasets]
        self.valid_loaders = [DataLoader(valid_dataset, config['batch_size'], shuffle=False) for valid_dataset in valid_datasets]
        self.test_loader = [DataLoader(test_dataset, config['batch_size'], shuffle=False) for test_dataset in test_datasets]
        self.mdl = auto_encoder_model(
            number_features=config['NUM_FEATURES'],
            layer_width=config['layer_width'],
            depth=config['depth'],
            IR_size=config['IR_SIZE'],
            dropout_rate=config['drop_out']
        ).to(self.device)
        self.lr = config['lr']
        self.loss_fn = nn.MSELoss()
        self.opt = optim.Adam(self.mdl.parameters(), lr=self.lr)

    def train(self):
        losses = []
        t_start_time = time.perf_counter()

        # Initialize early stopping parameters
        patience = 10
        early_stopping_counter = 0
        best_validation_loss = float('inf')
        best_model = None

        for idx in range(120):
            i_start_time = time.perf_counter()

            print(f"iteration [{idx + 1}/{120}]")

            self.mdl.train()

            for loader in self.train_loaders:
                for data_in, data_out, mask in loader:
                    data_in = data_in.to(self.device)
                    data_out = data_out.to(self.device)
                    mask = mask.to(self.device)

                    self.opt.zero_grad()

                    scores = self.mdl(data_in)
                    scores = scores * mask
                    data_out = data_out * mask

                    loss = self.loss_fn(scores, data_out)

                    loss.backward()
                    self.opt.step()

            self.mdl.eval()
            val_loss = 0.0
            with torch.no_grad():
                for loader in self.valid_loaders:
                    for data_in, data_out, mask in loader:
                        data_in = data_in.to(self.device)
                        data_out = data_out.to(self.device)
                        mask = mask.to(self.device)
                        
                        output = self.mdl(data_in)
                        val_loss += self.loss_fn(output, data_out)#.item()
                    # server_A1, server_A2 = calculate_a1_a2(self.mdl, [self.validation_loader], device)
                    # print(f"epochs: [{epochs + 1}/{n_epochs}] train_loss: {train_loss:.3f}",
                    #     "A1", round(server_A1, 3), "A2", round(server_A2, 3))
                val_loss /= len(self.valid_loaders)
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

            # i_end_time = time.perf_counter()
            # print_time(i_end_time, i_start_time)

        server_A1, server_A2 = calculate_a1_a2(self.mdl, self.test_loader, self.device)
        print("A1", round(server_A1, 3), "A2", round(server_A2, 3))
        losses.append((server_A1, server_A2))
       
        t_end_time = time.perf_counter()
        print_time(t_end_time, t_start_time)

        return losses, self.mdl
