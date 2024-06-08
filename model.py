import torch
import torch.nn as nn
import torch.nn.functional as F


class auto_encoder_model(nn.Module):
    def __init__(self, number_features, layer_width, depth, IR_size, dropout_rate):
        super(auto_encoder_model, self).__init__()
        self.number_features = number_features
        self.layer_width = layer_width
        self.depth = depth
        self.IR_size = IR_size
        self.dropout = nn.Dropout(dropout_rate)

        self.encode_layers = nn.ModuleList()
        self.decode_layers = nn.ModuleList()

        # Creating encoding layers
        prev_size = self.number_features
        for _ in range(self.depth):
            self.encode_layers.append(nn.Linear(prev_size, self.layer_width))
            prev_size += self.layer_width
        self.encode_layers.append(nn.Linear(prev_size, self.IR_size))

        # Creating decoding layers
        prev_size = self.IR_size
        for _ in range(self.depth):
            self.decode_layers.append(nn.Linear(prev_size, self.layer_width))
            prev_size += self.layer_width
        self.decode_layers.append(nn.Linear(prev_size, self.number_features))

    def forward(self, x):
        # Forward pass through encoding layers
        encoded = x
        for enc_layer in self.encode_layers[:-1]:
            encoded = torch.cat([encoded, F.relu(enc_layer(self.dropout(encoded)))], dim=1)
        encoded = self.encode_layers[-1](encoded)

        # Forward pass through decoding layers
        decoded = encoded
        for dec_layer in self.decode_layers[:-1]:
            decoded = torch.cat([decoded, F.relu(dec_layer(self.dropout(decoded)))], dim=1)
        return self.decode_layers[-1](decoded)
