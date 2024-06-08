from collections import OrderedDict
from model import auto_encoder_model


class Server:
    def __init__(self, config, device):
        self.mdl = auto_encoder_model(
            number_features=config['NUM_FEATURES'],
            layer_width=config['layer_width'],
            depth=config['depth'],
            IR_size=config['IR_SIZE'],
            dropout_rate=config['drop_out']
        ).to(device)

    def aggregate_models(self, clients_model, client_sizes):
        update_state = OrderedDict()

        all_rows = sum(client_sizes)

        for k, client_model in enumerate(clients_model):
            local_state = client_model.state_dict()
            for i, key in enumerate(self.mdl.state_dict().keys()):
                if k == 0:
                    update_state[key] = local_state[key] * client_sizes[k] / all_rows
                else:
                    update_state[key] += local_state[key] * client_sizes[k] / all_rows

        self.mdl.load_state_dict(update_state)
