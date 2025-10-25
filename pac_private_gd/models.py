from torch import nn

class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # initialize with zero weights
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)
    
class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes):
        super().__init__()
        layers = []
        prev_size = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_dim))
        self.network = nn.Sequential(*layers)

        # initialize with zero weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)

    def forward(self, x):
        return self.network(x)
