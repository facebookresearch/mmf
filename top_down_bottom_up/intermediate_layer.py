import torch.nn as nn

class inter_layer(nn.Module):
    def __init__(self, dim, n_layer):
        super(inter_layer, self).__init__()
        layers=[]
        for i in range(n_layer):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)