import torch.nn as nn


activation_functions = {
    'elu': nn.ELU(),
    'relu': nn.ReLU()
}

class MLP_Surrogate(nn.Module):
    
    def __init__(self, N_feat=1, activation='elu', N_hidden=10):
        super().__init__()
        self.layer1 = nn.Linear(N_feat, N_hidden)
        self.act1 = activation_functions[activation.lower()]
        self.layer2 = nn.Linear(N_hidden, N_hidden*2)
        self.act2 = activation_functions[activation.lower()]
        self.layer3 = nn.Linear(N_hidden*2, N_hidden)
        self.act3 = activation_functions[activation.lower()]
        self.output = nn.Linear(N_hidden, 1)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.output(x)
        return x

