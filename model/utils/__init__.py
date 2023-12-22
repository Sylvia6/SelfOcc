import torch


class LambdaLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)


class IdentityLayer(torch.nn.Module):
    def __init__(self):
        super(IdentityLayer, self).__init__()
    def forward(self, x):
        return x


activations = {
    'idd': IdentityLayer(),
    'relu': torch.nn.ReLU(inplace=True),
    'sigmoid': torch.nn.Sigmoid(),
    'tanh': torch.nn.Tanh(),
    'none': LambdaLayer(lambda x: x),
    'gelu': torch.nn.GELU(),
}