import torch

def define_device():

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device('cuda')

    return torch.device('cpu')

# setup runtime
device = define_device()

