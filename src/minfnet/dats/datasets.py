import torch
from torch.utils.data import Dataset

class MinfDataset(Dataset):

    def __init__(self, A_var, B_var, thetas):
        self.A = A_var
        self.B = B_var
        self.thetas = thetas
        self.B_perm = self.B[torch.randperm(self.B.size(0))]

    def __len__(self):
        return self.A.size(0)
    
    def __getitem__(self, idx):
        return self.A[idx], self.B[idx], self.B_perm[idx], self.thetas[idx]
    
