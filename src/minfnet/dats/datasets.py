import numpy as np
import torch
from torch.utils.data import Dataset

class MinfDataset(Dataset):

    def __init__(self, A_var, B_var, thetas):
        self.A = torch.tensor(A_var.reshape(-1, 1), dtype=torch.float32)
        self.B = torch.tensor(B_var.reshape(-1, 1), dtype=torch.float32)
        self.thetas = torch.tensor(thetas.reshape(-1, 1), dtype=torch.float32)
        self.B_perm = self.B[torch.randperm(len(self.B))]

    def __len__(self):
        return len(self.A)
    
    def __getitem__(self, idx):
        return self.A[idx], self.B[idx], self.B_perm[idx], self.thetas[idx]
    

class SurrDataset(Dataset):
    
    def __init__(self, theta, mi):
        self.theta = torch.tensor(theta.astype(np.float32).reshape(-1, 1))
        self.mi = torch.tensor(mi.astype(np.float32).reshape(-1, 1))
    
    def __len__(self):
        return len(self.theta)
        
    def __getitem__(self, idx):
        return self.theta[idx], self.mi[idx]
    
