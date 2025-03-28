import numpy as np
import torch
from torch.utils.data import Dataset

class MinfDataset(Dataset):

    def __init__(self, A_var, B_var):
        self.A = torch.tensor(A_var.reshape(-1, 1), dtype=torch.float32)
        self.B = torch.tensor(B_var.reshape(-1, 1), dtype=torch.float32)
        self.B_perm = self.B[torch.randperm(len(self.B))]

    def __len__(self):
        return len(self.A)
    
    def __getitem__(self, idx):
        return self.A[idx], self.B[idx], self.B_perm[idx]


class MinfCondDataset(MinfDataset):

    def __init__(self, A_var, B_var, thetas):
        super().__init__(A_var, B_var)
        if len(thetas.shape) == 1:
            thetas = thetas.reshape(-1, 1)
        self.thetas = torch.tensor(thetas, dtype=torch.float32)

    def __getitem__(self, idx):
        A, B, B_perm = super().__getitem__(idx)
        return A, B, B_perm, self.thetas[idx]



class InfinityMinfDataset(Dataset):

    def __init__(self, x_fn, theta_fn, y_fn, N):
        self.x_fn = x_fn
        self.theta_fn = theta_fn
        self.y_fn = y_fn
        self.N = N       

    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        x = self.x_fn()
        theta = self.theta_fn()
        y = self.y_fn(x, theta)
        x_wrong = self.x_fn()
        theta_wrong = self.theta_fn()
        y_wrong = self.y_fn(x_wrong, theta_wrong)
        return x, y, y_wrong, theta
    


class SurrDataset(Dataset):
    
    def __init__(self, theta, mi):
        if len(theta.shape) == 1:
            theta = theta.reshape(-1, 1)
        self.theta = torch.tensor(theta.astype(np.float32))
        self.mi = torch.tensor(mi.astype(np.float32).reshape(-1, 1))
    
    def __len__(self):
        return len(self.theta)
        
    def __getitem__(self, idx):
        return self.theta[idx], self.mi[idx]
    
