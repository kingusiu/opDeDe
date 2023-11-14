import math
import torch, torchvision
import pandas as pd
import numpy as np

import src.util as uti

##############################################
#       generate toy sensor inputs
# alpha ... particle shooting angle
# z ... binary hit/no-hit array
##############################################


def generate_sensor_responses(
        nb,
        r_sensor=0.05,
        B_N=16,
        epsilon=math.pi/10,
        sensor_config='ring',
        single_configuration=True):

    alpha = torch.rand(nb) * 2 * math.pi
    beta = alpha + (torch.rand(nb) - 0.5) * epsilon

    if sensor_config == 'random':
        ## Randomly placed sensors
        x=torch.rand(1 if single_configuration else nb, B_N)*2-1
        y=torch.rand(1 if single_configuration else nb, B_N)*2-1

        # x = torch.tensor([[ 0.3383,  0.6828, -0.8881, -0.1610,  0.1105, -0.5459, -0.1385, -0.6813,
        #     0.8355,  0.2302,  0.0901,  0.0475,  0.7599, -0.6489, -0.6065, -0.6200]])
        
        # y = torch.tensor([[-0.5504, -0.9351, -0.0326, -0.8577,  0.4813, -0.9587, -0.2622, -0.8422,
        #     0.2316,  0.7447, -0.1618, -0.4690,  0.4523,  0.8592,  0.8261, -0.0340]])

    elif sensor_config == 'ring':
        # Sensors placed on a circle
        theta = 2*math.pi/B_N
        x = 0.5*torch.cos(torch.arange(0, B_N) * theta)
        y = 0.5*torch.sin(torch.arange(0, B_N) * theta)

    elif sensor_config == 'linear':
        # Sensors placed on a line
        theta = math.pi/4
        x = (0.1 + torch.arange(0, B_N)) * torch.cos(torch.tensor(theta)) * r_sensor * 2
        y = x

    
    beta=beta[:,None]
    z=((-beta.sin() * x + beta.cos() * y).abs() <= r_sensor).float()

    valid = (((x*beta.cos() + y*beta.sin()) / torch.sqrt(x**2 + y**2)) >= 0)
    z = z * valid

    return alpha.unsqueeze(-1).to(uti.device), z.to(uti.device)


def generate_sensor_inputs(samples_N=int(1e5)):

    config = [0.09, 8, 0., 'ring', 'ring (large)']
    alpha_train, hits_train = generate_sensor_responses(nb=samples_N,r_sensor=config[0],B_N=config[1],epsilon=config[2],sensor_config=config[3])
    alpha_test, hits_test = generate_sensor_responses(nb=samples_N,r_sensor=config[0],B_N=config[1],epsilon=config[2],sensor_config=config[3])

    return alpha_train, hits_train, alpha_test, hits_test


##################################
#       read inputs from file
# x ... true energy
# y ... deposited energy
##################################

def read_inputs_from_file(file_path, b_label='sensor_energy'): 

    # import ipdb; ipdb.set_trace()

    df = pd.read_pickle(file_path)

    A = torch.from_numpy(df['true_energy'].to_numpy(dtype=np.float32)).unsqueeze(-1).to(uti.device)
    B = torch.from_numpy(df[b_label].to_numpy(dtype=np.float32)).to(uti.device)

    return A, B, A, B  # for the moment same dataset for test and train


################################################
#       generate random correlated variables
# x ... true energy
# y ... deposited energy
###############################################

def generate_random_variables(N=int(1e5), corr=0., means=[0.0, 0.0], stds=[1.0, 1.0], train_test_split=None):

    cov = [[stds[0]**2, stds[0]*stds[1]*corr], [stds[0]*stds[1]*corr, stds[1]**2]]
    A, B = np.random.multivariate_normal(means, cov, size=N).T
    A = torch.from_numpy(A).to(uti.device)
    B = torch.from_numpy(B).to(uti.device)

    return A[:train_test_split], B[:train_test_split], A[train_test_split:], B[train_test_split:]

