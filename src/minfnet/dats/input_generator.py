import math
import torch, torchvision
import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import multivariate_normal

from minfnet.util import runtime_util as rtut
# import src.util.runtime_util as rtut

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

    return alpha.unsqueeze(-1).to(rtut.device), z.to(rtut.device)


def generate_sensor_inputs(samples_N=int(1e5)):

    config = [0.09, 8, 0., 'ring', 'ring (large)']
    alpha_train, hits_train = generate_sensor_responses(nb=samples_N,r_sensor=config[0],B_N=config[1],epsilon=config[2],sensor_config=config[3])
    alpha_test, hits_test = generate_sensor_responses(nb=samples_N,r_sensor=config[0],B_N=config[1],epsilon=config[2],sensor_config=config[3])

    return alpha_train, hits_train, alpha_test, hits_test


def calc_train_test_split_N(N,train_test_split_share):
    if train_test_split_share:
        return int(N*train_test_split_share)
    return None


###########################################
#           read inputs from df
#   df                  ... dataframe with random variables A (R1) and B (RN, N=1..M)
#   a_label             ... label of variable A in df
#   b_label             ... label of variable B in df
#   train_test_split    ... float indicating train share or None, if None: train-sample == test-sample
###########################################

def read_inputs_from_df(df, a_label='true_energy', b_label='sensor_energy', train_test_split=None): 

    # import ipdb; ipdb.set_trace()

    train_test_split = calc_train_test_split_N(len(df),train_test_split)

    A = torch.from_numpy(df[a_label].to_numpy(dtype=np.float32)).unsqueeze(-1).to(rtut.device)
    B = torch.from_numpy(df[b_label].to_numpy(dtype=np.float32))
    B = B.to(rtut.device) if type(b_label) is list else B.unsqueeze(-1).to(rtut.device)

    return A[:train_test_split], B[:train_test_split], A[train_test_split:], B[train_test_split:]

def add_cumul_sum_layers(df,N_layers):

    sums = np.asarray([[sum(x[:i]) for x in df['sensor_energy']] for i in range(1,N_layers+1)])
    sum_col_names = [f'sum_{i}L' for i in range(1,N_layers+1)] 
    df[sum_col_names] = sums.T

    return df


# A in R^1, B in R^1 (energy sum)
def read_multilayer_calo_file_summed_E(file_path,N_layers=30):

    df = pd.read_pickle(file_path)
    df = df[['true_energy','total_dep_energy','sensor_energy']]

    df = add_cumul_sum_layers(df,N_layers)
    df.drop(['sensor_energy'],axis=1)

    return df

def add_per_layer_E_columns(df,N_layers):

    col_names = [f'E_L{i}' for i in range(1,N_layers+1)]
    df[col_names] = df['sensor_energy'].to_list()
    
    return df


# A in R^1, B in R^num_layers
def read_multilayer_calo_file_E_per_layer(file_path,N_layers=30):

    df = pd.read_pickle(file_path)
    df = df[['true_energy','total_dep_energy','sensor_energy']]

    df = add_per_layer_E_columns(df, N_layers)
    df.drop(['sensor_energy'],axis=1) 

    return df


def read_photon_hadron_dataframe(file_path_photons, file_path_hadrons, N_layers=30, sum_layers=True):

    df_photons = pd.read_pickle(file_path_photons)
    df_hadrons = pd.read_pickle(file_path_hadrons)

    df_photons.drop(['sensor_x', 'sensor_y', 'sensor_z', 'sensor_dx', 'sensor_dy', 'sensor_dz','sensor_copy_number'],axis=1)
    df_hadrons.drop(['sensor_x', 'sensor_y', 'sensor_z', 'sensor_dx', 'sensor_dy', 'sensor_dz','sensor_copy_number'],axis=1)

    df_photons['pid'] = 0
    df_hadrons['pid'] = 1

    df_all = pd.concat([df_photons, df_hadrons], ignore_index=True)

    if sum_layers == True:
        df = add_cumul_sum_layers(df_all,N_layers)
    else:
        df = add_per_layer_E_columns(df_all,N_layers)

    # shuffle
    df = df.sample(frac = 1)

    return df


################################################
#       generate random correlated variables
# x ... true energy
# y ... deposited energy
###############################################

def generate_random_variables(corr: float = 0., N: int = int(1e5), means: list = [0.0, 0.0], stds: list = [1.0, 1.0], train_test_split: float | None = None):

    train_test_split = calc_train_test_split_N(N,train_test_split)

    cov = [[stds[0]**2, stds[0]*stds[1]*corr], [stds[0]*stds[1]*corr, stds[1]**2]]
    normal = multivariate_normal(means, cov, allow_singular=True) 
    A, B = normal.rvs(size=N).astype(np.float32).T

    return A[:train_test_split], B[:train_test_split], A[train_test_split:], B[train_test_split:]



###############################################################
#       generate random bi-modal gaussian mixture variables
# x ... true energy
# y ... deposited energy
##############################################################

def samples_from_multivariate_multimodal_gaussian(mus: list | np.ndarray, covs: list | np.ndarray, N_samples: int = 100) -> np.ndarray:

    # set up distribution

    N_dims = len(mus[0])
    N_modes = len(mus)

    mixtures = [scipy.stats.multivariate_normal(mus[i], covs[i]) for i in range(N_modes)]

    # generate samples

    pick_mode = np.random.choice(N_modes, N_samples)
    N_samples_per_mode = [sum(pick_mode == i) for i in range(N_modes)]

    samples_per_mode = [mixtures[i].rvs(N_samples_per_mode[i]) for i in range(N_modes)]
    samples = np.concatenate(samples_per_mode)
    np.random.shuffle(samples)

    return samples


def generate_bimodal_gauss_mixture_samples(mus, N=int(1e5), train_test_split=None):

    train_test_split = calc_train_test_split_N(N,train_test_split)

    covs = [np.eye(2)]*3
    samples = samples_from_multivariate_multimodal_gaussian(mus, covs, N)
    A, B = samples[:,0].astype(np.float32), samples[:,1].astype(np.float32)
    A = torch.from_numpy(A).unsqueeze(-1).to(rtut.device)
    B = torch.from_numpy(B).unsqueeze(-1).to(rtut.device)

    return A[:train_test_split], B[:train_test_split], A[train_test_split:], B[train_test_split:]


###############################################################
#       generate noisy channel variables
# x ... true signal sent
# y ... signal received
##############################################################

def generate_noisy_channel_samples(N=int(1e5), noise_std_nominal=0.1, train_test_split=None):

    idx = calc_train_test_split_N(N,train_test_split)

    in_sig = np.linspace(0, 1, N).astype(np.float32)*4.0
    noise_std = np.abs(np.random.normal(noise_std_nominal,0.05,N)).astype(np.float32)
    noise = np.random.normal(0, noise_std, N).astype(np.float32)
    out_sig = in_sig + noise

    # return x,y,noise(=theta) for train and test
    return in_sig[:idx], out_sig[:idx], noise_std[:idx], in_sig[idx:], out_sig[idx:], noise_std[idx:]

def generate_two_theta_noisy_samples(N=int(1e5), t1_noise_nominal=0.1, t2_damp_nominal=1.1, train_test_split=None):

    idx = calc_train_test_split_N(N,train_test_split)

    in_sig = np.linspace(0, 1, N).astype(np.float32)*4.0

    noise_std = np.abs(np.random.normal(t1_noise_nominal,0.05,N)).astype(np.float32)
    noise = np.random.normal(0, noise_std, N).astype(np.float32)
    
    damp = np.abs(np.random.normal(t2_damp_nominal,0.05,N)).astype(np.float32)

    out_sig = in_sig + noise - np.log(damp)*in_sig

    return in_sig[:idx], out_sig[:idx], noise_std[:idx], damp[:idx], in_sig[idx:], out_sig[idx:], noise_std[idx:], damp[idx:]
