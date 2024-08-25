import torch
import numpy as np
import matplotlib.pyplot as plt
from heputl import logging as heplog

from minfnet.dats import datasets as dase
from minfnet.dats import input_generator as inge
from minfnet.ml import mime as modl
import yaml
import minfnet.util.runtime_util as rtut



logger = heplog.get_logger(__name__)

def plot_theta_vs_mi(theta, mi, scatter_thetas=False, plot_name=None, fig_dir=None):
    # plot theta vs mi
    sorted_indices = np.argsort(theta)
    sorted_theta = theta[sorted_indices]
    sorted_mi = mi[sorted_indices]

    plt.figure()
    plt.plot(sorted_theta, sorted_mi)
    if scatter_thetas:
        plt.scatter(sorted_theta, sorted_mi, color='red', marker='>')
    plt.xlabel('Theta')
    plt.ylabel('MI')
    plt.title('Theta vs MI')
    if plot_name is not None and fig_dir is not None:
        plt.savefig(f'{fig_dir}/{plot_name}.png')
    #plt.close()


def generate_constant_inputs(thetas, N_per_theta, theta_type):

    a_tests = []
    b_tests = []
    
    for theta in thetas:
        if theta_type == 'noise':
            A_test, B_test, *_ = inge.generate_noisy_channel_samples(N=N_per_theta, noise_std_nominal=theta, train_test_split=None)
        else:
            A_test, B_test, *_ = inge.generate_random_variables(N=N_per_theta, corr=theta, train_test_split=None)
        a_test_median = np.median(A_test, axis=0)
        b_test_median = np.median(B_test, axis=0)
        
        a_tests.append(a_test_median)
        b_tests.append(b_test_median)
    
    a_tests = torch.tensor(a_tests).reshape(-1, 1).to(rtut.device)
    b_tests = torch.tensor(b_tests).reshape(-1, 1).to(rtut.device)
    thetas = torch.tensor(thetas).reshape(-1,1).to(rtut.device)

    return a_tests, b_tests, thetas


exp_dir = '/afs/cern.ch/user/k/kiwoznia/opde/opDeDe/results/noisy_channel_test/20240821_run5'
minf = torch.load(exp_dir+'/disc_model20240821_run5.pt')

with open(exp_dir+'/config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


logger.info('loading data from ' + exp_dir)
result_ll = np.load(exp_dir+'/result_ll.npz')

# create surrogate dataset
dataset = dase.SurrDataset(result_ll['theta'], result_ll['mi'])


plot_theta_vs_mi(result_ll['theta'], result_ll['mi'])


#****************************************************************#
#                       find theta with best MI
#****************************************************************#

minf.eval()

theta = dataset.theta[np.random.randint(len(dataset.theta))].to(rtut.device)
theta_bounds = (min(dataset.theta), max(dataset.theta))
step_sz = torch.tensor(15.).to(rtut.device)
thetas = []
i = 0
N_per_theta = int(1e3)


while theta.item() >= theta_bounds[0] and theta.item() <= theta_bounds[1]:

    thetas.append(theta.item())
    logger.info(f'theta {i}: {theta.item():.3f}')

    # generate inputs
    if config['theta_type'] == 'noise':
        a_test, b_test, theta_test, *_ = inge.generate_noisy_channel_samples(N=N_per_theta, noise_std_nominal=theta.item(), train_test_split=None)
    else:
        a_test, b_test, *_ = inge.generate_random_variables(N=N_per_theta, corr=theta.item(), train_test_split=None)
        theta_test = np.random.normal(loc=theta, scale=0.1, size=a_test.shape)

    inputs = dase.MinfDataset(A_var=a_test, B_var=b_test, thetas=theta_test)
    dataloader = torch.utils.data.DataLoader(inputs, batch_size=N_per_theta, shuffle=False)

    batch = next(iter(dataloader))
    aa, bb, bbper, tt = [b.to(rtut.device) for b in batch]
    tt.requires_grad = True

    dep_ab = minf(aa,bb,tt)
    indep_ab = minf(aa,bbper,tt)
    loss = -modl.mutual_info(dep_ab=dep_ab, indep_ab=indep_ab)
    #loss.backward()
    grad = torch.autograd.grad(outputs=loss, inputs=tt, retain_graph=True)[0] # check for other tuple elememnts!
    theta = theta + step_sz * grad.mean() # increase mi (gradient pointing towards decreasing negative mi)
    i += 1
    if abs(theta.item() - thetas[-1]) < 1e-7:
        break   

# evaluate iterated thetas for their mi value

mis = []

for theta in thetas:

    if config['theta_type'] == 'noise':
        a_test, b_test, theta_test, *_ = inge.generate_noisy_channel_samples(N=N_per_theta, noise_std_nominal=theta, train_test_split=None)
    else:
        a_test, b_test, *_ = inge.generate_random_variables(N=N_per_theta, corr=theta.item(), train_test_split=None)
        theta_test = np.random.normal(loc=theta, scale=0.1, size=a_test.shape)

    dataset = dase.MinfDataset(A_var=a_test, B_var=b_test, thetas=theta_test)

    dep_ab = minf(dataset.A.to(rtut.device), dataset.B.to(rtut.device), dataset.thetas.to(rtut.device))
    indep_ab = minf(dataset.A.to(rtut.device), dataset.B_perm.to(rtut.device), dataset.thetas.to(rtut.device))
    mi = modl.mutual_info(dep_ab=dep_ab, indep_ab=indep_ab)

    mis.append(mi.item())

thetas_np = np.array(thetas)
mis_np = np.array(mis)

plot_theta_vs_mi(thetas_np, mis_np, scatter_thetas=True, plot_name='theta_vs_mi_descent_N'+str(N_per_theta)+'_largeStep', fig_dir=exp_dir)