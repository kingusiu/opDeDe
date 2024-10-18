import datetime
import os
import numpy as np
import yaml
import wandb
import torch

from minfnet.dats import input_generator as inge
from minfnet.dats import datasets as dase
from minfnet.ml import mime_cond as modl
from minfnet.util import runtime_util as rtut


if __name__ == 'main':

        #****************************************#
    #    runtime params
    #****************************************#

    config_path = '/afs/cern.ch/user/k/kiwoznia/opde/opDeDe/configs/noisy_channel.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    datestr = datetime.datetime.now().strftime('%Y%m%d') + '_run' + str(config['run_n'])
    result_dir = '/afs/cern.ch/user/k/kiwoznia/opde/opDeDe/results/noisy_channel_test/' + datestr
    os.makedirs(result_dir, exist_ok=True)

    # Save config file
    config_save_path = os.path.join(result_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)

    # wandb
    wandb.login()
    wandb.init(project="minfnet")

    #****************************************#
    #               build model 
    #****************************************#

    B_N = 1

    # create model
    model = modl.MI_Model(B_N=B_N, acti=config['activation'], acti_out=config['activation_out'])
    model.to(rtut.device)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    model.train()

    #****************************************#
    #              nominal theta loop   
    #****************************************#

    thetas = np.linspace(config['theta_min'],config['theta_max'],config['theta_step'])

    data_dict = {'A_train': [], 'B_train': [], 'theta_train': []}

    for i,theta in enumerate(thetas):

        #****************************************#
        #               load data   
        #****************************************#


        if config['theta_type'] == 'noisesqr': theta = theta**2
        A_train, B_train, theta_train, *_ = inge.generate_noisy_channel_samples(N=config['n_per_theta'], noise_std_nominal=theta, train_test_split=None)

        data_dict['A_train'].append(A_train)
        data_dict['B_train'].append(B_train)
        data_dict['theta_train'].append(theta_train)

        dataset_train = dase.MinfDataset(A_var=A_train, B_var=B_train, thetas=theta_train)
        train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)

                


