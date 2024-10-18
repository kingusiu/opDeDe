import datetime
import os
import numpy as np
import yaml
import wandb
import torch
from sklearn import feature_selection
from heputl import logging as heplog

from minfnet.dats import input_generator as inge
from minfnet.dats import datasets as dase
from minfnet.ml import mime_cond as modl
from minfnet.util import runtime_util as rtut

logger = heplog.get_logger(__name__)

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

    #****************************************#
    #              nominal theta loop   
    #****************************************#

    thetas = np.linspace(config['theta_min'],config['theta_max'],config['theta_step'])

    data_dict = {'A_train': [], 'B_train': [], 'theta_train': []}

    for i,theta in enumerate(thetas):

        #****************************************#
        #               load data   
        #****************************************#

        logger.info(f'Generating data for theta={theta}')


        if config['theta_type'] == 'noisesqr': theta = theta**2
        A_train, B_train, theta_train, *_ = inge.generate_noisy_channel_samples(N=config['n_per_theta'], noise_std_nominal=theta, train_test_split=None)

        data_dict['A_train'].append(A_train)
        data_dict['B_train'].append(B_train)
        data_dict['theta_train'].append(theta_train)

        dataset_train = dase.MinfDataset(A_var=A_train, B_var=B_train, thetas=theta_train)
        train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)

        #****************************************#
        #               build model
        #****************************************#

        # create model
        model = modl.MI_Model(B_N=B_N, acti=config['activation'], acti_out=config['activation_out'])
        model.to(rtut.device)

        # create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        model.train()
                
        #****************************************#
        #               train model
        #****************************************#

        train_mi = modl.train(model, train_dataloader, config['n_epochs'], optimizer)
        true_mi = feature_selection.mutual_info_regression(A_train.reshape(-1,1), B_train)[0]

        #****************************************#
        #              print results    
        # ****************************************#
        logger.info(f'theta1 {t1:.03f} / theta2 {t2:.03f}: \t train MI {train_mi:.04f} \t true MI {true_mi:.04f}')    
        result_ll.append([t1, t2, train_mi, true_mi])
    
    plot_inputs(data_dict['A_train'], data_dict['B_train'], tt1.flatten(), tt2.flatten(), plot_name='scatter_plot_inputs_train', fig_dir=result_dir)
    
    xlabel = 'Theta/noise level' if 'noise' in config['theta_type'] else 'Theta/correlation'
    plot_results(result_ll, plot_name='mi_vs_theta_train', fig_dir=result_dir)


