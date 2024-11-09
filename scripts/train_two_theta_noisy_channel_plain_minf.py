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
from minfnet.ml import mime as modl
from minfnet.util import runtime_util as rtut
from minfnet.util import plotting_util as plut
from minfnet.util import data_util as datu


logger = heplog.get_logger(__name__)



if __name__ == '__main__':

    #****************************************#
    #    runtime params
    #****************************************#

    config_path = '/home/users/w/wozniak/dev/opde/minf/opDeDe/configs/noisy_channel.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    datestr = datetime.datetime.now().strftime('%Y%m%d') + '_run' + str(config['run_n'])
    result_dir = '/home/users/w/wozniak/dev/opde/minf/opDeDe/results/noisy_channel_test/' + datestr
    os.makedirs(result_dir, exist_ok=True)

    # Save config file
    config_save_path = os.path.join(result_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)

    # wandb
    wandb.login()
    wandb.init(project="minfnet")

    #****************************************#
    #              nominal theta loop   
    #****************************************#

    B_N = 1

    N_per_theta = config['n_per_theta']

    tt1, tt2 = datu.make_two_theta_grid(config['theta_min'],config['theta_max'],config['theta_num'])

    result_ll = []

    data_dict = {'A_train': [], 'B_train': [], 'tt1_train': [], 'tt2_train': []}


    for t1, t2 in zip(tt1.flatten(), tt2.flatten()):
        
        logger.info(f'generating data for t1: {t1:.03f}, t2: {t2:.03f}')

        A_train, B_train, thetas_train, *_ = inge.generate_two_theta_noisy_samples(N=N_per_theta, t1_noise_nominal=t1, t2_damp_nominal=t2)

        data_dict['A_train'].append(A_train)
        data_dict['B_train'].append(B_train)
        data_dict['tt1_train'].append(thetas_train[:,0])
        data_dict['tt2_train'].append(thetas_train[:,1])

        dataset_train = dase.MinfDataset(A_var=A_train, B_var=B_train)
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

        # Compute square root error between train MI and true MI
        sqrt_error = np.sqrt((train_mi - true_mi) ** 2)
        logger.info(f'theta1 {t1:.03f}/ theta2 {t2:.03f}: \t train MI {train_mi:.04f} \t true MI {true_mi:.04f} \t sqrt error {sqrt_error:.04f}')

        result_ll.append([t1, t2, train_mi, true_mi, sqrt_error])

    result_arr = np.array(result_ll)

    plut.plot_inputs_two_theta(data_dict['A_train'], data_dict['B_train'], tt1.flatten(), tt2.flatten(), plot_name='scatter_plot_inputs_train', fig_dir=result_dir)

    #*************************************************************#
    #               plot theta vs MI with one theta fixed
    #*************************************************************#

    # Sort by tt1
    tt1_sort, tt2_sort, train_mi_sort, true_mi_sort, _ = datu.sort_results_by_theta(result_arr, sort_by_idx=0)
    # Slice arrays into chunks of N_theta
    tt1_chunks, tt2_chunks, train_mi_chunks, true_mi_chunks = [np.array_split(aa, config['theta_num']) for aa in [tt1_sort, tt2_sort, train_mi_sort, true_mi_sort]]

    unique_theta1 = np.unique(result_arr[:, 0])

    plut.plot_two_theta_vs_mi_with_one_theta_fixed(tt2_chunks, train_mi_chunks, true_mi_chunks, fixed_thetas=unique_theta1, fixed_theta_name='theta1', \
                                                   xlabel='theta2', plot_name='mi_vs_theta2_train_for_constant_theta1', fig_dir=result_dir)    

    # Sort by tt2
    tt1_sort, tt2_sort, train_mi_sort, true_mi_sort, _ = datu.sort_results_by_theta(result_arr, sort_by_idx=1)
    # Slice arrays into chunks of N_theta
    tt1_chunks, tt2_chunks, train_mi_chunks, true_mi_chunks = [np.array_split(aa, config['theta_num']) for aa in [tt1_sort, tt2_sort, train_mi_sort, true_mi_sort]]

    unique_theta2 = np.unique(result_arr[:, 1])

    plut.plot_two_theta_vs_mi_with_one_theta_fixed(tt1_chunks, train_mi_chunks, true_mi_chunks, fixed_thetas=unique_theta2, fixed_theta_name='theta2', \
                                                   xlabel='theta1', plot_name='mi_vs_theta1_train_for_constant_theta2', fig_dir=result_dir)
    
    logger.info(f'square root error: {np.mean(result_arr[:, 4]):.04f}')
    
    #*************************************************************#
    #               save results and model
    #*************************************************************#

    np.savez(os.path.join(result_dir, 'result_ll_train.npz'), theta1=result_arr[:, 0],theta2=result_arr[:, 1], mi=result_arr[:, 2], true_mi=result_arr[:, 3], sqrt_error=result_arr[:, 4])

    model_path = result_dir+'/disc_model'+datestr+'.pt'
    logger.info('saving model to ' + model_path)
    torch.save(model, model_path)
