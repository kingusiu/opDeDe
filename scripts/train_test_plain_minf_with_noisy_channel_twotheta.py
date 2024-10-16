import datetime
import os
import numpy as np
import torch
from sklearn import feature_selection
import wandb
import yaml

from minfnet.dats import input_generator as inge
from minfnet.dats import datasets as dase
from minfnet.ml import mime_cond as modl
from minfnet.util import runtime_util as rtut
from minfnet.util import string_constants as stco

from heputl import logging as heplog
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


logger = heplog.get_logger(__name__)

corrs_ll = lambda tmin,tmax,tstep: {f'corr {corr:.03f}': round(corr, 3) for corr in np.arange(tmin, tmax, tstep)}

def plot_inputs(A_list, B_list, t1_list, t2_list, plot_name='scatter_plot', fig_dir='results'):

    num_rows_cols = int(np.sqrt(len(t1_list)))
    fig, axs = plt.subplots(num_rows_cols, num_rows_cols, figsize=(6*len(t1_list), 8*len(t2_list)))
    
    for i in range(num_rows_cols):
        for j in range(num_rows_cols):
            idx = i * num_rows_cols + j
            axs[i, j].scatter(A_list[idx], B_list[idx])
            axs[i, j].set_xlabel('A')
            axs[i, j].set_ylabel('B')
            axs[i, j].set_title(f'theta1={t1_list[idx]:.03f}, theta2={t2_list[idx]:.03f}')
    
    plot_path = os.path.join(fig_dir, plot_name+'.png')
    logger.info(f'saving plot to {plot_path}')
    plt.savefig(plot_path)


def plot_histogram(thetas, thetas_nominal, plot_name='theta_histogram', fig_dir='results'):
    num_plots = len(thetas)
    fig, axs = plt.subplots(1, num_plots, figsize=(6*num_plots,8))
    for i, (theta, theta_nominal) in enumerate(zip(thetas, thetas_nominal)):
        axs[i].hist(theta, bins=50, alpha=0.5, label='theta distribution')
        axs[i].axvline(x=theta_nominal, color='red', linestyle='--', label=f'theta={theta_nominal:.03f}')
        axs[i].set_xlabel('Theta')
        axs[i].set_ylabel('Frequency')
        axs[i].legend()
    plot_path = os.path.join(fig_dir, plot_name+'.png')
    logger.info(f'saving histogram plot to {plot_path}')
    plt.savefig(plot_path)


def plot_results(result_ll, plot_name='mi_vs_theta', fig_dir='results'):

    result_ll = np.array(result_ll)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = result_ll[:, 0]
    y = result_ll[:, 1]
    z = result_ll[:, 2]

    ax.plot_trisurf(x, y, z, cmap='viridis')

    ax.set_xlabel('theta1: noise')
    ax.set_ylabel('theta2: damp')
    ax.set_zlabel('approx MI')

    logger.info(f'saving results plot to {fig_dir}/{plot_name}')
    plt.savefig(fig_dir + '/' + plot_name+'.png')



def make_two_theta_grid(theta_min, theta_max, theta_num):
    t1 = np.linspace(theta_min, theta_max, theta_num)
    t2 = np.linspace(1, theta_max, theta_num)
    random.shuffle(t1)
    random.shuffle(t2)
    tt1,tt2 = np.meshgrid(t1, t2)
    return tt1, tt2




def main():

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
    #               load data 
    #****************************************#
    N_per_theta = config['n_per_theta']

    tt1, tt2 = make_two_theta_grid(config['theta_min'],config['theta_max'],config['theta_step'])

    result_ll = []

    data_dict = {'A_train': [], 'B_train': [], 'tt1_train': [], 'tt2_train': []}

    for t1, t2 in zip(tt1.flatten(), tt2.flatten()):
        
        logger.info(f'generating data for t1: {t1:.03f}, t2: {t2:.03f}')

        A_train, B_train, thetas_train, *_ = inge.generate_two_theta_noisy_samples(N=N_per_theta, t1_noise_nominal=t1, t2_damp_nominal=t2)

        data_dict['A_train'].append(A_train)
        data_dict['B_train'].append(B_train)
        data_dict['tt1_train'].append(thetas_train[:,0])
        data_dict['tt2_train'].append(thetas_train[:,1])

        dataset_train = dase.MinfDataset(A_var=A_train, B_var=B_train, thetas=thetas_train)
        train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)

        #****************************************#
        #               train model
        #****************************************#
        train_acc_mi = modl.train(model, train_dataloader, config['n_epochs'], optimizer)
        train_true_mi = feature_selection.mutual_info_regression(A_train.reshape(-1,1), B_train)[0]

        #****************************************#
        #              print results    
        # ****************************************#
        logger.info(f'theta1 {t1:.03f} / theta2 {t2:.03f}: \t train MI {train_acc_mi:.04f} \t true MI {train_true_mi:.04f}')    
        result_ll.append([t1, t2, train_acc_mi, train_true_mi])
    
    plot_inputs(data_dict['A_train'], data_dict['B_train'], tt1.flatten(), tt2.flatten(), plot_name='scatter_plot_inputs_train', fig_dir=result_dir)
    
    xlabel = 'Theta/noise level' if 'noise' in config['theta_type'] else 'Theta/correlation'
    plot_results(result_ll, plot_name='mi_vs_theta_train', fig_dir=result_dir)
    plot_histogram(data_dict['tt1_train'], tt1.flatten(), plot_name='t1_train_histogram', fig_dir=result_dir)
    plot_histogram(data_dict['tt1_train'], tt2.flatten(), plot_name='t2_train_histogram', fig_dir=result_dir)

    result_ll = np.array(result_ll)
    np.savez(os.path.join(result_dir, 'result_ll_train.npz'), theta1=result_ll[:, 0],theta2=result_ll[:, 1], mi=result_ll[:, 2])

    model_path = result_dir+'/disc_model'+datestr+'.pt'
    logger.info('saving model to ' + model_path)
    torch.save(model, model_path)


    N_per_theta = config['n_per_theta']
    result_ll = []
    tt1_test, tt2_test = make_two_theta_grid(config['theta_min'], config['theta_max'], config['theta_step'])
    data_dict = {'A_test': [], 'B_test': [], 'tt1_test': [], 'tt2_test': []}
    for t1, t2 in zip(tt1_test.flatten(), tt2_test.flatten()):
        logger.info(f'generating data for t1: {t1:.03f}, t2: {t2:.03f}')
        A_test, B_test, thetas_test, *_ = inge.generate_two_theta_noisy_samples(N=N_per_theta, t1_noise_nominal=t1, t2_damp_nominal=t2)
        data_dict['A_test'].append(A_test)
        data_dict['B_test'].append(B_test)
        data_dict['tt1_test'].append(thetas_test[:, 0])
        data_dict['tt2_test'].append(thetas_test[:, 1])

        dataset_test = dase.MinfDataset(A_var=A_test, B_var=B_test, thetas=thetas_test)
        test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=config['batch_size'], shuffle=False)
        #****************************************#
        #               test model
        #****************************************#
        test_acc_mi = modl.test(model, test_dataloader)
        test_true_mi = feature_selection.mutual_info_regression(A_test.reshape(-1, 1), B_test)[0]
        wandb.log({"test mi": test_acc_mi})
        #****************************************#
        #              print results    
        # ****************************************#
        logger.info(f'theta1 {t1:.03f} / theta2 {t2:.03f}: \t test MI {test_acc_mi:.04f} \t true MI {test_true_mi:.04f}')    
        result_ll.append([t1, t2, test_acc_mi, test_true_mi])

    plot_inputs(data_dict['A_test'], data_dict['B_test'], tt1_test.flatten(), tt2_test.flatten(), plot_name='scatter_plot_inputs_test', fig_dir=result_dir)
    plot_results(result_ll, plot_name='mi_vs_theta_test', fig_dir=result_dir)
    plot_histogram(data_dict['tt1_test'], tt1_test.flatten(), plot_name='t1_test_histogram', fig_dir=result_dir)
    plot_histogram(data_dict['tt2_test'], tt2_test.flatten(), plot_name='t2_test_histogram', fig_dir=result_dir)

    result_ll = np.array(result_ll)
    np.savez(os.path.join(result_dir, 'result_ll_test.npz'), theta1=result_ll[:, 0],theta2=result_ll[:, 1], mi=result_ll[:, 2])


if __name__ == "__main__":
    main()