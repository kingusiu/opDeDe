import numpy as np
import datetime
import sys
import pandas as pd
import numpy as np
import math
import os
from collections import namedtuple
import argparse
from sklearn import preprocessing
import wandb

import src.input_generator as inge
import src.util.string_constants as stco
import src.ml.train_n_test as trte


if __name__ == '__main__':

    #*****************************************************#
    #   cmd arguments: number and type of input samples
    #*****************************************************#

    parser = argparse.ArgumentParser(description='read arguments for mutual information training and testing')
    parser.add_argument('-r', dest='run_n', type=int, help='experiment run number', default=1)
    parser.add_argument('-n', dest='N', type=int, help='number of samples', default=int(5e5))
    parser.add_argument('-in', dest='input_type', choices=['calo', 'pid_sum', 'random'], help='type of inputs: calorimeter read from file, toy sensors or random variables', default='pid_sum')

    args = parser.parse_args()


    run_config = {
        'calo' : stco.configs_calo,
        'pid_sum' : stco.configs_layer_sum,
        'random' : stco.configs_random        
    }

    #*****************************************************#
    # initialize wandb
    wandb.login()
    exp_conf = {
        'N' : int(5e5),
        'dat': 'pid_sum',
        'eps': 1e-7,
        'lr' : 1e-3,
        'batch_sz' : 512,
        'epochs' : 100,
    }

    #*****************************************************#
    #       approximate MI for each config instance
    #*****************************************************#

    result_ll = []
    columns = ['name', 'train ml mi', 'train appr mi', 'train true mi', 'test ml mi', 'test appr mi', 'test true mi']
    file_path_hadrons = '/eos/home-k/kiwoznia/data/rodem/opde/Feb24.pkl'
    file_path_photons = '/eos/home-k/kiwoznia/data/rodem/opde/Feb24_photons.pkl' 

    # read dataframe with all layer energy running sums once
    df = inge.read_photon_hadron_dataframe(file_path_photons, file_path_hadrons, N_layers=30, sum_layers=True)
    print(f'{len(df)} samples read')

    # normalize all energies (colum wise standard scale)
    #import ipdb; ipdb.set_trace()
    df_pid = df['pid']
    range_min, range_max = 1., 5.
    df = (df-df.min())/(df.max()-df.min()) * (range_max - range_min) + range_min
    df['pid'] = df_pid # reset the last column / not normalized
    # df = pd.DataFrame(preprocessing.StandardScaler().fit_transform(df.values), columns=df.columns)

    for design_i, (design_name, params) in enumerate(run_config[args.input_type].items()):

        with wandb.init(project='mi4opde_'+str(args.run_n)+'_'+design_name, config=exp_conf):

            print(f'running train and test for {design_name}')

            results = trte.run_experiment_per_detector_design(exp_conf,params,df,args.run_n,design_name)
            result_ll.append(results)
            wandb.log({"config_name":results[0], "train_acc_mi":results[1], "train_approx_mi":results[2], \
                "train_true_mi":results[3], "test_acc_mi":results[4], "test_approx_mi":results[5], "test_true_mi":results[6]})

    #****************************************#
    #               save results 
    #****************************************#

    datestr = datetime.datetime.now().strftime('_%Y%m%d')
    result_path = os.path.join(stco.result_dir,'results_MI'+str(args.run_n)+'_'+args.input_type+datestr+'.pkl')
    print(f'saving results to {result_path}')

    df = pd.DataFrame(result_ll, columns=columns)

    df.to_pickle(result_path)



