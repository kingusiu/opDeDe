import numpy as np
import datetime
import sys
import pandas as pd
import numpy as np
import math
import os
from collections import namedtuple
import argparse
from sklearn import feature_selection
from sklearn import preprocessing

import src.input_generator as inge
import src.util.runtime_util as rtut
import src.util.string_constants as stco
import src.util.math_util as maut
import src.ml.model as modl


if __name__ == '__main__':

    #*****************************************************#
    #   cmd arguments: number and type of input samples
    #*****************************************************#

    parser = argparse.ArgumentParser(description='read arguments for mutual information training and testing')
    parser.add_argument('-r', dest='run_n', type=int, help='experiment run number', default=1)
    parser.add_argument('-n', dest='N', type=int, help='number of samples', default=int(5e5))
    parser.add_argument('-in', dest='input_type', choices=['calo', 'pid_sum', 'random'], help='type of inputs: calorimeter read from file, toy sensors or random variables', default='pid_sum')

    args = parser.parse_args()


    config = {
        'calo' : stco.configs_calo,
        'pid_sum' : stco.configs_layer_sum,
        'random' : stco.configs_random        
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

    for config_name, params in config[args.input_type].items():

        print(f'running train and test for {config_name}')

        #****************************************#
        #               load data 
        #****************************************#

        A_train, B_train, A_test, B_test = inge.read_inputs_from_df(df, a_label='pid', b_label=params, train_test_split=0.5)

        # import ipdb; ipdb.set_trace()
        
        B_N = B_train.size(1)

        #****************************************#
        #               build model 
        #****************************************#

        # runtime params
        batch_size_min = 512
        batch_size = A_train.size(0) if A_train.size(0) < batch_size_min else batch_size_min
        nb_epochs = 100

        # create model
        model = modl.MI_Model(B_N=B_N)
        model.to(rtut.device)

        #****************************************#
        #               train model 
        #****************************************#

        train_acc_mi = modl.train(model, A_train, B_train, batch_size, nb_epochs)
        train_approx_mi = maut.mutual_info_from_xy(A_train,B_train)
        train_true_mi = feature_selection.mutual_info_regression(A_train.cpu().reshape(-1, 1), B_train.cpu().ravel())[0]

        #****************************************#
        #               test model 
        #****************************************#

        test_acc_mi = modl.test(model, A_test, B_test, batch_size)
        test_approx_mi = maut.mutual_info_from_xy(A_test,B_test)
        test_true_mi = feature_selection.mutual_info_regression(A_test.cpu().reshape(-1, 1), B_test.cpu().ravel())[0]

        #****************************************#
        #               collect results 
        #****************************************#

        result_ll.append([config_name, train_acc_mi, train_approx_mi, train_true_mi, test_acc_mi, test_approx_mi, test_true_mi])

        #****************************************#
        #               output results 
        #****************************************#

        result_str = f'{config_name}: \t train MI {train_acc_mi:.04f} (approx {train_approx_mi:.04f}, true {train_true_mi:.04f}) \t test MI {test_acc_mi:.04f} (approx {test_approx_mi:.04f}, true {test_true_mi:.04f})\n' 


    #****************************************#
    #               save results 
    #****************************************#

    datestr = datetime.datetime.now().strftime('_%Y%m%d')
    result_path = os.path.join(stco.result_dir,'results_MI'+str(run_n)+'_'+args.input_type+datestr+'.pkl')
    print(f'saving results to {result_path}')

    df = pd.DataFrame(result_ll, columns=columns)

    df.to_pickle(result_path)



