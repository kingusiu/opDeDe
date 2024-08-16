import sys
import pandas as pd
import numpy as np
import os
from collections import namedtuple
import argparse
from sklearn import feature_selection

from minfnet.dats import input_generator as inge
from minfnet.util import runtime_util as ruti
from minfnet.util string_constants as stco
from minfnet.util import math_util as maut
from minfnet.ml import mime as modl



def get_inputs(args, params):

    # create training and test inputs
    if args.input_type == 'calo':
        return inge.read_inputs_from_file(params, b_label='total_dep_energy', train_test_split=0.5)
    else:
        return inge.generate_random_variables(N=args.N, corr=params, train_test_split=0.5)


if __name__ == '__main__':

    #*****************************************************#
    #   cmd arguments: number and type of input samples
    #*****************************************************#

    parser = argparse.ArgumentParser(description='read arguments for mutual information training and testing')
    parser.add_argument('-n', dest='N', type=int, help='number of samples', default=int(5e5))
    parser.add_argument('-in', dest='input_type', choices=['calo', 'sensor', 'random'], help='type of inputs: calorimeter read from file, toy sensors or random variables', default='random')

    args = parser.parse_args()


    config = {
        'calo' : stco.configs_calo,
        'random' : stco.configs_random        
    }

    #*****************************************************#
    #       approximate MI for each config instance
    #*****************************************************#

    result_ll = []
    columns = ['name', 'corr', 'train ml mi', 'train appr mi', 'train true mi', 'test ml mi', 'test appr mi', 'test true mi'] + (['train E res', 'test E res'] if args.input_type == 'calo' else [])

    for config_name, params in config[args.input_type].items():

        print(f'running train and test for {config_name}')

        #****************************************#
        #               load data 
        #****************************************#

        A_train, B_train, A_test, B_test = get_inputs(args, params)
        
        B_N = B_train.size(1)

        #****************************************#
        #               build model 
        #****************************************#

        # runtime params
        batch_size = A_train.size(0) if A_train.size(0) < 256 else 256
        nb_epochs = 10

        # create model
        model = modl.MI_Model(B_N=B_N)
        model.to(ruti.device)

        #****************************************#
        #               train model 
        #****************************************#

        train_acc_mi = modl.train(model, A_train, B_train, batch_size, nb_epochs)
        # import ipdb; ipdb.set_trace()
        train_approx_mi = maut.mutual_info_from_xy(A_train,B_train)
        train_true_mi = feature_selection.mutual_info_regression(A_train.reshape(-1, 1), B_train.ravel())[0]

        #****************************************#
        #               test model 
        #****************************************#

        test_acc_mi = modl.test(model, A_test, B_test, batch_size)
        test_approx_mi = maut.mutual_info_from_xy(A_test,B_test)
        test_true_mi = feature_selection.mutual_info_regression(A_test.reshape(-1, 1), B_test.ravel())[0]

        #****************************************#
        #               collect results 
        #****************************************#

        result_ll.append([config_name, params, train_acc_mi, train_approx_mi, train_true_mi, test_acc_mi, test_approx_mi, test_true_mi])

        #****************************************#
        #               output results 
        #****************************************#

        result_str = f'{config_name}: \t train MI {train_acc_mi:.04f} (approx {train_approx_mi:.04f}, true {train_true_mi:.04f}) \t test MI {test_acc_mi:.04f} (approx {test_approx_mi:.04f}, true {test_true_mi:.04f})' 

        # import ipdb; ipdb.set_trace()
        # add energy resolution
        if args.input_type == 'calo':
            train_e_res = maut.energy_resolution(A_train.numpy(), B_train.numpy())
            test_e_res = maut.energy_resolution(A_test.numpy(), B_test.numpy())
            result_str += f'\t energy res train {train_e_res:.02f} \t energy res test {test_e_res:.02f}'
            result_ll[-1] += [train_e_res, test_e_res]
        
        print(result_str + '\n')


    #****************************************#
    #               save results 
    #****************************************#

    result_path = os.path.join(stco.result_dir,'results_'+args.input_type+'.h5')
    print(f'saving results to {result_path}')

    df = pd.DataFrame(result_ll, columns=columns)

    df.to_pickle(result_path)



