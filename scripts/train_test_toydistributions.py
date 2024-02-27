import sys
import pandas as pd
import numpy as np
import os
from collections import namedtuple
import argparse
import datetime
from sklearn import feature_selection
import matplotlib.pyplot as plt

import src.input_generator as inge
import src.util.runtime_util as rtut
import src.util.string_constants as stco
import src.util.math_util as maut
import src.ml.model as modl


def generate_input(args,params):

    if args.input_type == 'corr':
        return inge.generate_random_variables(corr=params, N=args.N, train_test_split=0.5)
    else:
        return inge.generate_bimodal_gauss_mixture_samples(params, N=args.N, train_test_split=0.5)



if __name__ == '__main__':

    #*****************************************************#
    #   cmd arguments: number and type of input samples
    #*****************************************************#

    parser = argparse.ArgumentParser(description='read arguments for mutual information training and testing')
    parser.add_argument('-n', dest='N', type=int, help='number of samples', default=int(1e5))
    parser.add_argument('-in', dest='input_type', choices=['corr','multimod'], help='type of toy inputs: correlated gauss or multimodal gauss', default='corr')

    args = parser.parse_args()

    config = {
        'corr' : stco.configs_random,
        'multimod' : stco.configs_multimod        
    }

    #*****************************************************#
    #       approximate MI for each config instance
    #*****************************************************#

    result_ll = []
    columns = ['name', 'corr', 'train ml mi', 'train appr mi', 'train true mi', 'test ml mi', 'test appr mi', 'test true mi']

    for config_name, params in config[args.input_type].items():

        print(f'running train and test for {config_name}')

        #****************************************#
        #               load data 
        #****************************************#

        A_train, B_train, A_test, B_test = generate_input(args,params)
        plt.scatter(A_train.cpu().numpy(),B_train.cpu().numpy()) 
        
        B_N = B_train.size(1)

        #****************************************#
        #               build model 
        #****************************************#

        # runtime params
        batch_size = A_train.size(0) if A_train.size(0) < 256 else 256
        nb_epochs = 30

        # create model
        model = modl.MI_Model(B_N=B_N)
        model.to(rtut.device)

        #****************************************#
        #               train model 
        #****************************************#

        train_acc_mi = modl.train(model, A_train, B_train, batch_size, nb_epochs)
        # import ipdb; ipdb.set_trace()
        train_approx_mi = maut.mutual_info_from_xy(A_train,B_train)
        train_true_mi = feature_selection.mutual_info_regression(A_train.cpu().numpy().reshape(-1, 1), B_train.cpu().numpy().ravel())[0]

        #****************************************#
        #               test model 
        #****************************************#

        test_acc_mi = modl.test(model, A_test, B_test, batch_size)
        test_approx_mi = maut.mutual_info_from_xy(A_test,B_test)
        test_true_mi = feature_selection.mutual_info_regression(A_test.cpu().numpy().reshape(-1, 1), B_test.cpu().numpy().ravel())[0]

        #****************************************#
        #               collect results 
        #****************************************#

        corr = params if args.input_type == 'corr' else np.corrcoef(A_test.cpu().numpy().ravel(),B_test.cpu().numpy().ravel())[0,1]

        result_ll.append([config_name, corr, train_acc_mi, train_approx_mi, train_true_mi, test_acc_mi, test_approx_mi, test_true_mi])

        #****************************************#
        #               output results 
        #****************************************#

        result_str = f'{config_name}: \t train MI {train_acc_mi:.04f} (approx {train_approx_mi:.04f}, true {train_true_mi:.04f}) \t test MI {test_acc_mi:.04f} (approx {test_approx_mi:.04f}, true {test_true_mi:.04f})\n' 


    #****************************************#
    #               save results 
    #****************************************#

    datestr = datetime.datetime.now().strftime('_%Y%m%d')
    result_path = os.path.join(stco.result_dir,'results_'+args.input_type+datestr+'.pkl')
    print(f'saving results to {result_path}')

    df = pd.DataFrame(result_ll, columns=columns)

    df.to_pickle(result_path)



