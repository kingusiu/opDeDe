import wandb
from sklearn import feature_selection

from minfnet.dats import input_generator as inge
from minfnet.ml import model as modl        
from minfnet.util import runtime_util as rtut
from minfnet.util import math_util as maut


def run_experiment_per_detector_design(config,data_config,df,run_n,design_name):

        #****************************************#
        #               load data 
        #****************************************#

        A_train, B_train, A_test, B_test = inge.read_inputs_from_df(df, a_label='pid', b_label=data_config, train_test_split=0.5)

        # import ipdb; ipdb.set_trace()
        
        B_N = B_train.size(1)

        #****************************************#
        #               build model 
        #****************************************#

        model = modl.MI_Model(B_N=B_N)
        model.to(rtut.device)

        #****************************************#
        #               train model 
        #****************************************#

        train_acc_mi = modl.train(model, A_train, B_train, config['batch_sz'], config['epochs'], lr=config['lr'], eps=config['eps'])

        train_approx_mi = maut.mutual_info_from_xy(A_train,B_train)
        train_true_mi = feature_selection.mutual_info_regression(A_train.cpu().reshape(-1, 1), B_train.cpu().ravel())[0]

        #****************************************#
        #               test model 
        #****************************************#

        test_acc_mi = modl.test(model, A_test, B_test, config['batch_sz'], eps=config['eps'])
        test_approx_mi = maut.mutual_info_from_xy(A_test,B_test)
        test_true_mi = feature_selection.mutual_info_regression(A_test.cpu().reshape(-1, 1), B_test.cpu().ravel())[0]


        #****************************************#
        #               output results 
        #****************************************#

        result_str = f'{design_name}: \t train MI {train_acc_mi:.04f} (approx {train_approx_mi:.04f}, true {train_true_mi:.04f}) \t test MI {test_acc_mi:.04f} (approx {test_approx_mi:.04f}, true {test_true_mi:.04f})\n' 

        #****************************************#
        #               collect results 
        #****************************************#

        return [design_name, train_acc_mi, train_approx_mi, train_true_mi, test_acc_mi, test_approx_mi, test_true_mi]



def run_experiment_per_detector_design_multiB(config,data_config,df,run_n,design_name):

        #****************************************#
        #               load data 
        #****************************************#

        A_train, B_train, A_test, B_test = inge.read_inputs_from_df(df, a_label='pid', b_label=data_config, train_test_split=0.5)

        # import ipdb; ipdb.set_trace()
        
        B_N = B_train.size(1)

        #****************************************#
        #               build model 
        #****************************************#

        model = modl.MI_Model(B_N=B_N)
        model.to(rtut.device)

        #****************************************#
        #               train model 
        #****************************************#

        train_acc_mi = modl.train(model, A_train, B_train, config['batch_sz'], config['epochs'], lr=config['lr'], eps=config['eps'])
        # import ipdb; ipdb.set_trace()
        train_true_mi = feature_selection.mutual_info_regression(B_train.cpu(), A_train.cpu().ravel())[0]

        #****************************************#
        #               test model 
        #****************************************#

        test_acc_mi = modl.test(model, A_test, B_test, config['batch_sz'], eps=config['eps'])
        test_true_mi = feature_selection.mutual_info_regression(B_train.cpu(), A_train.cpu().ravel())[0]

        #****************************************#
        #               collect results 
        #****************************************#

        return [design_name, train_acc_mi, train_true_mi, test_acc_mi, test_true_mi]


