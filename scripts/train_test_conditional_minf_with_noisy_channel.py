from minfnet.dats import input_generator as inge


def main():
    
    #****************************************#
    #               load data 
    #****************************************#

    A_train, B_train, A_test, B_test = inge.read_inputs_from_df(df, b_label=params, train_test_split=0.5)

    # import ipdb; ipdb.set_trace()
    
    B_N = B_train.size(1)
    
    #****************************************#
    #               build model 
    #****************************************#

    # runtime params
    batch_size_min = 1024
    batch_size = A_train.size(0) if A_train.size(0) < batch_size_min else batch_size_min
    nb_epochs = 150

    # create model
    model = modl.MI_Model(B_N=B_N)
    model.to(rtut.device)





if __name__ == "__main__":
    main()