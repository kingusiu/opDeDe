import torch
from minfnet.dats import input_generator as inge


def main():

    #****************************************#
    #    runtime params
    #****************************************#

    batch_size = 256
    
    #****************************************#
    #               load data 
    #****************************************#
    N_per_theta = int(1e3)
    B_N = 1

    A_train_all = []
    B_train_all = []
    A_test_all = []
    B_test_all = []
    theta_train_all = []
    theta_test_all = []

    for theta in [0.1, 0.5, 1.0, 1.5, 2.0]:
        A_train, B_train, A_test, B_test, theta_train, theta_test = inge.generate_noisy_channel_samples(N=N_per_theta, noise_std_nominal=theta, train_test_split=0.7)
        A_train_all.append(A_train)
        B_train_all.append(B_train)
        A_test_all.append(A_test)
        B_test_all.append(B_test)
        theta_train_all.append(theta_train)
        theta_test_all.append(theta_test)
# import ipdb; ipdb.set_trace()
    
    dataset_train = inge.MinfDataset(A_var=A_train_all, B_var=B_train_all, thetas=theta_train_all)
    dataset_test = inge.MinfDataset(A_var=A_test_all, B_var=B_test_all, thetas=theta_test_all)

    train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    
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