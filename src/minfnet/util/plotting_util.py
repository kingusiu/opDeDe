import os
from scipy.stats import multivariate_normal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from heputl import logging as heplog

from minfnet.dats import input_generator as inge
from minfnet.util import string_constants as stco

logger = heplog.get_logger(__name__)



class Animator():

    def __init__(self,result_df,N,regime):
        self.result_df = result_df
        self.regime = regime
        self.fig, self.axs = plt.subplots(1,2,figsize=(8,4))
        A_train, B_train, A_test, B_test = inge.generate_random_variables(corr=0, N=N)
        A_train, B_train, A_test, B_test = A_train.cpu().numpy(), B_train.cpu().numpy(), A_test.cpu().numpy(), B_test.cpu().numpy()
        self.ll1 = self.axs[0].plot(A_train, B_train, '.')[0]
        self.ll2 = self.axs[1].plot(result_df['train ml mi'][0],label='approx MI')[0]
        self.ll3 = self.axs[1].plot(result_df['train true mi'][0],label='true MI')[0]
        self.axs[0].set_aspect('equal')
        self.axs[1].set(xlim=[0, result_df['corr'].iloc[-1]], ylim=[0,result_df['train ml mi'].max()*1.1], ylabel='MI')
        self.axs[1].legend()
        plt.tight_layout() 

    def animate(self,frame):

        corr = self.result_df['corr'][frame]
        A_train, B_train, A_test, B_test = inge.generate_random_variables(corr, 800)
        A_train, B_train, A_test, B_test = A_train.cpu().numpy(), B_train.cpu().numpy(), A_test.cpu().numpy(), B_test.cpu().numpy()
        self.ll1.set_xdata(A_train)
        self.ll1.set_ydata(B_train)
        self.ll2.set_xdata(self.result_df['corr'][:frame+1])
        self.ll2.set_ydata(self.result_df['train ml mi'][:frame+1])
        self.ll3.set_xdata(self.result_df['corr'][:frame+1])
        self.ll3.set_ydata(self.result_df['train true mi'][:frame+1])
        return self.ll1, self.ll2, self.ll3



def animate_distribution_vs_mi(result_path,exclude_last=False):

    df = pd.read_pickle(result_path)
    if exclude_last: df = df[:-1]

    N = 800
    regime = 'test'
    animator = Animator(df,N,regime)
    animObj = animation.FuncAnimation(animator.fig, animator.animate, frames=len(df), repeat=False, interval=300, blit=True)

    ff = os.path.join(stco.result_dir,'animated_mi_'+regime+'.gif')
    print('saving training gif to '+ff)
    writergif = animation.PillowWriter(fps=2) 
    animObj.save(ff, writer=writergif)



class MultimodAnimator():

    def __init__(self,result_df,N,min_dat,max_dat):
        self.result_df = result_df
        self.n_designs = len(result_df)
        self.mus = list(stco.configs_multimod.values())
        self.fig, self.axs = plt.subplots(1,2,figsize=(8,4))
        A_train, B_train, A_test, B_test = inge.generate_bimodal_gauss_mixture_samples(self.mus[0], N=N)
        A_train, B_train, A_test, B_test = A_train.cpu().numpy(), B_train.cpu().numpy(), A_test.cpu().numpy(), B_test.cpu().numpy()
        self.ll1 = self.axs[0].plot(A_train, B_train, '.')[0]
        self.ll2 = self.axs[1].plot(result_df['train ml mi'][0],label='approx MI')[0]
        self.ll3 = self.axs[1].plot(result_df['train true mi'][0],label='true MI')[0]
        self.axs[0].set(xlim=[min_dat, max_dat], ylim=[min_dat,max_dat])
        self.axs[0].set_aspect('equal')
        self.axs[1].set(xlim=[0, self.n_designs], ylim=[0,result_df['train ml mi'].max()*1.1], ylabel='MI')
        self.axs[1].legend()
        plt.tight_layout() 

    def animate(self,frame):

        A_train, B_train, A_test, B_test = inge.generate_bimodal_gauss_mixture_samples(self.mus[frame], 1000)
        A_train, B_train, A_test, B_test = A_train.cpu().numpy(), B_train.cpu().numpy(), A_test.cpu().numpy(), B_test.cpu().numpy()
        self.ll1.set_xdata(A_train)
        self.ll1.set_ydata(B_train)
        self.ll2.set_xdata(np.arange(0,self.n_designs)[:frame+1])
        self.ll2.set_ydata(self.result_df['train ml mi'][:frame+1])
        self.ll3.set_xdata(np.arange(0,self.n_designs)[:frame+1])
        self.ll3.set_ydata(self.result_df['train true mi'][:frame+1])
        return self.ll1, self.ll2, self.ll3



def animate_multimod_vs_mi(result_path,exclude_last=False):

    df = pd.read_pickle(result_path)

    N = 800
    min_dat, max_dat = 0., 24.
    animator = MultimodAnimator(df,N, min_dat, max_dat)
    animObj = animation.FuncAnimation(animator.fig, animator.animate, frames=len(df), repeat=False, interval=300, blit=True)

    ff = os.path.join(stco.result_dir,'animated_mi_multimod.gif')
    print('saving training gif to '+ff)
    writergif = animation.PillowWriter(fps=2) 
    animObj.save(ff, writer=writergif)


def plot_inputs_one_theta(A_list, B_list, thetas, plot_name='scatter_plot', fig_dir='results'):

    num_cols = len(thetas)
    fig, axs = plt.subplots(1, num_cols, figsize=(5*num_cols, 5))
    
    for i in range(num_cols):
        axs[i].scatter(A_list[i], B_list[i], marker='.', s=12)
        axs[i].set_aspect(1./axs[i].get_data_ratio())
        # axs[i].set_aspect('equal', adjustable='box')  # Set equal aspect ratio
        axs[i].set_xlabel('sensor energy')
        axs[i].set_ylabel('true energy')
        axs[i].set_title(f'theta={thetas[i]:.03f}')
        #plt.axis('square') 
    
    plot_path = os.path.join(fig_dir, plot_name+'.png')
    logger.info(f'saving plot to {plot_path}')
    plt.savefig(plot_path)
    plt.close(fig)

def plot_inputs_two_theta(A_list, B_list, t1_list, t2_list, plot_name='scatter_plot', fig_dir='results'):

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


def plot_theta_vs_mi(theta, mi, truth=None, scatter_thetas=False, plot_name=None, fig_dir=None):
    # plot theta vs mi
    sorted_indices = np.argsort(theta)
    sorted_theta = theta[sorted_indices]
    sorted_mi = mi[sorted_indices]
    if truth is not None:
        sorted_truth = truth[sorted_indices]

    plt.figure(figsize=(7, 7))
    plt.plot(sorted_theta, sorted_mi, label='mime')
    if truth is not None:
        plt.plot(sorted_theta, sorted_truth, label='true mi')
    if scatter_thetas:
        plt.scatter(sorted_theta, sorted_mi, color='red', marker='>')
    plt.xlabel('Theta')
    plt.ylabel('MI')
    plt.title('Theta vs MI')
    plt.legend()
    if plot_name is not None and fig_dir is not None:
        plt.savefig(f'{fig_dir}/{plot_name}.png')
    # plt.show()
    plt.close()


def plot_results_two_theta(result_ll, plot_name='mi_vs_theta', fig_dir='results'):

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

