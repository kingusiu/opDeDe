import os
from scipy.stats import multivariate_normal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from minfnet.dats import input_generator as inge
from minfnet.util import string_constants as stco


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

