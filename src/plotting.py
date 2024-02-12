import os
from scipy.stats import multivariate_normal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import src.util as uti
import src.input_generator as inge
import src.string_constants as stco


class Animator():

    def __init__(self,result_df,N,regime):
        self.result_df = result_df
        self.regime = regime
        self.fig, self.axs = plt.subplots(1,2,figsize=(8,4))
        A_train, B_train, A_test, B_test = inge.generate_random_variables(N=N, corr=0)
        self.ll1 = self.axs[0].plot(A_train, B_train, '.')[0]
        self.ll2 = self.axs[1].plot(result_df['train ml mi'][0],label='approx MI')[0]
        self.ll3 = self.axs[1].plot(result_df['train true mi'][0],label='true MI')[0]
        self.axs[0].set_aspect('equal')
        self.axs[1].set(xlim=[0, result_df['corr'].iloc[-1]], ylim=[0,result_df['train ml mi'].max()*1.1], ylabel='MI')
        self.axs[1].legend()
        plt.tight_layout() 

    def animate(self,frame):

        corr = self.result_df['corr'][frame]
        A_train, B_train, A_test, B_test = inge.generate_random_variables(800, corr)
        self.ll1.set_xdata(A_train)
        self.ll1.set_ydata(B_train)
        self.ll2.set_xdata(self.result_df['corr'][:frame+1])
        self.ll2.set_ydata(self.result_df['train ml mi'][:frame+1])
        self.ll3.set_xdata(self.result_df['corr'][:frame+1])
        self.ll3.set_ydata(self.result_df['train true mi'][:frame+1])
        return self.ll1, self.ll2, self.ll3



def animate_distribution_vs_mi(result_path):

    df = pd.read_pickle(result_path)

    N = 800
    regime = 'test'
    animator = Animator(df,N,regime)
    animObj = animation.FuncAnimation(animator.fig, animator.animate, frames=len(df), repeat=False, interval=300, blit=True)

    ff = os.path.join(stco.result_dir,'animated_mi_'+regime+'.gif')
    print('saving training gif to '+ff)
    writergif = animation.PillowWriter(fps=3) 
    animObj.save(ff, writer=writergif)

