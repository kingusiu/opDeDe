import os
import sys
sys.path.insert(0, '..')
import src.plotting as pltt
import src.util.string_constants as stco

if __name__ == '__main__':

    #result_path = os.path.join(stco.result_dir,'results_corr_20240225.pkl')
    #pltt.animate_distribution_vs_mi(result_path,exclude_last=True)

    result_path = os.path.join(stco.result_dir,'results_multimod_20240227.pkl')
    pltt.animate_multimod_vs_mi(result_path,exclude_last=True)