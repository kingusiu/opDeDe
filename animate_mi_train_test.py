import os
import sys
sys.path.insert(0, '..')
import src.plotting as pltt
import src.string_constants as stco

if __name__ == '__main__':

    result_path = os.path.join(stco.result_dir,'results_random.h5')
    pltt.animate_distribution_vs_mi(result_path)