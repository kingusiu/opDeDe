
import random
import numpy as np


def make_two_theta_grid(theta_min, theta_max, theta_num):
 
    t1 = np.linspace(theta_min, theta_max, theta_num)
    t2 = np.linspace(1, theta_max, theta_num)
    random.shuffle(t1)
    random.shuffle(t2)
 
    return np.meshgrid(t1, t2)


def sort_results_by_theta(result_arr, sort_by_idx):

    sorted_indices = np.argsort(result_arr[:, sort_by_idx])

    return [aa[sorted_indices] for aa in result_arr.T]
    