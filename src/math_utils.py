import numpy as np

def entropy_from_xy(x,y):
	return - np.sum(	)


def energy_resolution(e_true, e_depo):
	# if e_depo is collected from multiple sensors, sum to total deposited energy
	e_depo = np.sum(e_depo,axis=1)
	# return mean of ratio divided by standard deviation of ratio
	ratio = e_depo / e_true
	return np.mean(ratio) / np.std(ratio)