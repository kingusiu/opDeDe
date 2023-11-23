import numpy as np

def entropy_from_probs(probs):
	return -np.sum(probs * np.log(probs, out=np.zeros(probs.shape), where=(probs!=0))) / np.log(2)

def mutual_info_from_xy(x,y,bins=10):
	p_x, *_ = np.histogram(x,bins=bins,density=True) 
	p_y, *_ = np.histogram(y,bins=bins,density=True) 
	p_xy, *_ = np.histogram2d(x, y, bins=bins, density=True)
	return entropy_from_probs(p_x) + entropy_from_probs(p_y) - entropy_from_probs(p_xy)


def energy_resolution(e_true, e_depo):
	# if e_depo is collected from multiple sensors, sum to total deposited energy
	# import ipdb; ipdb.set_trace()
	e_depo = np.sum(e_depo,axis=1).squeeze()
	# return mean of ratio divided by standard deviation of ratio
	ratio = e_depo / e_true.squeeze()
	return np.mean(ratio) / np.std(ratio)

