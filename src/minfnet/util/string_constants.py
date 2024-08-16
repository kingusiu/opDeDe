import os
import numpy as np

#****************************************#
#               configs 
#****************************************#

configs_calo = {
        'good calo': 'data/Feb24.pkl', # 'data/testfile_t0.pkl'
        'bad calo' : 'data/kinga2_badcalo_t0.pkl'
    }

configs_calo_medium = {
        'good calo': 'data/kinga2_t0.pkl', # 'data/testfile_t0.pkl'
        'bad calo' : 'data/kinga2_badcalo_t0.pkl'
    }

configs_calo_baby = {
        'good calo': 'data/testfile_t0.pkl',
        'bad calo' : 'data/testfile_t0.pkl'
    }

configs_random = {f'corr {corr:.03f}': round(corr, 3) for corr in np.arange(-1.0, 1.05, 0.2)}

N_layers = 30
configs_layer_sum = {f'{i} layers sum':f'sum_{i}L' for i in range(1,N_layers+1)}
configs_layer_multi = {f'{k} layers': [f'E_L{i}' for i in range(1,k+1)] for k in range(1,N_layers+1)}

min_pt, mid_pt, max_pt = 1., 10., 20.
n_dists = 15
mus1 = np.linspace(mid_pt,max_pt,n_dists)
mus1 = list(zip(mus1,np.full(len(mus1), max_pt)))
mus2 = np.linspace(mid_pt,min_pt,n_dists)
mus2 = list(zip(mus2,np.full(len(mus2), max_pt)))
mus3 = np.linspace(max_pt,min_pt,n_dists)
mus3 = list(zip(np.full(len(mus3), mid_pt),mus3))

configs_multimod = {f'mu ({mus[0][0]:0.2f},{mus[0][1]:0.2f}),({mus[1][0]:0.2f},{mus[1][1]:0.2f}),({mus[2][0]:0.2f},{mus[2][1]:0.2f})' : mus for mus in zip(mus1,mus2,mus3)}


#****************************************#
#               paths 
#****************************************#

input_dir = '/eos/home-k/kiwoznia/data/rodem/opde'
result_dir = 'results'

