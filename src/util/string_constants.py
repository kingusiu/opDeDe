import os
import numpy as np

#****************************************#
#               configs 
#****************************************#

configs_calo = {
        'good calo': 'data/kinga2_t0.pkl', # 'data/testfile_t0.pkl'
        'bad calo' : 'data/kinga2_badcalo_t0.pkl'
    }

configs_calo_baby = {
        'good calo': 'data/testfile_t0.pkl',
        'bad calo' : 'data/testfile_t0.pkl'
    }

configs_random = {f'corr {corr:.02f}': corr for corr in np.arange(0.,1.05,0.05)}


#****************************************#
#               paths 
#****************************************#

result_dir = 'results'

