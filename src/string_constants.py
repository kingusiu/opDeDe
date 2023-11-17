import numpy as np

configs_calo = {
        'good calo': 'data/kinga2_t0.pkl',
        'bad calo' : 'data/kinga2_badcalo_t0.pkl'
    }

configs_random = {f'corr {corr:.01f}': corr for corr in np.arange(0.,1.,0.1)}

