import numpy as np

import src.util.string_constants as stco



if __name__ == '__main__':
	

	#****************************************#
    #       plot true E vs sum sensor E 
    #****************************************#

    file_path = os.path.join(stco.input_dir,'Feb24.pkl') 

    # read dataframe with all layer energy running sums once
    df = inge.read_multilayer_calo_file_summed_E(file_path)
    print(f'{len(df)} samples read')

    b_label = 'layer_sum'

    A_train, B_train, _, _ = inge.read_inputs_from_df(df, b_label=b_label)
    A_train, B_train = A_train.cpu().numpy(), B_train.cpu().numpy()

    # plot 2D hist

