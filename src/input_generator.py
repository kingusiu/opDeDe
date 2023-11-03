import math
import torch, torchvision

import src.util as uti

##################################
#       generate inputs
# alpha ... particle shooting angle
# z ... binary hit/no-hit array
##################################


def generate(
        nb,
        r_sensor=0.05,
        nb_sensors=16,
        epsilon=math.pi/10,
        sensor_config='ring',
        single_configuration=True):

    alpha = torch.rand(nb) * 2 * math.pi
    beta = alpha + (torch.rand(nb) - 0.5) * epsilon

    if sensor_config == 'random':
        ## Randomly placed sensors
        x=torch.rand(1 if single_configuration else nb, nb_sensors)*2-1
        y=torch.rand(1 if single_configuration else nb, nb_sensors)*2-1

        # x = torch.tensor([[ 0.3383,  0.6828, -0.8881, -0.1610,  0.1105, -0.5459, -0.1385, -0.6813,
        #     0.8355,  0.2302,  0.0901,  0.0475,  0.7599, -0.6489, -0.6065, -0.6200]])
        
        # y = torch.tensor([[-0.5504, -0.9351, -0.0326, -0.8577,  0.4813, -0.9587, -0.2622, -0.8422,
        #     0.2316,  0.7447, -0.1618, -0.4690,  0.4523,  0.8592,  0.8261, -0.0340]])

    elif sensor_config == 'ring':
        # Sensors placed on a circle
        theta = 2*math.pi/nb_sensors
        x = 0.5*torch.cos(torch.arange(0, nb_sensors) * theta)
        y = 0.5*torch.sin(torch.arange(0, nb_sensors) * theta)

    elif sensor_config == 'linear':
        # Sensors placed on a line
        theta = math.pi/4
        x = (0.1 + torch.arange(0, nb_sensors)) * torch.cos(torch.tensor(theta)) * r_sensor * 2
        y = x

    
    beta=beta[:,None]
    z=((-beta.sin() * x + beta.cos() * y).abs() <= r_sensor).float()

    valid = (((x*beta.cos() + y*beta.sin()) / torch.sqrt(x**2 + y**2)) >= 0)
    z = z * valid

    return alpha.unsqueeze(-1).to(uti.device), z.to(uti.device)



##################################
#       read inputs from file
# x ... true energy
# y ... deposited energy
##################################

def read_inputs(file): 

    df = pd.read_pickle(file_path)

    return df['true_energy'], df['total_dep_energy']