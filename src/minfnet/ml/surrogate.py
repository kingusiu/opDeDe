import torch
import normflows as nf
import torch.nn as nn
from normflows.flows.neural_spline import autoregressive



activation_functions = {
    'elu': nn.ELU(),
    'relu': nn.ReLU()
}

class MLP_Surrogate(nn.Module):
    
    def __init__(self, N_feat=1, activation='elu', N_hidden=10):
        super().__init__()
        self.layer1 = nn.Linear(N_feat, N_hidden)
        self.act1 = activation_functions[activation.lower()]
        self.layer2 = nn.Linear(N_hidden, N_hidden*2)
        self.act2 = activation_functions[activation.lower()]
        self.layer3 = nn.Linear(N_hidden*2, N_hidden)
        self.act3 = activation_functions[activation.lower()]
        self.output = nn.Linear(N_hidden, 1)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.output(x)
        return x


# START_MODEL
class PiecewiseLinear(nn.Module):
    def __init__(self, n_conditions, xmin=0., xmax=20, nb=1000):
        super().__init__()
        self.xmin = xmin
        self.xmax = xmax
        self.nb = nb
        self.alpha = nn.Parameter(torch.tensor([xmin], dtype = torch.float))
        #mu = math.log((xmax - xmin) / nb)
        
        #self.xi = nn.Parameter(torch.empty(nb + 1).normal_(mu, 1e-4))
        self.condition_net = torch.nn.Sequential(
            torch.nn.Linear(n_conditions, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, nb + 1)
        )

    def forward(self, x, conditions):
        '''
        original implementation:
        x torch.Size([100]) : B
        y torch.Size([1002]) : nb+1
        u torch.Size([100]) : B
        n torch.Size([100]) : B
        a torch.Size([100]) : B
        out torch.Size([100]): B
        '''
        #since conditions change, this is now different for each batch element, add zero dimension everywhere
        xi = 0.1 * self.condition_net(conditions)
        #print("xi.shape, x.shape",xi.shape, x.shape)  # B x nb+1
        y = self.alpha + xi.exp().cumsum(1) # 0 -> 1 # B x nb+1
        #print("y.shape",y.shape)
        u = self.nb * (x - self.xmin) / (self.xmax - self.xmin) # B
        #print("u.shape",u.shape) 
        n = u.long().clamp(0, self.nb - 1) # B
        #print("n.shape",n.shape)
        a = (u - n).clamp(0, 1) # B
        #print("a.shape",a.shape)
        y0 = y.gather(1, n)  # Gather y values in dim 1 at indices n
        y1 = y.gather(1, n + 1)  # Gather y values in dim 1 at indices n + 1

        # now we need to use the right batch elment in y
        out = (1 - a) * y0 + a * y1
        
        return out


    def invert(self, y, conditions): #FIXME also w.r.t. dimensions
        # Generate xi from the condition input
        xi = 0.1 * self.condition_net(conditions)
        
        # Calculate ys using the cumulative sum of the exponential of xi
        ys = self.xmin + xi.exp().cumsum(dim=1)
        
        yy = y.view(-1, 1)
        k = torch.arange(self.nb, device=y.device).view(1, -1)
        
        # Ensure y values are within the valid range
        assert (y >= ys[:, 0]).all() and (y <= ys[:, -1]).all()
        
        yk = ys[:, :-1]
        ykp1 = ys[:, 1:]
        
        # Create masks to identify the correct intervals
        masks = (yy >= yk) & (yy < ykp1)
        
        # Calculate the inverse transformation within the identified intervals
        x = self.xmin + (self.xmax - self.xmin) / self.nb * ((masks.float() * (k + (yy - yk) / (ykp1 - yk))).sum(dim=1, keepdim=True))
        
        return x



class Flow_Surrogate(nn.Module):

    def __init__(self, data_dim, ctxt_dim, nodes_n, layers_n, K, tail_bound):

        super(Flow_Surrogate).__init__()

        # FLOW MODEL
        self.K = K

        self.data_dim = data_dim
        self.ctxt_dim = ctxt_dim
        self.nodes_n = nodes_n
        self.layers_n = layers_n
        self.tail_bound = tail_bound

        flows = []

        for i in range(K):
            # MADE                                            
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(self.data_dim, self.layers_n, self.nodes_n, 
                                                                    num_context_channels=self.ctxt_dim, tail_bound=self.tail_bound)]
            flows += [nf.flows.LULinearPermute(self.data_dim)]

        # context encoder
        self.context_encoder = torch.nn.Sequential(
                    torch.nn.Linear(self.ctxt_dim, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, self.data_dim * 2),
                )

        # Set base distribution
        self.q0 = nf.distributions.base.ConditionalDiagGaussian(self.data_dim, context_encoder=self.context_encoder)
        
        self.full_flow = nf.ConditionalNormalizingFlow(self.q0, flows)

    def forward(self, x, context):
        return self.full_flow(x, context)
    
