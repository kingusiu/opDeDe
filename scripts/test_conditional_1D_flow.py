import scipy
import torch

def ctxt_dist_sample(N):
    return scipy.stats.multivariate_normal(3,1.3).rvs(N)

def target_dist_sample(N):
    p, std = 0.3, 0.2
    result = torch.empty(N).normal_(0, std)
    result = result + torch.sign(torch.rand(result.size()) - p) / 2
    return result

######################################################################

# Training

nb_samples = 25000
nb_epochs = 250
batch_size = 100

model = miflo.PiecewiseLinear(nb = 1001, xmin = -4, xmax = 4)
#model.train()

train_input = sample_phi(nb_samples)

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

for k in range(nb_epochs):
    acc_loss = 0

    # START_OPTIMIZATION
    for input in train_input.split(batch_size):
        input.requires_grad_()
        output = model(input)

        derivatives, = torch.autograd.grad(
            output.sum(), input,
            retain_graph = True, create_graph = True
        )

        loss = ( 0.5 * (output**2 + math.log(2*math.pi)) - derivatives.log() ).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # END_OPTIMIZATION

acc_loss += loss.item()
if k%10 == 0: print(k, loss.item())

######################################################################

