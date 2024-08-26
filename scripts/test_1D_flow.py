import torch
import math
import matplotlib.pyplot as plt
import matplotlib.collections as mc

import minfnet.ml.miniflow_surrogate as miflo

######################################################################

# bimodal target distribution
def phi(x):
    p, std = 0.3, 0.2
    mu = (1 - p) * torch.exp(LogProba((x - 0.5) / std, math.log(1 / std))) + \
              p  * torch.exp(LogProba((x + 0.5) / std, math.log(1 / std)))
    return mu

# sample from bimodal target distribution
def sample_phi(nb):
    p, std = 0.3, 0.2
    result = torch.empty(nb).normal_(0, std)
    result = result + torch.sign(torch.rand(result.size()) - p) / 2
    return result

######################################################################

def LogProba(x, ldj):
    log_p = ldj - 0.5 * (x**2 + math.log(2*math.pi))
    return log_p


def sample_phi(nb):
    p, std = 0.3, 0.2
    result = torch.empty(nb).normal_(0, std)
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

input = torch.linspace(-3, 3, 175)

mu = phi(input) # true target distribution
mu_N = torch.exp(LogProba(input, 0)) # true base distribution

input.requires_grad_()
output = model(input)

grad = torch.autograd.grad(output.sum(), input)[0]
mu_hat = LogProba(output, grad.log()).detach().exp() # approximated target distribution


######################################################################
# FIGURES

result_path = '/afs/cern.ch/user/k/kiwoznia/opde/opDeDe/results/oneD_flow_test'

input = input.detach().numpy()
output = output.detach().numpy()
mu = mu.numpy()
mu_hat = mu_hat.numpy()


######################################################################

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# ax.set_xlim(-5, 5)
# ax.set_ylim(-5, 5)
# ax.set_aspect('equal')
# ax.axis('off')

ax.plot(input, output, '-', color = 'tab:red')

file_path = result_path + '/miniflow_mapping.pdf'
print(f'Saving {file_path}')
fig.savefig(file_path, bbox_inches = 'tight')

# plt.show()

######################################################################

green_dist = '#bfdfbf'

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# ax.set_xlim(-4.5, 4.5)
# ax.set_ylim(-0.1, 1.1)
lines = list(([(x_in.item(), 0), (x_out.item(), 0.5)]) for (x_in, x_out) in zip(input, output))
lc = mc.LineCollection(lines, color = 'tab:red', linewidth = 0.1)
ax.add_collection(lc)
ax.axis('off')

ax.fill_between(input,  0.52, mu_N * 0.2 + 0.52, color = green_dist)
ax.fill_between(input, -0.30, mu   * 0.2 - 0.30, color = green_dist)

file_path = result_path + '/miniflow_flow.pdf'
print(f'Saving {file_path}')
fig.savefig(file_path, bbox_inches = 'tight')

# plt.show()

######################################################################

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.axis('off')

ax.fill_between(input, 0, mu, color = green_dist)
# ax.plot(input, mu, '-', color = 'tab:blue')
# ax.step(input, mu_hat, '-', where = 'mid', color = 'tab:red')
ax.plot(input, mu_hat, '-', color = 'tab:red')

file_path = result_path + '/miniflow_dist.pdf'
print(f'Saving {file_path}')
fig.savefig(file_path, bbox_inches = 'tight')

# plt.show()

######################################################################
