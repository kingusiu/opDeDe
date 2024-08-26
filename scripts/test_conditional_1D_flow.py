from matplotlib import pyplot as plt
import scipy
import torch
import numpy as np
import math
import matplotlib.collections as mc


import minfnet.ml.miniflow_surrogate as miflo


def ctxt_dist_sample(N):
    return np.random.exponential(scale=0.5,size=N)

def LogProba(x, ldj):
    log_p = ldj - 0.5 * (x**2 + math.log(2*math.pi))
    return log_p


def target_dist_sample(N):
    p, std = 0.3, 0.2
    result = torch.empty(N).normal_(0, std)
    result = result + torch.sign(torch.rand(result.size()) - p) / 2
    return result

def target_dist_probs(x):
    p, std = 0.3, 0.2
    mu = (1 - p) * torch.exp(LogProba((x - 0.5) / std, math.log(1 / std))) + \
              p  * torch.exp(LogProba((x + 0.5) / std, math.log(1 / std)))
    return mu

######################################################################

def plot_dist(dist,plt_path,label):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(dist, bins=100, alpha=0.5, label=label)
    ax.legend()
    fig.savefig(plt_path)
##

# Training
n_conditions = 1
nb_samples = 25000
nb_epochs = 5
batch_size = 100

model = miflo.PiecewiseLinear(nb = 1001, xmin = -4, xmax = 4, n_conditions = n_conditions)
model.train()

train_input = target_dist_sample(nb_samples).reshape(-1, 1)
train_cond = torch.tensor(ctxt_dist_sample(nb_samples), dtype=torch.float32).reshape(-1, 1)

result_path = '/afs/cern.ch/user/k/kiwoznia/opde/opDeDe/results/cond_oneD_flow_test'
plot_dist(train_input.numpy().squeeze(), result_path + '/train_input_hist.png',label='train_input')
plot_dist(train_cond.numpy().squeeze(), result_path + '/train_cond_hist.png',label='train_cond')

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

for k in range(nb_epochs):
    acc_loss = 0

    # START_OPTIMIZATION
    for input, cond in zip(train_input.split(batch_size),train_cond.split(batch_size)):
        
        input.requires_grad_()
        output = model(input,cond)

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
    if k%5 == 0: print(k, loss.item())

######################################################################

model.eval()

input = torch.linspace(-3, 3, 175).reshape(-1, 1).float()
cond = torch.tensor(ctxt_dist_sample(len(input)), dtype=torch.float32).reshape(-1, 1)
target_probs = target_dist_probs(input)
base_probs = torch.exp(LogProba(input, 0))

input.requires_grad_()
output = model(input, cond)

grad = torch.autograd.grad(output.sum(), input)[0]
target_probs_hat = LogProba(output, grad.log()).detach().exp() # approximated target distribution probabilities

######################################################################
# FIGURES


input = input.detach().numpy().squeeze()
output = output.detach().numpy().squeeze()
target_probs = target_probs.numpy().squeeze()
target_probs_hat = target_probs_hat.numpy().squeeze()
base_probs = base_probs.numpy().squeeze()


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
lc = mc.LineCollection(lines, color='tab:red', linewidth=0.1)
ax.add_collection(lc)
ax.axis('off')

ax.fill_between(input, 0.52, base_probs * 0.2 + 0.52, color=green_dist)
ax.fill_between(input, -0.30, target_probs * 0.2 - 0.30, color=green_dist)

file_path = result_path + '/miniflow_flow.pdf'
print(f'Saving {file_path}')
fig.savefig(file_path, bbox_inches = 'tight')

# plt.show()

######################################################################

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.axis('off')

ax.fill_between(input, 0, target_probs, color = green_dist)
# ax.plot(input, mu, '-', color = 'tab:blue')
# ax.step(input, mu_hat, '-', where = 'mid', color = 'tab:red')
ax.plot(input, target_probs_hat, '-', color = 'tab:red')

file_path = result_path + '/miniflow_dist.pdf'
print(f'Saving {file_path}')
fig.savefig(file_path, bbox_inches = 'tight')

# plt.show()

######################################################################

