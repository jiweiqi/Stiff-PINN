# -*- coding: utf-8 -*-

import numpy as np
import torch
from tqdm import tqdm

from _utils import grad_norm, plot_pinn_y, to_np, K, ODE, get_solution, get_param_grad, grad_mean
from config import cuda_index, device, default_tensor_type
from pinn_model import PINN_Model

torch.set_default_tensor_type(default_tensor_type)
np.random.seed(2)
torch.manual_seed(2)

is_restart = False
use_annealing = False
n_grid_train = 1000
n_grid_test = 24
learning_rate = 1e-3
grad_max = 1e0

num_epochs = 100000
printout_freq = 100

# Solving Robertson
ode = ODE(3)

y_list = []
dydt_list = []
np.random.seed(0)

t_end = 1e5
n_steps = n_grid_train

t_np = np.logspace(start=-5, stop=np.log10(t_end), num=n_steps, endpoint=True)
n_steps = t_np.shape[0]

y0 = np.array([1.0, 0.0, 0.0])
y = get_solution(ode, y0, t_end, n_steps, t_np)
y_list.append(y[1:, :])

y_np = np.vstack(y_list)

np.savez('./Datasets/Robertson_PINN Dataset.npz', t=t_np, y=y_np)


# Training PINN
t_true = torch.from_numpy(t_np)
t_true.to(device=device)

y_true = torch.from_numpy(y_np)
y_true.to(device=device)

n_var = 3

# initial condition
y0 = y_true[0, :].view(-1, n_var).to(device=device)

# Scaling factor
y_scale = y_true.max(dim=0).values.to(device=device)
x_scale = t_true.max(dim=0).values.to(device=device)
w_res = torch.ones(n_var).to(device=device) * x_scale / y_scale
w_scale = torch.ones(n_var).to(device=device) * y_scale


checkpoint_path = 'models/robertson_stiff.pt'

net = PINN_Model(nodes=800, layers=3, y0=y0, w_scale=w_scale,
                 x_scale=x_scale).to(device=device)

criterion = torch.nn.L1Loss()

# def criterion(x, y, c=0.1):

#     out = torch.pow((x - y).abs(), c)
#     return out.mean()


optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                             weight_decay=1e-4)

loss_list = {}
key_list_loss = ['res_train', 'res_test',
                 'grad_norm', 'slope', 'res_0', 'res_1', 'res_2']
for key in key_list_loss:
    loss_list[key] = []

epoch_old = 0

# prepare data
t_end = t_true.max().item()
eps = 1e-30

# sampling equally in log-scale
x_train = torch.pow(10, (torch.rand(n_grid_train, device=device).unsqueeze(-1) - 0.5) * 10)  # noqa: E501
x_test = torch.pow(10, (torch.rand(n_grid_test, device=device).unsqueeze(-1) - 0.5) * 10)  # noqa: E501

if is_restart is True:
    checkpoint = torch.load(checkpoint_path + '.tar', map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_old = checkpoint['epoch']
    loss_list = checkpoint['loss_list']
    x_train = torch.Tensor(checkpoint['x_train']).to(device)
    x_test = torch.Tensor(checkpoint['x_test']).to(device)

# for plotting
sorted_index = x_train.sort(dim=0).indices.view(-1)

x_all = torch.cat([x_train, x_test], dim=0)
x_all.requires_grad = True
# x_all_repeat = x_all.repeat(n_var, 1)
# x_all_repeat.requires_grad = True

# n_total = n_grid_train + n_grid_test

# y_jac_indice = torch.empty([n_total * n_var, 2]).long()
# for i in range(n_var):
#     y_jac_indice[n_total * i:n_total *
#                  (i + 1), 0] = torch.arange(0, n_total) + n_total * i
#     y_jac_indice[n_total * i:n_total * (i + 1), 1] = i

loss_res_train = torch.Tensor(3).to(device=device)

for epoch in tqdm(range(num_epochs)):
    if is_restart:
        if epoch < epoch_old:
            continue

    # y_all_repeat = net(x_all_repeat)

    # use jacobian tricks
    # dydt_all = torch.autograd.grad(outputs=y_all_repeat[y_jac_indice[:, 0],
    #                                                     y_jac_indice[:, 1]].sum(),
    #                                inputs=x_all_repeat,
    #                                retain_graph=True,
    #                                create_graph=True,
    #                                allow_unused=True)[0].view(n_var, -1).T

    # y_all = y_all_repeat[:n_total]

    y_all = net(x_all)
    shape = y_all.shape

    rhs_all = torch.Tensor(shape).to(device=device)
    dydt_all = torch.Tensor(shape).to(device=device)

    for i in range(3):
        dydt_all[:, i] = torch.autograd.grad(outputs=y_all[:, i].sum(),
                                             inputs=x_all,
                                             retain_graph=True,
                                             create_graph=True,
                                             allow_unused=True)[0].view(1, -1)

    rhs_all[:, 0] = -K[0] * y_all[:, 0] + K[2] * y_all[:, 1] * y_all[:, 2]
    rhs_all[:, 1] = K[0] * y_all[:, 0] - K[2] * y_all[:, 1] * y_all[:, 2] \
        - K[1] * y_all[:, 1] * y_all[:, 1]
    rhs_all[:, 2] = K[1] * y_all[:, 1] * y_all[:, 1]

    y_train = y_all[:n_grid_train, :]
    y_test = y_all[n_grid_train:, :]
    rhs_train = rhs_all[:n_grid_train, :]
    rhs_test = rhs_all[n_grid_train:, :]
    dydt_train = dydt_all[:n_grid_train, :]
    dydt_test = dydt_all[n_grid_train:, :]

    loss_train = criterion(dydt_train, rhs_train)
    loss_test = criterion(dydt_test, rhs_test)

    optimizer.zero_grad()
    loss_train.backward()

    grad_sum = grad_norm(net)

    torch.nn.utils.clip_grad_norm_(
        net.parameters(), max_norm=grad_max, norm_type=2.0)

    optimizer.step()

    slope = net.get_slope()

    loss_list['res_train'].append(loss_train.item())
    loss_list['res_test'].append(loss_test.item())
    loss_list['res_0'].append(loss_res_train[0].item())
    loss_list['res_1'].append(loss_res_train[1].item())
    loss_list['res_2'].append(loss_res_train[2].item())
    loss_list['slope'].append(slope)
    loss_list['grad_norm'].append(grad_sum)

    if epoch % printout_freq == 0:
        print('\n @epoch {} cuda {} slope {:.2f} grad_norm {:.2e}'.format(
            epoch, cuda_index, slope, grad_sum))

        print(['{} = {:.2e}'.format(key, loss_list[key][epoch])
               for key in key_list_loss])

        # plot here
        plot_pinn_y(to_np(x_train[sorted_index]),
                    to_np(y_train[sorted_index]),
                    to_np(t_true),
                    to_np(y_true),
                    to_np(dydt_train[sorted_index]),
                    to_np(rhs_train[sorted_index]),
                    loss_list,
                    x_scale,
                    t_np[0])

        torch.save({'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_list': loss_list,
                    'x_train': to_np(x_train),
                    'x_test': to_np(x_test),
                    }, checkpoint_path + '.tar')

        torch.save(net, checkpoint_path)
