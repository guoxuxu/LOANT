import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from functools import partial


def objective(A, b, c, var):
    error = torch.mm(torch.mm(var.t(), A), var) + torch.mm(b.t(), var) + c
    return error


def gradient_descent(var, extra=False, gamma=0.001, first_order=True):
    error = objective(A, b, c, var)
    print(error)
    d_error_dx = torch.autograd.grad(outputs=error, inputs=var)[0]
    if extra is True:
        new_var = var - gamma * d_error_dx
        new_error = objective(A, b, c, new_var)
        if first_order is True:
            d_error_dx = torch.autograd.grad(outputs=new_error, inputs=var)[0]
        else:
            hessian = torch.autograd.functional.hessian(partial(objective, A, b, c), var)
            matrix = torch.eye(2) - gamma * hessian.squeeze()
            d_error_dx = torch.mm(torch.autograd.grad(outputs=new_error, inputs=new_var)[0].t(), matrix).t()
        error = new_error
    return d_error_dx, error


def train(x, lr, extra=False, gamma=0.001, first_order=True):

    x0 = np.arange(-5, 15, 0.5) / 100
    x1 = np.arange(-30, -10, 0.5) / 100

    error_vals = np.zeros(shape=(x0.size, x1.size))

    for i, value1 in enumerate(x0):
        for j, value2 in enumerate(x1):
            w_temp = torch.tensor([[value1], [value2]], dtype=torch.float32)
            error = objective(A, b, c, w_temp)
            error_vals[j, i] = error.item()

    levels = np.sort(np.array(np.arange(error_vals.min()-0.4, error_vals.min()+0.4, 0.03).tolist()))

    old_x, errors = [], []
    for step in range(0, 6):
        gradient, error = gradient_descent(x, extra, gamma, first_order)

        old_x.append(x.detach().numpy())
        errors.append(error.item())

        new_x = x - lr * gradient
        print("Step: {} - Error: {:.4f}, gradient:{}, x: {} -> {}".format(step, error.item(), gradient.reshape(-1).detach().numpy(), init_x.reshape(-1).detach().numpy(), x.reshape(-1).detach().numpy()))

        # Stopping Condition
        if abs(new_x - x).sum().item() < 1e-10:
            print('Gradient Descent has converged')
            break
        x = new_x

    print('Final x=', x)

    all_ws = np.array(old_x)

    for i in range(len(all_ws) - 1):
        plt.annotate('', xy=all_ws[i + 1, :], xytext=all_ws[i, :],
                     arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                     va='center', ha='center')

    CS = plt.gca().contour(x0, x1, error_vals, levels)
    plt.gca().clabel(CS, CS.levels[0::4], inline=True, fontsize=17)
    plt.xlabel("w0", fontsize=17)
    plt.ylabel("w1", fontsize=17)
    plt.xticks(np.arange(min(x0), max(x0)+0.05, 0.05), fontsize=17)
    plt.yticks(np.arange(min(x1), max(x1)+0.05, 0.05), fontsize=17)
    plt.show()
    print()


if __name__ == '__main__':
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.cuda.manual_seed(1)

    lamb = torch.FloatTensor([
        [40, 0],
        [0, 1]
    ])

    oth_a = torch.rand(2)
    oth_b = torch.rand(2)
    cos = torch.dot(oth_a, oth_b) / (oth_a.norm() * oth_b.norm())
    oth_b = oth_b - oth_b.norm() * cos * (oth_a / oth_a.norm())
    oth_b = oth_b/oth_b.norm()
    oth_a = oth_a/oth_a.norm()
    Q = torch.stack([oth_a, oth_b], dim=0).t()

    # weights
    A = torch.mm(torch.mm(Q, lamb), Q.t())
    b = torch.rand(2,1)
    c = torch.rand(1)

    # input
    x = torch.FloatTensor([[0.0], [-0.15]]).requires_grad_(True)
    print('initial x=', x)

    train(x=x, lr=0.025, extra=False, gamma=0.01, first_order=True)
    train(x=x, lr=0.025, extra=True, gamma=0.01, first_order=True)
    train(x=x, lr=0.1, extra=True, gamma=0.01, first_order=False)