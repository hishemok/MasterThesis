from hamiltonian import HamiltonianModel
from optimizer import  custom_loss, optimize_with_restarts, random_theta_init, basin_hopping_optimize
from analysis import plot_parity_spectrum
import torch
import numpy as np

def move_free(n=3, optimize_method="basin_hopping", with_restarts=False):
    model = HamiltonianModel(n=n)

    if optimize_method == "basin_hopping":
        optimized_theta, _ = basin_hopping_optimize(
            model, custom_loss, random_theta_init, optim_w_restarts=with_restarts)
    else:
        optimized_theta, _ = optimize_with_restarts(model, custom_loss, random_theta_init, iters=500, lr=0.01)

    print("Optimized parameters:", optimized_theta)
       # Reconstruct full theta for plotting
    optimized_params = model.map_theta(optimized_theta)
    
    full_theta = []
    full_theta.append(optimized_params['t'].item())
    full_theta.extend(optimized_params['U'].tolist())
    full_theta.extend(optimized_params['eps'].tolist())
    full_theta.append(optimized_params['Delta'].item()) 

    plot_parity_spectrum(n, full_theta, model)

def somefixed(n = 3, fixed = None, optimize_method="basin_hopping", with_restarts=False):

    if fixed is None:
        U = [1.0] * (n - 1)
        t = [1.0] * (n - 1)
        fixed = {
            "U": torch.tensor(U),  # fixed interaction strengths
            "t": torch.tensor(t)
        }


    model = HamiltonianModel(n=n, fixed_params=fixed)

    if optimize_method == "basin_hopping":
        optimized_theta, loss = basin_hopping_optimize(
            model, custom_loss, random_theta_init, optim_w_restarts=with_restarts, steps=30)
    else:
        optimized_theta, loss = optimize_with_restarts(model, custom_loss, random_theta_init, iters=600, lr=0.01, restarts=5)

    print("Optimized parameters with some fixed:", optimized_theta)

    optimized_params = model.map_theta(optimized_theta)

    # Reconstruct full theta for plotting
    full_theta = []
    full_theta.append(optimized_params['t'].item())
    full_theta.extend(optimized_params['U'].tolist())
    full_theta.extend(optimized_params['eps'].tolist())
    full_theta.append(optimized_params['Delta'].item()) 

    plot_parity_spectrum(n, full_theta, model)

def all_fixed(n=3, params = None):

    if params is None:
        U = [1.0] * (n - 1)
        params = {
            "U": torch.tensor(U), 
            "t": torch.tensor(1.0),
            "eps": torch.tensor([0.0] * n),
            "Delta": torch.tensor(1.0)
        }
    model = HamiltonianModel(n=n, fixed_params=params)

    # No optimization needed, just build and plot
    full_theta = []
    full_theta.append(params['t'].item())
    full_theta.extend(params['U'].tolist())
    full_theta.extend(params['eps'].tolist())
    full_theta.append(params['Delta'].item())
    plot_parity_spectrum(n, full_theta, model)


if __name__ == "__main__":
    # move_free()
    # somefixed(n=2)
    # somefixed(n=3) 
    # somefixed(n=4)

    U_vals = [1.0]
    for U in U_vals:
        print(f"\nOptimizing with U fixed to {U} for n=2,3,4")
        U = [U] * (2 - 1)  # max n=4
        # fixed_params = {
        #     "U": torch.tensor(U)
        #     # "t": torch.tensor(1.0)
        # }
        # somefixed(n=2, fixed=fixed_params, optimize_method="basin_hopping", with_restarts=False)

        # U = [U[0]] * (3 - 1)
        # fixed_params = {
        #     "U": torch.tensor(U)
        #     # "t": torch.tensor(1.0)
        # }
        # somefixed(n=3, fixed=fixed_params, optimize_method="basin_hopping", with_restarts=False)
        U = [U[0]] * (4 - 1)
        fixed_params = {
            "U": torch.tensor(U)
            # "t": torch.tensor(1.0)
        }
        somefixed(n=4, fixed=fixed_params, optimize_method="basin_hopping", with_restarts=True)

    # move_free(n=2)
    # move_free(n=3)
    # move_free(n=4)
    # params = {
    #     "t": torch.tensor(6.207209916537057),
    #     "U": torch.tensor([1.0,1.0]),
    #     "eps": torch.tensor([-2.75826954e-03  5.71615884e+00 -1.00638620e+00]),
    #     "Delta": torch.tensor(1.8170338617393165)
    # }
    # all_fixed(n=3, params=params)
    # params = {
    #     "t": torch.tensor(1.0),
    #     "U": torch.tensor([1.0]),
    #     "eps": torch.tensor([0.0, 0.0]),
    #     "Delta": torch.tensor(1.0)
    # }
    # all_fixed(n=2, params=params)
