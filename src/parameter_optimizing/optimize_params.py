from hamiltonian import HamiltonianModel
from optimizer import optimize, custom_loss, optimize_with_restarts, random_theta_init
from analysis import plot_parity_spectrum
import torch
import numpy as np

def move_free(n=3):
    model = HamiltonianModel(n=n)

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

def somefixed(n = 3, fixed = None):

    if fixed is None:
        U = [1.0] * (n - 1)
        fixed = {
            "U": torch.tensor(U),  # fixed interaction strengths
            "t": torch.tensor(1.0)
        }


    model = HamiltonianModel(n=n, fixed_params=fixed)

    optimized_theta, _ = optimize_with_restarts(model, custom_loss, random_theta_init, iters=600, lr=0.01, restarts=5)

    print("Optimized parameters with some fixed:", optimized_theta)

    optimized_params = model.map_theta(optimized_theta)

    # Reconstruct full theta for plotting
    full_theta = []
    full_theta.append(optimized_params['t'].item())
    full_theta.extend(optimized_params['U'].tolist())
    full_theta.extend(optimized_params['eps'].tolist())
    full_theta.append(optimized_params['Delta'].item()) 

    plot_parity_spectrum(n, full_theta, model)

if __name__ == "__main__":
    # move_free()
    # somefixed(n=2)
    # somefixed(n=3) 
    # somefixed(n=4)

    # U_vals = [0.0, 1.0, 5.0, 10.0]
    # for U in U_vals:
    #     print(f"\nOptimizing with U fixed to {U} for n=2,3,4")
    #     U = [U] * (2 - 1)  # max n=4
    #     fixed_params = {
    #         "U": torch.tensor(U),
    #         "t": torch.tensor(1.0)
    #     }
    #     somefixed(n=2, fixed=fixed_params)

    #     U = [U[0]] * (3 - 1)
    #     fixed_params = {
    #         "U": torch.tensor(U),
    #         "t": torch.tensor(1.0)
    #     }
    #     somefixed(n=3, fixed=fixed_params)
    #     U = [U[0]] * (4 - 1)
    #     fixed_params = {
    #         "U": torch.tensor(U),
    #         "t": torch.tensor(1.0)
    #     }
    #     somefixed(n=4, fixed=fixed_params)

    # # move_free(n=2)
    move_free(n=3)
    move_free(n=4)