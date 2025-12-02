import torch
from hamiltonian import HamiltonianModel
from optimizer import optimize_with_restarts, random_theta_init, basin_hopping_optimize
from analysis import analyze, write_configs_to_file


def run_model(n=3, configs=None, method="basin_hopping", with_restarts=False, steps=30,
    plot=True,  # controls whether analysis plots are generated
    write_to_file=False  # controls whether to write configuration to file
):
    model = HamiltonianModel(n=n, param_configs=configs)

    theta0 = model.get_tensor()
    model.pretty_print_params(theta0)


    if method == "basin_hopping":
        optimized_theta, loss = basin_hopping_optimize(
            model=model,
            steps=steps,
            local_iters=300,
            hop_size=0.25,
            T=1.0,
            verbose=True,
            restarts=6 if with_restarts else 0,
            restart_perturb_scale=0.8
        )
    elif method == "adam":
        optimized_theta, loss = optimize_with_restarts(
            model=model,
            restarts=5 if with_restarts else 0,
            iters=6,
            lr=0.05,
            restart_perturb_scale=0.1
        )
    else:
        raise ValueError(f"Unknown optimize method: {method}")
    
    print("\nRaw optimized θ:", optimized_theta)


    optimized_theta = torch.tensor(optimized_theta, dtype=torch.float64)
    adjusted_theta = model.adjust_tensor(optimized_theta)

    param_dict = model.tensor_to_dict(adjusted_theta)
    physical_params = model.get_physical_parameters(param_dict)

    print("\n► Physical parameters:")
    model.pretty_print_params(adjusted_theta)

    full_theta = model.build_full_theta(adjusted_theta)

    if plot:
        analyze(model, adjusted_theta)

    if write_to_file:
        optimized_theta_dict = model.tensor_to_dict(adjusted_theta)
        optimized_theta_dict = {k: v.cpu().numpy().tolist() for k, v in optimized_theta_dict.items()}
        write_configs_to_file(
            optimized_theta_dict,
            n,
            parameter_configs=configs,
            loss=loss
        )

    return full_theta, physical_params, loss

def plot_known_parameters(n, theta_dict):
    model = HamiltonianModel(n=n, param_configs=None)

    # Convert dict → tensor
    theta_tensor = model.dict_to_tensor(theta_dict)

    # Convert tensor → dict (enforces correct shaping)
    param_dict = model.tensor_to_dict(theta_tensor)

    # Convert to physical parameters
    phys = model.get_physical_parameters(param_dict)

    print("► Physical parameters:")
    model.pretty_print_params(theta_tensor)

    # analyze expects a tensor, NOT a dict
    analyze(model, theta_tensor)

if __name__ == "__main__":
    config = {
            "t": {"mode": "homogeneous", "fixed": 1.0},
            "U": {"mode": "homogeneous", "fixed": 0.1},
            "eps": {"mode": "inhomogeneous", "fixed": None},
            "Delta": {"mode": "homogeneous", "fixed": None}
        }
    n = 3
    U_vals = [0.0, 0.5, 1.0, 2.0]
    for n in [2,3,4]:
        print(f"\n=== Running for n = {n} ===")
        for U in U_vals:
            print(f"\n=== Running for U = {U} ===")
            config["U"]["fixed"] = U
            full_theta, physical_params, loss = run_model(n=n, configs=config, method="basin_hopping", with_restarts=True, steps=30, plot=False, write_to_file=True)
