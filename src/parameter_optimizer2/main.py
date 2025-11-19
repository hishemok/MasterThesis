import torch
from hamiltonian import HamiltonianModel
from optimizer import optimize_with_restarts, random_theta_init, basin_hopping_optimize


def run_model(
    n=3,
    configs=None,
    method="basin_hopping",
    with_restarts=False,
    steps=30,
    plot=True  # controls whether analysis plots are generated
):
    # -----------------------------------------------------
    # 1. Construct model
    # -----------------------------------------------------
    model = HamiltonianModel(n=n, param_configs=configs)

    # -----------------------------------------------------
    # 2. Initial tensor to optimize
    # -----------------------------------------------------
    theta0 = model.get_tensor()
    model.pretty_print_params(theta0)

    # -----------------------------------------------------
    # 3. Choose optimization strategy
    # -----------------------------------------------------
    if method == "basin_hopping":
        optimized_theta, loss = basin_hopping_optimize(
            model=model,
            steps=steps,
            local_iters=500,
            hop_size=0.5,
            T=1.0,
            verbose=True,
            restarts=6 if with_restarts else 0,
            restart_perturb_scale=0.1
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

    # -----------------------------------------------------
    # 4. Convert optimized θ → adjusted θ → physical params
    # -----------------------------------------------------
    optimized_theta = torch.tensor(optimized_theta, dtype=torch.float64)
    adjusted_theta = model.adjust_tensor(optimized_theta)

    param_dict = model.tensor_to_dict(adjusted_theta)
    physical_params = model.get_physical_parameters(param_dict)

    print("\n► Physical parameters:")
    model.pretty_print_params(adjusted_theta)

    # -----------------------------------------------------
    # 5. Build full theta (optional)
    # -----------------------------------------------------
    full_theta = model.build_full_theta(adjusted_theta)

    # -----------------------------------------------------
    # 6. Analysis and plotting
    # -----------------------------------------------------
    if plot:
        from analysis import analyze
        analyze(model, adjusted_theta)

    return full_theta, physical_params, loss


if __name__ == "__main__":
    config = {
            "t": {"mode": "homogeneous", "fixed": None},
            "U": {"mode": "homogeneous", "fixed": 1},
            "eps": {"mode": "inhomogeneous", "fixed": None},
            "Delta": {"mode": "inhomogeneous", "fixed": None}
        }
    
    run_model(
        n=4,
        configs=config,
        method="basin_hopping",
        with_restarts=True,
        steps=30
    )