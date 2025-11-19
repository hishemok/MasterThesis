import torch
import numpy as np
from operators import parity_operator_torch, device
from measurements import calculate_parities, charge_difference_torch, Majorana_polarization_torch

def optimize_local(model, initial_theta=None, lr=0.05, max_iters=300, verbose=True):
    """
    Minimal local optimizer.
    Works directly on a starting tensor (initial_theta) if provided.
    Uses adjust_tensor() to enforce physical constraints.
    """
    theta = initial_theta if initial_theta is not None else model.get_tensor()
    theta = theta.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    P = parity_operator_torch(model.n)

    for it in range(max_iters):
        opt.zero_grad()

        # Adjust to physical parameter space
        adjusted = model.adjust_tensor(theta)
        dict_params = model.tensor_to_dict(adjusted)
        phys_params = model.get_physical_parameters(dict_params)
        H = model.build(phys_params)

        loss = model.loss(H, P)
        loss.backward()
        opt.step()

        if verbose and (it % 20 == 0 or it == max_iters - 1):
            print(f"iter {it:4d} loss = {loss.item():.6f}")

    final_adjusted = model.adjust_tensor(theta)
    final_dict = model.tensor_to_dict(final_adjusted)
    final_phys = model.get_physical_parameters(final_dict)

    return theta.detach(), final_phys


def optimize_with_restarts(model,
                           restarts=6,
                           iters=300,
                           lr=0.05,
                           verbose=False,
                           restart_perturb_scale=0.08,
                           basin_theta=None):
    """
    Restarted local optimization.
    Each restart starts from basin_theta + random perturbation, or model.get_tensor() if None.
    """
    best_loss = float("inf")
    best_theta = None
    P = parity_operator_torch(model.n)

    # Convert basin_theta to numpy if given
    if basin_theta is not None:
        if isinstance(basin_theta, torch.Tensor):
            basin_theta_np = basin_theta.detach().cpu().numpy().copy()
        else:
            basin_theta_np = np.asarray(basin_theta, dtype=float)
    else:
        basin_theta_np = None

    for r in range(restarts):
        # --- starting point ---
        theta0 = model.get_tensor().detach().cpu().numpy()


        if basin_theta_np is not None:
            theta0 = basin_theta_np.copy()

        # perturb
        perturb = (np.random.rand(theta0.size) - 0.5) * 2.0 * restart_perturb_scale
        theta0 = theta0 + perturb

        # convert back to tensor
        initial_theta = torch.tensor(theta0, dtype=torch.float64, device=model.device)

        # --- run local optimization ---
        final_theta, final_phys = optimize_local(
            model,
            initial_theta=initial_theta,
            lr=lr,
            max_iters=iters,
            verbose=False
        )

        # compute final loss
        H = model.build(final_phys)
        final_loss = model.loss(H, P).item()

        if final_loss < best_loss:
            best_loss = final_loss
            best_theta = final_theta.detach().cpu().numpy().copy()

        if verbose:
            print(f"Restart {r+1}/{restarts}: loss = {final_loss:.6f}")

    if verbose:
        print("\nBest restart loss =", best_loss)

    return best_theta, best_loss


def basin_hopping_optimize(
        model,
        steps=30,
        local_iters=300,
        hop_size=0.25,
        T=1.0,
        lr=0.05,
        verbose=True,
        restarts=6,
        restart_perturb_scale=0.08
    ):
    """
    Basin-hopping using optimize_with_restarts.
    """
    theta = model.get_tensor().detach().clone()
    best_theta = theta.clone()
    P = parity_operator_torch(model.n)

    param_dict = model.tensor_to_dict(theta)
    H = model.build(param_dict)
    best_loss = model.loss(H, P).item()

    if verbose:
        print(f"Initial basin loss: {best_loss:.6e}")

    for step in range(steps):
        # Hop in parameter space
        hop = hop_size * torch.randn_like(theta)
        trial_theta = (theta + hop).detach().clone()

        # Run local optimization with restarts
        trial_best_theta, trial_best_loss = optimize_with_restarts(
            model,
            restarts=restarts,
            iters=local_iters,
            lr=lr,
            verbose=False,
            restart_perturb_scale=restart_perturb_scale,
            basin_theta=trial_theta.detach().cpu().numpy()
        )

        # Metropolis acceptance
        dL = trial_best_loss - best_loss
        if dL < 0 or np.random.rand() < np.exp(-dL / T):
            theta = torch.tensor(trial_best_theta, dtype=torch.float64, device=model.device)
            best_theta = theta.clone()
            best_loss = trial_best_loss
            if verbose:
                print(f"[Step {step:3d}] ACCEPTED → Loss: {best_loss:.6e}")
        else:
            if verbose:
                print(f"[Step {step:3d}] Rejected (trial: {trial_best_loss:.6e})")

    if verbose:
        print("\n=== Basin Hopping Finished ===")
        print(f"Best loss: {best_loss:.6e}")

    return best_theta.clone(), best_loss

def random_theta_init(n):
    rng = np.random.default_rng()
    # U: Coulomb interaction (uniform-ish but with slight random variation)
    U0 = 1 + rng.random()      # around 1–2
    U = U0 * (0.9 + 0.2 * rng.random(n - 1))

    # t: hopping (positive, order unity)
    t0 = 0.5 + rng.random()    # 0.5–1.5
    t = t0 * (0.9 + 0.2 * rng.random(n - 1))

    # Delta: pairing (positive, maybe slightly smaller than t)
    D0 = 0.5 * t0 + rng.random()
    Delta = D0 * (0.9 + 0.2 * rng.random(n - 1))

    # eps: onsite energies (center sites deeper)
    eps = np.zeros(n)
    eps[0] = -0.5 * U0 * (0.5 + rng.random())
    eps[-1] = -0.5 * U0 * (0.5 + rng.random())
    if n > 2:
        eps[1:-1] = -U0 * (0.5 + rng.random())

    # pack into a single θ vector
    theta0 = np.concatenate([t, U, eps, Delta]).astype(np.float64)
    return theta0
