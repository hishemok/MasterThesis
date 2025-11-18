import torch
import numpy as np
from operators import parity_operator_torch, device
from measurements import calculate_parities, charge_difference_torch, Majorana_polarization_torch
from analysis import print_params


import torch
import numpy as np

# ---------------------------
# local Adam optimizer (single run)
# ---------------------------
def optimize_local(model, loss_fn, theta_init, iters=400, lr=0.05, verbose=False, keep_equal=None):
    """
    Local Adam optimization starting exactly at theta_init.
    theta_init: 1D numpy array OR torch tensor (will be converted to torch tensor).
    Returns: (theta_numpy, final_loss_float)
    """
    # ensure torch tensor on double
    loss = torch.tensor(0.0)  # Initialize loss to avoid reference before assignment
    if not isinstance(theta_init, torch.Tensor):
        theta_t = torch.tensor(theta_init, dtype=torch.float64, requires_grad=True)
    else:
        theta_t = theta_init.detach().clone().to(dtype=torch.float64).requires_grad_(True)

    optimizer = torch.optim.Adam([theta_t], lr=lr)
    P = parity_operator_torch(model.n)

    for it in range(iters):
        optimizer.zero_grad()
        theta_t_mod = theta_t.clone()  # keep original tensor for autograd

        idx = 0
        n = model.n

        if 't' not in model.fixed_params:
            if "t" in keep_equal:
                t_val = torch.mean(theta_t[idx:idx+(n-1)])
                theta_t_mod[idx:idx+(n-1)] = t_val
            idx += n-1
        if 'U' not in model.fixed_params:
            if "U" in keep_equal:
                U_val = torch.mean(theta_t[idx:idx+(n-1)])
                theta_t_mod[idx:idx+(n-1)] = U_val
            idx += n-1
        if 'eps' not in model.fixed_params:
            if "eps" in keep_equal:
                eps_val = torch.mean(theta_t[idx:idx+n])
                theta_t_mod[idx:idx+n] = eps_val
            idx += n
        if 'Delta' not in model.fixed_params:
            if "Delta" in keep_equal:
                D_val = torch.mean(theta_t[idx:idx+(n-1)])
                theta_t_mod[idx:idx+(n-1)] = D_val
            idx += n-1

        # now map the modified theta to parameters
        params = model.map_theta(theta_t_mod)
        H = model.build(params)
        loss = loss_fn(H, P, model.n, theta=theta_t_mod)
        loss.backward()
        optimizer.step()


    final_loss = float(loss.detach().cpu().item())
    return theta_t.detach().cpu().numpy().copy(), final_loss


# ---------------------------
# Adam with restarts (starts FROM basin_theta and perturbs around it for each restart)
# ---------------------------
def optimize_with_restarts(model,
                           loss_fn,
                           theta_init_fn,         # fallback function to create theta if basin_theta is None
                           restarts=6,
                           iters=300,
                           lr=0.05,
                           verbose=False,
                           basin_theta=None,
                           restart_perturb_scale=0.08, keep_equal=None):
    """
    Adam with restarts that PRESERVES the basin_theta and perturbs around it.

    - If basin_theta is None -> use theta_init_fn(model.n) to draw initial seeds.
    - If basin_theta is provided (numpy array or torch tensor), each restart
      starts at basin_theta + small_random_perturbation (scale controlled by restart_perturb_scale).
    Returns (best_theta_numpy, best_loss_float).
    """
    best_loss = float("inf")
    best_theta = None

    # make sure basin_theta is numpy array if provided
    if basin_theta is not None:
        if isinstance(basin_theta, torch.Tensor):
            basin_theta_np = basin_theta.detach().cpu().numpy().copy()
        else:
            basin_theta_np = np.asarray(basin_theta).astype(float)
    else:
        basin_theta_np = None

    for r in range(restarts):
        if basin_theta_np is None:
            theta0 = theta_init_fn(model.n)   # user-supplied random init function
        else:

            perturb = (np.random.rand(basin_theta_np.size) - 0.5) * 2.0 * restart_perturb_scale
            theta0 = basin_theta_np + perturb

        theta_res, loss_res = optimize_local(model, loss_fn, theta0, iters=iters, lr=lr, verbose=False, keep_equal=keep_equal)

        if loss_res < best_loss:
            best_loss = loss_res
            best_theta = theta_res.copy()

        if verbose:
            print(f"  Restart {r:2d}: final loss {loss_res:.6e}")

    if verbose:
        print(f"  -> best restart loss: {best_loss:.6e}")

    return best_theta, best_loss


# ---------------------------
# Basin-hopping that calls optimize_with_restarts, preserving the hop
# ---------------------------
def basin_hopping_optimize(model,
                           loss_fn,
                           theta_init_fn,
                           steps=30,
                           local_iters=300,
                           hop_size=0.25,
                           T=1.0,
                           lr=0.05,
                           verbose=True,
                           optim_w_restarts=True,
                           restarts=5,
                           restart_perturb_scale=0.08,
                           keep_equal=None):
    """
    Basin-hopping + Adam-with-restarts (Option 2).
    - theta_init_fn: function(n) -> numpy initial theta (used to seed the very first theta).
    - The 'hop' is preserved: trial_theta = current_theta + hop; all restarts are generated
      as small perturbations around trial_theta (not from scratch).
    - optimize_with_restarts returns best local minimizer started around the hopped point.
    """
    # initialize
    theta0 = theta_init_fn(model.n).astype(float)
    theta = torch.tensor(theta0, dtype=torch.float64)          # current basin center (torch)
    P = parity_operator_torch(model.n)

    # compute initial loss
    theta_t = theta.clone().requires_grad_(True)
    params = model.map_theta(theta_t, keep_equal=keep_equal)
    H = model.build(params)
    best_loss = float(loss_fn(H, P, model.n, theta=theta_t).detach())
    best_theta = theta.detach().cpu().numpy().copy()

    if verbose:
        print(f"Initial loss: {best_loss:.6e}")

    for s in range(steps):
        #basin hop 
        hop = hop_size * torch.randn_like(theta)
        trial_theta = (theta + hop).detach().cpu().numpy().copy()   # numpy for passing into restarts

        #locally optimize starting around trial_theta using Adam-with-restarts (keeps hop)
        if optim_w_restarts:
            trial_theta_best, trial_loss = optimize_with_restarts(
                model,
                loss_fn,
                theta_init_fn,
                restarts=restarts,
                iters=local_iters,
                lr=lr,
                verbose=False,
                basin_theta=trial_theta,
                restart_perturb_scale=restart_perturb_scale,
                keep_equal=keep_equal
            )
        else:
            # single local run starting exactly at trial_theta
            trial_theta_best, trial_loss = optimize_local(
                model,
                loss_fn,
                trial_theta,
                iters=local_iters,
                lr=lr,
                verbose=False,
                keep_equal=keep_equal
            )

        #Metropolis/accept-reject
        loss_diff = trial_loss - best_loss
        accept = False
        if loss_diff < 0:
            accept = True
        else:
            prob = float(np.exp(-loss_diff / float(T)))
            if np.random.rand() < prob:
                accept = True

        if accept:
            theta = torch.tensor(trial_theta_best, dtype=torch.float64)
            best_loss = trial_loss
            best_theta = trial_theta_best.copy()
            if verbose:
                print(f"[Step {s:3d}] Accepted new minimum: {best_loss:.6e}")
        else:
            if verbose:
                print(f"[Step {s:3d}] Rejected (trial {trial_loss:.6e})")

    # final
    if verbose:
        print("\nBest solution:")
        print(f"Loss = {best_loss:.6e}")
        # you can keep your print_params function
        try:
            print_params(best_theta, model.n, model.fixed_params)
        except Exception:
            # fallback if print_params not available in this scope
            pass

    return best_theta.copy(), best_loss


def custom_loss(H, P, n, theta=None, weight_vec=None, gap_target=0.5, gap_weights=None):

    evals, evecs = torch.linalg.eigh(H)  # evals sorted ascending
    even_states, odd_states, even_vecs, odd_vecs = calculate_parities(evals, evecs, P)

    n_elements = min(len(even_states), len(odd_states))
    if n_elements == 0:
        raise ValueError("No states in one of the parity sectors \n Proceed to next restart")  # bad configuration Penalty
    if len(even_states) != len(odd_states):
        raise ValueError("Unequal number of even and odd states \n Bad configuration, proceed to next restart")
    
    if weight_vec is None:
        weight_array = np.linspace(3, 0.1, 2*n_elements-1)**2# First half for Degeneracy, second half for gaps
        weight_vec = torch.tensor(weight_array, device=device)

    penalty_array = torch.zeros(2*n_elements-1, device=device) # First half for Degeneracy, second half for gaps
    deg_terms = torch.abs(even_states[:n_elements] - odd_states[:n_elements]) # Cut out unequal lengths
    w = weight_vec.to(device)
    degeneracy_terms = w[:n_elements] * deg_terms
    penalty_array[:n_elements] = degeneracy_terms

    even_gaps = even_states[1:n_elements] - even_states[:n_elements-1]
    odd_gaps = odd_states[1:n_elements] - odd_states[:n_elements-1]
    worst_gaps = torch.min(torch.stack([even_gaps, odd_gaps]), dim=0).values

    gap_penalties = torch.nn.functional.softplus(gap_target - worst_gaps)
    gap_terms = w[n_elements:] * gap_penalties
    penalty_array[n_elements:] = gap_terms

    charge_diff = charge_difference_torch(even_vecs, odd_vecs, n)

    MP_penalty = MP_Penalty(even_vecs, odd_vecs, n)

    total_penalty = torch.sum(penalty_array) + charge_diff + MP_penalty

    mean_energy = evals.abs().mean().detach() + 1e-8
    normalized_loss = total_penalty / mean_energy

    return normalized_loss.real



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


def MP_Penalty(even_vecs, odd_vecs, n):
    """
    Majorana Polarization penalty:
    ideally +1 on first site, -1 on last site, 0 elsewhere.
    even_vecs, odd_vecs: (dim, num_states)
    n: number of sites
    MP shape: (num_states, 2n)
    """
    MP = Majorana_polarization_torch(even_vecs, odd_vecs, n)

    target_MP = torch.zeros_like(MP)
    target_MP[:, 0] = 1.0
    target_MP[:, -1] = -1.0

    # both signs
    penalty_pos = torch.abs(torch.sum((MP - target_MP)**2))
    penalty_neg = torch.abs(torch.sum((MP + target_MP)**2))

    penalty = torch.min(penalty_pos, penalty_neg)
    return penalty


