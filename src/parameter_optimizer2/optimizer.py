import torch
import numpy as np
from operators import parity_operator_torch, device
from measurements import calculate_parities, charge_difference_torch, Majorana_polarization_torch


def optimize_local(model, lr=0.05, max_iters=300, verbose=True):
    """
    Minimal local optimizer.
    Works directly on model.get_tensor(), using model.adjust_tensor()
    and model.get_physical_parameters() to enforce physical constraints.
    """

    theta = model.get_tensor().clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)

    for it in range(max_iters):
        opt.zero_grad()

        # Enforce physical parameter rules (positivity, ordering, etc.)
        adjusted = model.adjust_tensor(theta)

        # Interpret adjusted tensor as physical parameters
        dict_params = model.tensor_to_dict(adjusted)
        phys_params = model.get_physical_parameters(dict_params)

        # Build Hamiltonian from the physical parameters
        H = model.build_H(phys_params)

        # Compute loss
        loss = model.loss(H)

        # Backprop
        loss.backward()
        opt.step()

        if verbose and (it % 20 == 0 or it == max_iters - 1):
            print(f"iter {it:4d} loss = {loss.item():.6f}")

    # Final physical parameters
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
    Restarted local optimization for the NEW model structure.

    - model.get_tensor() gives the starting point.
    - model.adjust_tensor() enforces constraints.
    - model.tensor_to_dict(), get_physical_parameters(), build_H() etc. are used inside optimize_local().

    If basin_theta is given:
        each restart = basin_theta + random_perturbation
    Else:
        each restart starts from model.get_tensor() + small noise.
    """

    best_loss = float("inf")
    best_theta = None

    # Convert basin_theta to numpy if provided
    if basin_theta is not None:
        if isinstance(basin_theta, torch.Tensor):
            basin_theta_np = basin_theta.detach().cpu().numpy().copy()
        else:
            basin_theta_np = np.asarray(basin_theta, dtype=float)
    else:
        basin_theta_np = None

    for r in range(restarts):

        # --- NEW: create a fresh model copy for each restart ---
        m = model.copy()  # you MUST implement model.copy() (shallow is fine)

        # --- Choose starting tensor ---
        theta0 = m.get_tensor().detach().cpu().numpy()

        if basin_theta_np is not None:
            theta0 = basin_theta_np.copy()
        
        # add perturbation
        perturb = (np.random.rand(theta0.size) - 0.5) * 2.0 * restart_perturb_scale
        theta0 = theta0 + perturb

        # inject starting vector
        m.set_tensor(torch.tensor(theta0, dtype=torch.float64, device=m.device))

        # --- run local optimization ---
        final_theta, final_phys = optimize_local(
            m, lr=lr, max_iters=iters, verbose=False
        )

        # compute final loss
        H = m.build_H(final_phys)
        final_loss = m.loss(H).item()

        # keep best
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
    Basin-hopping for the NEW optimizer structure.

    - Uses model.get_tensor()
    - Uses model.set_tensor()
    - Uses model.build_H() and model.loss()
    - Calls optimize_with_restarts()

    No theta_init_fn. No old loss_fn signatures.
    """

    # -------------------------------------
    # INITIALIZE
    # -------------------------------------
    best_model = model.copy()
    theta = best_model.get_tensor().detach().clone()
    best_theta = theta.clone()

    H = best_model.build_H(best_model.adjust_tensor(theta))
    best_loss = best_model.loss(H).item()

    if verbose:
        print(f"Initial basin loss: {best_loss:.6e}")

    # -------------------------------------
    # MAIN LOOP
    # -------------------------------------
    for step in range(steps):

        # -------------------------
        # Hop: θ_trial = θ + random_step
        # -------------------------
        hop = hop_size * torch.randn_like(theta)
        trial_theta = (theta + hop).detach().clone()

        # -------------------------
        # Create fresh model for local search
        # -------------------------
        m = model.copy()
        m.set_tensor(trial_theta.clone())

        # -------------------------
        # Local optimization with RESTARTS
        # -------------------------
        trial_best_theta, trial_best_loss = optimize_with_restarts(
            m,
            restarts=restarts,
            iters=local_iters,
            lr=lr,
            verbose=False,
            restart_perturb_scale=restart_perturb_scale,
            basin_theta=trial_theta.detach().cpu().numpy()
        )

        # -------------------------
        # Acceptance test (Metropolis)
        # -------------------------
        dL = trial_best_loss - best_loss

        if dL < 0:
            accept = True
        else:
            accept = (np.random.rand() < np.exp(-dL / T))

        if accept:
            theta = torch.tensor(trial_best_theta, dtype=torch.float64, device=model.device)
            best_theta = theta.clone()
            best_loss = trial_best_loss
            best_model = m.copy()

            if verbose:
                print(f"[Step {step:3d}] ACCEPTED → Loss: {best_loss:.6e}")

        else:
            if verbose:
                print(f"[Step {step:3d}] Rejected (trial: {trial_best_loss:.6e})")

    if verbose:
        print("\n=== Basin Hopping Finished ===")
        print(f"Best loss: {best_loss:.6e}")

    return best_theta.clone(), best_loss



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


