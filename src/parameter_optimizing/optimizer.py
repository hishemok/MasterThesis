import torch
import numpy as np
from operators import parity_operator_torch, device
from measurements import calculate_parities, charge_difference_torch, Majorana_polarization_torch
from analysis import print_params



def optimize(model, loss_fn, theta_init, iters=500, lr=0.05):
    theta = torch.tensor(theta_init, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=lr)
    for i in range(iters):
        optimizer.zero_grad()
        H = model.build(model.map_theta(theta))
        P = parity_operator_torch(model.n)
        loss = loss_fn(H, P, model.n, theta=theta)
        loss.backward()
        optimizer.step()
    return theta.detach()


def optimize_with_restarts(model, loss_fn, theta_init_fn, restarts=8, iters=600, lr=0.05, verbose=True):
    """
    Optimizes model parameters with multiple restarts.
    
    model: HamiltonianModel instance
    loss_fn: callable (H, P, n, theta) -> loss
    theta_init_fn: function that returns a new random θ_init (torch tensor)
    """
    best_loss = float("inf")
    best_theta = None
    loss = torch.tensor(0.0)  # Initialize loss to avoid reference before assignment
    for r in range(restarts):

        theta_init = theta_init_fn(model.n)
        theta = torch.tensor(theta_init, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([theta], lr=lr)
        for it in range(iters):
            optimizer.zero_grad()
            params = model.map_theta(theta)
            H = model.build(params)
            P = parity_operator_torch(model.n)
            loss = loss_fn(H, P, model.n, theta=theta)
            loss.backward()
            optimizer.step()

        final_loss = float(loss.detach().cpu())
        if final_loss < best_loss:
            best_loss = final_loss
            best_theta = theta.detach().cpu().numpy().copy()

        if verbose:
            print(f"Restart {r}, final loss {final_loss:.6e}")

    print(f"\nBest loss: {best_loss:.6e}")
    print_params(best_theta, model.n, model.fixed_params)
    return best_theta, best_loss


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
    rnd = rng.random(5)

    U0 = 1 + rnd[0]
    U = [U0] * (n - 1)
    eps_ends = [-U0/2 * rnd[1], -U0/2 * rnd[1]]
    eps_mids = [-U0 * rnd[2]] * (n - 2) if n > 2 else []
    eps = np.array([eps_ends[0], *eps_mids, eps_ends[1]])
    t0 = 0.5 + rnd[3] * 0.5
    D0 = t0 + U0/2 + rnd[4]

    theta0 = np.concatenate([[t0], U, eps, [D0]]).astype(np.float64)
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


def basin_hopping_optimize(model, loss_fn, theta_init_fn,
    steps=30,           # number of basin hops
    local_iters=300,    # Adam steps for each local opt
    hop_size=0.3,       # how far to jump
    T=1.0,              # temperature for Metropolis acceptance
    lr=0.05,
    verbose=True
):
    """
    Basin-hopping optimizer:
    - Random hop in parameter space
    - Local minimization using Adam
    - Accept/reject based on Metropolis criterion
    """

    # --- initialize ---
    theta = theta_init_fn(model.n).astype(float)
    theta = torch.tensor(theta, dtype=torch.float64)
    P = parity_operator_torch(model.n)

    # Evaluate initial loss
    theta_t = theta.clone().requires_grad_(True)
    params = model.map_theta(theta_t)
    H = model.build(params)
    best_loss = float(loss_fn(H, P, model.n, theta=theta_t).detach())
    best_theta = theta.clone().detach()
    loss = torch.tensor(0.0)

    if verbose:
        print(f"Initial loss: {best_loss:.6e}")

    # --- basin hopping loop ---
    for s in range(steps):

        # ----- 1) Random hop -----
        hop = hop_size * torch.randn_like(theta)
        trial_theta = theta + hop

        # ----- 2) Local optimization from trial_theta -----
        trial_theta_t = trial_theta.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([trial_theta_t], lr=lr)

        for _ in range(local_iters):
            optimizer.zero_grad()
            params = model.map_theta(trial_theta_t)
            H = model.build(params)
            loss = loss_fn(H, P, model.n, theta=trial_theta_t)
            loss.backward()
            optimizer.step()

        trial_loss = float(loss.detach())

        # ----- 3) Metropolis acceptance rule -----
        loss_diff = trial_loss - best_loss

        accept = False
        if loss_diff < 0:
            accept = True
        else:
            prob = torch.exp(torch.tensor(-loss_diff / T)).item()
            if torch.rand(1).item() < prob:
                accept = True

        # ----- 4) Accept or reject -----
        if accept:
            theta = trial_theta_t.detach().clone()
            best_loss = trial_loss
            best_theta = theta.clone()

            if verbose:
                print(f"[Step {s}] Accepted new minimum: {best_loss:.6e}")
        else:
            if verbose:
                print(f"[Step {s}] Rejected: {trial_loss:.6e}")

    # --- Final results ---
    print("\nBest solution:")
    print(f"Loss = {best_loss:.6e}")
    print_params(best_theta.numpy(), model.n, model.fixed_params)

    return best_theta.numpy(), best_loss
