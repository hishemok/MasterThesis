import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from functools import reduce

from n_particles_degen import creation_annihilation, charge_difference, Majorana_polarization, parity_operator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def to_torch(mat_np, device=device):
    return torch.from_numpy(mat_np).to(device=device, dtype=torch.complex128)


def build_H_torch(n, params, cre_t, ann_t, num_t):

    t = params[0]
    U = params[1:1+(n-1)]
    eps = params[1+(n-1):1+(n-1)+n]
    Delta = params[-1]
    dim = 2**n
    H = torch.zeros((dim,dim), dtype=torch.complex128, device=device)

    for i in range(n):
        # onsite 
        H += eps[i] * num_t[i]
        if i < n - 1:
            # hopping
            H += t * (cre_t[i] @ ann_t[i+1] + cre_t[i+1] @ ann_t[i])
            # pairing
            H += Delta * (cre_t[i] @ cre_t[i+1] + ann_t[i+1] @ ann_t[i])
            # Coulomb term
            H += U[i] * (num_t[i] @ num_t[i+1])
    return H

def softclip(x, minval, maxval):
    return minval + 0.5*(torch.tanh(x) + 1.0)*(maxval - minval)
def map_params_to_physical(theta, n):
    """
    Map raw parameters (any real numbers) to physical ranges.
    """
    Umin, Umax = 0.1, 10.0
    epsmin, epsmax = -5.0, 5.0
    tmin, tmax = 0.1, 5.0
    Dmin, Dmax = 0.0, 5.0

    t     = softclip(theta[0], tmin, tmax)
    U     = softclip(theta[1:1+(n-1)], Umin, Umax)
    eps   = softclip(theta[1+(n-1):1+(n-1)+n], epsmin, epsmax)
    Delta = softclip(theta[-1], Dmin, Dmax)
    
    
    t = t.unsqueeze(0)       # make scalar tensor shape (1,)
    Delta = Delta.unsqueeze(0)
    thetamapped = torch.cat((t, U, eps, Delta))

    return thetamapped



def parity_operator_torch(n):
    P = torch.eye(2**n, dtype=torch.complex128)
    for j in range(n):
        cd, c = creation_annihilation(j, n)
        cd_t = to_torch(cd, device=device)
        c_t = to_torch(c, device=device)
        n_i = cd_t @ c_t
        P = P @ (torch.eye(2**n, dtype=torch.complex128, device=device) - 2 * n_i)
    return P


def precompute_ops(n):
    # returns lists of creation, annihilation, number (numpy)
    cre = []
    ann = []
    num = []
    for j in range(n):
        cd, c = creation_annihilation(j, n)
        cre.append(cd)
        ann.append(c)
        num.append(cd @ c)
    return cre, ann, num

def calculate_parities(evals, evecs, P):
    # evecs: columns are eigenvectors
    parities = torch.real(torch.sum(torch.conj(evecs) * (P @ evecs), dim=0))
    even_mask = parities >= 0
    odd_mask  = parities < 0

    E_even = evals[even_mask]
    E_odd  = evals[odd_mask]

    # Return also eigenvectors if needed
    even_vecs = evecs[:, even_mask]
    odd_vecs  = evecs[:, odd_mask]

    return E_even, E_odd, even_vecs, odd_vecs



def charge_difference_torch(even_state, odd_state, n):
    """Compute the charge difference between even and odd states.
    ∑_n |⟨e_k|n_n|e_k⟩ - ⟨o_k|n_n|o_k⟩|
    """
    # accumulate a real scalar (sum of magnitudes), keep as float tensor
    charge_diff = torch.tensor(0.0, dtype=torch.float64, device=device)
    n_states = even_state.shape[1]
    for site in range(n):
        f_dag_i, f_i = creation_annihilation(site, n)
        f_dag_i_t = to_torch(f_dag_i, device=device)
        f_i_t = to_torch(f_i, device=device)
        n_i = f_dag_i_t @ f_i_t
        exp_even = torch.zeros((), dtype=torch.complex128, device=device)
        exp_odd = torch.zeros((), dtype=torch.complex128, device=device)
        for k in range(n_states):
            even_vec = even_state[:, k]
            odd_vec = odd_state[:, k]
            exp_even += torch.vdot(even_vec, n_i @ even_vec)
            exp_odd  += torch.vdot(odd_vec,  n_i @ odd_vec)
        charge_diff += torch.abs(exp_even - exp_odd).to(dtype=torch.float64)
    return charge_diff


def construct_Gamma_operators_torch(j,n, device=device):
    """Construct the Majorana operators Γ^s_j for site j in a chain of n sites."""
    f_dag_j, f_j = creation_annihilation(j, n)
    f_dag_j_t = to_torch(f_dag_j, device=device)
    f_j_t = to_torch(f_j, device=device)
    Gamma_1 = f_dag_j_t + f_j_t
    Gamma_2 = 1j * (f_dag_j_t - f_j_t)
    return Gamma_1, Gamma_2

def Majorana_polarization_torch(even_vecs, odd_vecs, n):
    """
    Compute local Majorana polarization M_j for each site j.
    M_j = sum_s (<o|Γ^s_j|e>)^2 / sum_s |<o|Γ^s_j|e>|^2
    """
    M = torch.zeros((even_vecs.shape[1], n), dtype=torch.complex128, device=device)

    for j in range(n):
        Gamma_1, Gamma_2 = construct_Gamma_operators_torch(j, n, device=device)
        for i in range(even_vecs.shape[1]):
            e_k = even_vecs[:, i]
            o_k = odd_vecs[:, i]
            term1 = torch.vdot(o_k, Gamma_1 @ e_k)
            term2 = torch.vdot(o_k, Gamma_2 @ e_k)
            M[i, j] = term1**2 + term2**2 #/ denominator if denominator != 0 else 0.0
    return M


def degeneracy_and_gap_loss(H, P, n, weight_vec=None, gap_target=0.5, gap_weights=None, theta=None):
    # H: Hermitian torch matrix
    H = 0.5 * (H + H.conj().T) ## Just because the optimizer can introduce small non-Hermiticities

    evals, evecs = torch.linalg.eigh(H)  # evals sorted ascending

    even_states, odd_states, even_vecs, odd_vecs = calculate_parities(evals, evecs, P)

    # Just in case
    E_even, _ = torch.sort(even_states)
    E_odd,  _ = torch.sort(odd_states)

    m = min(len(E_even), len(E_odd)) # Minimum number of elements
    if m == 0:
        raise  ValueError("No states in one of the parity sectors \n Proceed to next restart")  # bad configuration Penalty

    if len(E_even) != len(E_odd):
        # print("Warning: unequal number of even and odd states:", len(E_even), len(E_odd))
        raise ValueError("Unequal number of even and odd states \n Bad configuration, proceed to next restart")


    # # default weights
    if weight_vec is None:
        weight_array = np.linspace(1, 0.1, 2*m-1)**2# First half for Degeneracy, second half for gaps
        weight_vec = torch.tensor(weight_array, device=device)

    penalty_array = torch.zeros(2*m-1, device=device) # First half for Degeneracy, second half for gaps

    deg_terms = torch.abs(E_even - E_odd) # Cut out unequal lengths
    w = weight_vec.to(device)
    degeneracy_terms = w[:m] * deg_terms
    penalty_array[:m] = degeneracy_terms


    even_gaps = E_even[1:] - E_even[:-1]
    odd_gaps = E_odd[1:] - E_odd[:-1]
    worst_gaps = torch.min(torch.stack([even_gaps, odd_gaps]), dim=0).values

    # If gap < 0.5 penalize, else 0
    gap_penalties = torch.nn.functional.softplus(gap_target - worst_gaps)
    gap_terms = w[m:] * gap_penalties
    penalty_array[m:] = gap_terms


    Majorana_metric = Majorana_polarization_torch(even_vecs, odd_vecs, n)
    charge_diff = charge_difference_torch(even_vecs, odd_vecs, n)

    Majorana_metric_penalty = torch.sum(torch.nn.functional.softplus(torch.abs(Majorana_metric) - 1e-5))

    charge_diff_penalty = charge_diff

    total_penalty = torch.sum(penalty_array) + charge_diff_penalty #+ Majorana_metric_penalty #+ charge_diff_penalty
    # return total_penalty
    mean_energy = evals.abs().mean().detach() + 1e-8
    normalized_loss = total_penalty / mean_energy

    # Optional: add a small weight decay on θ if desired (keeps optimizer stable)
    if theta is not None:
        normalized_loss = normalized_loss + 1e-3 * (theta.norm() ** 2)

    return normalized_loss


def print_params(theta, n):
    t = theta[0]
    U = theta[1:1+(n-1)]
    eps = theta[1+(n-1):1+(n-1)+n]
    Delta = theta[-1]
    print(f"t = {t:.4f}")
    print(f"U = {U}")
    print(f"ε = {eps}")
    print(f"Δ = {Delta:.4f}")


def optimize_params(n, device=device, restarts=8, iters=600):
    # precompute ops
    cre_np, ann_np, num_np = precompute_ops(n)
    cre_t = [to_torch(m, device=device) for m in cre_np]
    ann_t = [to_torch(m, device=device) for m in ann_np]
    num_t = [to_torch(m, device=device) for m in num_np]

    best_loss = 1e9
    best_theta = None

    for r in range(restarts):
        # initialize torch parameter vector (t, U0..U_{n-2}, eps0..eps_{n-1}, Delta)
        # choose sensible ranges: t ~ [0.2,2], U~[0.2,2], eps ~ [-1,1], Delta ~ [0,2]
        rng = np.random.RandomState(seed=r)
        rnd1 = rng.rand()
        rnd2 = rng.rand()
        rnd3 = rng.rand()
        rnd4 = rng.rand()
        rnd5 = rng.rand()

        U0 = 1 + rnd1
        U = [U0 for _ in range(n-1)]

        eps_ends = [-U0/2 * rnd2 , -U0/2  * rnd2]
        if n > 2:
            eps_mids = [-U0 * rnd3] * (n-2)
            eps0 = np.array([eps_ends[0] , *eps_mids , eps_ends[1]])
        else:   
            eps0 = eps_ends

        t0 = 0.5 + rnd4 * 0.5
        D0 = t0 + U0/2 + rnd5

        theta0 = np.concatenate([[t0], U, eps0, [D0]]).astype(np.float64)
        theta = torch.tensor(theta0, dtype=torch.float64, device=device, requires_grad=True)

        loss = torch.tensor(0.0, dtype=torch.float64, device=device)
        optimizer = torch.optim.Adam([theta], lr=0.05)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=1e-3)

        for it in range(iters):
            optimizer.zero_grad()
            # theta_phys = map_params_to_physical(theta, n)
            H = build_H_torch(n, theta, cre_t, ann_t, num_t)  # accepts real theta
            P = parity_operator_torch(n)
            loss = degeneracy_and_gap_loss(H, P, n, theta=theta)

            loss.backward()
            optimizer.step()
  
        with torch.no_grad():
            final_loss = float(loss.cpu().item())
            if final_loss < best_loss:
                best_loss = final_loss
                best_theta = theta.detach().cpu().numpy().copy()
        print(f"Restart {r}, final loss {final_loss:.6e}")

    print("Best loss", best_loss)
    print_params(best_theta, n)
    return best_theta, best_loss


def plot_parity_spectrum(n, theta):
    cre_np, ann_np, num_np = precompute_ops(n)
    # build H numeric numpy
    t = theta[0]; U = theta[1:1+(n-1)]; eps = theta[1+(n-1):1+(n-1)+n]; Delta = theta[-1]
    # build H
    H = np.zeros((2**n,2**n), dtype=complex)
    for i in range(n):
        H += eps[i] * num_np[i]
    for i in range(n-1):
        H += t * (cre_np[i] @ ann_np[i+1] + cre_np[i+1] @ ann_np[i])
        H += Delta * (cre_np[i] @ cre_np[i+1] + ann_np[i+1] @ ann_np[i])
        H += U[i] * (num_np[i] @ num_np[i+1])


    evals, evecs = np.linalg.eigh(H)

    P_np = parity_operator(n)
    parities = [np.real(np.vdot(ev, P_np @ ev)) for ev in evecs.T]
    even_vecs = []
    odd_vecs = []
    print(f"{n}-dot system eigenvalues and their parity sectors:")
    for i, (E,p) in enumerate(zip(evals, parities)):
        if abs(p) < .9:
            print(f"Warning: Eigenvalue {E:.4f} has ambiguous parity {p:.4f}")
        sector = "even" if p > .9 else "odd"
        if sector == "even":
            even_vecs.append(evecs[:, i])
        else:
            odd_vecs.append(evecs[:, i])
        print(f"Eigenvalue {E:.4f} belongs to {sector} parity sector, with parity {p:.4f}")
    print("")
    odd_vecs = np.array(odd_vecs).T
    even_vecs = np.array(even_vecs).T

    MP_np = Majorana_polarization(even_vecs, odd_vecs, n)
    Majorana_metric = MP_np


    for i in range(Majorana_metric.shape[0]):
        plt.plot(range(n), Majorana_metric[i], 'o-', label=f'|M̃|_{i}')
    plt.xlabel('Site index')
    plt.ylabel('Majorana localization')
    plt.title(f'Majorana localization profile for {n}-dot system')
    plt.axhline(0, color='k', linestyle='--')
    plt.legend()
    plt.show()

    evens = [(E, v) for E, v, p in zip(evals, evecs.T, parities) if p >= 0]
    odds  = [(E, v) for E, v, p in zip(evals, evecs.T, parities) if p <  0]
    y_even = [E for E,_ in evens]
    y_odd = [E for E,_ in odds]

    Total_energy_difference = sum([abs(Ee - Eo) for Ee, Eo in zip(y_even, y_odd)])
    print(f"Total energy difference between even and odd states: {Total_energy_difference:.6e}\n")
    degeneracy_lines = []
    for i, Ee in enumerate(y_even):
        Eo = y_odd[i]
        if abs(Ee - Eo) < 1e-2:
            degeneracy_lines.append(Ee)


    charge_diff = charge_difference(even_vecs, odd_vecs, n)


    print(f"Charge difference between even and odd states: {charge_diff:.6e}\n")


    t = theta[0]
    U = theta[1:1+(n-1)]
    eps = theta[1+(n-1):1+(n-1)+n]
    Delta = theta[-1]

    fig, axs = plt.subplots(2, 1, figsize=(10,4), gridspec_kw={'height_ratios':[2,1]})
    ax1, ax2 = axs

    # -----------------------------
    # Left panel: parity-resolved energy spectrum
    # -----------------------------
    ax1.hlines(y_even, -0.2, 0.2, color='tab:blue', label='Even')
    ax1.hlines(y_odd,  0.8, 1.2, color='tab:red', label='Odd')
    ax1.hlines(degeneracy_lines, 0.2, 0.8, color='tab:gray', linestyles='dashed', label='Degeneracies')
    ax1.set_xticks([0,1])
    ax1.set_xticklabels(['Even','Odd'])
    ax1.set_title(f"Parity-resolved spectrum ({n}-dot)")
    ax1.set_ylabel("Energy")
    ax1.legend(frameon=False)

    # -----------------------------
    # Right panel: schematic of QD–SC–QD chain
    # -----------------------------
    ax2.set_xlim(-0.5, n-0.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.axis('off')

    dot_y = 0
    sc_height = 0.5
    sc_width = 0.8

    # Draw dots (quantum dots)
    for i in range(n):
        ax2.scatter(i, dot_y, s=800, color='tab:blue', edgecolor='black', zorder=3)
        ax2.text(i, dot_y-0.5, f"$\\epsilon_{i}$={eps[i]:.2f}", ha='center', fontsize=9)

    # Draw superconductors (rectangles between dots)
    for i in range(n-1):
        x = (i + (i+1))/2
        rect = plt.Rectangle((x - sc_width/2, dot_y - sc_height/2), sc_width, sc_height,
                             color='lightgray', ec='black', lw=1.2, zorder=2)
        ax2.add_patch(rect)

        # Coupling t and Δ inside SC
        ax2.text(x, dot_y, f"t={t:.2f}\nΔ={Delta:.2f}", ha='center', va='center', fontsize=8)

        # U label above connection
        ax2.text(x, dot_y+0.6, f"$U_{i}$={U[i]:.2f}", ha='center', fontsize=9, color='tab:purple')

    ax2.set_title("Optimized QD–SC–QD setup")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # n = 2
    # best_theta, best_loss = optimize_params(n=n, device='cpu', restarts=5, iters=600)
    # plot_parity_spectrum(n, best_theta)
    for n in range(2,5):
        print("Optimizing for n =", n)
        best_theta, best_loss = optimize_params(n=n, device=device, restarts=5, iters=600)
        plot_parity_spectrum(n, best_theta)