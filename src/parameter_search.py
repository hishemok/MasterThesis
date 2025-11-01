import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from functools import reduce

from n_particles_degen import n_dot_Hamiltonian, n_dot_sweetspot, σ_x, σ_y, σ_z, I, tensor_product, sigma_site, creation_annihilation

def to_torch(mat_np, device='cpu'):
    return torch.from_numpy(mat_np).to(device=device, dtype=torch.complex128)


def build_H_torch(n, params, cre_t, ann_t, num_t):
    # params: torch tensor (length 1 + (n-1) + n + 1)
    # mapping:
    # t = params[0]
    # U_i = params[1:1+(n-1)]
    # eps_i = params[1+(n-1):1+(n-1)+n]
    # Delta = params[-1]
    t = params[0]
    U = params[1].item()
    #params[1:1+(n-1)]
    # print(U)
    U = torch.full((n-1,), U, dtype=params.dtype, device=params.device)
    eps = params[1+(n-1):1+(n-1)+n]
    Delta = params[-1]
    dim = 2**n
    H = torch.zeros((dim,dim), dtype=torch.complex128, device=params.device)

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


def parity_operator_torch(n):
    P = torch.eye(2**n, dtype=torch.complex128)
    for j in range(n):
        cd, c = creation_annihilation(j, n)
        cd_t = to_torch(cd, device=P.device)
        c_t = to_torch(c, device=P.device)
        n_i = cd_t @ c_t
        P = P @ (torch.eye(2**n, dtype=torch.complex128, device=P.device) - 2 * n_i)
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

    return E_even, E_odd

def Majorana_purity_metric(n, eigenvecs, create_t, ann_t):
    """Used to calculate the Majorana purity metric for a set of eigenvectors
        M_n = ( ⟨c_n†⟩ - ⟨c_n⟩  ) / ( ⟨c_n†c_n⟩ + ⟨c_nc_n†⟩ )
        M -> 0 for pure Majorana state localized on site n
        M -> 1 for delocalized fermionic state
        M = ∑_n M_n
    ."""
    M = torch.zeros(n, dtype=torch.complex128, device=eigenvecs.device)
    for i in range(n):
        cd_t = create_t[i]
        c_t = ann_t[i]
        exp_cd = torch.sum(torch.conj(eigenvecs) * (cd_t @ eigenvecs), dim=0)
        exp_c  = torch.sum(torch.conj(eigenvecs) * (c_t @ eigenvecs), dim=0)
        exp_n  = torch.sum(torch.conj(eigenvecs) * (cd_t @ c_t @ eigenvecs), dim=0)
        exp_nbar = torch.sum(torch.conj(eigenvecs) * (c_t @ cd_t @ eigenvecs), dim=0)

        M[i] = torch.mean(torch.abs((exp_cd - exp_c) / (exp_n + exp_nbar + 1e-12)))

    M_total = torch.sum(torch.abs(M))
    return M_total.real

def degeneracy_and_gap_loss(H, P, n, weight_vec=None, gap_target=0.5, gap_weights=None):
    # H: Hermitian torch matrix
    evals, evecs = torch.linalg.eigh(H)  # evals sorted ascending

    even_states, odd_states = calculate_parities(evals, evecs, P)

    # Just in case
    E_even, _ = torch.sort(even_states)
    E_odd,  _ = torch.sort(odd_states)

    m = min(len(E_even), len(E_odd)) # Minimum number of elements
    if m == 0:
        return torch.tensor(1e6, device=H.device)  # bad configuration Penalty

    if len(E_even) != len(E_odd):
        print("Warning: unequal number of even and odd states:", len(E_even), len(E_odd))

    # default weights
    if weight_vec is None:
        weight_array = np.linspace(1, 1000, m)[::-1].copy()
        weight_vec = torch.tensor(weight_array, device=H.device)


    deg_terms = torch.abs(E_even[:m] - E_odd[:m]) # Cut out unequal lengths
    w = weight_vec.to(H.device)
    Ldeg = torch.sum(w * deg_terms) / torch.sum(w)

    # gap penalty: enforce minimal gaps in the full spectrum for the first p gaps
    p = min(6, evals.numel()-1)
    gaps = evals[1:] - evals[:-1]
    if gap_weights is None:
        gap_weights = torch.linspace(10.0, 1.0, steps=p, device=H.device)
    gap_pen = torch.sum(gap_weights * torch.nn.functional.softplus(gap_target - gaps[:p]))

    return 10 * Ldeg + 0.1 * gap_pen




def degeneracy_and_gap_loss2(H, P, n, weight_vec=None, gap_target=0.5, gap_weights=None, theta=None):
    # H: Hermitian torch matrix
    H = 0.5 * (H + H.conj().T) ## Just because the optimizer can introduce small non-Hermiticities

    evals, evecs = torch.linalg.eigh(H)  # evals sorted ascending

    even_states, odd_states = calculate_parities(evals, evecs, P)

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
        weight_vec = torch.tensor(weight_array, device=H.device)

    penalty_array = torch.zeros(2*m-1, device=H.device) # First half for Degeneracy, second half for gaps

    deg_terms = torch.abs(E_even - E_odd) # Cut out unequal lengths
    w = weight_vec.to(H.device)
    degeneracy_terms = w[:m] * deg_terms
    penalty_array[:m] = degeneracy_terms


    even_gaps = E_even[1:] - E_even[:-1]
    odd_gaps = E_odd[1:] - E_odd[:-1]
    worst_gaps = torch.min(torch.stack([even_gaps, odd_gaps]), dim=0).values

    # If gap < 0.5 penalize, else 0
    gap_penalties = torch.nn.functional.softplus(gap_target - worst_gaps)
    gap_terms = w[m:] * gap_penalties
    penalty_array[m:] = gap_terms

    cre_t, ann_t, _ = precompute_ops(n)
    cre_t = [to_torch(m, device=H.device) for m in cre_t]
    ann_t = [to_torch(m, device=H.device) for m in ann_t]   
    # Calculate Majorana purity metric for the eigenvectors
    Majorana_metric = Majorana_purity_metric(n, evecs, cre_t, ann_t)
    # Optimal is 0, so we can add it directly to the penalty
    total_penalty = torch.sum(penalty_array) + Majorana_metric
    return total_penalty


def print_params(theta, n):
    t = theta[0]
    U = theta[1:1+(n-1)]
    eps = theta[1+(n-1):1+(n-1)+n]
    Delta = theta[-1]
    print(f"t = {t:.4f}")
    print(f"U = {U}")
    print(f"ε = {eps}")
    print(f"Δ = {Delta:.4f}")




def optimize_params(n, device='cpu', restarts=8, iters=600):
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
        t0 = 0.5 + rng.rand()*1.5
        U0 = 0.5 + abs(rng.rand(n-1)*1.5)
        eps0 = -0.5 + rng.rand(n)*1.0
        D0 = 0.5 + rng.rand()*1.5
        
        theta0 = np.concatenate([[t0], U0, eps0, [D0]]).astype(np.float64)
        theta = torch.tensor(theta0, dtype=torch.float64, device=device, requires_grad=True)

        optimizer = torch.optim.Adam([theta], lr=0.05)
        for it in range(iters):
            optimizer.zero_grad()
            H = build_H_torch(n, theta, cre_t, ann_t, num_t)  # accepts real theta
            P = parity_operator_torch(n)
            loss = degeneracy_and_gap_loss(H, P, n)

            # --- Enforce known sweet-spot relations ---
            t = theta[0]
            U = theta[1:1+(n-1)]
            eps = theta[1+(n-1):1+(n-1)+n]
            Delta = theta[-1]

            eps_penalty = torch.mean((eps[0] + U[0]/2)**2 + (eps[-1] + U[-1]/2)**2)
            if n > 2:
                eps_penalty += torch.mean((eps[1:-1] + U[:-1])**2)
            Delta_penalty = torch.mean((Delta - (t + U[0]/2))**2)

            # Scale the penalty term
            loss += 0.1 *  (eps_penalty + Delta_penalty)
            # ------------------------------------------

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

def optimize_params2(n, device='cpu', restarts=8, iters=600):
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
        t0 = 0.5 + rng.rand()*.5
        # U0 = 0.5 + abs(rng.rand(n-1)*1.5)
        # eps0 = -0.5 + rng.rand(n)*1.0
        # D0 = 0.5 + rng.rand()*1.5
        U0 =  0.5 + abs(rng.rand()*5)
        U = [U0 for _ in range(n-1)]

        rnd1 = rng.rand()
        rnd2 = rng.rand()
        rnd3 = rng.rand()
        eps_ends = [-U0/2 * rnd1, -U0/2 * rnd1]
        if n > 2:
            eps_mids = [-U0 * rnd2] * (n-2)
            eps0 = np.array([eps_ends[0] , *eps_mids , eps_ends[1]])
        else:   
            eps0 = eps_ends
        D0 = t0 + U0/2 + rnd3

        theta0 = np.concatenate([[t0], U, eps0, [D0]]).astype(np.float64)
        theta = torch.tensor(theta0, dtype=torch.float64, device=device, requires_grad=True)

        optimizer = torch.optim.Adam([theta], lr=0.05)
        # optimizer = torch.optim.LBFGS([theta], lr=0.5, max_iter=100)
        # def closure():
        #     optimizer.zero_grad()
        #     with torch.no_grad():
        #         theta.data.clamp_(-10, 10)
        #     H = build_H_torch(n, theta, cre_t, ann_t, num_t)
        #     P = parity_operator_torch(n)
        #     loss = degeneracy_and_gap_loss2(H, P, n, theta=theta)
        #     loss.backward()
        #     return loss

        # # Outer loop for restarts / convergence monitoring
        # for it in range(iters):
        #     loss = optimizer.step(closure)  # LBFGS needs the closure here
    
        for it in range(iters):
            optimizer.zero_grad()
            H = build_H_torch(n, theta, cre_t, ann_t, num_t)  # accepts real theta
            P = parity_operator_torch(n)
            loss = degeneracy_and_gap_loss2(H, P, n)

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
    P = parity_operator_torch(n).cpu().numpy()
    parities = [np.real(np.vdot(ev, P @ ev)) for ev in evecs.T]

    print(f"{n}-dot system eigenvalues and their parity sectors:")
    for i, (E,p) in enumerate(zip(evals, parities)):
        if abs(p) < .9:
            print(f"Warning: Eigenvalue {E:.4f} has ambiguous parity {p:.4f}")
        sector = "even" if p > .9 else "odd"
        print(f"Eigenvalue {E:.4f} belongs to {sector} parity sector, with parity {p:.4f}")
    print("")

    cre_t, ann_t, _ = precompute_ops(n)
    cre_t = [to_torch(m, device=H.device) for m in cre_t]
    ann_t = [to_torch(m, device=H.device) for m in ann_t]   
    # Calculate Majorana purity metric for the eigenvectors
    Majorana_metric = Majorana_purity_metric(n, to_torch(evecs) , cre_t, ann_t)
    print(f"Majorana purity metric: {Majorana_metric:.6f}\n")


    evens = [(E, v) for E, v, p in zip(evals, evecs.T, parities) if p >= 0]
    odds  = [(E, v) for E, v, p in zip(evals, evecs.T, parities) if p <  0]
    y_even = [E for E,_ in evens]
    y_odd = [E for E,_ in odds]

    degeneracy_lines = []
    for i, Ee in enumerate(y_even):
        Eo = y_odd[i]
        if abs(Ee - Eo) < 1e-3:
            degeneracy_lines.append(Ee)


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
        best_theta, best_loss = optimize_params2(n=n, device='cpu', restarts=5, iters=600)
        plot_parity_spectrum(n, best_theta)