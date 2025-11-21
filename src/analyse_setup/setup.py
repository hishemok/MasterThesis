from get_configuration import get_configuration, get_best_config
from tools import *
import sympy as sp
from sympy import symbols
import numpy as np



def symbolic_hamiltonian(n_sites):
    epsilons = symbols(f'ϵ0:{n_sites}')
    t_values = symbols(f't0:{n_sites-1}')
    delta_values = symbols(rf'Δ0:{n_sites-1}')
    U_values = symbols(f'U0:{n_sites}')
    # print(epsilons, t_values, delta_values, U_values)

    ## Sympy Hamiltonian construction
    H = sp.zeros(2**n_sites)     
    for i in range(n_sites):
        f_dag_i, f_i = creation_annihilation_sympy(i, n_sites)
        n_i = f_dag_i * f_i
        H += epsilons[i] * n_i
        if i < n_sites - 1:
            f_dag_j, f_j = creation_annihilation_sympy(i + 1, n_sites)
            n_j = f_dag_j * f_j
            # hopping
            H += t_values[i] * (f_dag_i * f_j + f_dag_j * f_i)

            # pairing
            H += delta_values[i] * (f_dag_i * f_dag_j + f_j * f_i)

            # Coulomb term
            H += U_values[i] * n_i * n_j
    return sp.simplify(H)


def symbolic_hamiltonian_to_np(n_sites, param_dict):
    H_sym = symbolic_hamiltonian(n_sites)
    
    n = int(param_dict["n"])
    theta = param_dict["theta"]
    t_vals = np.array(theta["t"], dtype=float)
    U_vals = np.array(theta["U"], dtype=float)
    epsilons = np.array(theta["eps"], dtype=float)
    delta_vals = np.array(theta["Delta"], dtype=float)

    if n != n_sites:
        
        raise ValueError(f"Parameter dictionary n={n} does not match n_sites={n_sites}")
    if len(t_vals) != n - 1:
        if len(t_vals) == 1:
            t_vals = np.repeat(t_vals, n - 1)
            print(f"Expanded t_vals to: {t_vals}")
        else:
            raise ValueError(f"Length of t_vals {len(t_vals)} does not match n_sites-1={n_sites-1}")
    if len(U_vals) != n:
        if len(U_vals) == 1:
            U_vals = np.repeat(U_vals, n)
        else:
            raise ValueError(f"Length of U_vals {len(U_vals)} does not match n_sites={n_sites}")
    if len(epsilons) != n:
        if len(epsilons) == 1:
            epsilons = np.repeat(epsilons, n)
        else:
            raise ValueError(f"Length of epsilons {len(epsilons)} does not match n_sites={n_sites}")
    if len(delta_vals) != n - 1:
        if len(delta_vals) == 1:
            delta_vals = np.repeat(delta_vals, n - 1)
        else:
            raise ValueError(f"Length of delta_vals {len(delta_vals)} does not match n_sites-1={n_sites-1}")

    subs_dict = {}
    for i in range(n_sites):
        subs_dict[sp.symbols(f'ϵ{i}')] = epsilons[i]
        subs_dict[sp.symbols(f'U{i}')] = U_vals[i]
        if i < n_sites - 1:
            subs_dict[sp.symbols(f't{i}')] = t_vals[i]
            subs_dict[sp.symbols(rf'Δ{i}')] = delta_vals[i]
    H_num = H_sym.subs(subs_dict)
    H_np = np.array(H_num).astype(np.complex128)
    print(f"\nNumerical Hamiltonian for n={n_sites}:\n", H_np)
    return H_np



def classify_parities(eigenvalues, eigenvectors, n_sites):
    """Classify eigenstates by their parity using the parity operator."""
    P = parity_operator(n_sites)
    odd_states = []
    even_states = []
    parity_labels = []
    for i in range(len(eigenvalues)):
        state = eigenvectors[:, i]
        parity = state.conj().T @ P @ state
        if np.isclose(parity, 1):
            parity_labels.append("even")
            even_states.append(state)
        elif np.isclose(parity, -1):
            parity_labels.append("odd")
            odd_states.append(state)
        else:
            parity_labels.append("unknown")
    return parity_labels, even_states, odd_states

def majorana_operators(n):
    """Construct Majorana operators for n sites."""
    majoranas = []
    for j in range(n):
        f_dag_j, f_j = creation_annihilation(j, n)
        Gamma_1 = f_dag_j + f_j
        Gamma_2 = 1j * (f_dag_j - f_j)
        majoranas.append((Gamma_1, Gamma_2))
    return majoranas

def check_majorana_localization(even_state, odd_state, n):
    """Check the localization of Majorana modes by computing the overlap with site operators."""
    majoranas = majorana_operators(n)
    overlaps = []
    for j in range(n):
        Gamma_1, Gamma_2 = majoranas[j]
        overlap_1 = np.abs(even_state.conj().T @ Gamma_1 @ odd_state)
        overlap_2 = np.abs(even_state.conj().T @ Gamma_2 @ odd_state)
        overlaps.append((overlap_1, overlap_2))
    return overlaps



def classify_parities_and_find_ground_pair(eigvals, eigvecs, n_sites, tol=1e-8):
    P = parity_operator(n_sites)
    dim = eigvecs.shape[0]
    par_vals = np.array([np.vdot(eigvecs[:, i], P @ eigvecs[:, i]).real for i in range(eigvecs.shape[1])])

    even_idx = np.where(np.abs(par_vals - 1) < tol)[0]
    odd_idx  = np.where(np.abs(par_vals + 1) < tol)[0]
    # if some parities are numerically close but not exact, we still assign them
    
    if even_idx.size == 0 or odd_idx.size == 0:
        raise RuntimeError("No states detected in one parity sector. Check parity operator or tol.")

    # choose lowest-energy even and odd separately
    lowest_even_idx = even_idx[np.argmin(eigvals[even_idx])]
    lowest_odd_idx  = odd_idx[np.argmin(eigvals[odd_idx])]
    ground_pair = (lowest_even_idx, lowest_odd_idx)

    return {
        "par_vals": par_vals,
        "even_idx": even_idx,
        "odd_idx": odd_idx,
        "lowest_even_idx": lowest_even_idx,
        "lowest_odd_idx": lowest_odd_idx,
        "ground_pair": ground_pair
    }

def site_majorana_ops(n):
    """Return Γ_j = c_j + c_j^† and Γp_j = i(c_j - c_j^†) lists"""
    Gammas = []
    Gammas_p = []
    for j in range(n):
        fdag, f = creation_annihilation(j, n)
        Gammas.append(f + fdag)                # hermitian
        Gammas_p.append(1j*(f - fdag))         # hermitian
    return Gammas, Gammas_p

def compute_majorana_matrix_elements(eigvecs, idx_even, idx_odd, n_sites):
    psi_e = eigvecs[:, idx_even]
    psi_o = eigvecs[:, idx_odd]
    Gammas, Gammas_p = site_majorana_ops(n_sites)

    M = np.zeros(n_sites, dtype=complex)
    Mp = np.zeros(n_sites, dtype=complex)
    for j in range(n_sites):
        M[j] = np.vdot(psi_e, Gammas[j] @ psi_o)
        Mp[j] = np.vdot(psi_e, Gammas_p[j] @ psi_o)
    return M, Mp

def compute_local_densities(eigvec, n_ops):
    return np.array([np.vdot(eigvec, n_ops[j] @ eigvec).real for j in range(len(n_ops))])

def compute_spectral_weights(eigvecs, psi_gs, c_ops, low_idx):
    # returns list of lists: for each site j -> list of (state_idx, weight, energy)
    weights = {j: [] for j in range(len(c_ops))}
    for j in range(len(c_ops)):
        for idx in low_idx:
            psi = eigvecs[:, idx]
            amp = np.vdot(psi, c_ops[j] @ psi_gs)
            weights[j].append((idx, np.abs(amp)**2))
    return weights

def project_low_energy_subspace(H_np, basis_vecs):
    # basis_vecs: list/array of column vectors (dim x m)
    B = np.column_stack(basis_vecs)
    H_eff = B.conj().T @ (H_np @ B)
    return H_eff

def find_optimal_majorana_via_svd(eigvecs, even_idx, odd_idx, Gammas):
    """
    Build matrix A_{j, (e,o)} = <even_e | Gamma_j | odd_o> for chosen low even/odd sets.
    Do SVD to find linear combination of Gammas (vector over sites) that best maps even->odd.
    """
    # use only the lowest even and odd for now or several if provided
    Elist = np.atleast_1d(even_idx)
    Olist = np.atleast_1d(odd_idx)
    m = len(Gammas)
    A = np.zeros((m, len(Elist)*len(Olist)), dtype=complex)
    col = 0
    for e_idx in Elist:
        for o_idx in Olist:
            psi_e = eigvecs[:, e_idx]
            psi_o = eigvecs[:, o_idx]
            for j in range(m):
                A[j, col] = np.vdot(psi_e, Gammas[j] @ psi_o)
            col += 1
    # SVD on A
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    # U columns are site-combination principal vectors. s[0] large -> clear localized operator.
    best_site_vector = U[:, 0]      # coefficients for Gammas
    return best_site_vector, s

# ---------- main analysis function ----------
def analyze_H(H_np, n_sites, n_low=8, tol=1e-8, verbose=True):
    dim = H_np.shape[0]
    eigvals, eigvecs = np.linalg.eigh(H_np)
    info = classify_parities_and_find_ground_pair(eigvals, eigvecs, n_sites, tol=tol)
    idx_e, idx_o = info["lowest_even_idx"], info["lowest_odd_idx"]
    if verbose:
        print("Lowest even idx, E:", idx_e, eigvals[idx_e])
        print("Lowest odd  idx, E:", idx_o, eigvals[idx_o])
        print("Even-odd splitting:", abs(eigvals[idx_e] - eigvals[idx_o]))

    # operators
    c_ops = [creation_annihilation(j, n_sites)[1] for j in range(n_sites)]   # annihilation
    cdag_ops = [creation_annihilation(j, n_sites)[0] for j in range(n_sites)]
    n_ops = [cdag_ops[j] @ c_ops[j] for j in range(n_sites)]
    Gammas, Gammas_p = site_majorana_ops(n_sites)

    # majorana matrix elements between GS pair
    M, Mp = compute_majorana_matrix_elements(eigvecs, idx_e, idx_o, n_sites)
    if verbose:
        for j in range(n_sites):
            print(f"Site {j}: |<e|Γ|o>| = {abs(M[j]):.4e}, |<e|Γ'|o>| = {abs(Mp[j]):.4e}")

    # local densities (for even and odd GS)
    n_even = compute_local_densities(eigvecs[:, idx_e], n_ops)
    n_odd  = compute_local_densities(eigvecs[:, idx_o], n_ops)
    if verbose:
        print("Local occupancies (even GS):", np.round(n_even, 4))
        print("Local occupancies (odd  GS):", np.round(n_odd, 4))
        print("Delta n (even-odd):", np.round(n_even - n_odd, 4))

    # spectral weights on a few lowest eigenstates
    low_idx = np.argsort(eigvals)[:min(n_low, len(eigvals))]
    spec_weights = compute_spectral_weights(eigvecs, eigvecs[:, idx_e], c_ops, low_idx)
    if verbose:
        for j in range(n_sites):
            print(f"Site {j} spectral weights (idx, weight):", spec_weights[j])

    # H_eff projection into small subspace [even_gs, odd_gs, next_low] optionally
    # choose a candidate mid state (lowest excited state that's not the ground pair)
    sorted_idx = np.argsort(eigvals)
    # remove the chosen even/odd ground indices from sorted list
    sorted_low = [i for i in sorted_idx if i not in (idx_e, idx_o)]
    mid_idx = sorted_low[0] if len(sorted_low) > 0 else None
    if mid_idx is not None:
        basis = [eigvecs[:, idx_e], eigvecs[:, idx_o], eigvecs[:, mid_idx]]
        H_eff = project_low_energy_subspace(H_np, basis)
        if verbose:
            print("Projected H_eff (3x3) in basis [even, odd, mid]:\n", np.round(H_eff, 6))

    # SVD to find best local Majorana from site Gammas
    best_site_vector, singular_vals = find_optimal_majorana_via_svd(eigvecs, [idx_e], [idx_o], Gammas)
    if verbose:
        print("Singular values of operator mapping even->odd:", singular_vals)
        print("Best site-coefficients for Gamma (complex) (localization profile):", np.round(best_site_vector, 4))

    results = {
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "ground_even_idx": idx_e,
        "ground_odd_idx": idx_o,
        "M": M,
        "Mp": Mp,
        "n_even": n_even,
        "n_odd": n_odd,
        "spec_weights": spec_weights,
        "mid_idx": mid_idx,
        "H_eff": H_eff if mid_idx is not None else None, #type:ignore
        "best_site_vector": best_site_vector,
        "singular_vals": singular_vals
    }
    return results

# ---------- Example usage ----------
# results = analyze_H(Hnum, n_sites=3, n_low=8, verbose=True)
# print("Interpret |M_j| (large on ends -> Majorana localized on ends).")

if __name__ == "__main__":
    n_sites = 3
    H = symbolic_hamiltonian(n_sites)
    sp.pprint(H)
  
    best_config3 = get_best_config(3)

    Hnum = symbolic_hamiltonian_to_np(n_sites, best_config3)
    eigvals, eigvecs = np.linalg.eigh(Hnum)
    # parities, even_states, odd_states = classify_parities(eigvals, eigvecs, n_sites)
    # for i in range(len(eigvals)):
    #     print(f"Eigenvalue: {eigvals[i]:.4f}, Parity: {parities[i]}")
    results = analyze_H(Hnum, n_sites, n_low=8, verbose=True)
    # for i in range(len(even_states)):
    #     even_state = even_states[i]
    #     odd_state = odd_states[i]
    #     charge_diff = check_majorana_localization(even_state, odd_state, n_sites)
    #     print(f"Charge difference between even and odd state {i}: {charge_diff}")