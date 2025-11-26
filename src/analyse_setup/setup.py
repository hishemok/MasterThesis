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
    # print(f"\nNumerical Hamiltonian for n={n_sites}:\n", H_np)
    return H_np


def classify_states(eigenvalues, eigenvectors, n_sites):

    dim = eigenvectors.shape[0]

    # Build fermionic operators
    fermionic_ops = site_fermionic_operators(n_sites)
    n_ops   = [ops[2] for ops in fermionic_ops]
    c_ops   = [ops[1] for ops in fermionic_ops]
    cdag_ops= [ops[0] for ops in fermionic_ops]

    # Build Majorana operators
    majorana_ops = site_majorana_operators(n_sites)
    gamma1_ops = [ops[0] for ops in majorana_ops]
    gamma2_ops = [ops[1] for ops in majorana_ops]

    # Sort eigenstates into even/odd parity sectors
    parity_labels, even_states, odd_states, even_vals, odd_vals = classify_parities(
        eigenvalues, eigenvectors, n_sites
    )

    GS_even = even_states[0]
    GS_odd  = odd_states[0]

    gamma, gamma_tilde, igammagamma_tilde = operator_basis(GS_even, GS_odd)

    results = {}

    # Electron like?
    occupancies = np.zeros((dim, n_sites))
    for s in range(dim):
        v = eigenvectors[:, s]
        for i in range(n_sites):
            occupancies[s,i] = np.real(v.conj().T @ n_ops[i] @ v)
    results["electron_occupancy"] = occupancies

    # Majorana expectations
    maj_strength_1 = np.zeros(n_sites)
    maj_strength_2 = np.zeros(n_sites)
    for i in range(n_sites):
        maj_strength_1[i] = np.abs(GS_even.conj().T @ gamma1_ops[i] @ GS_odd)
        maj_strength_2[i] = np.abs(GS_even.conj().T @ gamma2_ops[i] @ GS_odd)
    results["majorana_transition_gamma1"] = maj_strength_1
    results["majorana_transition_gamma2"] = maj_strength_2

    # Andreev pairing
    pairings = np.zeros((dim, n_sites-1))
    for s in range(dim):
        v = eigenvectors[:, s]
        for i in range(n_sites-1):
            pairings[s,i] = np.abs(v.conj().T @ c_ops[i] @ c_ops[i+1] @ v)
    results["andreev_pairings"] = pairings

    return results



def operator_basis(even,odd):
    gamma = np.outer(even, odd.conj()) + np.outer(odd, even.conj())
    gamma_tilde = 1j * (np.outer(even, odd.conj()) - np.outer(odd, even.conj()))
    igammagamma_tilde = np.outer(even, even.conj()) - np.outer(odd, odd.conj())

    ## verify properties
    gamma2 = gamma @ gamma
    gamma_tilde2 = gamma_tilde @ gamma_tilde
    if not np.allclose(gamma2, gamma_tilde2, atol=1e-8):
        print("Warning: γ² and γ̃² are not equal")
    if not np.allclose(gamma2, np.outer(even, even.conj()) + np.outer(odd, odd.conj())  , atol=1e-8):
        print("Warning: γ² is not equal to identity operator in ground state subspace")

    return gamma, gamma_tilde, igammagamma_tilde

if __name__ == "__main__":
    n_sites = 3
    H = symbolic_hamiltonian(n_sites)
    # sp.pprint(H)
  
    best_config3 = get_best_config(3)

    Hnum = symbolic_hamiltonian_to_np(n_sites, best_config3)
    eigvals, eigvecs = np.linalg.eigh(Hnum)

    results = classify_states(eigvals, eigvecs, n_sites)
    electron_occupancies = results["electron_occupancy"]
    majorana_overlaps_gamma1 = results["majorana_transition_gamma1"]
    majorana_overlaps_gamma2 = results["majorana_transition_gamma2"]
    Andreev_pairings = results["andreev_pairings"]
    parity_labels, _, _, _, _ = classify_parities(eigvals, eigvecs, n_sites)

    # global info
    print("Majorana γ1 localization:", majorana_overlaps_gamma1)
    print("Majorana γ2 localization:", majorana_overlaps_gamma2)

    dim = eigvecs.shape[0]
    # per-state info
    for s in range(dim):
        print(f"--- State {s} ---")
        print(f"Energy = {eigvals[s]:.6f}, Parity = {parity_labels[s]}")
        print(f"  Occupancies      : {electron_occupancies[s]}")
        print(f"  Andreev pairing  : {Andreev_pairings[s]}")
