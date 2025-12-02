from get_configuration import get_configuration, get_best_config
from tools import *
import sympy as sp
from sympy import symbols
import numpy as np
import matplotlib.pyplot as plt


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

    results = {}

    ## electron like?
    even_electron_weight = np.zeros((n_sites,len(even_states)))
    odd_electron_weight = np.zeros((n_sites,len(odd_states)))
    #Dim: [n_sites, len(even_states)] = [i,s]
    #Holds: ⟨v_s| n_i |v_s⟩
    for i in range(n_sites):
        n_i = n_ops[i]
        for s in range(len(even_states)):
            e = even_states[s]
            even_electron_weight[i,s] = np.real(e.conj().T @ n_i @ e)
            o = odd_states[s]
            odd_electron_weight[i,s] = np.real(o.conj().T @ n_i @ o)
    results["even_electron_occupancy"] = even_electron_weight 
    results["odd_electron_occupancy"] = odd_electron_weight

    ### Majorana matrix elements
    maj_strength_1 = np.zeros((n_sites,len(odd_states)))  # ⟨ o_s | γ₁_i | e_s ⟩
    maj_strength_2 = np.zeros((n_sites,len(odd_states)))  # ⟨ o_s | γ₂_i | e_s ⟩
    for i in range(n_sites):
        gamma_1 = gamma1_ops[i]
        gamma_2 = gamma2_ops[i]
        for s in range(len(odd_states)):
            even_v = even_states[s]
            odd_v  = odd_states[s]
            maj_strength_1[i,s] = np.abs(even_v.conj().T @ gamma_1 @ odd_v)
            maj_strength_2[i,s] = np.abs(even_v.conj().T @ gamma_2 @ odd_v)
    results["majorana_transition_gamma1"] = maj_strength_1
    results["majorana_transition_gamma2"] = maj_strength_2

    ### Andreev pairing
    even_pairings = np.zeros((n_sites-1, len(even_states)))  # ⟨ v_s | c_i c_{i+1} | v_s ⟩
    odd_pairings  = np.zeros((n_sites-1, len(odd_states)))   # ⟨ v_s | c_i c_{i+1} | v_s ⟩
    for i in range(n_sites - 1):
        c_i = c_ops[i]
        c_j = c_ops[i + 1]
        for s in range(len(even_states)):
            v = even_states[s]
            even_pairings[i,s] = np.abs(v.conj().T @ c_i @ c_j @ v)
            v = odd_states[s]
            odd_pairings[i,s] = np.abs(v.conj().T @ c_i @ c_j @ v)
    results["andreev_pairings_even"] = even_pairings
    results["andreev_pairings_odd"] = odd_pairings

    ## Hoppings
    even_hoppings = np.zeros((n_sites - 1, len(even_states)))  # ⟨ v_s | c_i† c_{i+1} | v_s ⟩
    odd_hoppings  = np.zeros((n_sites - 1, len(odd_states)))   # ⟨ v_s | c_i† c_{i+1} | v_s ⟩
    for i in range(n_sites - 1):    
        cdag_i = cdag_ops[i]
        c_j    = c_ops[i + 1]
        for s in range(len(even_states)):
            v = even_states[s]
            even_hoppings[i,s] = np.abs(v.conj().T @ cdag_i @ c_j @ v)
            v = odd_states[s]
            odd_hoppings[i,s] = np.abs(v.conj().T @ cdag_i @ c_j @ v)
    results["hoppings_even"] = even_hoppings
    results["hoppings_odd"] = odd_hoppings

    ### Charge difference between even and odd states
    charge_differences = np.zeros((len(even_states),))
    for s in range(len(even_states)):
        even_v = even_states[s]
        odd_v  = odd_states[s]
        charge_even = 0.0
        charge_odd = 0.0
        for i in range(n_sites):
            n_i = n_ops[i]
            charge_even += np.real(even_v.conj().T @ n_i @ even_v)
            charge_odd  += np.real(odd_v.conj().T @ n_i @ odd_v)
            charge_differences[s] += np.abs(charge_even - charge_odd)
    results["charge_differences"] = charge_differences

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
    n_sites = 4
    H = symbolic_hamiltonian(n_sites)
    # sp.pprint(H)
  
    configurations = get_configuration()
    print(f"Available configurations for n={n_sites}:")
    # print(configurations["n3"][2])
    for config in configurations[f"n{n_sites}"]:
        print(f"Loss: {config['loss']}, Theta: {config['theta']}")

    # best_config3 = get_best_config(2)
    # print("Best configuration for n=3:")
    # print(best_config3["theta"])


    Hnum = symbolic_hamiltonian_to_np(n_sites, configurations[f"n{n_sites}"][0])
    eigvals, eigvecs = np.linalg.eigh(Hnum)

    results = classify_states(eigvals, eigvecs, n_sites)
    even_electron_occupancy = results["even_electron_occupancy"].T # [n_sites, len(even_states)].T ⟨v_s| n_i |v_s⟩
    odd_electron_occupancy = results["odd_electron_occupancy"].T # [n_sites, len(odd_states)].T ⟨v_s| n_i |v_s⟩
    maj1 = results["majorana_transition_gamma1"].T # [n_sites, len(odd_states)].T ⟨ o_s | γ₁_i | e_s ⟩
    maj2 = results["majorana_transition_gamma2"].T # [n_sites, len(odd_states)].T ⟨ o_s | γ₂_i | e_s ⟩
    pairings_even = results["andreev_pairings_even"].T # [n_sites-1, len(even_states))].T ⟨ v_s | c_i c_{i+1} | v_s ⟩
    pairings_odd  = results["andreev_pairings_odd"].T  # [n_sites-1, len(odd_states))].T ⟨ v_s | c_i c_{i+1} | v_s ⟩
    hoppings_even = results["hoppings_even"].T       # [n_sites-1, len(even_states))].T ⟨ v_s | c_i† c_{i+1} | v_s ⟩
    hoppings_odd  = results["hoppings_odd"].T        # [n_sites-1, len(odd_states))].T ⟨ v_s | c_i† c_{i+1} | v_s ⟩
    charge_differences = results["charge_differences"]  # [len(even_states)]  |⟨v_s| Q |v_s⟩ - ⟨u_s| Q |u_s⟩|
   
    print("Charge differences between even and odd states:")
    print(charge_differences)



    majorana_ticks_sites = np.arange(n_sites)
    majorana_ticks_labels1 = [f"⟨o|γ1,{i}|e⟩ " for i in range(n_sites)]
    majorana_ticks_labels2 = [f"⟨o|γ2,{i}|e⟩ " for i in range(n_sites)]

    electron_ticks_sites = np.arange(n_sites)
    electron_ticks_label_even = [f"⟨e|n_{i}|e⟩" for i in range(n_sites)]
    electron_ticks_label_odd  = [f"⟨o|n_{i}|o⟩" for i in range(n_sites)]

    andreev_n_bonds = n_sites - 1
    xticks_sites = np.arange(andreev_n_bonds)
    andreev_even = [f"⟨e|c_{i} c_{i+1}|e⟩" for i in range(andreev_n_bonds)]
    andreev_odd  = [f"⟨o|c_{i} c_{i+1}|o⟩" for i in range(andreev_n_bonds)]

    hoppings_even_tics = [f"⟨e|c†_{i} c_{i+1}|e⟩" for i in range(andreev_n_bonds)]
    hoppings_odd_tics  = [f"⟨o|c†_{i} c_{i+1}|o⟩" for i in range(andreev_n_bonds)]


    y_ticks_sites = np.arange(len(even_electron_occupancy))
    y_ticks_labels = [f"E_{i}" for i in range(len(even_electron_occupancy))]

    plt.subplot(4,2,1)
    plt.imshow(maj1, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("Majorana Transition Amplitudes γ₁")
    # plt.xlabel("Odd Eigenstate Index")
    plt.yticks(ticks=y_ticks_sites, labels=y_ticks_labels)
    plt.xticks(ticks=majorana_ticks_sites, labels=majorana_ticks_labels1)
    plt.ylabel("Energy level Index")

    plt.subplot(4,2,2)
    plt.imshow(maj2, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("Majorana Transition Amplitudes γ₂")
    # plt.xlabel("Odd Eigenstate Index")
    plt.yticks(ticks=y_ticks_sites, labels=y_ticks_labels)
    plt.xticks(ticks=majorana_ticks_sites, labels=majorana_ticks_labels2)
    plt.ylabel("Energy level Index")

    plt.subplot(4,2,3)
    plt.imshow(even_electron_occupancy, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("Even Parity Electron Occupancy")
    # plt.xlabel("Even Eigenstate Index")
    plt.xticks(ticks=electron_ticks_sites, labels=electron_ticks_label_even)
    plt.yticks(ticks=y_ticks_sites, labels=y_ticks_labels)
    plt.ylabel("Energy level Index")

    plt.subplot(4,2,4)
    plt.imshow(odd_electron_occupancy, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("Odd Parity Electron Occupancy")
    # plt.xlabel("Odd Eigenstate Index")
    plt.xticks(ticks=electron_ticks_sites, labels=electron_ticks_label_odd)
    plt.yticks(ticks=y_ticks_sites, labels=y_ticks_labels)
    plt.ylabel("Energy level Index")

    plt.subplot(4,2,5)
    plt.imshow(pairings_even, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("Andreev Pairings Even")       
    # plt.xlabel("Eigenstate Index")
    plt.xticks(ticks=xticks_sites, labels=andreev_even)
    plt.yticks(ticks=y_ticks_sites, labels=y_ticks_labels)
    plt.ylabel("Bond Index")

    plt.subplot(4,2,6)
    plt.imshow(pairings_odd, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("Andreev Pairings Odd")         
    # plt.xlabel("Eigenstate Index")
    plt.xticks(ticks=xticks_sites, labels=andreev_odd)
    plt.yticks(ticks=y_ticks_sites, labels=y_ticks_labels)
    plt.ylabel("Bond Index")

    plt.subplot(4,2,7)
    plt.imshow(hoppings_even, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("Hoppings Even")           
    # plt.xlabel("Eigenstate Index")
    plt.xticks(ticks=xticks_sites, labels=hoppings_even_tics)
    plt.yticks(ticks=y_ticks_sites, labels=y_ticks_labels)
    plt.ylabel("Bond Index")

    plt.subplot(4,2,8)
    plt.imshow(hoppings_odd, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("Hoppings Odd")             
    # plt.xlabel("Eigenstate Index")
    plt.xticks(ticks=xticks_sites, labels=hoppings_odd_tics)
    plt.yticks(ticks=y_ticks_sites, labels=y_ticks_labels)
    plt.ylabel("Bond Index")

    plt.tight_layout()
    plt.show()
