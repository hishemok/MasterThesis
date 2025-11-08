import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


"""
TODO: 
Next majorana check functions:
Charge diff: ∑_n |⟨e_k|n_n|e_k⟩ - ⟨o_k|n_n|o_k⟩|
M tilde: \tilde{M_k^n} = ∑_s ⟨o_k|Γ^n_s|e_k⟩; should go between 1 and -1 and also 0 where the majorana is not well localized. Plot for k energy levels where x-axis is site n.
Check non-interacting against interacting.
"""

## Pauli spin matrices
σ_x = np.array([[0, 1], 
                [1, 0]])
σ_y = np.array([[0, -1j], 
                [1j, 0]])
σ_z = np.array([[1, 0], 
                [0, -1]])
I = np.eye(2)

def tensor_product(*args):
    """Compute the tensor product of multiple matrices."""
    result = args[0]
    for mat in args[1:]:
        result = np.kron(result, mat)
    return result   

def sigma_site(j, n, operator):
    """Return the operator for site j in a chain of n sites."""
    ops = [I] * n
    ops[j] = operator
    return tensor_product(*ops)


def creation_annihilation(j,n):
    """Return the creation and annihilation operators for site j in a chain of n sites, after Jordan-Wigner transformation."""
    f_dag = (sigma_site(j, n, σ_x) + 1j * sigma_site(j, n, σ_y)) * 0.5
    f = (sigma_site(j, n, σ_x) - 1j * sigma_site(j, n, σ_y)) * 0.5

    Zops = np.eye(2**n) 
    for i in range(0, j):
        Zops @= sigma_site(i, n, -σ_z)

    create = Zops @ f_dag 
    annihilate = Zops @ f 

    return create, annihilate

def creation_annihilation_sympy(j, n):
    """Return symbolic JW creation/annihilation operators for site j."""
    f_dag_np, f_np = creation_annihilation(j, n)
    # Convert to SymPy matrices
    f_dag = sp.Matrix(f_dag_np)
    f = sp.Matrix(f_np)
    return f_dag, f

def n_dot_Hamiltonian(n=2):
    t, Δ = sp.symbols('t Δ', real=True)
    H = sp.zeros(2**n)
    ϵs = sp.symbols(f'ϵ0:{n}')
    Us = sp.symbols(f'U0:{n-1}')


    dots = n 
    for i in range(dots):
        f_dag_i, f_i = creation_annihilation_sympy(i, n)
        n_i = f_dag_i * f_i
        H += ϵs[i] * n_i
        if i < dots - 1:
            f_dag_j, f_j = creation_annihilation_sympy(i + 1, n)
            n_j = f_dag_j * f_j
            # hopping
            H += t * (f_dag_i * f_j + f_dag_j * f_i)

            # pairing
            H += Δ * (f_dag_i * f_dag_j + f_j * f_i)

            # Coulomb term
            H += Us[i] * n_i * n_j
    return sp.simplify(H)


def parity_operator(N):
    """Construct the parity operator for N dots."""
    P = np.eye(2**N, dtype=complex)
    for i in range(N):
        f_dag_i, f_i = creation_annihilation_sympy(i, N)
        n_i = f_dag_i * f_i
        P = P @ (np.eye(2**N) - 2 * n_i)
    return P

def charge_difference(even_state, odd_state, n):
    """Compute the charge difference between even and odd states.
    ∑_n |⟨e_k|n_n|e_k⟩ - ⟨o_k|n_n|o_k⟩|
    """
    charge_diff = 0
    for i in range(n):
        f_dag_i, f_i = creation_annihilation(i, n)
        n_i = f_dag_i @ f_i
        exp_even = np.vdot(even_state, n_i @ even_state)
        exp_odd  = np.vdot(odd_state,  n_i @ odd_state)
        charge_diff += np.abs(exp_even - exp_odd)
    return charge_diff.real



def construct_Gamma_operators(j,n):
    """Construct the Majorana operators Γ^s_j for site j in a chain of n sites."""
    f_dag_j, f_j = creation_annihilation(j, n)
    Gamma_1 = f_dag_j + f_j
    Gamma_2 = 1j * (f_dag_j - f_j)
    return Gamma_1, Gamma_2

def Majorana_polarization(even_vecs, odd_vecs, n):
    """
    Compute local Majorana polarization M_j for each site j.
    M_j = sum_s (<o|Γ^s_j|e>)^2 / sum_s |<o|Γ^s_j|e>|^2
    """
    M = np.zeros(n, dtype=complex)

    for j in range(n):
        Gamma_1, Gamma_2 = construct_Gamma_operators(j, n)
        numerator = 0.0 + 0.0j
        denominator = 0.0
        for i in range(even_vecs.shape[1]):
            e_k = even_vecs[:, i]
            o_k = odd_vecs[:, i]
            term1 = np.vdot(o_k, Gamma_1 @ e_k)
            term2 = np.vdot(o_k, Gamma_2 @ e_k)
            numerator   += term1**2 + term2**2
            denominator += abs(term1)**2 + abs(term2)**2
        M[j] = numerator / denominator if denominator != 0 else 0.0
    return M 

def n_dot_sweetspot(n, params):
    """
    Generalized n-dot sweet spot analysis.
    n: number of dots
    params: dictionary with keys 'U', 't', 'Δ', 'ϵ' containing lists of values
    ϵ should be a list of length n
    U should be a list of length n-1
    t and Δ are scalars
    """ 
    Hn = n_dot_Hamiltonian(n=n)

    # Symbols
    ϵs = sp.symbols(f'ϵ0:{n}')
    Us = sp.symbols(f'U0:{n-1}')
    Δ, t = sp.symbols('Δ t', real=True)

    subs_dict = {Δ: params['Δ'], t: params['t']}

    # Add epsilon substitutions
    subs_dict.update({ϵs[i]: params['ϵ'][i] for i in range(n)})

    # Add U substitutions
    subs_dict.update({Us[i]: params['U'][i] for i in range(n - 1)})

    # Substitute into the Hamiltonian
    Hn_num = Hn.subs(subs_dict)

    # Convert sympy.Matrix to numpy array (float complex)
    Hn_np = np.array(Hn_num).astype(np.complex128)

    # Eigen decomposition
    eigvals_np, eigvecs_np = np.linalg.eigh(Hn_np)

    P = parity_operator(n)
    parities = []
    for vec in eigvecs_np.T:
        parity = np.vdot(vec, P @ vec)
        parities.append(np.real_if_close(parity))

    print(f"{n}-dot system eigenvalues and their parity sectors:")
    for i, (E,p) in enumerate(zip(eigvals_np, parities)):
        if abs(p) < .9:
            print(f"Warning: Eigenvalue {E:.4f} has ambiguous parity {p:.4f}")
        sector = "even" if p > 0.9 else "odd"
        print(f"Eigenvalue {E:.4f} belongs to {sector} parity sector, with parity {p:.4f}")
    print("")

    even_states = [(E, v) for E, v, p in zip(eigvals_np, eigvecs_np.T, parities) if p > .1]
    odd_states  = [(E, v) for E, v, p in zip(eigvals_np, eigvecs_np.T, parities) if p < -.1]

    # Compute Majorana localization
    even_vecs = np.array([v for _, v in even_states]).T
    odd_vecs  =  np.array([v for _, v in odd_states]).T

    charge_diff = charge_difference(even_vecs, odd_vecs, n)
    print(f"Charge difference between lowest even and odd states: {charge_diff:.4f}")


    M = Majorana_polarization(even_vecs, odd_vecs, n)
    plt.plot(range(n), M, 'o-', label='|M̃₀|')
    plt.xlabel('Site index')
    plt.ylabel('Majorana localization')
    plt.title(f'Majorana localization profile for {n}-dot system')
    plt.axhline(0, color='k', linestyle='--')
    plt.legend()
    plt.show()

    # Plot energy levels by parity
    plt.figure(figsize=(6,4))
    y_even = [E for E, _ in even_states]
    y_odd  = [E for E, _ in odd_states]

    degeneracy_lines = []
    for i, Ee in enumerate(y_even):
        Eo = y_odd[i]
        if abs(Ee - Eo) < 1e-5:
            degeneracy_lines.append(Ee)


    plt.hlines(y_even, xmin=-0.2, xmax=0.2, color='blue', label='Even parity')
    plt.hlines(y_odd,  xmin=0.8, xmax=1.2, color='red',  label='Odd parity')
    plt.hlines(degeneracy_lines, xmin=-0.2, xmax=0.8, color='gray', linestyles='dashed', label='Degenerate levels')

    plt.xlim(-0.5, 1.5)
    plt.xticks([0, 1], ['Even', 'Odd'])
    plt.ylabel("Energy eigenvalues")
    plt.title(f"Parity-resolved energy spectrum for {n}-dot system")
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    return eigvals_np, parities

def two_dot_sweetspot(n,params):
    Hn = n_dot_Hamiltonian(n=n)

    # Symbols
    ϵs = sp.symbols(f'ϵ0:{n}')
    Us = sp.symbols(f'U0:{n-1}')
    Δ, t = sp.symbols('Δ t', real=True)

    subs_dict = {Δ: params['Δ'], t: params['t']}

    # Add epsilon substitutions
    subs_dict.update({ϵs[i]: params['ϵ'][i] for i in range(n)})

    # Add U substitutions
    subs_dict.update({Us[i]: params['U'][i] for i in range(n - 1)})

    # Substitute into the Hamiltonian
    Hn_num = Hn.subs(subs_dict)

    # Convert sympy.Matrix to numpy array (float complex)
    Hn_np = np.array(Hn_num).astype(np.complex128)

    # Eigen decomposition
    eigvals_np, eigvecs_np = np.linalg.eigh(Hn_np)
    P = parity_operator(n)
    parities = []
    for vec in eigvecs_np.T:
        parity = np.vdot(vec, P @ vec)
        parities.append(np.real_if_close(parity))

    
    even_states = [(E, v) for E, v, p in zip(eigvals_np, eigvecs_np.T, parities) if p > .9]
    odd_states  = [(E, v) for E, v, p in zip(eigvals_np, eigvecs_np.T, parities) if p < -.9]

    degeneracies = []
    for i, (Ee, ve) in enumerate(even_states):
        for j, (Eo, vo) in enumerate(odd_states):
            degeneracies.append((abs(Ee - Eo), Ee, Eo))
    
    degeneracy_sum = sum([d[0] for d in degeneracies])
    return degeneracy_sum







if __name__ == "__main__":
    # two_dot_sweetspot()

    U0_val = 10
    t_val = 0.1
    Δ_val = t_val + U0_val/2
    ϵ0_val = -U0_val/2
    ϵ1_val = -U0_val

    params = {
        'U': [U0_val],
        't': t_val,
        'Δ': Δ_val,
        'ϵ': [ϵ0_val, ϵ0_val]
    }
    n_dot_sweetspot(2, params)

    # three_dot_sweetspot()

    # U0_val = 1
    # t_val = 1
    # Δ_val = t_val + U0_val/2
    # ϵ0_val = -U0_val/2
    # ϵ1_val = -U0_val/2
    params = {
        'U': [U0_val, U0_val],
        't': t_val,
        'Δ': Δ_val,
        'ϵ': [ϵ0_val, ϵ1_val, ϵ0_val]
    }
    n_dot_sweetspot(3, params)


    # four_dot_sweetspot()

    # U0_val = 1
    # t_val = 1
    # Δ_val = t_val + U0_val/2
    # ϵ0_val = -U0_val/2
    # ϵ1_val = -U0_val/2
    params = {
        'U': [U0_val, U0_val, U0_val],
        't': t_val,
        'Δ': Δ_val,
        'ϵ': [ϵ0_val, ϵ1_val, ϵ1_val, ϵ0_val]
    }
    n_dot_sweetspot(4, params)


    # params = {
    #     'U': [U0_val, U0_val , U0_val, U0_val],
    #     't': t_val,
    #     'Δ': Δ_val,
    #     'ϵ': [ϵ0_val, ϵ1_val, ϵ1_val, ϵ1_val, ϵ0_val]
    # }
    # n_dot_sweetspot(5, params)