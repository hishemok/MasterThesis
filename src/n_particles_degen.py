import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# μ = sp.symbols('μ')
# t = sp.symbols('t', real=True)
# Δ = sp.symbols('Δ', real=True)
# U = sp.symbols('U', real=True)

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
        


def two_dot_sweetspot():
    H2 = n_dot_Hamiltonian(n=2)  

    # Symbols
    ϵs = sp.symbols('ϵ0 ϵ1')
    Us = sp.symbols('U0')
    Δ, t = sp.symbols('Δ t', real=True)

    # Sweet spot parameters
    U0_val = 1
    t_val = 1
    Δ_val = t_val + U0_val/2
    ϵ0_val = -U0_val/2
    ϵ1_val = -U0_val/2

    # Substitute parameters into symbolic matrix
    H2_num = H2.subs({Δ: Δ_val, t: t_val, ϵs[0]: ϵ0_val, ϵs[1]: ϵ1_val, Us: U0_val})

    # Convert sympy.Matrix to numpy array (float complex)
    H2_np = np.array(H2_num).astype(np.complex128)

    # Eigen decomposition
    eigvals_np, eigvecs_np = np.linalg.eigh(H2_np)

    P = parity_operator(2)
    parities = []
    for vec in eigvecs_np.T:
        parity = np.vdot(vec, P @ vec)
        parities.append(np.real_if_close(parity))
    print("Two-dot system eigenvalues and their parity sectors:")
    for i, (E,p) in enumerate(zip(eigvals_np, parities)):
        sector = "even" if np.isclose(p, 1) else "odd"
        print(f"Eigenvalue {E:.4f} belongs to {sector} parity sector")
    print("")

    even_states = [(E, v) for E, v, p in zip(eigvals_np, eigvecs_np.T, parities) if np.isclose(p, 1)]
    odd_states  = [(E, v) for E, v, p in zip(eigvals_np, eigvecs_np.T, parities) if np.isclose(p, -1)]

    # Plot energy levels by parity
    plt.figure(figsize=(6,4))
    y_even = [E for E, _ in even_states]
    y_odd  = [E for E, _ in odd_states]

    plt.hlines(y_even, xmin=-0.2, xmax=0.2, color='blue', label='Even parity')
    plt.hlines(y_odd,  xmin=0.8, xmax=1.2, color='red',  label='Odd parity')

    plt.xlim(-0.5, 1.5)
    plt.xticks([0, 1], ['Even', 'Odd'])
    plt.ylabel("Energy eigenvalues")
    plt.title("Parity-resolved energy spectrum for 2-dot system")
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    return eigvals_np, parities
    



def three_dot_sweetspot():
    H3 = n_dot_Hamiltonian(n=3)

    # Symbols
    ϵs = sp.symbols('ϵ0 ϵ1 ϵ2')
    Us = sp.symbols('U0 U1')
    Δ, t = sp.symbols('Δ t', real=True)

    # Sweet spot parameters
    U0_val = 1
    U1_val = 1
    t_val = 1
    Δ_val = t_val + U0_val/2
    ϵ0_val = -U0_val/2
    ϵ1_val = -U0_val
    ϵ2_val = -U0_val/2

    # Substitute parameters into symbolic matrix
    H3_num = H3.subs({Δ: Δ_val, t: t_val, ϵs[0]: ϵ0_val, ϵs[1]: ϵ1_val, ϵs[2]: ϵ2_val, Us[0]: U0_val, Us[1]: U1_val})

    # Convert sympy.Matrix to numpy array (float complex)
    H3_np = np.array(H3_num).astype(np.complex128)

    # Eigen decomposition
    eigvals_np, eigvecs_np = np.linalg.eigh(H3_np)

    P = parity_operator(3)
    parities = []
    for vec in eigvecs_np.T:
        parity = np.vdot(vec, P @ vec)
        parities.append(np.real_if_close(parity))

    print("Three-dot system eigenvalues and their parity sectors:")
    for i, (E,p) in enumerate(zip(eigvals_np, parities)):
        if abs(p) < .9:
            print(f"Warning: Eigenvalue {E:.4f} has ambiguous parity {p:.4f}")
        sector = "even" if p > .9 else "odd"
        print(f"Eigenvalue {E:.4f} belongs to {sector} parity sector, with parity {p:.4f}")
    print("")

    even_states = [(E, v) for E, v, p in zip(eigvals_np, eigvecs_np.T, parities) if p > .9]
    odd_states  = [(E, v) for E, v, p in zip(eigvals_np, eigvecs_np.T, parities) if p < -.9]

    # Plot energy levels by parity
    plt.figure(figsize=(6,4))
    y_even = [E for E, _ in even_states]
    y_odd  = [E for E, _ in odd_states]

    plt.hlines(y_even, xmin=-0.2, xmax=0.2, color='blue', label='Even parity')
    plt.hlines(y_odd,  xmin=0.8, xmax=1.2, color='red',  label='Odd parity')

    plt.xlim(-0.5, 1.5)
    plt.xticks([0, 1], ['Even', 'Odd'])
    plt.ylabel("Energy eigenvalues")
    plt.title("Parity-resolved energy spectrum for 3-dot system")
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    return eigvals_np, parities
    

def four_dot_sweetspot():
    H4 = n_dot_Hamiltonian(n=4)

    # Symbols
    ϵs = sp.symbols('ϵ0 ϵ1 ϵ2 ϵ3')
    Us = sp.symbols('U0 U1 U2')
    Δ, t = sp.symbols('Δ t', real=True)

    # Sweet spot parameters
    U0_val = 1
    U1_val = 1
    U2_val = 1
    t_val = 1
    Δ_val = t_val + U0_val/2
    ϵ0_val = -U0_val/2
    ϵ1_val = -U0_val
    ϵ2_val = -U0_val
    ϵ3_val = -U0_val/2

    # Substitute parameters into symbolic matrix
    H4_num = H4.subs({Δ: Δ_val, t: t_val, ϵs[0]: ϵ0_val, ϵs[1]: ϵ1_val, ϵs[2]: ϵ2_val, ϵs[3]: ϵ3_val, Us[0]: U0_val, Us[1]: U1_val, Us[2]: U2_val})

    # Convert sympy.Matrix to numpy array (float complex)
    H4_np = np.array(H4_num).astype(np.complex128)

    # Eigen decomposition
    eigvals_np, eigvecs_np = np.linalg.eigh(H4_np)

    P = parity_operator(4)
    parities = []
    for vec in eigvecs_np.T:
        parity = np.vdot(vec, P @ vec)
        parities.append(np.real_if_close(parity))

    print("Four-dot system eigenvalues and their parity sectors:")
    for i, (E,p) in enumerate(zip(eigvals_np, parities)):
        if abs(p) < .9:
            print(f"Warning: Eigenvalue {E:.4f} has ambiguous parity {p:.4f}")
        sector = "even" if p > .9 else "odd"
        print(f"Eigenvalue {E:.4f} belongs to {sector} parity sector, with parity {p:.4f}")
    print("")

    even_states = [(E, v) for E, v, p in zip(eigvals_np, eigvecs_np.T, parities) if p > .9]
    odd_states  = [(E, v) for E, v, p in zip(eigvals_np, eigvecs_np.T, parities) if p < -.9]

    # Plot energy levels by parity
    plt.figure(figsize=(6,4))
    y_even = [E for E, _ in even_states]
    y_odd  = [E for E, _ in odd_states]

    plt.hlines(y_even, xmin=-0.2, xmax=0.2, color='blue', label='Even parity')
    plt.hlines(y_odd,  xmin=0.8, xmax=1.2, color='red',  label='Odd parity')

    plt.xlim(-0.5, 1.5)
    plt.xticks([0, 1], ['Even', 'Odd'])
    plt.ylabel("Energy eigenvalues")
    plt.title("Parity-resolved energy spectrum for 4-dot system")
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    return eigvals_np, parities

if __name__ == "__main__":
    two_dot_sweetspot()
    three_dot_sweetspot()
    four_dot_sweetspot()
