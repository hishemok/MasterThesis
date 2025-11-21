import numpy as np
import sympy as sp

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


def parity_operator(N):
    """Construct the parity operator for N dots."""
    P = np.eye(2**N, dtype=complex)
    for i in range(N):
        f_dag_i, f_i = creation_annihilation(i, N)
        n_i = f_dag_i @ f_i
        P = P @ (np.eye(2**N) - 2 * n_i)
    return P