import numpy as np
import sympy as sp
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_torch(mat_np, device=device):
    return torch.from_numpy(mat_np).to(device=device, dtype=torch.complex128)

def from_torch(mat_torch):
    return mat_torch.cpu().numpy()



def tensor_product(*args):
    """Compute the tensor product of multiple matrices."""
    result = args[0]
    for mat in args[1:]:
        result = np.kron(result, mat)
    return result   

def sigma_site(j, n, operator):
    """Return the operator for site j in a chain of n sites."""
    I = np.eye(2)
    ops = [I] * n
    ops[j] = operator
    return tensor_product(*ops)


def creation_annihilation(j,n):
    """Return the creation and annihilation operators for site j in a chain of n sites, after Jordan-Wigner transformation."""

    σ_x = np.array([[0, 1], 
                    [1, 0]])
    σ_y = np.array([[0, -1j], 
                    [1j, 0]])
    σ_z = np.array([[1, 0], 
                    [0, -1]])
    f_dag = (sigma_site(j, n, σ_x) + 1j * sigma_site(j, n, σ_y)) * 0.5
    f = (sigma_site(j, n, σ_x) - 1j * sigma_site(j, n, σ_y)) * 0.5

    Zops = np.eye(2**n) 
    for i in range(0, j):
        Zops @= sigma_site(i, n, -σ_z)

    create = Zops @ f_dag 
    annihilate = Zops @ f 

    return create, annihilate


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

def parity_operator_torch(n):
    P = torch.eye(2**n, dtype=torch.complex128)
    for j in range(n):
        cd, c = creation_annihilation(j, n)
        cd_t = to_torch(cd, device=device)
        c_t = to_torch(c, device=device)
        n_i = cd_t @ c_t
        P = P @ (torch.eye(2**n, dtype=torch.complex128, device=device) - 2 * n_i)
    return P

def parity_operator(N):
    """Construct the parity operator for N dots."""
    P = np.eye(2**N, dtype=complex)
    for i in range(N):
        f_dag_i, f_i = creation_annihilation(i, N)
        n_i = f_dag_i @ f_i
        P = P @ (np.eye(2**N) - 2 * n_i)
    return P


def construct_Gamma_operators_torch(j,n, device=device):
    """Construct the Majorana operators Γ^s_j for site j in a chain of n sites."""
    f_dag_j, f_j = creation_annihilation(j, n)
    f_dag_j_t = to_torch(f_dag_j, device=device)
    f_j_t = to_torch(f_j, device=device)
    Gamma_1 = f_dag_j_t + f_j_t
    Gamma_2 = 1j * (f_dag_j_t - f_j_t)
    return Gamma_1, Gamma_2

def construct_Gamma_operators(j,n):
    """Construct the Majorana operators Γ^s_j for site j in a chain of n sites."""
    f_dag_j, f_j = creation_annihilation(j, n)
    Gamma_1 = f_dag_j + f_j
    Gamma_2 = 1j * (f_dag_j - f_j)
    return Gamma_1, Gamma_2
