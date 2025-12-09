from get_setup import params_for_n_site_Hamiltonian
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

### Tools

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


def total_parity(num_t):
    P = np.eye(num_t[0].shape[0], dtype=complex)
    for n_op in num_t:
        P = P @ (np.eye(P.shape[0]) - 2 * n_op)
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


def big_H(n, duplicate, t_vals, U_vals, eps_vals, delta_vals):

    big_N = duplicate * n  # total number of sites in the super Hamiltonian
    H = np.zeros((2**(big_N), 2**(big_N)), dtype=complex)
 
    cre_t, ann_t, num_t = precompute_ops(big_N)

    for j in range(n - 1):
        for d in range(duplicate):
            H += -t_vals[j] * (ann_t[j + d*n] @ cre_t[j + 1 + d*n] + cre_t[j + d*n] @ ann_t[j + 1 + d*n])
            H += delta_vals[j] * (cre_t[j + d*n] @ cre_t[j + 1 + d*n] + ann_t[j + 1 + d*n] @ ann_t[j + d*n])
    for j in range(n):
        for d in range(duplicate):
            H += eps_vals[j] * num_t[j + d*n]
    for j in range(n - 1):
        for d in range(duplicate):
            H += U_vals[j] * num_t[j + d*n] @ num_t[j + 1 + d*n]

    
    return H





if __name__ == "__main__":
    n_sites = 3

    pars, extras = params_for_n_site_Hamiltonian(n_sites, configs=None, specified_vals={"U": [0.1]}, path="configuration.json")
  
    t, U, eps, Delta = pars

    H = big_H(n_sites, 3,  t, U, eps, Delta)

    evals, evecs = np.linalg.eigh(H)

    min_idxs = np.where(np.isclose(evals, min(evals), atol=1e-2))
    print("Ground state energies and vectors:")
    for idx in min_idxs:
        print(f"Energy: {evals[idx]}")
        # print(f"Vector: {evecs[:, idx]}")

    _, _, num_t = precompute_ops(n_sites * 3)

    P = total_parity(num_t)

    parities = []
    for k in range(len(evals)):
        psi = evecs[:, k]
        parity = np.vdot(psi, P @ psi)
        parities.append(np.real_if_close(parity))

    print(list(zip(evals, parities)))
    even_states = []
    odd_states = []
    for i in range(len(list(zip(evals, parities)))):
        if np.isclose(parities[i], 1):
            even_states.append((evals[i], evecs[:, i]))
        elif np.isclose(parities[i], -1):   
            odd_states.append((evals[i], evecs[:, i]))


    def cluster_energies(evals, tol=1e-6):
        cluster = []
        counts = []
        count = 0
        i = 1
        while i < len(evals) - 1:
            for j in range(i+1, len(evals)):
                if abs(evals[j] - evals[i]) < tol:
                    count += 1
                else: 
                    counts.append(count)
                    count = 0
                    cluster.append(evals[i:j])
                    i = j
                    break 
            i += 1
                    
        return cluster, counts
    tolerance = 1e-2
    even_clusters, even_counts = cluster_energies([state[0] for state in even_states], tol=tolerance)
    odd_clusters, odd_counts = cluster_energies([state[0] for state in odd_states], tol=tolerance)
    print(len(even_clusters), len(odd_clusters))

    # print("Even clusters and counts:", list(zip(even_clusters, even_counts)))
    # print("Odd clusters and counts:", list(zip(odd_clusters, odd_counts)))
    # for i in range(len(even_clusters)):
        # print(f"Even cluster {i}: Energies = {even_clusters[i]}, Count = {even_counts[i]+1}")
    for i in range(len(even_clusters)):
        plt.hlines(min(even_clusters[i]), -0.2, 0.4, colors='b')
        # print(min(even_clusters[i]))
        # print(even_counts[i]+1)
        plt.text(0.41, min(even_clusters[i]), f"{even_counts[i]+1}", va="center", fontsize=12)
    for i in range(len(odd_clusters)):
        plt.hlines(min(odd_clusters[i]), 0.6, 1.2, colors='r')
        plt.text(1.21, min(odd_clusters[i]), f"{odd_counts[i]+1}", va="center", fontsize=12)
    plt.title("Energy spectrum (even vs odd parity)")
    plt.xlabel("Parity sector")
    plt.ylabel("Energy")
    plt.xticks([0.1, 1.0], ["Even", "Odd"])
    plt.tight_layout()
    plt.show()

