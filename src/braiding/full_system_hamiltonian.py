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
    """Calculate total parity operator from number operators."""
    # precompute_ops returns a list, so iterate directly
    P = np.eye(num_t[0].shape[0], dtype=np.complex128)
    for n_op in num_t:
        P = P @ (np.eye(P.shape[0], dtype=np.complex128) - 2 * n_op)
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


# def big_H(n, dup, t_vals, U_vals, eps_vals, delta_vals, t_couple1=0.0, delta_couple1=0.0, t_couple2=0.0, delta_couple2=0.0, eps_detune=None,
#           couple_A=None, couple_B=None, couple_C=None, couple_D=None):
#     """
#     n: sites per PMM
#     dup: number of PMMs
#     couple_A, couple_B: (PMM_index, local_site_index)
#     t_couple: hopping strength between couple_A and couple_B
#     delta_couple: pairing strength between couple_A and couple_B
#     eps_detune: dictionary {PMM_index: detune_value}
#     """

#     big_N = dup * n
#     dim = 2**big_N
#     H = np.zeros((dim, dim), dtype=complex)





#     cre, ann, num = precompute_ops(big_N)

#     eps_full = [eps_vals.copy() for _ in range(dup)]

#     # Apply detuning to specific PMMs
#     if eps_detune is not None:
#         for pm_idx, value in eps_detune.items():
#             for j in range(n):
#                 eps_full[pm_idx][j] = value

#     # Add intra-PMM terms
#     for d in range(dup):
#         off = d * n
#         for j in range(n - 1):
#             H += -t_vals[j] * (cre[off+j] @ ann[off+j+1] +
#                                ann[off+j] @ cre[off+j+1])

#             H += delta_vals[j] * (cre[off+j] @ cre[off+j+1] +
#                                   ann[off+j+1] @ ann[off+j])

#             H += U_vals[j] * num[off+j] @ num[off+j+1]

#         for j in range(n):
#             H += eps_full[d][j] * num[off+j]

#     # Add inter-PMM coupling
#     if couple_A is not None and couple_B is not None:
#         pmA, siteA = couple_A
#         pmB, siteB = couple_B

#         iA = pmA*n + siteA
#         iB = pmB*n + siteB

#         if t_couple1 != 0:
#             H += -t_couple1 * (cre[iA] @ ann[iB] + ann[iA] @ cre[iB])
#         if delta_couple1 != 0:
#             H += delta_couple1 * (cre[iA] @ cre[iB] +
#                                  ann[iA] @ ann[iB])

#     if couple_C is not None and couple_D is not None:
#         pmC, siteC = couple_C
#         pmD, siteD = couple_D

#         iC = pmC*n + siteC
#         iD = pmD*n + siteD

#         if t_couple2 != 0:
#             H += -t_couple2 * (cre[iC] @ ann[iD] + ann[iC] @ cre[iD])
#         if delta_couple2 != 0:
#             H += delta_couple2 * (cre[iC] @ cre[iD] +
#                                  ann[iC] @ ann[iD])


#     return H
import time


def big_H(n, dup, t_vals, U_vals, eps_vals, delta_vals,
               couplings=(), eps_detune=None, operators=None):

    start_time = time.time()
    big_N = dup * n
    dim = 2**big_N
    H = np.zeros((dim, dim), dtype=complex)

    if operators is None:
        cre, ann, num = precompute_ops(big_N)
        hop_ops = {}
        pair_ops = {}
        dens_ops = {}
        for i in range(big_N):
            for j in range(i+1, big_N):
                hop_ops[(i,j)] = cre[i] @ ann[j] + ann[i] @ cre[j]
                pair_ops[(i,j)] = cre[i] @ cre[j] + ann[j] @ ann[i]
                dens_ops[(i,j)] = num[i] @ num[j]
    else:
        cre = operators["cre"]
        ann = operators["ann"]
        num = operators["num"]
        hop_ops = operators["hop"]
        pair_ops = operators["pair"]
        dens_ops = operators["dens"]
    # cre, ann, num = precompute_ops(big_N)


    # # Precompute two-site operators
    # hop_ops = {}
    # pair_ops = {}
    # dens_ops = {}

    # end_time = time.time()  
    # print("Precomputed single-site operators in {:.4f} seconds.".format(end_time - start_time))

    # start_time = time.time()
    # for i in range(big_N):
    #     for j in range(i+1, big_N):
    #         hop_ops[(i,j)] = cre[i] @ ann[j] + ann[i] @ cre[j]
    #         pair_ops[(i,j)] = cre[i] @ cre[j] + ann[j] @ ann[i]
    #         dens_ops[(i,j)] = num[i] @ num[j]
    # end_time = time.time()
    # print("Precomputed two-site operators in {:.4f} seconds.".format(end_time - start_time))


    eps_full = np.tile(eps_vals, (dup,1))
    if eps_detune:
        for pm_idx, val in eps_detune.items():
            eps_full[pm_idx, :] = val


    # Intra PMM terms
    for d in range(dup):
        off = d * n
        for j in range(n-1):
            i, k = off+j, off+j+1
            H += -t_vals[j]   * hop_ops[(i,k)]
            H +=  delta_vals[j] * pair_ops[(i,k)]
            H +=  U_vals[j]   * dens_ops[(i,k)]

        for j in range(n):
            H += eps_full[d,j] * num[off+j]



    #  Inter PMM couplings
    for cA, cB, t_c, d_c in couplings:
        if cA is None or cB is None:
            continue

        i = cA[0]*n + cA[1]
        j = cB[0]*n + cB[1]
        key = (min(i,j), max(i,j))

        if t_c != 0:
            H += -t_c * hop_ops[key]
        if d_c != 0:
            H +=  d_c * pair_ops[key]
    return H



def extract_effective_H(big_H, n, dup, target):
    dim = 2**n
    dims = [dim] * dup

    H_tensor = big_H.reshape(dims + dims)

    # Keep indices for target subsystem
    # Left index:  target
    # Right index: target + dup
    left = target
    right = target + dup

    # Build index string for einsum
    # example for dup=3:   H[i0,i1,i2, j0,j1,j2]
    indices = ''.join(chr(ord('a') + i) for i in range(2*dup))

    # Output uses only target-left and target-right
    out = indices[left] + indices[right]

    # Trace all other pairs by equating indices
    for d in range(dup):
        if d != target:
            iL = indices[d]
            iR = indices[d + dup]
            out += ''   # no output
            indices = indices.replace(iR, iL)  # equate right to left

    # Build output tensor:
    H_eff = np.einsum(f"{indices}->{out}", H_tensor)

    return H_eff


if __name__ == "__main__":
    n_sites = 3

    pars, extras = params_for_n_site_Hamiltonian(n_sites, configs=None, specified_vals={"U": [0.1]}, path="configuration.json")
  
    t, U, eps, Delta = pars
    # H = big_H(n_sites, 3,  t, U, eps, Delta, t_couple=1, delta_couple=1, from_site=1, to_site=2)

    # H = big_H(n_sites, 3, t, U, eps, Delta,
    #       couple_A=(0,2),   # PMM 0, site 2
    #       couple_B=(1,0),   # PMM 1, site 0
    #       t_couple1=1,
    #       delta_couple1=1)
    big_N = n_sites * 3
    cre, ann, num = precompute_ops(big_N)
    hop_ops = {}
    pair_ops = {}
    dens_ops = {}

    for i in range(big_N):
        for j in range(i+1, big_N):
            hop_ops[(i,j)] = cre[i] @ ann[j] + ann[i] @ cre[j]
            pair_ops[(i,j)] = cre[i] @ cre[j] + ann[j] @ ann[i]
            dens_ops[(i,j)] = num[i] @ num[j]
    
    operators = {"cre": cre, "ann": ann, "num": num,
                 "hop": hop_ops, "pair": pair_ops, "dens": dens_ops}


    couple_A = (0, 2)  # PMM 0, site 2
    couple_B = (1, 0)  # PMM 1, site 0
    t_couple1 = 1.0
    delta_couple1 = 1.0
    couple_C = (1, 0)  # PMM 1, site 2
    couple_D = (2, 0)  # PMM 2, site 0
    t_couple2 = 1.0
    delta_couple2 = 1.0

    H = big_H(
        n_sites, 3,
        t, U, eps, Delta,
        couplings=[
            (couple_A, couple_B, t_couple1, delta_couple1),
            (couple_C, couple_D, t_couple2, delta_couple2)
        ],
        eps_detune={1: 2.0},
        operators=operators
    )



    H = big_H(n_sites, 3, t, U, eps, Delta,
              eps_detune={1: 1.0})  # Detune PMM 1 by +1.0

    # eff_dim = 2**n_sites
    # arm_map = {
    #     "A": list(range(0,eff_dim)),
    #     "B": list(range(eff_dim, 2*eff_dim)),
    #     "C": list(range(2*eff_dim, 3*eff_dim))
    # }
    # print(arm_map)
    # set1 = extract_effective_H(H, "A", arm_map)
    # set2 = extract_effective_H(H, "B", arm_map)
    # set3 = extract_effective_H(H, "C", arm_map)

    set1 = extract_effective_H(H, n_sites, 3, target=0)
    set2 = extract_effective_H(H, n_sites, 3, target=1)
    set3 = extract_effective_H(H, n_sites, 3, target=2)
    sets = [set1, set2, set3]

    evals, evecs = np.linalg.eigh(H)

    min_idxs = np.where(np.isclose(evals, min(evals), atol=1e-2))
    # print("Ground state energies and vectors:")
    # for idx in min_idxs:
    #     print(f"Energy: {evals[idx]}")
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

