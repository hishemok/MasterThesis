from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence
from full_system_hamiltonian import *

from hamiltonian_builder import BraidingHamiltonianBuilder, default_config_path
import numpy as np
import matplotlib.pyplot as plt
from explore_hamiltonian_values import calculate_parities_optimized, get_parity_operator

try:
    from .get_setup import pull_configurations
except ImportError:
    from get_setup import pull_configurations


n = 3
dupes = 3
big_N = n * dupes

def precompute_operators(n, dup):
    big_N = n * dup
    operators = {}
    cre, ann, num = precompute_ops(big_N)
    operators['cre'] = cre
    operators['ann'] = ann
    operators['num'] = num
    hop_ops = {}
    pair_ops = {}
    dens_ops = {}
    for d in range(dup):
        off = d * n
        for i in range(n-1):
            a, b = off+i, off+i+1
            hop_ops[(a,b)] = cre[a] @ ann[b] + cre[b] @ ann[a] 
            pair_ops[(a,b)] = cre[a] @ cre[b] + ann[b] @ ann[a]
            dens_ops[(a,b)] = num[a] @ num[b]
    # Inter PMM coupling terms
    # Inner dots only for now
    if dup > 1:
        hop_ops[(0,n)] = cre[0] @ ann[n] + ann[0] @ cre[n] #A0 - B0
        hop_ops[(0,2*n)] = cre[0] @ ann[2*n] + ann[0] @ cre[2*n] #A0 - C0
    # hop_ops[(n,2*n)] = cre[n] @ ann[2*n] + ann[n] @ cre[2*n] #B0 - C0
    operators['hop'] = hop_ops
    operators['pair'] = pair_ops
    operators['dens'] = dens_ops
    return operators



def subsystem_operators(n, dupes):
    big_N = n * dupes
    cre, ann, num = precompute_ops(big_N)
    subsys_operators = {}
    subsys_operators['cre'] = cre
    subsys_operators['ann'] = ann
    subsys_operators['num'] = num
    hop_ops = {}
    pair_ops = {}
    dens_ops = {}
    for i in range(dupes):
        off = i * n
        for j in range(n-1):
            a, b = off+j, off+j+1
            hop_ops[(a,b)] = cre[a] @ ann[b] + ann[a] @ cre[b]
            pair_ops[(a,b)] = cre[a] @ cre[b] + ann[b] @ ann[a]
            dens_ops[(a,b)] = num[a] @ num[b]
    subsys_operators['hop'] = hop_ops
    subsys_operators['pair'] = pair_ops
    subsys_operators['dens'] = dens_ops

    return subsys_operators


def subsys_parity_op(sites = 3):
    print("Calculating parity operator for subsystem with", sites, "sites")
    num = precompute_ops(sites)[2]
    print("Number operators shape:", num[0].shape)
    dim = num[0].shape[0]
    I = np.eye(dim, dtype=complex)
    P = I.copy()
    for i in range(sites):
        P = P @ (I - 2 * num[i])
    return P

def tensorprod(mats = []):
    result = np.array([[1]], dtype=complex)
    for M in mats:
        result = np.kron(result, M)
    return result
    
def pauli_Z():
    return np.array([[1,0],[0,-1]], dtype=complex)

def build_JW_string(subsys_before_sites):
    """
    subsys_before_sites: number of sites before the subsystem
    Returns: JW string as a sparse diagonal matrix
    """
    mat = np.array([[1]], dtype=complex)
    Z = pauli_Z()
    I = np.eye(2, dtype=complex)

    # Multiply by Z on each preceding site
    for _ in range(subsys_before_sites):
        mat = np.kron(mat, -Z)

    return mat



def pair_even_odd(E_even, E_odd):
    pairs = []

    for i, Ee in enumerate(E_even):
        j = np.argmin(np.abs(E_odd - Ee))
        pairs.append((i, j))

    return pairs

def construct_majoranas(evecs, ovecs, E_even, E_odd, n=1):

    dim = evecs.shape[0]
    gamma1 = np.zeros((dim, dim), dtype=complex)
    gamma2 = np.zeros((dim, dim), dtype=complex)

    pairs = pair_even_odd(E_even, E_odd)

    for i in range(n):

        ei, oi = pairs[i]

        e_v = evecs[:, ei]
        o_v = ovecs[:, oi]

        gamma1 += np.outer(o_v, e_v.conj()) + np.outer(e_v, o_v.conj())
        gamma2 += 1j*(np.outer(o_v, e_v.conj()) - np.outer(e_v, o_v.conj()))

    return gamma1, gamma2

def subsys_parity_oper(sites = 3):
    subsystem_operators_ = subsystem_operators(n, 1)
    # print(subsystem_operators_["num"])
    num = subsystem_operators_["num"]
    
    num = precompute_ops(sites)[2]
    dim = num[0].shape[0]
    I = np.eye(dim, dtype=complex)
    P = I.copy()
    for i in range(sites):
        P = P @ (I - 2 * num[i])
    return P



def get_full_gammas(levels_to_include = 4,verbose=False):
    JW_A = np.eye(2**3)  # nothing before A
    JW_B = build_JW_string(3)            # sees all A sites
    JW_C = build_JW_string(6)            # sees all A+B sites

    builder = BraidingHamiltonianBuilder(
        n_sites=3,
        dupes=3,
        specified_vals={"U": [0.1]},
        config_path=default_config_path(),
    )

    builder_sub = BraidingHamiltonianBuilder(
        n_sites=3,
        dupes=1,
        specified_vals={"U": [0.1]},
        config_path=default_config_path(),
    )

    h_a = builder_sub.full_system_hamiltonian()
    h_a_eigvals, h_a_eigvecs = np.linalg.eigh(h_a)
    h_b = builder_sub.full_system_hamiltonian()
    h_b_eigvals, h_b_eigvecs = np.linalg.eigh(h_b)
    h_c = builder_sub.full_system_hamiltonian()
    h_c_eigvals, h_c_eigvecs = np.linalg.eigh(h_c)
    h_full = builder.full_system_hamiltonian()
    h_full_eigvals, h_full_eigvecs = np.linalg.eigh(h_full)


    subsys_parity = subsys_parity_oper(sites=n)
    # print("Subsystem parity", subsys_parity.shape)
    even_energies_A, odd_energies_A, even_vec_A, odd_vec_A = calculate_parities_optimized(h_a_eigvecs, h_a_eigvals, subsys_parity)
    even_energies_B, odd_energies_B, even_vec_B, odd_vec_B = calculate_parities_optimized(h_b_eigvecs, h_b_eigvals, subsys_parity)
    even_energies_C, odd_energies_C, even_vec_C, odd_vec_C = calculate_parities_optimized(h_c_eigvecs, h_c_eigvals, subsys_parity)

    # print("Even energies A:", even_energies_A)


    gamma_A1, gamma_A2 = construct_majoranas(even_vec_A, odd_vec_A, even_energies_A, odd_energies_A, n=levels_to_include)
    gamma_B1, gamma_B2 = construct_majoranas(even_vec_B, odd_vec_B, even_energies_B, odd_energies_B, n=levels_to_include)
    gamma_C1, gamma_C2 = construct_majoranas(even_vec_C, odd_vec_C, even_energies_C, odd_energies_C, n=levels_to_include)
    # pretty_print_matrix(gamma_A1@gamma_A1, "γ_A1")
    # print("γ² check:", np.linalg.norm(gamma_A1 @ gamma_A1 - np.eye(gamma_A1.shape[0])))
    # print("anticommutation:",
    #       np.linalg.norm(gamma_A1 @ gamma_A2 + gamma_A2 @ gamma_A1))

    gamma_A1_full = tensorprod([gamma_A1, np.eye(2**6)])
    gamma_A2_full = tensorprod([gamma_A2, np.eye(2**6)])
    gamma_B1_full = tensorprod([JW_B, gamma_B1, np.eye(2**3)])
    gamma_B2_full = tensorprod([JW_B, gamma_B2, np.eye(2**3)])
    gamma_C1_full = tensorprod([JW_C, gamma_C1])  
    gamma_C2_full = tensorprod([JW_C, gamma_C2])


    if verbose:
        print("HA |e> = E|e>?:",np.linalg.norm(h_a @ even_vec_A[:,0] - even_energies_A[0]*even_vec_A[:,0]))
        print("HA |o> = E|o>?:",np.linalg.norm(h_a @ odd_vec_A[:,0] - odd_energies_A[0]*odd_vec_A[:,0]))
        print("[H_full, gamma_A1_full]:\n", np.linalg.norm(h_full @ gamma_A1_full - gamma_A1_full @ h_full))
        print("[h_full, gamma_A2_full]:\n", np.linalg.norm(h_full @ gamma_A2_full - gamma_A2_full @ h_full))
        print("[h_full, gamma_B1_full]:\n", np.linalg.norm(h_full @ gamma_B1_full - gamma_B1_full @ h_full))
        print("[h_full, gamma_B2_full]:\n", np.linalg.norm(h_full @ gamma_B2_full - gamma_B2_full @ h_full))
        print("[h_full, gamma_C1_full]:\n", np.linalg.norm(h_full @ gamma_C1_full - gamma_C1_full @ h_full))
        print("[h_full, gamma_C2_full]:\n", np.linalg.norm(h_full @ gamma_C2_full - gamma_C2_full @ h_full))
        print("[H_A_sub, gamma_A1]:\n", np.linalg.norm(h_a @ gamma_A1 - gamma_A1 @ h_a))
        print("[H_A_sub, gamma_A2]:\n", np.linalg.norm(h_a @ gamma_A2 - gamma_A2 @ h_a))
        print("[H_B_sub, gamma_B1]:\n", np.linalg.norm(h_b @ gamma_B1 - gamma_B1 @ h_b))
        print("[H_B_sub, gamma_B2]:\n", np.linalg.norm(h_b @ gamma_B2 - gamma_B2 @ h_b))
        print("[H_C_sub, gamma_C1]:\n", np.linalg.norm(h_c @ gamma_C1 - gamma_C1 @ h_c))
        print("[H_C_sub, gamma_C2]:\n", np.linalg.norm(h_c @ gamma_C2 - gamma_C2 @ h_c))

    return (gamma_A1_full, gamma_A2_full), (gamma_B1_full, gamma_B2_full), (gamma_C1_full, gamma_C2_full)


if __name__ == "__main__":
    get_full_gammas()


