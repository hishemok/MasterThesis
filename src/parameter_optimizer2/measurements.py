import numpy as np
import torch
from operators import to_torch, creation_annihilation, device, construct_Gamma_operators_torch, construct_Gamma_operators



def calculate_parities(evals, evecs, P):
    # evecs: columns are eigenvectors
    parities = torch.real(torch.sum(torch.conj(evecs) * (P @ evecs), dim=0))
    even_mask = parities >= 0
    odd_mask  = parities < 0

    E_even = evals[even_mask]
    E_odd  = evals[odd_mask]

    # Return also eigenvectors if needed
    even_vecs = evecs[:, even_mask]
    odd_vecs  = evecs[:, odd_mask]

    return E_even, E_odd, even_vecs, odd_vecs

def charge_difference_torch(even_state, odd_state, n):
    """Compute the charge difference between even and odd states.
    ∑_n |⟨e_k|n_n|e_k⟩ - ⟨o_k|n_n|o_k⟩|
    """
    # accumulate a real scalar (sum of magnitudes), keep as float tensor
    charge_diff = torch.tensor(0.0, dtype=torch.float64, device=device)
    n_states = even_state.shape[1]
    for site in range(n):
        f_dag_i, f_i = creation_annihilation(site, n)
        f_dag_i_t = to_torch(f_dag_i, device=device)
        f_i_t = to_torch(f_i, device=device)
        n_i = f_dag_i_t @ f_i_t
        exp_even = torch.zeros((), dtype=torch.complex128, device=device)
        exp_odd = torch.zeros((), dtype=torch.complex128, device=device)
        for k in range(n_states):
            even_vec = even_state[:, k]
            odd_vec = odd_state[:, k]
            exp_even += torch.vdot(even_vec, n_i @ even_vec)
            exp_odd  += torch.vdot(odd_vec,  n_i @ odd_vec)
        charge_diff += torch.abs(exp_even - exp_odd).to(dtype=torch.float64)
    return charge_diff


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



def Majorana_polarization_torch(even_vecs, odd_vecs, n):
    """
    Compute local Majorana polarization M_j for each site j.
    M_j = sum_s (<o|Γ^s_j|e>)^2 / sum_s |<o|Γ^s_j|e>|^2
    """
    M = torch.zeros((even_vecs.shape[1], n), dtype=torch.complex128, device=device)

    for j in range(n):
        Gamma_1, Gamma_2 = construct_Gamma_operators_torch(j, n, device=device)
        for i in range(even_vecs.shape[1]):
            e_k = even_vecs[:, i]
            o_k = odd_vecs[:, i]
            term1 = torch.vdot(o_k, Gamma_1 @ e_k)
            term2 = torch.vdot(o_k, Gamma_2 @ e_k)
            M[i, j] = term1**2 + term2**2 #/ denominator if denominator != 0 else 0.0
    return M

def Majorana_polarization(even_vecs, odd_vecs, n, model):
    """
    Compute local Majorana polarization M_j for each site j.
    M_j = sum_s (<o|Γ^s_j|e>)^2 / sum_s |<o|Γ^s_j|e>|^2
    """
    M = np.zeros((even_vecs.shape[1], n), dtype=complex)

    for j in range(n):
        Gamma_1, Gamma_2 = construct_Gamma_operators(j, n)
        for i in range(even_vecs.shape[1]):
            e_k = even_vecs[:, i]
            o_k = odd_vecs[:, i]
            term1 = np.vdot(o_k, Gamma_1 @ e_k)
            term2 = np.vdot(o_k, Gamma_2 @ e_k)
            M[i, j] = term1**2 + term2**2
    return M 



def MP_Penalty(even_vecs, odd_vecs, n):
    """
    Majorana Polarization penalty:
    ideally +1 on first site, -1 on last site, 0 elsewhere.
    even_vecs, odd_vecs: (dim, num_states)
    n: number of sites
    MP shape: (num_states, 2n)
    """
    MP = Majorana_polarization_torch(even_vecs, odd_vecs, n)

    target_MP = torch.zeros_like(MP)
    target_MP[:, 0] = 1.0
    target_MP[:, -1] = -1.0

    # both signs
    penalty_pos = torch.abs(torch.sum((MP - target_MP)**2))
    penalty_neg = torch.abs(torch.sum((MP + target_MP)**2))

    penalty = torch.min(penalty_pos, penalty_neg)
    return penalty

