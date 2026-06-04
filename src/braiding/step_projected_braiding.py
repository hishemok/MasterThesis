import numpy as np
import matplotlib.pyplot as plt
from get_mzm_JW import get_full_gammas as get_majoranas_JW
from get_mzm_JW import precompute_operators
from remake_majoranas3 import make_majoranas_for_B_and_C_with_projection_dim
from scipy.linalg import expm
from tqdm import tqdm
from hamiltonian_builder import BraidingHamiltonianBuilder, default_config_path
from braiding_model import delta_pulse
from extended_projection_braiding import normalize_projected_majorana
from pathlib import Path

cwd = Path(__file__).parent

specified_vals = {"U": [0.0]}

builder = BraidingHamiltonianBuilder(
    n_sites=3,
    dupes=3,
    specified_vals=specified_vals,
    config_path=default_config_path(),
)
h_full = builder.full_system_hamiltonian()
eigvals, eigvecs = np.linalg.eigh(h_full)


(gamma_A1_full, gamma_A2_full), (gamma_B1_full, gamma_B2_full), (gamma_C1_full, gamma_C2_full) = get_majoranas_JW(levels_to_include=4, specified_vals=specified_vals)


#plot mister
def plot_energy_levels(even_energies, odd_energies, title='Energy levels colored by parity'):
    xmin_even,xmax_even = 0,1
    xmin_odd,xmax_odd = 2,3
    plt.figure(figsize=(10, 6))
    plt.hlines(even_energies, xmin_even, xmax_even, color='blue', label='Even parity energies')
    plt.hlines(odd_energies, xmin_odd, xmax_odd, color='orange', label='Odd parity energies')
    plt.xlabel('State index')
    plt.ylabel('Energy')
    plt.title(title)
    plt.legend()
    plt.show()

def split_evals_by_parity(evals, evecs, parity, parity_tol=1e-3, energy_tol=1e-8):
    even_energies = []
    odd_energies = []
    used = np.zeros(len(evals), dtype=bool)

    for i in range(len(evals)):
        if used[i]:
            continue

        close_idxs = np.where(np.isclose(evals, evals[i], atol=energy_tol))[0]
        close_idxs = [j for j in close_idxs if not used[j]]
        for j in close_idxs:
            used[j] = True

        subspace = evecs[:, close_idxs]
        parity_subspace = subspace.conj().T @ parity @ subspace
        parity_subspace = 0.5 * (parity_subspace + parity_subspace.conj().T)
        parity_vals = np.linalg.eigvalsh(parity_subspace)

        for parity_val in parity_vals:
            if np.isclose(parity_val, 1, atol=parity_tol):
                even_energies.append(evals[i])
            elif np.isclose(parity_val, -1, atol=parity_tol):
                odd_energies.append(evals[i])
            else:
                raise ValueError(f"Unexpected parity value: {parity_val}")

    return np.array(even_energies), np.array(odd_energies)

def plot_energy_levels_from_evals(evals, evecs, parity, title='Energy levels colored by parity'):
    even_energies, odd_energies = split_evals_by_parity(evals, evecs, parity)
    plot_energy_levels(even_energies, odd_energies, title=title)
    return even_energies, odd_energies

def even_odd_splitter(eigvecs, eigvals, subsys_parity):
    even_energies = []
    odd_energies = []
    even_vecs = []
    odd_vecs = []
    even_idxs = []
    odd_idxs = []
    tol = 1e-3
    for i in range(len(eigvals)):
        parity = np.vdot(eigvecs[:, i], subsys_parity @ eigvecs[:, i])
        if np.isclose(parity, 1, atol=tol):
            even_energies.append(eigvals[i])
            even_vecs.append(eigvecs[:, i])
            even_idxs.append(i)
        elif np.isclose(parity, -1, atol=tol):
            odd_energies.append(eigvals[i])
            odd_vecs.append(eigvecs[:, i])
            odd_idxs.append(i)
        else:
            raise ValueError(f"Unexpected parity value: {parity}")
    return (
        np.array(even_energies),
        np.array(odd_energies),
        np.array(even_vecs).T,
        np.array(odd_vecs).T,
        np.array(even_idxs),
        np.array(odd_idxs),
    )


ops = precompute_operators(n=3, dup=3)
def parity_op(ops, sites = 3):
    num = ops["num"]
    
    dim = num[0].shape[0]
    I = np.eye(dim, dtype=complex)
    P = I.copy()
    for i in range(sites):
        P = P @ (I - 2 * num[i])
    return P


full_parity = parity_op(ops, sites=9)
even_energies, odd_energies, even_vecs, odd_vecs, even_idxs, odd_idxs = even_odd_splitter(eigvecs, eigvals, full_parity)
print(even_energies)

def find_close_groups(even_energies, odd_energies, even_idxs, odd_idxs):
    groups = []
    used = np.zeros(min(len(even_energies), len(odd_energies)), dtype=bool)

    for i in range(len(used)):
        if used[i]:
            continue

        e = float(even_energies[i])
        o = float(odd_energies[i])

        group_even_idxs = []
        group_odd_idxs = []
        for j in range(i, len(used)):
            e2 = float(even_energies[j])
            o2 = float(odd_energies[j])
            if np.isclose(e, e2, atol=1e-2) and np.isclose(o, o2, atol=1e-2):
                used[j] = True
                group_even_idxs.append(even_idxs[j])
                group_odd_idxs.append(odd_idxs[j])

        group = np.array(sorted(group_even_idxs + group_odd_idxs))
        print(len(groups), len(group_even_idxs), "energies in even group")
        print(len(groups), len(group_odd_idxs), "energies in odd group")
        groups.append(group)

    for group in groups:
        print(group)
    return groups

def find_group_bases(groups, eigvecs):
    bases = []
    for group in groups:
        basis = eigvecs[:, group]
        
        bases.append(basis)
    return bases


def project_majoranas(bases, gamma_A1_full, gamma_A2_full, gamma_B1_full, gamma_B2_full, gamma_C1_full, gamma_C2_full):
    projected_majoranas = []
    for i, basis in enumerate(bases):
        A1_proj = normalize_projected_majorana(basis.conj().T @ gamma_A1_full @ basis, f"A1 group {i}")
        A2_proj = normalize_projected_majorana(basis.conj().T @ gamma_A2_full @ basis, f"A2 group {i}")
        B1_proj = normalize_projected_majorana(basis.conj().T @ gamma_B1_full @ basis, f"B1 group {i}")
        B2_proj = normalize_projected_majorana(basis.conj().T @ gamma_B2_full @ basis, f"B2 group {i}")
        C1_proj = normalize_projected_majorana(basis.conj().T @ gamma_C1_full @ basis, f"C1 group {i}")
        C2_proj = normalize_projected_majorana(basis.conj().T @ gamma_C2_full @ basis, f"C2 group {i}")

        projected_majoranas.append((A1_proj, A2_proj, B1_proj, B2_proj, C1_proj, C2_proj))
    return projected_majoranas

def majorana_checks(projected_majoranas):
    for i, (A1_proj, A2_proj, B1_proj, B2_proj, C1_proj, C2_proj) in enumerate(projected_majoranas):
        #check norms
        for name, gamma in zip(["A1", "A2", "B1", "B2", "C1", "C2"], [A1_proj, A2_proj, B1_proj, B2_proj, C1_proj, C2_proj]):
            if not np.allclose(gamma @ gamma, np.eye(gamma.shape[0])):
                print(f"Warning: {name} in group {i} does not have unit norm.")

        for name, gamma in zip(["A1", "A2", "B1", "B2", "C1", "C2"], [A1_proj, A2_proj, B1_proj, B2_proj, C1_proj, C2_proj]):
            gamma_squared = gamma @ gamma
            identity = np.eye(gamma.shape[0])
            if not np.allclose(gamma_squared, identity):
                print(f"Warning: {name} in group {i} does not square to identity. Max deviation: {np.max(np.abs(gamma_squared - identity))}")
            for other_name, other_gamma in zip(["A1", "A2", "B1", "B2", "C1", "C2"], [A1_proj, A2_proj, B1_proj, B2_proj, C1_proj, C2_proj]):
                if name != other_name:
                    anticommutator = gamma @ other_gamma + other_gamma @ gamma
                    if not np.allclose(anticommutator, np.zeros_like(anticommutator)):
                        print(f"Warning: {name} and {other_name} in group {i} do not anticommute. Max deviation: {np.max(np.abs(anticommutator))}")

groups = find_close_groups(even_energies, odd_energies, even_idxs, odd_idxs)
bases = find_group_bases(groups, eigvecs)
projected_majorana_groups = project_majoranas(bases, gamma_A1_full, gamma_A2_full, gamma_B1_full, gamma_B2_full, gamma_C1_full, gamma_C2_full)
majorana_checks(projected_majorana_groups)


def get_braiding_terms(projected_majoranas, idx):
    """idx needs to range from 0 to len(projected_majoranas) (15)"""
    A1_proj, A2_proj, B1_proj, B2_proj, C1_proj, C2_proj = projected_majoranas[idx]
    TA = hermitian_part(1j * A1_proj @ A2_proj)
    TB = hermitian_part(1j* A1_proj @ B2_proj)
    TC = hermitian_part(1j* A1_proj @ C2_proj)
    return TA, TB, TC

T_total = 1.0
Delta_max = 1.0
Delta_min = 0.0
Width = T_total / 3
S = 20 / Width

def hermitian_part(matrix):
    return 0.5 * (matrix + matrix.conj().T)

def build_projected_hamiltonian(t, term_a, term_b, term_c, static_term=None):


    delta_1 = (
        delta_pulse(t, 0, Width, S, Delta_max, Delta_min)
        + delta_pulse(t, T_total, Width, S, Delta_max, Delta_min)
        - Delta_min
    )
    delta_2 = delta_pulse(t, T_total / 3, Width, S, Delta_max, Delta_min)
    delta_3 = delta_pulse(t, 2 * T_total / 3, Width, S, Delta_max, Delta_min)

    hamiltonian = delta_1 * term_a + delta_2 * term_b + delta_3 * term_c
    if static_term is not None:
        hamiltonian = hamiltonian + static_term
    return hamiltonian, (delta_1, delta_2, delta_3)

def plot_projected_hamiltonian_levels(idx, t=0.0):
    TA, TB, TC = get_braiding_terms(projected_majorana_groups, idx)
    hamiltonian, deltas = build_projected_hamiltonian(t, TA, TB, TC)
    evals, evecs = np.linalg.eigh(hamiltonian)
    parity_projected = bases[idx].conj().T @ full_parity @ bases[idx]
    title = f"Group {idx} projected Hamiltonian at t={t:.3f}"
    return plot_energy_levels_from_evals(evals, evecs, parity_projected, title=title)

def evolve_system(n_steps, projected_majoranas, idx):
    TA, TB, TC = get_braiding_terms(projected_majoranas, idx)
    time_arr = np.linspace(0, T_total, n_steps)
    dt = time_arr[1] - time_arr[0]

    transport_dim = TA.shape[0] // 2
    group_basis = bases[idx]
    static_term = group_basis.conj().T @ h_full @ group_basis
    static_term = hermitian_part(static_term)
    
    U_kato = np.eye(TA.shape[0], dtype=complex)
    H_t, deltas = build_projected_hamiltonian(0, TA, TB, TC, static_term=static_term)
    evals, evecs = np.linalg.eigh(H_t)


    initial_basis = evecs[:, :transport_dim]
    basis = initial_basis  # Start in the subspace of the lowest transport_dim eigenstates

    parity_op = group_basis.conj().T @ full_parity @ group_basis
    for t in tqdm(time_arr[1:], desc="Evolving system"):
        H_t, deltas = build_projected_hamiltonian(t, TA, TB, TC, static_term=static_term)

        evals, evecs = np.linalg.eigh(H_t)
        print()
        if np.abs(t - T_total/3) < 1e-2 or np.abs(t - 2*T_total/3) < 1e-2:
            print(evals)
            plot_energy_levels_from_evals(evals, evecs, parity_op, title=f"Group {idx} projected Hamiltonian at t={t:.3f}")
        # exit(0)

        next_basis = evecs[:, :transport_dim]
        projector = basis @ basis.conj().T
        # print(projector)
        next_projector = next_basis @ next_basis.conj().T
        d_projector = (next_projector - projector) / dt
        kato_generator = projector @ d_projector - d_projector @ projector
        U_kato = expm(-dt * kato_generator) @ U_kato
        basis = next_basis


    return U_kato, initial_basis


def get_target_unitary(projected_majoranas, idx):
    A1_proj, A2_proj, B1_proj, B2_proj, C1_proj, C2_proj = projected_majoranas[idx]
    return np.array(expm(-np.pi/4 * B2_proj @ C2_proj))


def unitary_overlap(U, V):
    d = U.shape[0]
    return abs(np.trace(V.conj().T @ U)) / d

evolve_system(300, projected_majorana_groups, idx=0)
# results = {}
# for idx in range(0, len(projected_majorana_groups)):
#     U_kato, initial_basis = evolve_system(300, projected_majorana_groups, idx=idx)
    
#     U_target = get_target_unitary(projected_majorana_groups, idx=idx)
#     U_kato_sub = initial_basis.conj().T @ U_kato @ initial_basis
#     U_target_sub = initial_basis.conj().T @ U_target @ initial_basis
#     results[idx] = unitary_overlap(U_kato_sub, U_target_sub)
#     print(f"Overlap for group {idx}:", results[idx])

# print("Overlap results for all groups:")
# for idx, overlap_value in results.items():
#     print(f"Group {idx}: Overlap = {overlap_value:.4f}")


# # plot_projected_hamiltonian_levels(3, t=0.0)

# #Write results to file
# cwd_path = Path.cwd()
# results_path = cwd_path / f"braiding_results_step_projected_braiding_U={specified_vals['U'][0]}.txt"

# with open(results_path, "w") as f:
#     f.write("Group\tOverlap\n")
#     for idx, overlap_value in results.items():
#         f.write(f"{idx}\t{overlap_value:.10f}\n")
