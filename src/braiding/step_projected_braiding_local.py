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




def get_energy_group_ticks(even_energies, odd_energies, energy_tol=1e-2):
    group_energies = []
    group_sizes = []
    used = np.zeros(min(len(even_energies), len(odd_energies)), dtype=bool)

    for i in range(len(used)):
        if used[i]:
            continue

        group = []
        for j in range(i, len(used)):
            if (
                np.isclose(even_energies[i], even_energies[j], atol=energy_tol)
                and np.isclose(odd_energies[i], odd_energies[j], atol=energy_tol)
            ):
                used[j] = True
                group.extend((even_energies[j], odd_energies[j]))

        group_energies.append(np.mean(group))
        group_sizes.append(len(group))

    return group_energies, group_sizes


def save_energy_level_figure(fig, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.08, dpi=220)


def spread_group_label_positions(group_energies, y_limits, min_gap_fraction=0.026):
    lower, upper = y_limits
    padding = 0.015 * (upper - lower)
    lower += padding
    upper -= padding
    min_gap = min_gap_fraction * (upper - lower)
    positions = np.array(group_energies, dtype=float)

    for i in range(1, len(positions)):
        positions[i] = max(positions[i], positions[i - 1] + min_gap)

    if positions[-1] > upper:
        positions -= positions[-1] - upper
    for i in range(len(positions) - 2, -1, -1):
        positions[i] = min(positions[i], positions[i + 1] - min_gap)
    if positions[0] < lower:
        positions += lower - positions[0]

    return positions


def plot_energy_levels(
    even_energies,
    odd_energies,
    title="Energy levels colored by parity",
    output_path=None,
    show=True,
    figsize=(7.2, 6.4),
    degeneracy_tol=1e-2,
):
    xmin_even,xmax_even = 0,1
    xmin_odd,xmax_odd = 2,3
    lw1 = 3.0
    fig, energy_axis = plt.subplots(figsize=figsize)
    energy_axis.hlines(even_energies, xmin_even, xmax_even, color='navy', label='Even parity energies', linewidth=lw1)
    energy_axis.hlines(odd_energies, xmin_odd, xmax_odd, color='firebrick', label='Odd parity energies', linewidth=lw1)

    degeneracy_label = "Degenerate parity pair"
    for even_energy, odd_energy in zip(even_energies, odd_energies):
        if np.isclose(even_energy, odd_energy, atol=degeneracy_tol, rtol=0.0):
            energy_axis.plot(
                [xmax_even, xmin_odd],
                [even_energy, odd_energy],
                color="gray",
                linestyle="--",
                linewidth=1.5,
                label=degeneracy_label,
            )
            degeneracy_label = None

    energy_axis.set_xlabel('Parity sector', fontsize=18)
    energy_axis.set_ylabel('Energy', fontsize=18)
    energy_axis.set_xticks([0.5, 2.5], ['Even', 'Odd'], fontsize=14)
    energy_axis.tick_params(axis="y", labelsize=14)
    energy_axis.set_title(title, fontsize=18, pad=38)
    energy_axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    group_energies, group_sizes = get_energy_group_ticks(even_energies, odd_energies)
    group_label_positions = spread_group_label_positions(group_energies, energy_axis.get_ylim())
    right_edge = energy_axis.get_xlim()[1]
    for group_energy, label_position in zip(group_energies, group_label_positions):
        if not np.isclose(group_energy, label_position):
            energy_axis.plot(
                [xmax_odd, right_edge],
                [group_energy, label_position],
                color="dimgray",
                linewidth=0.7,
                alpha=0.65,
                clip_on=False,
            )

    group_size_axis = energy_axis.twinx()
    group_size_axis.set_ylim(energy_axis.get_ylim())
    group_size_axis.set_yticks(group_label_positions)
    group_size_axis.set_yticklabels(group_sizes, fontsize=12)
    group_size_color = "dimgray"

    group_size_axis.set_ylabel(
        "Group size",
        fontsize=18,
        color=group_size_color,
    )
    group_size_axis.tick_params(
        axis="y",
        labelsize=12,
        colors=group_size_color,
    )
    group_size_axis.spines["right"].set_color(group_size_color)

    fig.tight_layout()
    if output_path is not None:
        save_energy_level_figure(fig, output_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

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

def plot_energy_levels_from_evals(
    evals,
    evecs,
    parity,
    title="Energy levels colored by parity",
    output_path=None,
    show=True,
    figsize=(7.2, 6.4),
):
    even_energies, odd_energies = split_evals_by_parity(evals, evecs, parity)
    plot_energy_levels(
        even_energies,
        odd_energies,
        title=title,
        output_path=output_path,
        show=show,
        figsize=figsize,
    )
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



def project_operator(operator, basis):
    return basis.conj().T @ operator @ basis

def projected_local_majoranas(cdag, c, basis):
    gamma_plus = project_operator(cdag + c, basis)
    gamma_minus = project_operator(1j * (cdag - c), basis)
    return gamma_plus, gamma_minus

def select_physical_majorana(gamma_plus, gamma_minus,mode = "minus_only" ):
    if mode == "minus_only":
        return gamma_minus
    if mode == "plus_only":
        return gamma_plus
    if mode == "plus_minus":
        return gamma_plus + gamma_minus
    raise ValueError(f"Unknown mode={mode}.")

def parity_op(ops, sites = 3):
    num = ops["num"]
    
    dim = num[0].shape[0]
    I = np.eye(dim, dtype=complex)
    P = I.copy()
    for i in range(sites):
        P = P @ (I - 2 * num[i])
    return P




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

    # for group in groups:
    #     print(group)
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

def project_local_majoranas(bases, operators, mode = "minus_only", B_INNER_SITE = 3, C_INNER_SITE = 6):
    local_majoranas = []
    for i, basis in enumerate(bases):
        B_plus, B_minus = projected_local_majoranas(
            operators["cre"][B_INNER_SITE],
            operators["ann"][B_INNER_SITE],
            basis,
        )
        C_plus, C_minus = projected_local_majoranas(
            operators["cre"][C_INNER_SITE],
            operators["ann"][C_INNER_SITE],
            basis,
        )

        B_local = normalize_projected_majorana(select_physical_majorana(B_plus, B_minus, mode), f"B local group {i}")
        C_local = normalize_projected_majorana(select_physical_majorana(C_plus, C_minus, mode), f"C local group {i}")
        local_majoranas.append((B_local, C_local))
    return local_majoranas

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


def get_braiding_terms(projected_majoranas, local_majoranas, idx):
    """idx needs to range from 0 to len(projected_majoranas)"""
    A1_proj, A2_proj, B1_proj, B2_proj, C1_proj, C2_proj = projected_majoranas[idx]
    B_local, C_local = local_majoranas[idx]
    TA = hermitian_part(1j * A1_proj @ A2_proj)
    TB = hermitian_part(1j* A1_proj @ B_local)
    TC = hermitian_part(1j* A1_proj @ C_local)
    return TA, TB, TC

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

def plot_projected_hamiltonian_levels(idx, t=0.0, title=None, output_path=None, show=True):
    TA, TB, TC = get_braiding_terms(projected_majorana_groups, local_majorana_groups, idx)
    hamiltonian, deltas = build_projected_hamiltonian(t, TA, TB, TC)
    evals, evecs = np.linalg.eigh(hamiltonian)
    parity_projected = bases[idx].conj().T @ full_parity @ bases[idx]
    if title is None:
        title = f"Group {idx} projected Hamiltonian at t={t:.3f}"
    return plot_energy_levels_from_evals(
        evals,
        evecs,
        parity_projected,
        title=title,
        output_path=output_path,
        show=show,
    )

def evolve_system(n_steps, projected_majoranas, local_majoranas, idx):
    TA, TB, TC = get_braiding_terms(projected_majoranas, local_majoranas, idx)
    time_arr = np.linspace(0, T_total, n_steps)
    dt = time_arr[1] - time_arr[0]

    transport_dim = TA.shape[0] // 2
    print(f"Transport dimension: {transport_dim}")
    group_basis = bases[idx]
    static_term = group_basis.conj().T @ h_full @ group_basis
    static_term = hermitian_part(static_term)
    
    U_kato = np.eye(TA.shape[0], dtype=complex)
    H_t, deltas = build_projected_hamiltonian(0, TA, TB, TC, static_term=static_term)
    evals, evecs = np.linalg.eigh(H_t)

    plot_projected_hamiltonian_levels(idx, t=0.0)
    exit()


    initial_basis = evecs[:, :transport_dim]
    basis = initial_basis  # Start in the subspace of the lowest transport_dim eigenstates

    for t in tqdm(time_arr[1:], desc="Evolving system"):
        H_t, deltas = build_projected_hamiltonian(t, TA, TB, TC, static_term=static_term)

        evals, evecs = np.linalg.eigh(H_t)
        # print(evals)
        # even_energies, odd_energies, even_vecs, odd_vecs, even_idxs, odd_idxs = even_odd_splitter(evecs, evals,  bases[idx].conj().T @ full_parity @ bases[idx])
        # # g_unit = find_close_groups(even_energies, odd_energies, even_idxs, odd_idxs)
        # print(f"Group {idx} at time {t:.3f}: found {len(g_unit)} groups in projected Hamiltonian")
        # plot_projected_hamiltonian_levels(idx, t=t)
        # exit()
        next_basis = evecs[:, :transport_dim]
        projector = basis @ basis.conj().T
        # print(projector)
        next_projector = next_basis @ next_basis.conj().T
        d_projector = (next_projector - projector) / dt
        kato_generator = projector @ d_projector - d_projector @ projector
        U_kato = expm(-dt * kato_generator) @ U_kato
        basis = next_basis


    return U_kato, initial_basis, static_term


def get_ideal_target_unitary(projected_majoranas, idx, mode="minus_only"):

    """ DOES THIS PART MAKE SENSE?"""
    A1_proj, A2_proj, B1_proj, B2_proj, C1_proj, C2_proj = projected_majoranas[idx]
    if mode == "minus_only":
        return np.array(expm(-np.pi/4 * B2_proj @ C2_proj))
    elif mode == "plus_only":
        return np.array(expm(-np.pi/4 * B1_proj @ C1_proj))
    elif mode == "plus_minus":
        return np.array(expm(-np.pi/4 * B1_proj @ C2_proj))
    elif mode == "minus_plus":
        return np.array(expm(-np.pi/4 * B2_proj @ C1_proj))
    else:
        raise ValueError(f"Unknown mode={mode}.")

def get_local_target_unitary(local_majoranas, idx):
    B_local, C_local = local_majoranas[idx]
    return np.array(expm(-np.pi/4 * B_local @ C_local))


def matrix_overlap(A, B):
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    if np.isclose(denom, 0.0):
        return 0.0
    return abs(np.trace(B.conj().T @ A)) / denom

def fit_to_ideal_pair(local_gamma, gamma_plus, gamma_minus):
    ideal_gammas = [gamma_plus, gamma_minus]
    gram = np.array(
        [[np.trace(g1.conj().T @ g2) for g2 in ideal_gammas] for g1 in ideal_gammas],
        dtype=complex,
    )
    rhs = np.array([np.trace(g.conj().T @ local_gamma) for g in ideal_gammas], dtype=complex)
    coeffs = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    fit = coeffs[0] * gamma_plus + coeffs[1] * gamma_minus

    overlap = matrix_overlap(local_gamma, fit)
    error = np.linalg.norm(local_gamma - fit) / np.linalg.norm(local_gamma)
    return coeffs, overlap, error

def get_local_majorana_fit(projected_majoranas, local_majoranas, idx):
    A1_proj, A2_proj, B1_proj, B2_proj, C1_proj, C2_proj = projected_majoranas[idx]
    B_local, C_local = local_majoranas[idx]

    B_coeffs, B_overlap, B_error = fit_to_ideal_pair(B_local, B1_proj, B2_proj)
    C_coeffs, C_overlap, C_error = fit_to_ideal_pair(C_local, C1_proj, C2_proj)
    return B_coeffs, B_overlap, B_error, C_coeffs, C_overlap, C_error

def format_coeffs(coeffs):
    return f"[{coeffs[0].real:+.4f}{coeffs[0].imag:+.4f}j, {coeffs[1].real:+.4f}{coeffs[1].imag:+.4f}j]"

def unitary_overlap(U, V):
    d = U.shape[0]
    return abs(np.trace(V.conj().T @ U)) / d

def phase_aligned_error(U, V):
    overlap = np.trace(V.conj().T @ U)
    phase = 0.0 if np.isclose(abs(overlap), 0.0) else np.angle(overlap)
    return np.linalg.norm(U - np.exp(1j * phase) * V) / np.sqrt(U.shape[0])


def generate_thesis_energy_level_figures(
    output_dir=None,
    u_values=(0.0, 0.1, 0.5, 1.0, 2.0),
):
    global T_total, Delta_max, Delta_min, Width, S

    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "texmex" / "figs"
    output_dir = Path(output_dir)

    T_total = 1.0
    Delta_max = 1.0
    Delta_min = 0.0
    Width = T_total / 3
    S = 20 / Width

    for u_value in u_values:
        print(f"Generating energy-level figures for U={u_value:g}")
        specified_vals = {"U": [u_value]}
        builder = BraidingHamiltonianBuilder(
            n_sites=3,
            dupes=3,
            specified_vals=specified_vals,
            config_path=default_config_path(),
        )
        hamiltonian_full = builder.full_system_hamiltonian()
        eigvals, eigvecs = np.linalg.eigh(hamiltonian_full)
        operators = builder.get_operators()
        parity_full = parity_op(operators, sites=9)
        u_slug = str(u_value).replace(".", "p")

        plot_energy_levels_from_evals(
            eigvals,
            eigvecs,
            parity_full,
            title=fr"Full-system spectrum, $U={u_value:g}$",
            output_path=output_dir / f"full_system_spectrum_U_{u_slug}",
            show=False,
            figsize=(7.2, 8.0),
        )

        gamma_groups = get_majoranas_JW(
            levels_to_include=4,
            specified_vals=specified_vals,
        )
        (gamma_A1, gamma_A2), (gamma_B1, gamma_B2), (gamma_C1, gamma_C2) = gamma_groups
        even_energies, odd_energies, _, _, even_idxs, odd_idxs = even_odd_splitter(
            eigvecs,
            eigvals,
            parity_full,
        )
        groups_for_u = find_close_groups(even_energies, odd_energies, even_idxs, odd_idxs)
        bases_for_u = find_group_bases(groups_for_u, eigvecs)
        projected_majoranas_for_u = project_majoranas(
            bases_for_u,
            gamma_A1,
            gamma_A2,
            gamma_B1,
            gamma_B2,
            gamma_C1,
            gamma_C2,
        )
        local_majoranas_for_u = project_local_majoranas(bases_for_u, operators)

        TA, TB, TC = get_braiding_terms(projected_majoranas_for_u, local_majoranas_for_u, idx=0)
        projected_hamiltonian, _ = build_projected_hamiltonian(0.0, TA, TB, TC)
        projected_evals, projected_evecs = np.linalg.eigh(projected_hamiltonian)
        projected_parity = bases_for_u[0].conj().T @ parity_full @ bases_for_u[0]
        plot_energy_levels_from_evals(
            projected_evals,
            projected_evecs,
            projected_parity,
            title=fr"Projected ground-group spectrum, $U={u_value:g}$",
            output_path=output_dir / f"projected_ground_group_spectrum_U_{u_slug}",
            show=False,
            figsize=(7.2, 6.4),
        )


if __name__ == "__main__":
    for uval in (0.0, 0.1, 0.5, 1.0, 2.0):
        print(f"\n\n=== Running scan for U={uval} ===")
        specified_vals = {"U": [uval]}
        mode = "minus_only"  # Options: "minus_only", "plus_only", "plus_minus"
        B_INNER_SITE = 3
        C_INNER_SITE = 6

        builder = BraidingHamiltonianBuilder(
            n_sites=3,
            dupes=3,
            specified_vals=specified_vals,
            config_path=default_config_path(),
        )
        h_full = builder.full_system_hamiltonian()
        eigvals, eigvecs = np.linalg.eigh(h_full)
        operators = builder.get_operators()

        plot_energy_levels_from_evals(eigvals, eigvecs, parity_op(operators, sites=9), title=f"Full Hamiltonian Energy Levels for U={uval}")
        exit()

        (gamma_A1_full, gamma_A2_full), (gamma_B1_full, gamma_B2_full), (gamma_C1_full, gamma_C2_full) = get_majoranas_JW(levels_to_include=4, specified_vals=specified_vals)


        ops = precompute_operators(n=3, dup=3)
        full_parity = parity_op(ops, sites=9)
        even_energies, odd_energies, even_vecs, odd_vecs, even_idxs, odd_idxs = even_odd_splitter(eigvecs, eigvals, full_parity)
        print(even_energies)

        groups = find_close_groups(even_energies, odd_energies, even_idxs, odd_idxs)
        bases = find_group_bases(groups, eigvecs)
        projected_majorana_groups = project_majoranas(bases, gamma_A1_full, gamma_A2_full, gamma_B1_full, gamma_B2_full, gamma_C1_full, gamma_C2_full)
        local_majorana_groups = project_local_majoranas(bases, operators, mode=mode)
        majorana_checks(projected_majorana_groups)



        T_total = 1.0
        Delta_max = 1.0
        Delta_min = 0.0
        Width = T_total / 3
        S = 20 / Width

        ideal_target_results = {
            "minus_only": {},
            "plus_only": {},
            "plus_minus": {},
            "minus_plus": {}
        }


        local_target_results = {}
        for idx in range(0, len(projected_majorana_groups)):
            U_kato, initial_basis, static_term = evolve_system(300, projected_majorana_groups, local_majorana_groups, idx=idx)
            B_coeffs, B_fit_overlap, B_fit_error, C_coeffs, C_fit_overlap, C_fit_error = get_local_majorana_fit(
                projected_majorana_groups,
                local_majorana_groups,
                idx,
            )
            
            U_ideal_target_minus = get_ideal_target_unitary(projected_majorana_groups, idx=idx, mode="minus_only")
            U_ideal_target_plus = get_ideal_target_unitary(projected_majorana_groups, idx=idx, mode="plus_only")
            U_ideal_target_plus_minus = get_ideal_target_unitary(projected_majorana_groups, idx=idx, mode="plus_minus")
            U_ideal_target_minus_plus = get_ideal_target_unitary(projected_majorana_groups, idx=idx, mode="minus_plus")
            U_local_target = get_local_target_unitary(local_majorana_groups, idx=idx)

            U_kato_sub = initial_basis.conj().T @ U_kato @ initial_basis
            U_ideal_target_minus_sub = initial_basis.conj().T @ U_ideal_target_minus @ initial_basis
            U_ideal_target_plus_sub = initial_basis.conj().T @ U_ideal_target_plus @ initial_basis
            U_ideal_target_plus_minus_sub = initial_basis.conj().T @ U_ideal_target_plus_minus @ initial_basis
            U_ideal_target_minus_plus_sub = initial_basis.conj().T @ U_ideal_target_minus_plus @ initial_basis
            U_local_target_sub = initial_basis.conj().T @ U_local_target @ initial_basis

            ideal_target_results["minus_only"][idx] = unitary_overlap(U_kato_sub, U_ideal_target_minus_sub)
            ideal_target_results["plus_only"][idx] = unitary_overlap(U_kato_sub, U_ideal_target_plus_sub)
            ideal_target_results["plus_minus"][idx] = unitary_overlap(U_kato_sub, U_ideal_target_plus_minus_sub)
            ideal_target_results["minus_plus"][idx] = unitary_overlap(U_kato_sub, U_ideal_target_minus_plus_sub)
            local_target_results[idx] = unitary_overlap(U_kato_sub, U_local_target_sub)


            print(
                f"local target overlap = {local_target_results[idx]:.10f}\n"
                f"Group {idx}: ideal target overlap = {ideal_target_results['minus_only'][idx]:.10f},\n "
                f"ideal target plus_only overlap = {ideal_target_results['plus_only'][idx]:.10f},\n "
                f"ideal target plus_minus overlap = {ideal_target_results['plus_minus'][idx]:.10f}, \n"
                f"ideal target minus_plus overlap = {ideal_target_results['minus_plus'][idx]:.10f}\n"
                f"B local fit in [B1, B2]: overlap = {B_fit_overlap:.10f}, error = {B_fit_error:.3e}, coeffs = {format_coeffs(B_coeffs)}\n"
                f"C local fit in [C1, C2]: overlap = {C_fit_overlap:.10f}, error = {C_fit_error:.3e}, coeffs = {format_coeffs(C_coeffs)}"
            )

        print("Local-operator overlap results for all groups:")
        for idx in local_target_results:
                print(f"local target = {local_target_results[idx]:.10f}")
                for mode in ideal_target_results:
                    print(f"Group {idx}: ideal target {mode}: {ideal_target_results[mode][idx]:.10f}")
            


        #Write results to file
        cwd_path = Path.cwd()
        results_path = cwd_path / f"braiding_results_step_projected_braiding_local_U={specified_vals['U'][0]}.txt"

        with open(results_path, "w") as f:
            f.write(
                "Group\tIdeal Minus\tIdeal Plus\tIdeal Plus-Minus\t"
                "Ideal Minus-Plus\tLocal Target Overlap\n"
            )
            for idx in local_target_results:
                f.write(
                    f"{idx}\t"
                    f"{ideal_target_results['minus_only'][idx]:.10f}\t"
                    f"{ideal_target_results['plus_only'][idx]:.10f}\t"
                    f"{ideal_target_results['plus_minus'][idx]:.10f}\t"
                    f"{ideal_target_results['minus_plus'][idx]:.10f}\t"
                    f"{local_target_results[idx]:.10f}\n"
                )


        # plot_projected_hamiltonian_levels(3, t=0.0)
