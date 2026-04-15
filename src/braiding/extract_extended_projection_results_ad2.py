from get_mzm_JW import get_full_gammas
import numpy as np
from scipy.linalg import expm
from tqdm import tqdm
from hamiltonian_builder import BraidingHamiltonianBuilder, default_config_path
from braiding_model import delta_pulse
from full_system_hamiltonian import precompute_ops
from extended_projection_braiding import normalize_projected_majorana, build_total_parity_full, build_projection_stack


def project_operator(operator_full, basis):
    return basis.conj().T @ operator_full @ basis


def project_and_normalize(operator_full, basis):
    operator_sub = project_operator(operator_full, basis)
    return normalize_projected_majorana(operator_sub, "projected_operator")


def hermitian_part(operator):
    return 0.5 * (operator + operator.conj().T)


def build_hamiltonian(t, T_total, delta_max, delta_min, steepness, width, term_a, term_b, term_c):
    delta_1 = delta_pulse(t, 0, width, steepness, delta_max, delta_min) + delta_pulse(
        t, T_total, width, steepness, delta_max, delta_min
    ) - delta_min
    delta_2 = delta_pulse(t, T_total / 3, width, steepness, delta_max, delta_min)
    delta_3 = delta_pulse(t, 2 * T_total / 3, width, steepness, delta_max, delta_min)
    hamiltonian = delta_1 * term_a + delta_2 * term_b + delta_3 * term_c
    return hamiltonian, (delta_1, delta_2, delta_3)


def evolve_system(
    t_total,
    delta_max,
    delta_min,
    steepness,
    width,
    term_a,
    term_b,
    term_c,
    n_points=1000,
    transport_dim=None,
    verbose=False,
):
    times = np.linspace(0, t_total, n_points)
    dt = times[1] - times[0] if n_points > 1 else t_total

    dim = term_a.shape[0]
    if transport_dim is None:
        transport_dim = dim // 2
    if not 0 < transport_dim < dim:
        raise ValueError(f"transport_dim must be between 1 and {dim - 1}, got {transport_dim}.")

    energies = np.zeros((n_points, dim))
    couplings = np.zeros((n_points, 3))
    u_kato = np.eye(dim, dtype=complex)

    hamiltonian, couplings[0] = build_hamiltonian(times[0], t_total, delta_max, delta_min, steepness, width, term_a, term_b, term_c)
    evals, evecs = np.linalg.eigh(hamiltonian)
    energies[0] = evals
    basis = evecs[:, :transport_dim]

    if verbose:
        print("Analyzing ad2 projected braid...")
    for idx in tqdm(range(1, len(times)), total=len(times) - 1, disable=not verbose):
        hamiltonian, couplings[idx] = build_hamiltonian(
            times[idx],
            t_total,
            delta_max,
            delta_min,
            steepness,
            width,
            term_a,
            term_b,
            term_c,
        )
        evals, evecs = np.linalg.eigh(hamiltonian)
        energies[idx] = evals
        next_basis = evecs[:, :transport_dim]
        projector = basis @ basis.conj().T
        next_projector = next_basis @ next_basis.conj().T
        d_projector = (next_projector - projector) / dt
        kato_generator = projector @ d_projector - d_projector @ projector
        u_kato = expm(-dt * kato_generator) @ u_kato
        basis = next_basis

    return times, energies, couplings, u_kato


def get_ground_manifold_data(t_total, delta_max, delta_min, steepness, width, term_a, term_b, term_c, transport_dim):
    h0, _ = build_hamiltonian(0.0, t_total, delta_max, delta_min, steepness, width, term_a, term_b, term_c)
    h1, _ = build_hamiltonian(t_total, t_total, delta_max, delta_min, steepness, width, term_a, term_b, term_c)
    evals_0, evecs_0 = np.linalg.eigh(h0)
    evals_1, evecs_1 = np.linalg.eigh(h1)
    v0 = evecs_0[:, :transport_dim]
    v1 = evecs_1[:, :transport_dim]
    p0 = v0 @ v0.conj().T
    p1 = v1 @ v1.conj().T
    return {
        "evals_0": evals_0,
        "evals_1": evals_1,
        "V0": v0,
        "VT": v1,
        "P0": p0,
        "PT": p1,
    }


def phase_aligned_error(unitary, target):
    overlap = np.trace(target.conj().T @ unitary)
    phase = 0.0 if np.isclose(np.abs(overlap), 0.0) else np.angle(overlap)
    return np.linalg.norm(unitary - np.exp(1j * phase) * target)


def check_single_exchange(u_kato, gamma_list, verbose=False):
    expected_maps = [
        ("γ2 -> -γ3", gamma_list[2], -gamma_list[3]),
        ("γ3 ->  γ2", gamma_list[3], gamma_list[2]),
        ("γ1 ->  γ1", gamma_list[1], gamma_list[1]),
        ("γ0 ->  γ0", gamma_list[0], gamma_list[0]),
    ]

    errors = {}
    if verbose:
        print("\nSingle-exchange checks")
    for label, source, target in expected_maps:
        transformed = u_kato.conj().T @ source @ u_kato
        error = np.linalg.norm(transformed - target)
        errors[label] = error
        if verbose:
            print(f"{label}: {error:.2e}")

    return {
        "errors": errors,
        "max_error": max(errors.values()) if errors else 0.0,
    }


def format_single_exchange_errors(single_exchange):
    ordered_labels = [
        "γ2 -> -γ3",
        "γ3 ->  γ2",
        "γ1 ->  γ1",
        "γ0 ->  γ0",
    ]
    return ", ".join(
        f"{label}={single_exchange['errors'][label]:.2e}"
        for label in ordered_labels
    )


def check_parity_resolved_gate(u_kato, v0, parity_op, gamma2, gamma3, verbose=False):
    u_ground = v0.conj().T @ u_kato @ v0
    parity_ground = v0.conj().T @ parity_op @ v0

    parity_vals, parity_vecs = np.linalg.eigh(parity_ground)
    u_parity = parity_vecs.conj().T @ u_ground @ parity_vecs

    odd_indices = np.flatnonzero(parity_vals < -1e-8)
    even_indices = np.flatnonzero(parity_vals > 1e-8)
    mixed_indices = np.flatnonzero(np.abs(parity_vals) <= 1e-8)

    off_block = (
        np.linalg.norm(u_parity[np.ix_(odd_indices, even_indices)])
        + np.linalg.norm(u_parity[np.ix_(even_indices, odd_indices)])
    )
    odd_block = u_parity[np.ix_(odd_indices, odd_indices)]
    even_block = u_parity[np.ix_(even_indices, even_indices)]

    u_target = expm(-0.25 * np.pi * (gamma2 @ gamma3))
    u_target_ground = v0.conj().T @ u_target @ v0
    u_target_parity = parity_vecs.conj().T @ u_target_ground @ parity_vecs
    odd_target = u_target_parity[np.ix_(odd_indices, odd_indices)]
    even_target = u_target_parity[np.ix_(even_indices, even_indices)]

    odd_error = phase_aligned_error(odd_block, odd_target)
    even_error = phase_aligned_error(even_block, even_target)

    if verbose:
        print("\nParity-resolved gate checks")
        print(
            "parity counts in transported manifold: "
            f"odd={len(odd_indices)}, even={len(even_indices)}, mixed={len(mixed_indices)}"
        )
        print(f"off-block leakage in parity basis: {off_block:.2e}")
        print(f"odd-block target error:  {odd_error:.2e}")
        print(f"even-block target error: {even_error:.2e}")

    return {
        "off_block_leakage": off_block,
        "odd_target_error": odd_error,
        "even_target_error": even_error,
        "odd_dim": len(odd_indices),
        "even_dim": len(even_indices),
        "mixed_dim": len(mixed_indices),
    }


def get_single_majorana_components(term, gamma_labels, gamma_ops):
    dim = term.shape[0]
    rows = []

    for label, gamma in zip(gamma_labels, gamma_ops):
        coeff = np.trace(gamma.conj().T @ term).real / dim
        rows.append((abs(coeff), coeff, label))

    rows.sort(reverse=True, key=lambda item: item[0])
    return rows


def print_single_majorana_decomposition(rows, title, max_terms=None, verbose=False):
    if not verbose:
        return

    print(f"\n{title}")
    if max_terms is None:
        max_terms = len(rows)

    for _, coeff, label in rows[:max_terms]:
        print(f"{label}: {coeff:.3e}")


def main():
    specified_vals = {"U": [0.1]}
    verbose = False
    make_plots = False
    max_cumulative_checks = 16
    n_points = 300

    # Match braiding_model_ad2.py: use the local "y" Majorana on the junction sites.
    b_site = 3  # B1
    c_site = 6  # C1

    cre_ops, ann_ops, _ = precompute_ops(9)
    chi_b_full = 1j * (cre_ops[b_site] - ann_ops[b_site])
    chi_c_full = 1j * (cre_ops[c_site] - ann_ops[c_site])

    (gamma_a1_full, gamma_a2_full), (gamma_b1_full, gamma_b2_full), (gamma_c1_full, gamma_c2_full) = get_full_gammas(
        levels_to_include=4,
        verbose=verbose,
        specified_vals=specified_vals,
    )

    builder = BraidingHamiltonianBuilder(
        n_sites=3,
        dupes=3,
        specified_vals=specified_vals,
        config_path=default_config_path(),
    )

    h_full = builder.full_system_hamiltonian()
    eigvals, eigvecs = np.linalg.eigh(h_full)

    energy_tol = 1e-2
    parity_full = build_total_parity_full(builder)
    projection_blocks = build_projection_stack(eigvals, eigvecs, parity_full, energy_tol=energy_tol)
    basis_stack = [block["basis"] for block in projection_blocks]
    cumulative_stack = [np.hstack(basis_stack[: idx + 1]) for idx in range(len(basis_stack))]

    if verbose:
        print(f"Built {len(basis_stack)} energy-manifold basis blocks with energy_tol={energy_tol:g}.")

    all_stored_vals = []

    for block, basis in zip(projection_blocks[:max_cumulative_checks], cumulative_stack[:max_cumulative_checks]):
        dim_sub = basis.shape[1]
        transport_dim = dim_sub // 2

        if dim_sub % 2 != 0:
            if verbose:
                print(f"Skipping odd-dimensional projection with dim={dim_sub}.")
            continue

        gamma_a1 = project_and_normalize(gamma_a1_full, basis)
        gamma_a2 = project_and_normalize(gamma_a2_full, basis)
        gamma_b1 = project_and_normalize(gamma_b1_full, basis)
        gamma_b2 = project_and_normalize(gamma_b2_full, basis)
        gamma_c1 = project_and_normalize(gamma_c1_full, basis)
        gamma_c2 = project_and_normalize(gamma_c2_full, basis)

        gamma_labels = ["γA1", "γA2", "γB1", "γB2", "γC1", "γC2"]
        gamma_ops = [gamma_a1, gamma_a2, gamma_b1, gamma_b2, gamma_c1, gamma_c2]

        projected_local_majoranas = {
            "χ_B": project_and_normalize(chi_b_full, basis),
            "χ_C": project_and_normalize(chi_c_full, basis),
        }

        if verbose:
            for label, operator in projected_local_majoranas.items():
                rows = get_single_majorana_components(operator, gamma_labels, gamma_ops)
                print_single_majorana_decomposition(rows, f"{label} projected into the low-energy basis", max_terms=3, verbose=verbose)

        gamma0, gamma1, gamma2, gamma3 = gamma_a1, gamma_a2, gamma_b2, gamma_c2
        term_a = 1j * gamma0 @ gamma1
        term_b = hermitian_part(project_operator(1j * gamma_a1_full @ chi_b_full, basis))
        term_c = hermitian_part(project_operator(1j * gamma_a1_full @ chi_c_full, basis))

        t_total = 1.0
        delta_max = 1.0
        delta_min = 0.0
        width = t_total / 3.0
        steepness = 20.0 / width

        times, energies, couplings, u_kato = evolve_system(
            t_total,
            delta_max,
            delta_min,
            steepness,
            width,
            term_a,
            term_b,
            term_c,
            n_points=n_points,
            transport_dim=transport_dim,
            verbose=verbose,
        )

        if make_plots:
            from braiding_model import plot_results

            plot_results(times, energies, couplings)

        gamma_list = [gamma0, gamma1, gamma2, gamma3]
        parity_projected = project_operator(parity_full, basis)
        ground_data = get_ground_manifold_data(
            t_total,
            delta_max,
            delta_min,
            steepness,
            width,
            term_a,
            term_b,
            term_c,
            transport_dim,
        )

        single_exchange = check_single_exchange(u_kato, gamma_list, verbose=verbose)
        parity_gate = check_parity_resolved_gate(
            u_kato,
            ground_data["V0"],
            parity_projected,
            gamma2,
            gamma3,
            verbose=verbose,
        )

        stored = {
            "block_name": block["name"],
            "projection_dim": dim_sub,
            "manifold_dim": block["dim"],
            "braid_error": single_exchange["max_error"],
            "single_exchange_errors": {
                key: float(value) for key, value in single_exchange["errors"].items()
            },
            "off_block_leakage": parity_gate["off_block_leakage"],
            "odd_target_error": parity_gate["odd_target_error"],
            "even_target_error": parity_gate["even_target_error"],
        }
        all_stored_vals.append(stored)

        print(
            f"{block['name']} cum_dim={dim_sub} "
            f"max_braid_error={single_exchange['max_error']:.2e}"
            f"\n braid_errors: {format_single_exchange_errors(single_exchange)}"
        )

    if verbose:
        print("\nSummary of ad2 cumulative projection checks")
        for vals in all_stored_vals:
            print(
                f"{vals['block_name']} cum_dim={vals['projection_dim']} "
                f"braid_error={vals['braid_error']:.2e} "
                f"braid_errors={vals['single_exchange_errors']} "
                f"odd_error={vals['odd_target_error']:.2e} "
                f"even_error={vals['even_target_error']:.2e}"
            )


if __name__ == "__main__":
    main()
