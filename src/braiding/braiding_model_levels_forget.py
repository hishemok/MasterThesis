from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from explore_hamiltonian_values import calculate_parities_optimized
from get_mzm_JW import build_JW_string, construct_majoranas, subsys_parity_oper, tensorprod
from hamiltonian_builder import BraidingHamiltonianBuilder, default_config_path


def delta_pulse(t: float, t_peak: float, width: float, steepness: float, delta_max: float, delta_min: float) -> float:
    t_start = t_peak - width / 2.0
    t_end = t_peak + width / 2.0

    rise = 1.0 / (1.0 + np.exp(-steepness * (t - t_start)))
    fall = 1.0 / (1.0 + np.exp(steepness * (t - t_end)))
    return delta_min + (delta_max - delta_min) * rise * fall


def build_effective_hamiltonian(
    t: float,
    total_time: float,
    delta_max: float,
    delta_min: float,
    steepness: float,
    width: float,
    gamma0: np.ndarray,
    gamma1: np.ndarray,
    gamma2: np.ndarray,
    gamma3: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    delta_1 = (
        delta_pulse(t, 0.0, width, steepness, delta_max, delta_min)
        + delta_pulse(t, total_time, width, steepness, delta_max, delta_min)
        - delta_min
    )
    delta_2 = delta_pulse(t, total_time / 3.0, width, steepness, delta_max, delta_min)
    delta_3 = delta_pulse(t, 2.0 * total_time / 3.0, width, steepness, delta_max, delta_min)

    hamiltonian = (
        delta_1 * 1j * gamma0 @ gamma1
        + delta_2 * 1j * gamma0 @ gamma2
        + delta_3 * 1j * gamma0 @ gamma3
    )
    return hamiltonian, (delta_1, delta_2, delta_3)


def project_and_normalize(operator: np.ndarray, basis: np.ndarray) -> np.ndarray:
    projected = basis.conj().T @ operator @ basis
    dim = projected.shape[0]
    norm = np.sqrt(np.trace(projected @ projected).real / dim)
    return projected / norm


def build_total_parity_full(builder: BraidingHamiltonianBuilder) -> np.ndarray:
    number_ops = builder.get_operators()["num"]
    dim = number_ops[0].shape[0]
    identity = np.eye(dim, dtype=complex)
    parity = identity.copy()

    for number_op in number_ops:
        parity = parity @ (identity - 2 * number_op)

    return parity


def phase_aligned_error(unitary: np.ndarray, target: np.ndarray) -> float:
    overlap = np.trace(target.conj().T @ unitary)
    phase = 0.0 if np.isclose(np.abs(overlap), 0.0) else np.angle(overlap)
    return np.linalg.norm(unitary - np.exp(1j * phase) * target)


def build_full_gamma_cache(
    levels_list: list[int],
    config_path,
) -> dict[int, tuple[np.ndarray, ...]]:
    builder_sub = BraidingHamiltonianBuilder(
        n_sites=3,
        dupes=1,
        specified_vals={"U": [0.1]},
        config_path=config_path,
    )
    h_sub = builder_sub.full_system_hamiltonian()
    eigvals_sub, eigvecs_sub = np.linalg.eigh(h_sub)
    parity_sub = subsys_parity_oper(sites=builder_sub.n_sites)
    even_energies, odd_energies, even_vecs, odd_vecs = calculate_parities_optimized(
        eigvecs_sub, eigvals_sub, parity_sub
    )

    jw_a = np.eye(2**3, dtype=complex)
    jw_b = build_JW_string(3)
    jw_c = build_JW_string(6)

    gamma_cache = {}
    for levels_to_include in levels_list:
        gamma_a1, gamma_a2 = construct_majoranas(
            even_vecs, odd_vecs, even_energies, odd_energies, n=levels_to_include
        )
        gamma_b1, gamma_b2 = construct_majoranas(
            even_vecs, odd_vecs, even_energies, odd_energies, n=levels_to_include
        )
        gamma_c1, gamma_c2 = construct_majoranas(
            even_vecs, odd_vecs, even_energies, odd_energies, n=levels_to_include
        )

        gamma_cache[levels_to_include] = (
            tensorprod([gamma_a1, np.eye(2**6, dtype=complex)]),
            tensorprod([gamma_a2, np.eye(2**6, dtype=complex)]),
            tensorprod([jw_b, gamma_b1, np.eye(2**3, dtype=complex)]),
            tensorprod([jw_b, gamma_b2, np.eye(2**3, dtype=complex)]),
            tensorprod([jw_c, gamma_c1]),
            tensorprod([jw_c, gamma_c2]),
        )

    return gamma_cache


@dataclass
class SweepResult:
    projection_dim: int
    levels_to_include: int
    negative_dim: int
    max_hermiticity_error: float
    max_parity_commutator: float
    max_square_error: float
    max_anticommutator_error: float
    max_ground_splitting: float
    min_gap: float
    unitary_error: float
    transport_error: float
    closure_error: float
    exchange_23_error: float
    exchange_32_error: float
    spectator_1_error: float
    spectator_0_error: float
    double_2_error: float
    double_3_error: float
    double_1_error: float
    double_0_error: float
    four_exchange_error: float
    parity_offblock_error: float
    block_target_error_max: float


def analyze_case(
    projection_dim: int,
    levels_to_include: int,
    builder: BraidingHamiltonianBuilder,
    eigenvectors_full: np.ndarray,
    parity_full: np.ndarray,
    gamma_cache: dict[int, tuple[np.ndarray, ...]],
    total_time: float,
    delta_max: float,
    delta_min: float,
    width: float,
    steepness: float,
    n_points: int,
    energy_tol: float = 1e-10,
) -> SweepResult:
    basis = eigenvectors_full[:, :projection_dim]
    parity_projected = basis.conj().T @ parity_full @ basis

    full_gammas = gamma_cache[levels_to_include]
    projected_gammas = [project_and_normalize(gamma, basis) for gamma in full_gammas]
    gamma0, gamma1, gamma2, _, gamma3, _ = projected_gammas

    identity = np.eye(projection_dim, dtype=complex)
    max_square_error = max(np.linalg.norm(gamma @ gamma - identity) for gamma in projected_gammas)
    max_anticommutator_error = max(
        np.linalg.norm(projected_gammas[i] @ projected_gammas[j] + projected_gammas[j] @ projected_gammas[i])
        for i in range(len(projected_gammas))
        for j in range(i + 1, len(projected_gammas))
    )

    times = np.linspace(0.0, total_time, n_points)
    dt = times[1] - times[0]

    h0, _ = build_effective_hamiltonian(
        0.0, total_time, delta_max, delta_min, steepness, width, gamma0, gamma1, gamma2, gamma3
    )
    evals_0, evecs_0 = np.linalg.eigh(h0)
    negative_dim = int(np.count_nonzero(evals_0 < -energy_tol))
    if not 0 < negative_dim < projection_dim:
        raise ValueError(
            f"Projection dim {projection_dim} with levels={levels_to_include} gave invalid "
            f"negative-energy count {negative_dim}."
        )

    v_negative = evecs_0[:, :negative_dim]
    p0 = v_negative @ v_negative.conj().T
    u_kato = np.eye(projection_dim, dtype=complex)

    max_hermiticity_error = np.linalg.norm(h0 - h0.conj().T)
    max_parity_commutator = np.linalg.norm(h0 @ parity_projected - parity_projected @ h0)
    max_ground_splitting = evals_0[negative_dim - 1] - evals_0[0]
    min_gap = evals_0[negative_dim] - evals_0[negative_dim - 1]

    for t in times[1:]:
        hamiltonian, _ = build_effective_hamiltonian(
            t, total_time, delta_max, delta_min, steepness, width, gamma0, gamma1, gamma2, gamma3
        )
        evals_t, evecs_t = np.linalg.eigh(hamiltonian)
        negative_dim_t = int(np.count_nonzero(evals_t < -energy_tol))
        if negative_dim_t != negative_dim:
            raise ValueError(
                f"Negative-energy manifold changed from {negative_dim} to {negative_dim_t} "
                f"for projection dim {projection_dim} and levels={levels_to_include}."
            )

        max_hermiticity_error = max(max_hermiticity_error, np.linalg.norm(hamiltonian - hamiltonian.conj().T))
        max_parity_commutator = max(
            max_parity_commutator,
            np.linalg.norm(hamiltonian @ parity_projected - parity_projected @ hamiltonian),
        )
        max_ground_splitting = max(max_ground_splitting, evals_t[negative_dim - 1] - evals_t[0])
        min_gap = min(min_gap, evals_t[negative_dim] - evals_t[negative_dim - 1])

        w_negative = evecs_t[:, :negative_dim]
        projector = v_negative @ v_negative.conj().T
        projector_next = w_negative @ w_negative.conj().T
        generator = projector @ ((projector_next - projector) / dt) - ((projector_next - projector) / dt) @ projector

        u_kato = expm(-dt * generator) @ u_kato
        v_negative = w_negative

    pt = v_negative @ v_negative.conj().T
    unitary_error = np.linalg.norm(u_kato.conj().T @ u_kato - identity)
    transport_error = np.linalg.norm(u_kato @ p0 @ u_kato.conj().T - pt)
    closure_error = np.linalg.norm(pt - p0)

    transformed_gamma2 = u_kato.conj().T @ gamma2 @ u_kato
    transformed_gamma3 = u_kato.conj().T @ gamma3 @ u_kato
    transformed_gamma1 = u_kato.conj().T @ gamma1 @ u_kato
    transformed_gamma0 = u_kato.conj().T @ gamma0 @ u_kato

    exchange_23_error = np.linalg.norm(transformed_gamma2 + gamma3)
    exchange_32_error = np.linalg.norm(transformed_gamma3 - gamma2)
    spectator_1_error = np.linalg.norm(transformed_gamma1 - gamma1)
    spectator_0_error = np.linalg.norm(transformed_gamma0 - gamma0)

    u_double = u_kato @ u_kato
    double_2_error = np.linalg.norm(u_double.conj().T @ gamma2 @ u_double + gamma2)
    double_3_error = np.linalg.norm(u_double.conj().T @ gamma3 @ u_double + gamma3)
    double_1_error = np.linalg.norm(u_double.conj().T @ gamma1 @ u_double - gamma1)
    double_0_error = np.linalg.norm(u_double.conj().T @ gamma0 @ u_double - gamma0)

    u_four = u_double @ u_double
    four_exchange_error = max(
        np.linalg.norm(u_four.conj().T @ gamma @ u_four - gamma)
        for gamma in (gamma0, gamma1, gamma2, gamma3)
    )

    u_ground = evecs_0[:, :negative_dim].conj().T @ u_kato @ evecs_0[:, :negative_dim]
    parity_ground = evecs_0[:, :negative_dim].conj().T @ parity_projected @ evecs_0[:, :negative_dim]
    parity_vals, parity_vecs = np.linalg.eigh(parity_ground)
    order = np.argsort(parity_vals)
    parity_vals = parity_vals[order]
    parity_vecs = parity_vecs[:, order]
    negative_parity_dim = int(np.count_nonzero(parity_vals < 0.0))
    positive_parity_dim = parity_vals.size - negative_parity_dim

    if negative_parity_dim == 0 or positive_parity_dim == 0:
        parity_offblock_error = np.nan
        block_target_error_max = np.nan
    else:
        u_parity = parity_vecs.conj().T @ u_ground @ parity_vecs
        target_full = expm(-0.25 * np.pi * (gamma2 @ gamma3))
        target_ground = evecs_0[:, :negative_dim].conj().T @ target_full @ evecs_0[:, :negative_dim]
        target_parity = parity_vecs.conj().T @ target_ground @ parity_vecs

        parity_offblock_error = (
            np.linalg.norm(u_parity[:negative_parity_dim, negative_parity_dim:])
            + np.linalg.norm(u_parity[negative_parity_dim:, :negative_parity_dim])
        )
        odd_error = phase_aligned_error(
            u_parity[:negative_parity_dim, :negative_parity_dim],
            target_parity[:negative_parity_dim, :negative_parity_dim],
        )
        even_error = phase_aligned_error(
            u_parity[negative_parity_dim:, negative_parity_dim:],
            target_parity[negative_parity_dim:, negative_parity_dim:],
        )
        block_target_error_max = max(odd_error, even_error)

    return SweepResult(
        projection_dim=projection_dim,
        levels_to_include=levels_to_include,
        negative_dim=negative_dim,
        max_hermiticity_error=max_hermiticity_error,
        max_parity_commutator=max_parity_commutator,
        max_square_error=max_square_error,
        max_anticommutator_error=max_anticommutator_error,
        max_ground_splitting=max_ground_splitting,
        min_gap=min_gap,
        unitary_error=unitary_error,
        transport_error=transport_error,
        closure_error=closure_error,
        exchange_23_error=exchange_23_error,
        exchange_32_error=exchange_32_error,
        spectator_1_error=spectator_1_error,
        spectator_0_error=spectator_0_error,
        double_2_error=double_2_error,
        double_3_error=double_3_error,
        double_1_error=double_1_error,
        double_0_error=double_0_error,
        four_exchange_error=four_exchange_error,
        parity_offblock_error=parity_offblock_error,
        block_target_error_max=block_target_error_max,
    )


def print_result(result: SweepResult) -> None:
    print(
        f"M={result.projection_dim:>3}  levels={result.levels_to_include}  neg={result.negative_dim:>2}  "
        f"sq={result.max_square_error:.2e}  anti={result.max_anticommutator_error:.2e}  "
        f"split={result.max_ground_splitting:.2e}  gap={result.min_gap:.3e}"
    )
    print(
        f"           U={result.unitary_error:.2e}  transp={result.transport_error:.2e}  "
        f"close={result.closure_error:.2e}  parity={result.max_parity_commutator:.2e}"
    )
    print(
        f"           braid23={result.exchange_23_error:.2e}  braid32={result.exchange_32_error:.2e}  "
        f"spec1={result.spectator_1_error:.2e}  spec0={result.spectator_0_error:.2e}"
    )
    print(
        f"           dbl2={result.double_2_error:.2e}  dbl3={result.double_3_error:.2e}  "
        f"dbl1={result.double_1_error:.2e}  dbl0={result.double_0_error:.2e}  "
        f"four={result.four_exchange_error:.2e}"
    )
    print(
        f"           parity_off={result.parity_offblock_error:.2e}  "
        f"target={result.block_target_error_max:.2e}"
    )


def print_failure(projection_dim: int, levels_to_include: int, error: Exception) -> None:
    print(f"M={projection_dim:>3}  levels={levels_to_include}  FAILED")
    print(f"           {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the projected effective braiding model over larger Hilbert-space projections "
            "and different numbers of subsystem parity pairs used in the Majorana construction."
        )
    )
    parser.add_argument("--projection-dims", nargs="+", type=int, default=[8, 16, 24, 32])
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--n-points", type=int, default=600)
    parser.add_argument("--t-total", type=float, default=1000.0)
    parser.add_argument("--delta-max", type=float, default=1.0)
    parser.add_argument("--delta-min", type=float, default=0.0)
    parser.add_argument("--width-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument(
        "--steepness-scale",
        type=float,
        default=20.0,
        help="The pulse steepness is set to steepness_scale / width to match braiding_model.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = default_config_path()

    builder = BraidingHamiltonianBuilder(
        n_sites=3,
        dupes=3,
        specified_vals={"U": [0.1]},
        config_path=config_path,
    )
    full_hamiltonian = builder.full_system_hamiltonian()
    _, eigenvectors_full = np.linalg.eigh(full_hamiltonian)
    parity_full = build_total_parity_full(builder)
    gamma_cache = build_full_gamma_cache(args.levels, config_path)

    width = args.width_fraction * args.t_total
    steepness = args.steepness_scale / width

    print("This sweep enlarges the projected effective model.")
    print("It tests when the gamma-built braid stops behaving ideally as the projection dimension grows.")
    print("It does not replace a fully microscopic time-dependent braid simulation.\n")

    results = []
    for projection_dim in args.projection_dims:
        for levels_to_include in args.levels:
            print(f"Running M={projection_dim}, levels={levels_to_include}...")
            try:
                result = analyze_case(
                    projection_dim=projection_dim,
                    levels_to_include=levels_to_include,
                    builder=builder,
                    eigenvectors_full=eigenvectors_full,
                    parity_full=parity_full,
                    gamma_cache=gamma_cache,
                    total_time=args.t_total,
                    delta_max=args.delta_max,
                    delta_min=args.delta_min,
                    width=width,
                    steepness=steepness,
                    n_points=args.n_points,
                )
            except Exception as error:
                print_failure(projection_dim, levels_to_include, error)
                print()
                continue

            results.append(result)
            print_result(result)
            print()

    if len(args.levels) > 1:
        print("Level-to-level differences at fixed projection dimension")
        grouped = {}
        for result in results:
            grouped.setdefault(result.projection_dim, {})[result.levels_to_include] = result

        base_level = min(args.levels)
        for projection_dim in args.projection_dims:
            if base_level not in grouped.get(projection_dim, {}):
                continue
            base = grouped[projection_dim][base_level]
            for levels_to_include in sorted(args.levels):
                if levels_to_include == base_level or levels_to_include not in grouped[projection_dim]:
                    continue
                other = grouped[projection_dim][levels_to_include]
                print(
                    f"M={projection_dim:>3}: levels {base_level} -> {levels_to_include}  "
                    f"target delta={other.block_target_error_max - base.block_target_error_max:+.2e}  "
                    f"split delta={other.max_ground_splitting - base.max_ground_splitting:+.2e}  "
                    f"gap delta={other.min_gap - base.min_gap:+.2e}"
                )


if __name__ == "__main__":
    main()


"""

Level-to-level differences at fixed projection dimension
M=  8: levels 1 -> 2  target delta=+3.56e-16  split delta=+2.22e-16  gap delta=+4.44e-16
M=  8: levels 1 -> 3  target delta=+6.82e-16  split delta=-1.11e-16  gap delta=+1.11e-15
M=  8: levels 1 -> 4  target delta=+5.41e-16  split delta=-1.11e-16  gap delta=+4.44e-16
M= 32: levels 1 -> 2  target delta=-3.69e-01  split delta=-1.33e+00  gap delta=+1.41e+00
M= 32: levels 1 -> 3  target delta=-3.69e-01  split delta=-1.33e+00  gap delta=+1.41e+00
M= 32: levels 1 -> 4  target delta=-3.69e-01  split delta=-1.33e+00  gap delta=+1.41e+00
M=256: levels 1 -> 2  target delta=-3.11e-01  split delta=-1.07e+00  gap delta=-4.42e-09
M=256: levels 1 -> 3  target delta=-9.85e-01  split delta=-1.36e+00  gap delta=+7.09e-05
M=256: levels 1 -> 4  target delta=-1.54e+00  split delta=-2.46e+00  gap delta=+1.41e+00
M=512: levels 1 -> 2  target delta=+4.48e-01  split delta=-2.00e+00  gap delta=-8.25e-09
M=512: levels 1 -> 3  target delta=-6.95e-01  split delta=-2.67e+00  gap delta=-1.10e-08
M=512: levels 1 -> 4  target delta=-2.61e+00  split delta=-4.00e+00  gap delta=+1.41e+00
"""