import csv
from pathlib import Path

from get_mzm_JW import get_full_gammas as get_basic_full_gammas
from remake_majoranas3 import make_majoranas_for_B_and_C_with_projection_dim
import numpy as np
from scipy.linalg import expm
from tqdm import tqdm
from hamiltonian_builder import BraidingHamiltonianBuilder, default_config_path
from braiding_model import delta_pulse
from extended_projection_braiding import normalize_projected_majorana


U_VALUES = [0.1]
PROJECTION_LEVELS = [8]
COMPONENT_LEVELS = 4
N_POINTS = 300
VERBOSE = True
RESULTS_OUTPUT_PATH = Path(__file__).with_name("retry2_projection_scan_results.txt")
TRANSPORT_GAP_SCAN_POINTS = 41
TRANSPORT_DIM_STEP = 4
PHYSICAL_COUPLING_MODE = "minus_only"

T_TOTAL = 1.0
DELTA_MAX = 1.0
DELTA_MIN = 0.0
WIDTH = T_TOTAL / 3
STEEPNESS = 20 / WIDTH

B_INNER_SITE = 3
C_INNER_SITE = 6

OPERATOR_ACTION_COLUMNS = [
    ("gamma2 -> -gamma3", "gamma2_to_minus_gamma3"),
    ("gamma3 -> gamma2", "gamma3_to_gamma2"),
    ("gamma1 -> gamma1", "gamma1_to_gamma1"),
    ("gamma0 -> gamma0", "gamma0_to_gamma0"),
]


def build_projected_hamiltonian(t, term_a, term_b, term_c, static_term=None):
    delta_1 = (
        delta_pulse(t, 0, WIDTH, STEEPNESS, DELTA_MAX, DELTA_MIN)
        + delta_pulse(t, T_TOTAL, WIDTH, STEEPNESS, DELTA_MAX, DELTA_MIN)
        - DELTA_MIN
    )
    delta_2 = delta_pulse(t, T_TOTAL / 3, WIDTH, STEEPNESS, DELTA_MAX, DELTA_MIN)
    delta_3 = delta_pulse(t, 2 * T_TOTAL / 3, WIDTH, STEEPNESS, DELTA_MAX, DELTA_MIN)

    hamiltonian = delta_1 * term_a + delta_2 * term_b + delta_3 * term_c
    if static_term is not None:
        hamiltonian = hamiltonian + static_term
    return hamiltonian, (delta_1, delta_2, delta_3)


def get_projection_basis(eigenvectors, levels_to_include):
    basis = eigenvectors[:, :levels_to_include]
    overlap = basis.conj().T @ basis
    if not np.allclose(overlap, np.eye(levels_to_include, dtype=complex)):
        raise ValueError("Projection basis is not orthonormal: V†V != I")
    return basis


def project_operator(operator, projection_basis):
    return projection_basis.conj().T @ operator @ projection_basis


def projected_majoranas(cdag, c, projection_basis):
    gamma_plus = project_operator(cdag + c, projection_basis)
    gamma_minus = project_operator(1j * (cdag - c), projection_basis)
    return gamma_plus, gamma_minus


def select_physical_majorana(gamma_plus, gamma_minus):
    if PHYSICAL_COUPLING_MODE == "minus_only":
        return gamma_minus
    if PHYSICAL_COUPLING_MODE == "plus_only":
        return gamma_plus
    if PHYSICAL_COUPLING_MODE == "plus_minus":
        return gamma_plus + gamma_minus
    raise ValueError(f"Unknown PHYSICAL_COUPLING_MODE={PHYSICAL_COUPLING_MODE!r}.")


def evolve_system(term_a, term_b, term_c, static_term=None, transport_dim=None):
    times = np.linspace(0, T_TOTAL, N_POINTS)
    dt = times[1] - times[0] if N_POINTS > 1 else T_TOTAL

    dim = term_a.shape[0]
    if transport_dim is None:
        transport_dim = dim // 2

    energies = np.zeros((N_POINTS, dim))
    couplings = np.zeros((N_POINTS, 3))
    u_kato = np.eye(dim, dtype=complex)

    hamiltonian, couplings[0] = build_projected_hamiltonian(
        times[0],
        term_a,
        term_b,
        term_c,
        static_term=static_term,
    )
    evals, evecs = np.linalg.eigh(hamiltonian)
    energies[0] = evals
    basis = evecs[:, :transport_dim]

    for index in tqdm(range(1, N_POINTS), disable=not VERBOSE):
        hamiltonian, couplings[index] = build_projected_hamiltonian(
            times[index],
            term_a,
            term_b,
            term_c,
            static_term=static_term,
        )
        evals, evecs = np.linalg.eigh(hamiltonian)
        energies[index] = evals

        next_basis = evecs[:, :transport_dim]
        projector = basis @ basis.conj().T
        next_projector = next_basis @ next_basis.conj().T
        d_projector = (next_projector - projector) / dt
        kato_generator = projector @ d_projector - d_projector @ projector
        u_kato = expm(-dt * kato_generator) @ u_kato
        basis = next_basis

    return times, energies, couplings, u_kato


def candidate_transport_dims(dim):
    max_transport_dim = dim // 2
    candidates = list(range(TRANSPORT_DIM_STEP, max_transport_dim + 1, TRANSPORT_DIM_STEP))
    if not candidates:
        candidates = [max_transport_dim]
    return candidates


def scan_transport_gaps(term_a, term_b, term_c, static_term=None):
    dim = term_a.shape[0]
    min_gaps = np.full(dim - 1, np.inf)

    for time in np.linspace(0.0, T_TOTAL, TRANSPORT_GAP_SCAN_POINTS):
        hamiltonian, _ = build_projected_hamiltonian(
            time,
            term_a,
            term_b,
            term_c,
            static_term=static_term,
        )
        evals = np.linalg.eigvalsh(hamiltonian)
        min_gaps = np.minimum(min_gaps, np.diff(evals))

    return min_gaps


def choose_transport_dim(term_a, term_b, term_c, static_term=None):
    dim = term_a.shape[0]
    candidates = candidate_transport_dims(dim)
    min_gaps = scan_transport_gaps(term_a, term_b, term_c, static_term=static_term)

    transport_dim = max(candidates, key=lambda candidate: (min_gaps[candidate - 1], -candidate))
    ranked = sorted(
        ((candidate, float(min_gaps[candidate - 1])) for candidate in candidates),
        key=lambda item: item[1],
        reverse=True,
    )

    print("transport dimension chosen by spectral gap:")
    print(
        f"  selected transport_dim={transport_dim}, "
        f"min gap above band={min_gaps[transport_dim - 1]:.4e}"
    )
    preview = ", ".join(f"{candidate}:{gap:.2e}" for candidate, gap in ranked[:5])
    print(f"  best candidate gaps dim:gap = {preview}")

    return transport_dim


def normalized_error(left, right):
    return np.linalg.norm(left - right) / np.sqrt(left.shape[0])


def check_operator_action(u_kato, gamma0, gamma1, gamma2, gamma3):
    checks = [
        ("gamma2 -> -gamma3", gamma2, -gamma3),
        ("gamma3 -> gamma2", gamma3, gamma2),
        ("gamma1 -> gamma1", gamma1, gamma1),
        ("gamma0 -> gamma0", gamma0, gamma0),
    ]

    errors = {}
    for label, source, target in checks:
        transformed = u_kato.conj().T @ source @ u_kato
        errors[label] = normalized_error(transformed, target)
        print(f"  {label}: {errors[label]:.4e}")

    return {
        "errors": errors,
        "max_error": max(errors.values()),
    }


def check_operator_action_in_basis(u_kato, transport_basis, gamma0, gamma1, gamma2, gamma3):
    checks = [
        ("gamma2 -> -gamma3", gamma2, -gamma3),
        ("gamma3 -> gamma2", gamma3, gamma2),
        ("gamma1 -> gamma1", gamma1, gamma1),
        ("gamma0 -> gamma0", gamma0, gamma0),
    ]

    errors = {}
    norm = np.sqrt(transport_basis.shape[1])
    for label, source, target in checks:
        transformed = u_kato.conj().T @ source @ u_kato
        subspace_error = transport_basis.conj().T @ (transformed - target) @ transport_basis
        errors[label] = np.linalg.norm(subspace_error) / norm
        print(f"  {label}: {errors[label]:.4e}")

    return {
        "errors": errors,
        "max_error": max(errors.values()),
    }


def check_majorana_algebra(gammas):
    identity = np.eye(gammas[0].shape[0], dtype=complex)
    square_errors = {}
    for index, gamma in enumerate(gammas):
        label = f"gamma{index}"
        square_errors[label] = normalized_error(gamma @ gamma, identity)
        print(f"  {label}^2 - I: {square_errors[label]:.4e}")

    anticommutator_errors = {}
    for left in range(len(gammas)):
        for right in range(left + 1, len(gammas)):
            label = f"gamma{left}_gamma{right}"
            anticommutator = gammas[left] @ gammas[right] + gammas[right] @ gammas[left]
            anticommutator_errors[label] = np.linalg.norm(anticommutator) / np.sqrt(anticommutator.shape[0])
            print(
                f"  {{gamma{left}, gamma{right}}}: "
                f"{anticommutator_errors[label]:.4e}"
            )

    return {
        "square_errors": square_errors,
        "max_square_error": max(square_errors.values()),
        "anticommutator_errors": anticommutator_errors,
        "max_anticommutator_error": max(anticommutator_errors.values()),
    }


def phase_aligned_error(unitary, target):
    overlap = np.trace(target.conj().T @ unitary)
    phase = 0.0 if np.isclose(np.abs(overlap), 0.0) else np.angle(overlap)
    return np.linalg.norm(unitary - np.exp(1j * phase) * target)


def get_initial_transport_basis(term_a, term_b, term_c, static_term, transport_dim):
    hamiltonian_0, _ = build_projected_hamiltonian(
        0.0,
        term_a,
        term_b,
        term_c,
        static_term=static_term,
    )
    _, evecs_0 = np.linalg.eigh(hamiltonian_0)
    return evecs_0[:, :transport_dim]


def compare_to_target_gate(u_kato, transport_basis, gamma2, gamma3):
    u_subspace = transport_basis.conj().T @ u_kato @ transport_basis
    target_full = expm(-0.25 * np.pi * (gamma2 @ gamma3))
    target_subspace = transport_basis.conj().T @ target_full @ transport_basis
    return phase_aligned_error(u_subspace, target_subspace)


def compare_target_gates(transport_basis, gamma2_left, gamma3_left, gamma2_right, gamma3_right):
    target_left = expm(-0.25 * np.pi * (gamma2_left @ gamma3_left))
    target_right = expm(-0.25 * np.pi * (gamma2_right @ gamma3_right))
    target_left_subspace = transport_basis.conj().T @ target_left @ transport_basis
    target_right_subspace = transport_basis.conj().T @ target_right @ transport_basis
    return phase_aligned_error(target_left_subspace, target_right_subspace)


def operator_mismatch_in_basis(transport_basis, left, right):
    mismatch = transport_basis.conj().T @ (left - right) @ transport_basis
    return np.linalg.norm(mismatch) / np.sqrt(transport_basis.shape[1])


def compare_transport_unitaries(u_reference, u_test, transport_basis):
    u_reference_subspace = transport_basis.conj().T @ u_reference @ transport_basis
    u_test_subspace = transport_basis.conj().T @ u_test @ transport_basis
    return phase_aligned_error(u_test_subspace, u_reference_subspace)


def flatten_operator_action(prefix, action_check, normalization_dim):
    row = {}
    for label, suffix in OPERATOR_ACTION_COLUMNS:
        normalized_value = float(action_check["errors"][label])
        row[f"{prefix}_{suffix}_error"] = float(normalized_value * np.sqrt(normalization_dim))
        row[f"{prefix}_{suffix}_error_normalized"] = normalized_value
    normalized_max = float(action_check["max_error"])
    row[f"{prefix}_max_error"] = float(normalized_max * np.sqrt(normalization_dim))
    row[f"{prefix}_max_error_normalized"] = normalized_max
    return row


def flatten_algebra_check(prefix, algebra_check):
    row = {
        f"{prefix}_max_square_error_normalized": float(algebra_check["max_square_error"]),
        f"{prefix}_max_anticommutator_error_normalized": float(algebra_check["max_anticommutator_error"]),
    }
    for label, value in algebra_check["square_errors"].items():
        row[f"{prefix}_{label}_square_error_normalized"] = float(value)
    return row


def result_fieldnames():
    fieldnames = [
        "interaction_u",
        "projection_level",
        "transport_dim",
        "component_levels",
        "tocheck",
        "b_matrix_error",
        "b_matrix_error_normalized",
        "c_matrix_error",
        "c_matrix_error_normalized",
        "ideal_target_gate_error",
        "ideal_target_gate_error_normalized",
        "physical_target_gate_error_in_ideal_basis",
        "physical_target_gate_error_in_ideal_basis_normalized",
        "physical_target_gate_error_in_physical_basis",
        "physical_target_gate_error_in_physical_basis_normalized",
        "physical_vs_ideal_target_gate_error_in_ideal_basis",
        "physical_vs_ideal_target_gate_error_in_ideal_basis_normalized",
        "physical_target_gate_error_against_physical_target_in_ideal_basis",
        "physical_target_gate_error_against_physical_target_in_ideal_basis_normalized",
        "physical_target_gate_error_against_ideal_target_in_physical_basis",
        "physical_target_gate_error_against_ideal_target_in_physical_basis_normalized",
        "ideal_vs_physical_target_gate_error_in_ideal_basis",
        "ideal_vs_physical_target_gate_error_in_ideal_basis_normalized",
        "ideal_vs_physical_target_gate_error_in_physical_basis",
        "ideal_vs_physical_target_gate_error_in_physical_basis_normalized",
        "gamma2_ideal_vs_physical_error_in_ideal_basis_normalized",
        "gamma3_ideal_vs_physical_error_in_ideal_basis_normalized",
        "gamma2_ideal_vs_physical_error_in_physical_basis_normalized",
        "gamma3_ideal_vs_physical_error_in_physical_basis_normalized",
    ]
    for prefix in (
        "ideal_single_exchange",
        "physical_single_exchange_in_ideal_basis",
        "physical_single_exchange_in_physical_basis",
        "ideal_transport_single_exchange",
        "physical_transport_single_exchange_in_ideal_basis",
        "physical_transport_single_exchange_in_physical_basis",
    ):
        for _, suffix in OPERATOR_ACTION_COLUMNS:
            fieldnames.append(f"{prefix}_{suffix}_error")
            fieldnames.append(f"{prefix}_{suffix}_error_normalized")
        fieldnames.append(f"{prefix}_max_error")
        fieldnames.append(f"{prefix}_max_error_normalized")
    for prefix in ("ideal_majorana_algebra", "physical_majorana_algebra"):
        fieldnames.extend(
            [
                f"{prefix}_gamma0_square_error_normalized",
                f"{prefix}_gamma1_square_error_normalized",
                f"{prefix}_gamma2_square_error_normalized",
                f"{prefix}_gamma3_square_error_normalized",
                f"{prefix}_max_square_error_normalized",
                f"{prefix}_max_anticommutator_error_normalized",
            ]
        )
    return fieldnames


def save_results_table(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_fieldnames(), delimiter="\t")
        writer.writeheader()
        writer.writerows(results)


def load_results_table(output_path):
    if not output_path.exists():
        return []

    rows = []
    int_fields = {"projection_level", "transport_dim", "component_levels"}
    string_fields = {"tocheck"}
    with output_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for raw_row in reader:
            row = {}
            for key, value in raw_row.items():
                if key in string_fields:
                    row[key] = value
                elif key in int_fields:
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def result_key(row):
    return (float(row["interaction_u"]), int(row["projection_level"]))


def run_one_case(u_value, projection_level):
    print(f"\nU={u_value}, projection level={projection_level}")
    specified_vals = {"U": [u_value]}

    builder = BraidingHamiltonianBuilder(
        n_sites=3,
        dupes=3,
        specified_vals=specified_vals,
        config_path=default_config_path(),
    )
    h_full = builder.full_system_hamiltonian()
    eigvals, eigvecs = np.linalg.eigh(h_full)
    projection_basis = get_projection_basis(eigvecs, projection_level)

    h_static_bc_full = builder.subsystem_hamiltonian("B") + builder.subsystem_hamiltonian("C")
    h_static_bc_projected = project_operator(h_static_bc_full, projection_basis)



    (gamma_a1_full, gamma_a2_full), _, _ = get_basic_full_gammas(
        levels_to_include=COMPONENT_LEVELS,
        verbose=False,
        specified_vals=specified_vals,
    )


    gamma0 = normalize_projected_majorana(project_operator(gamma_a1_full, projection_basis), "gamma0")
    gamma1 = normalize_projected_majorana(project_operator(gamma_a2_full, projection_basis), "gamma1")

    b_result, c_result = make_majoranas_for_B_and_C_with_projection_dim(
        projection_dim=projection_level,
        specified_vals=specified_vals,
        projection_basis=projection_basis,
        component_levels=COMPONENT_LEVELS,
        verbose=VERBOSE,
        tocheck="Both",
    )
    gamma2 = normalize_projected_majorana(b_result["gamma_projected"], "gamma2")
    gamma3 = normalize_projected_majorana(c_result["gamma_projected"], "gamma3")

    operators = builder.get_operators()
    gamma_b1_phys, gamma_b2_phys = projected_majoranas(
        operators["cre"][B_INNER_SITE],
        operators["ann"][B_INNER_SITE],
        projection_basis,
    )

    gamma_c_phys, gamma_c2_phys = projected_majoranas(
        operators["cre"][C_INNER_SITE],
        operators["ann"][C_INNER_SITE],
        projection_basis,
    )

    gamma_b_phys = select_physical_majorana(gamma_b1_phys, gamma_b2_phys)
    gamma_c_phys = select_physical_majorana(gamma_c_phys, gamma_c2_phys)

    gamma2_phys = normalize_projected_majorana(gamma_b_phys, "gamma2_phys")
    gamma3_phys = normalize_projected_majorana(gamma_c_phys, "gamma3_phys")

    term_a = 1j * (gamma0 @ gamma1)
    term_b_ideal = 1j * (gamma0 @ gamma2)
    term_c_ideal = 1j * (gamma0 @ gamma3)
    term_b_phys = 1j * (gamma0 @ gamma2_phys)
    term_c_phys = 1j * (gamma0 @ gamma3_phys)
    transport_dim = choose_transport_dim(
        term_a,
        term_b_ideal,
        term_c_ideal,
        static_term=h_static_bc_projected,
    )

    print("remake_majoranas3 matrix errors:")
    print(f"  B : {b_result['matrix_error']:.4e}")
    print(f"  C : {c_result['matrix_error']:.4e}")

    print("ideal Majorana algebra:")
    ideal_algebra_check = check_majorana_algebra([gamma0, gamma1, gamma2, gamma3])

    print("physical Majorana algebra:")
    physical_algebra_check = check_majorana_algebra([gamma0, gamma1, gamma2_phys, gamma3_phys])

    print("evolving ideal system...")
    _, _, _, u_kato_ideal = evolve_system(
        term_a,
        term_b_ideal,
        term_c_ideal,
        static_term=h_static_bc_projected,
        transport_dim=transport_dim,
    )
    print("ideal braid action:")
    ideal_operator_action = check_operator_action(u_kato_ideal, gamma0, gamma1, gamma2, gamma3)

    print("evolving physical system...")
    _, _, _, u_kato_phys = evolve_system(
        term_a,
        term_b_phys,
        term_c_phys,
        static_term=h_static_bc_projected,
        transport_dim=transport_dim,
    )
    print("physical braid action, checked against physical Majoranas:")
    physical_operator_action_physical_basis = check_operator_action(
        u_kato_phys,
        gamma0,
        gamma1,
        gamma2_phys,
        gamma3_phys,
    )
    physical_operator_action_ideal_basis = check_operator_action(
        u_kato_phys,
        gamma0,
        gamma1,
        gamma2,
        gamma3,
    )

    ideal_basis = get_initial_transport_basis(
        term_a,
        term_b_ideal,
        term_c_ideal,
        h_static_bc_projected,
        transport_dim,
    )
    physical_basis = get_initial_transport_basis(
        term_a,
        term_b_phys,
        term_c_phys,
        h_static_bc_projected,
        transport_dim,
    )
    ideal_target_error = compare_to_target_gate(u_kato_ideal, ideal_basis, gamma2, gamma3)
    physical_target_error_ideal_basis = compare_to_target_gate(u_kato_phys, ideal_basis, gamma2, gamma3)
    physical_target_error_physical_basis = compare_to_target_gate(
        u_kato_phys,
        physical_basis,
        gamma2_phys,
        gamma3_phys,
    )
    physical_target_error_physical_target_ideal_basis = compare_to_target_gate(
        u_kato_phys,
        ideal_basis,
        gamma2_phys,
        gamma3_phys,
    )
    physical_target_error_ideal_target_physical_basis = compare_to_target_gate(
        u_kato_phys,
        physical_basis,
        gamma2,
        gamma3,
    )
    ideal_vs_physical_target_error_ideal_basis = compare_target_gates(
        ideal_basis,
        gamma2,
        gamma3,
        gamma2_phys,
        gamma3_phys,
    )
    ideal_vs_physical_target_error_physical_basis = compare_target_gates(
        physical_basis,
        gamma2,
        gamma3,
        gamma2_phys,
        gamma3_phys,
    )
    gamma2_ideal_vs_physical_error_ideal_basis = operator_mismatch_in_basis(
        ideal_basis,
        gamma2,
        gamma2_phys,
    )
    gamma3_ideal_vs_physical_error_ideal_basis = operator_mismatch_in_basis(
        ideal_basis,
        gamma3,
        gamma3_phys,
    )
    gamma2_ideal_vs_physical_error_physical_basis = operator_mismatch_in_basis(
        physical_basis,
        gamma2,
        gamma2_phys,
    )
    gamma3_ideal_vs_physical_error_physical_basis = operator_mismatch_in_basis(
        physical_basis,
        gamma3,
        gamma3_phys,
    )
    physical_vs_ideal_target_error_ideal_basis = compare_transport_unitaries(
        u_kato_ideal,
        u_kato_phys,
        ideal_basis,
    )

    print("ideal braid action in transported band:")
    ideal_transport_operator_action = check_operator_action_in_basis(
        u_kato_ideal,
        ideal_basis,
        gamma0,
        gamma1,
        gamma2,
        gamma3,
    )
    print("physical braid action in transported physical band:")
    physical_transport_operator_action_physical_basis = check_operator_action_in_basis(
        u_kato_phys,
        physical_basis,
        gamma0,
        gamma1,
        gamma2_phys,
        gamma3_phys,
    )
    print("physical braid action in transported ideal band:")
    physical_transport_operator_action_ideal_basis = check_operator_action_in_basis(
        u_kato_phys,
        ideal_basis,
        gamma0,
        gamma1,
        gamma2,
        gamma3,
    )

    row = {
        "interaction_u": float(u_value),
        "projection_level": int(projection_level),
        "transport_dim": int(transport_dim),
        "component_levels": int(COMPONENT_LEVELS),
        "tocheck": "Both",
        "b_matrix_error": float(b_result["matrix_error"]),
        "b_matrix_error_normalized": float(b_result["matrix_error"] / np.sqrt(projection_level)),
        "c_matrix_error": float(c_result["matrix_error"]),
        "c_matrix_error_normalized": float(c_result["matrix_error"] / np.sqrt(projection_level)),
        "ideal_target_gate_error": float(ideal_target_error),
        "ideal_target_gate_error_normalized": float(ideal_target_error / np.sqrt(transport_dim)),
        "physical_target_gate_error_in_ideal_basis": float(physical_target_error_ideal_basis),
        "physical_target_gate_error_in_ideal_basis_normalized": float(
            physical_target_error_ideal_basis / np.sqrt(transport_dim)
        ),
        "physical_target_gate_error_in_physical_basis": float(physical_target_error_physical_basis),
        "physical_target_gate_error_in_physical_basis_normalized": float(
            physical_target_error_physical_basis / np.sqrt(transport_dim)
        ),
        "physical_vs_ideal_target_gate_error_in_ideal_basis": float(physical_vs_ideal_target_error_ideal_basis),
        "physical_vs_ideal_target_gate_error_in_ideal_basis_normalized": float(
            physical_vs_ideal_target_error_ideal_basis / np.sqrt(transport_dim)
        ),
        "physical_target_gate_error_against_physical_target_in_ideal_basis": float(
            physical_target_error_physical_target_ideal_basis
        ),
        "physical_target_gate_error_against_physical_target_in_ideal_basis_normalized": float(
            physical_target_error_physical_target_ideal_basis / np.sqrt(transport_dim)
        ),
        "physical_target_gate_error_against_ideal_target_in_physical_basis": float(
            physical_target_error_ideal_target_physical_basis
        ),
        "physical_target_gate_error_against_ideal_target_in_physical_basis_normalized": float(
            physical_target_error_ideal_target_physical_basis / np.sqrt(transport_dim)
        ),
        "ideal_vs_physical_target_gate_error_in_ideal_basis": float(ideal_vs_physical_target_error_ideal_basis),
        "ideal_vs_physical_target_gate_error_in_ideal_basis_normalized": float(
            ideal_vs_physical_target_error_ideal_basis / np.sqrt(transport_dim)
        ),
        "ideal_vs_physical_target_gate_error_in_physical_basis": float(ideal_vs_physical_target_error_physical_basis),
        "ideal_vs_physical_target_gate_error_in_physical_basis_normalized": float(
            ideal_vs_physical_target_error_physical_basis / np.sqrt(transport_dim)
        ),
        "gamma2_ideal_vs_physical_error_in_ideal_basis_normalized": float(
            gamma2_ideal_vs_physical_error_ideal_basis
        ),
        "gamma3_ideal_vs_physical_error_in_ideal_basis_normalized": float(
            gamma3_ideal_vs_physical_error_ideal_basis
        ),
        "gamma2_ideal_vs_physical_error_in_physical_basis_normalized": float(
            gamma2_ideal_vs_physical_error_physical_basis
        ),
        "gamma3_ideal_vs_physical_error_in_physical_basis_normalized": float(
            gamma3_ideal_vs_physical_error_physical_basis
        ),
    }
    row.update(flatten_operator_action("ideal_single_exchange", ideal_operator_action, projection_level))
    row.update(
        flatten_operator_action(
            "physical_single_exchange_in_ideal_basis",
            physical_operator_action_ideal_basis,
            projection_level,
        )
    )
    row.update(
        flatten_operator_action(
            "physical_single_exchange_in_physical_basis",
            physical_operator_action_physical_basis,
            projection_level,
        )
    )
    row.update(
        flatten_operator_action(
            "ideal_transport_single_exchange",
            ideal_transport_operator_action,
            transport_dim,
        )
    )
    row.update(
        flatten_operator_action(
            "physical_transport_single_exchange_in_ideal_basis",
            physical_transport_operator_action_ideal_basis,
            transport_dim,
        )
    )
    row.update(
        flatten_operator_action(
            "physical_transport_single_exchange_in_physical_basis",
            physical_transport_operator_action_physical_basis,
            transport_dim,
        )
    )
    row.update(flatten_algebra_check("ideal_majorana_algebra", ideal_algebra_check))
    row.update(flatten_algebra_check("physical_majorana_algebra", physical_algebra_check))
    return row


def main(U_VALUES=U_VALUES, PROJECTION_LEVELS=PROJECTION_LEVELS, COMPONENT_LEVELS=COMPONENT_LEVELS, N_POINTS=N_POINTS):
    print("retry2 constants:")
    print(f"  U_VALUES={U_VALUES}")
    print(f"  PROJECTION_LEVELS={PROJECTION_LEVELS}")
    print(f"  COMPONENT_LEVELS={COMPONENT_LEVELS}")
    print(f"  N_POINTS={N_POINTS}")
    print(f"  TRANSPORT_GAP_SCAN_POINTS={TRANSPORT_GAP_SCAN_POINTS}")
    print(f"  PHYSICAL_COUPLING_MODE={PHYSICAL_COUPLING_MODE}")
    print(f"  RESULTS_OUTPUT_PATH={RESULTS_OUTPUT_PATH}")

    results = load_results_table(RESULTS_OUTPUT_PATH)
    done = {result_key(row) for row in results}
    if results:
        print(f"  loaded {len(results)} existing rows from {RESULTS_OUTPUT_PATH}")

    for u_value in U_VALUES:
        for projection_level in PROJECTION_LEVELS:
            key = (float(u_value), int(projection_level))
            # if key in done:
            #     print(f"skipping existing row: U={u_value}, projection level={projection_level}")
            #     continue

            
            results.append(run_one_case(u_value, projection_level))
            results.sort(key=result_key)
            save_results_table(results, RESULTS_OUTPUT_PATH)
            done.add(key)
            print(f"saved {len(results)} rows to {RESULTS_OUTPUT_PATH}")


if __name__ == "__main__":
    run_one_case(0.1, 8)
    main(U_VALUES=[0.0], PROJECTION_LEVELS=[8, 32, 56], )
