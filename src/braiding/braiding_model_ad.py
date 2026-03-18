""" More realistic braiding test using projected microscopic junction operators.

The old sweet-spot model used
    H_eff(t) = Δ1(t) i γ0 γ1 + Δ2(t) i γ0 γ2 + Δ3(t) i γ0 γ3
with hand-picked Majoranas inside the projected low-energy space.

Here I keep the same pulse order, but replace the external ideal terms by
projected microscopic junction terms:
    T_AB = P^† H_AB^junc P
    T_AC = P^† H_AC^junc P

so the effective Hamiltonian becomes
    H_eff(t) = Δ1(t) i γ0 γ1 + λ_AB(t) T_AB + λ_AC(t) T_AC

If the microscopic couplings are close to the ideal braid picture, then T_AB
and T_AC should mostly line up with the desired Majorana bilinears. If not,
other bilinears should appear in their decomposition, and the braid checks
should get worse.
"""


from get_mzm_JW import get_full_gammas
import numpy as np
from scipy.linalg import expm
from tqdm import tqdm
from hamiltonian_builder import BraidingHamiltonianBuilder, default_config_path
from braiding_model import delta_pulse, plot_results


def project_operator(operator_full, basis):
    return basis.conj().T @ operator_full @ basis


def project_and_normalize(operator_full, basis):
    operator_sub = project_operator(operator_full, basis)
    dim = operator_sub.shape[0]
    operator_sub /= np.sqrt(np.trace(operator_sub @ operator_sub).real / dim)
    return operator_sub


def flatten_site(subsystem, site, n_sites=3):
    return subsystem * n_sites + site


def build_junction_operator(operators, site_a, site_b, t_couple=0.0, delta_couple=0.0):
    left = min(site_a, site_b)
    right = max(site_a, site_b)
    key = (left, right)

    dim = operators["num"][0].shape[0]
    junction = np.zeros((dim, dim), dtype=complex)

    if t_couple != 0:
        junction += -float(t_couple) * operators["hop"][key]
    if delta_couple != 0:
        junction += float(delta_couple) * operators["pair"][key]

    return 0.5 * (junction + junction.conj().T)


def build_projected_hamiltonian(t, T_total, Δ_max, Δ_min, s, width, T_A, T_AB, T_AC):
    Δ1 = delta_pulse(t, 0, width, s, Δ_max, Δ_min) + delta_pulse(t, T_total, width, s, Δ_max, Δ_min) - Δ_min
    Δ2 = delta_pulse(t, T_total / 3, width, s, Δ_max, Δ_min)
    Δ3 = delta_pulse(t, 2 * T_total / 3, width, s, Δ_max, Δ_min)

    H = Δ1 * T_A + Δ2 * T_AB + Δ3 * T_AC
    return H, (Δ1, Δ2, Δ3)


def evolve_projected_system(T_total, Δ_max, Δ_min, s, width, T_A, T_AB, T_AC, n_points=1000):
    times = np.linspace(0, T_total, n_points)
    dt = T_total / n_points

    energies = np.zeros((n_points, 8))
    couplings = np.zeros((n_points, 3))
    U_kato = np.eye(8, dtype=complex)

    H0, coupling0 = build_projected_hamiltonian(0, T_total, Δ_max, Δ_min, s, width, T_A, T_AB, T_AC)
    evals, evecs = np.linalg.eigh(H0)
    energies[0] = evals
    couplings[0] = coupling0

    V = evecs[:, :4]

    print("Analyzing projected microscopic braid...")
    for i in tqdm(range(1, len(times)), total=n_points):
        t = times[i]
        H, couplings[i] = build_projected_hamiltonian(t, T_total, Δ_max, Δ_min, s, width, T_A, T_AB, T_AC)

        evals, evecs = np.linalg.eigh(H)
        energies[i] = evals

        W = evecs[:, :4]
        P = V @ V.conj().T
        P_next = W @ W.conj().T
        K = P @ ((P_next - P) / dt) - ((P_next - P) / dt) @ P
        U_kato = expm(-dt * K) @ U_kato
        V = W

    return times, energies, couplings, U_kato


def build_total_parity_projected(builder, basis):
    operators = builder.get_operators()
    num_ops = operators["num"]
    dim_full = num_ops[0].shape[0]
    identity_full = np.eye(dim_full, dtype=complex)
    parity_full = identity_full.copy()

    for number_op in num_ops:
        parity_full = parity_full @ (identity_full - 2 * number_op)

    return basis.conj().T @ parity_full @ basis


def get_ground_manifold_data(T_total, Δ_max, Δ_min, s, width, T_A, T_AB, T_AC):
    H0, _ = build_projected_hamiltonian(0, T_total, Δ_max, Δ_min, s, width, T_A, T_AB, T_AC)
    HT, _ = build_projected_hamiltonian(T_total, T_total, Δ_max, Δ_min, s, width, T_A, T_AB, T_AC)

    evals_0, evecs_0 = np.linalg.eigh(H0)
    evals_T, evecs_T = np.linalg.eigh(HT)

    V0 = evecs_0[:, :4]
    VT = evecs_T[:, :4]

    P0 = V0 @ V0.conj().T
    PT = VT @ VT.conj().T

    return {
        "evals_0": evals_0,
        "evals_T": evals_T,
        "V0": V0,
        "VT": VT,
        "P0": P0,
        "PT": PT,
    }


def phase_aligned_error(U, target):
    overlap = np.trace(target.conj().T @ U)
    phase = 0.0 if np.isclose(np.abs(overlap), 0.0) else np.angle(overlap)
    return np.linalg.norm(U - np.exp(1j * phase) * target)


def check_majorana_algebra(gamma_list):
    print("\nMajorana algebra checks")
    dim = gamma_list[0].shape[0]
    identity = np.eye(dim, dtype=complex)

    for i, gamma in enumerate(gamma_list):
        hermitian_error = np.linalg.norm(gamma - gamma.conj().T)
        square_error = np.linalg.norm(gamma @ gamma - identity)
        print(
            f"γ{i}: ||γ - γ†|| = {hermitian_error:.2e}, "
            f"||γ² - I|| = {square_error:.2e}"
        )

    for i in range(len(gamma_list)):
        for j in range(i + 1, len(gamma_list)):
            anticommutator_error = np.linalg.norm(
                gamma_list[i] @ gamma_list[j] + gamma_list[j] @ gamma_list[i]
            )
            print(f"{{γ{i}, γ{j}}}: {anticommutator_error:.2e}")


def check_path_properties(times, energies, parity_op, T_total, Δ_max, Δ_min, s, width, T_A, T_AB, T_AC):
    max_hermiticity_error = 0.0
    max_parity_commutator = 0.0

    for t in times:
        H, _ = build_projected_hamiltonian(t, T_total, Δ_max, Δ_min, s, width, T_A, T_AB, T_AC)
        max_hermiticity_error = max(max_hermiticity_error, np.linalg.norm(H - H.conj().T))
        max_parity_commutator = max(
            max_parity_commutator,
            np.linalg.norm(H @ parity_op - parity_op @ H),
        )

    ground_splitting = np.max(energies[:, 3] - energies[:, 0])
    min_gap = np.min(energies[:, 4] - energies[:, 3])

    print("\nPath checks")
    print(f"max_t ||H(t) - H(t)†|| = {max_hermiticity_error:.2e}")
    print(f"max_t ||[H(t), P_tot]|| = {max_parity_commutator:.2e}")
    print(f"max_t (E3 - E0) = {ground_splitting:.2e}")
    print(f"min_t (E4 - E3) = {min_gap:.2e}")


def check_kato_transport(U_kato, P0, PT):
    dim = U_kato.shape[0]
    identity = np.eye(dim, dtype=complex)
    unitary_error = np.linalg.norm(U_kato.conj().T @ U_kato - identity)
    transport_error = np.linalg.norm(U_kato @ P0 @ U_kato.conj().T - PT)
    loop_closure_error = np.linalg.norm(PT - P0)

    print("\nKato transport checks")
    print(f"||U†U - I|| = {unitary_error:.2e}")
    print(f"||U P0 U† - PT|| = {transport_error:.2e}")
    print(f"||PT - P0|| = {loop_closure_error:.2e}")


def check_single_exchange(U_kato, gamma_list):
    expected_maps = [
        ("γ2 -> -γ3", gamma_list[2], -gamma_list[3]),
        ("γ3 ->  γ2", gamma_list[3], gamma_list[2]),
        ("γ1 ->  γ1", gamma_list[1], gamma_list[1]),
        ("γ0 ->  γ0", gamma_list[0], gamma_list[0]),
    ]

    print("\nSingle-exchange checks")
    for label, source, target in expected_maps:
        transformed = U_kato.conj().T @ source @ U_kato
        error = np.linalg.norm(transformed - target)
        print(f"{label}: {error:.2e}")


def check_double_exchange(U_kato, gamma_list):
    U_double = U_kato @ U_kato
    expected_maps = [
        ("γ2 -> -γ2", gamma_list[2], -gamma_list[2]),
        ("γ3 -> -γ3", gamma_list[3], -gamma_list[3]),
        ("γ1 ->  γ1", gamma_list[1], gamma_list[1]),
        ("γ0 ->  γ0", gamma_list[0], gamma_list[0]),
    ]

    print("\nDouble-exchange checks")
    for label, source, target in expected_maps:
        transformed = U_double.conj().T @ source @ U_double
        error = np.linalg.norm(transformed - target)
        print(f"{label}: {error:.2e}")


def check_four_exchanges(U_kato, gamma_list):
    U_four = U_kato @ U_kato @ U_kato @ U_kato

    print("\nFour-exchange checks")
    for i, gamma in enumerate(gamma_list):
        transformed = U_four.conj().T @ gamma @ U_four
        print(f"γ{i} -> γ{i}: {np.linalg.norm(transformed - gamma):.2e}")


def check_parity_resolved_gate(U_kato, V0, parity_op, γ2, γ3):
    U_ground = V0.conj().T @ U_kato @ V0
    parity_ground = V0.conj().T @ parity_op @ V0

    parity_vals, parity_vecs = np.linalg.eigh(parity_ground)
    U_parity = parity_vecs.conj().T @ U_ground @ parity_vecs

    off_block = np.linalg.norm(U_parity[:2, 2:]) + np.linalg.norm(U_parity[2:, :2])
    odd_block = U_parity[:2, :2]
    even_block = U_parity[2:, 2:]

    U_target = expm(-0.25 * np.pi * (γ2 @ γ3))
    U_target_ground = V0.conj().T @ U_target @ V0
    U_target_parity = parity_vecs.conj().T @ U_target_ground @ parity_vecs
    odd_target = U_target_parity[:2, :2]
    even_target = U_target_parity[2:, 2:]

    print("\nParity-resolved gate checks")
    print(f"parity eigenvalues in GS manifold: {np.round(parity_vals, 8)}")
    print(f"off-block leakage in parity basis: {off_block:.2e}")
    print(f"odd-block eigenvalues:  {np.round(np.linalg.eigvals(odd_block), 8)}")
    print(f"even-block eigenvalues: {np.round(np.linalg.eigvals(even_block), 8)}")
    print(f"odd-block target error:  {phase_aligned_error(odd_block, odd_target):.2e}")
    print(f"even-block target error: {phase_aligned_error(even_block, even_target):.2e}")


def get_bilinear_components(term, gamma_labels, gamma_ops):
    dim = term.shape[0]
    rows = []

    for i in range(len(gamma_ops)):
        for j in range(i + 1, len(gamma_ops)):
            basis_op = 1j * gamma_ops[i] @ gamma_ops[j]
            coeff = np.trace(basis_op.conj().T @ term).real / dim
            rows.append((abs(coeff), coeff, i, j, f"i {gamma_labels[i]} {gamma_labels[j]}"))

    rows.sort(reverse=True, key=lambda item: item[0])
    return rows


def print_bilinear_decomposition(rows, title, max_terms=None):
    print(f"\n{title}")
    if max_terms is None:
        max_terms = len(rows)

    for _, coeff, _, _, label in rows[:max_terms]:
        print(f"{label}: {coeff:.3e}")


def get_partner_index(index):
    pair_map = {
        0: 1,
        1: 0,
        2: 3,
        3: 2,
        4: 5,
        5: 4,
    }
    return pair_map[index]


def choose_active_majoranas(rows_AB, rows_AC, gamma_labels, gamma_ops):
    _, coeff_AB, i_AB, j_AB, label_AB = rows_AB[0]
    _, coeff_AC, i_AC, j_AC, label_AC = rows_AC[0]

    pair_AB = {i_AB, j_AB}
    pair_AC = {i_AC, j_AC}
    common = pair_AB & pair_AC

    if len(common) != 1:
        raise ValueError(
            "Could not identify a unique shared Majorana between the dominant AB and AC junction terms."
        )

    γ0_idx = common.pop()
    γ2_idx = j_AB if i_AB == γ0_idx else i_AB
    γ3_idx = j_AC if i_AC == γ0_idx else i_AC
    γ1_idx = get_partner_index(γ0_idx)

    print("\nActive Majoranas chosen from dominant projected junction terms")
    print(f"AB dominant term: {label_AB} = {coeff_AB:.3e}")
    print(f"AC dominant term: {label_AC} = {coeff_AC:.3e}")
    print(f"Shared Majorana γ0 = {gamma_labels[γ0_idx]}")
    print(f"Partner on same subsystem γ1 = {gamma_labels[γ1_idx]}")
    print(f"Exchange pair: γ2 = {gamma_labels[γ2_idx]}, γ3 = {gamma_labels[γ3_idx]}")

    return {
        "γ0_idx": γ0_idx,
        "γ1_idx": γ1_idx,
        "γ2_idx": γ2_idx,
        "γ3_idx": γ3_idx,
        "γ0": gamma_ops[γ0_idx],
        "γ1": gamma_ops[γ1_idx],
        "γ2": gamma_ops[γ2_idx],
        "γ3": gamma_ops[γ3_idx],
    }


builder = BraidingHamiltonianBuilder(
    n_sites=3,
    dupes=3,
    specified_vals={"U": [0.1]},
    config_path=default_config_path(),
)


(gamma_A1_full, gamma_A2_full), (gamma_B1_full, gamma_B2_full), (gamma_C1_full, gamma_C2_full) = get_full_gammas(
    levels_to_include=4,
    verbose=False,
)

H_full = builder.full_system_hamiltonian()
_, eigvecs = np.linalg.eigh(H_full)
V_ref = eigvecs[:, :8]

gamma_A1_sub = project_and_normalize(gamma_A1_full, V_ref)
gamma_A2_sub = project_and_normalize(gamma_A2_full, V_ref)
gamma_B1_sub = project_and_normalize(gamma_B1_full, V_ref)
gamma_B2_sub = project_and_normalize(gamma_B2_full, V_ref)
gamma_C1_sub = project_and_normalize(gamma_C1_full, V_ref)
gamma_C2_sub = project_and_normalize(gamma_C2_full, V_ref)

operators = builder.get_operators()
A_edge = flatten_site(0, 2)
B_edge = flatten_site(1, 0)
C_edge = flatten_site(2, 0)

t_AB = builder.t[0]
delta_AB = builder.Delta[0]
t_AC = builder.t[0]
delta_AC = builder.Delta[0]

junction_AB_full = build_junction_operator(operators, A_edge, B_edge, t_couple=t_AB, delta_couple=delta_AB)
junction_AC_full = build_junction_operator(operators, A_edge, C_edge, t_couple=t_AC, delta_couple=delta_AC)

gamma_labels = ["γA1", "γA2", "γB1", "γB2", "γC1", "γC2"]
gamma_ops = [gamma_A1_sub, gamma_A2_sub, gamma_B1_sub, gamma_B2_sub, gamma_C1_sub, gamma_C2_sub]

T_AB = project_operator(junction_AB_full, V_ref)
T_AC = project_operator(junction_AC_full, V_ref)

rows_AB = get_bilinear_components(T_AB, gamma_labels, gamma_ops)
rows_AC = get_bilinear_components(T_AC, gamma_labels, gamma_ops)

print_bilinear_decomposition(rows_AB, "Projected AB junction decomposition", max_terms=6)
print_bilinear_decomposition(rows_AC, "Projected AC junction decomposition", max_terms=6)

active = choose_active_majoranas(rows_AB, rows_AC, gamma_labels, gamma_ops)
γ0, γ1, γ2, γ3 = active["γ0"], active["γ1"], active["γ2"], active["γ3"]

T_A = 1j * γ0 @ γ1


T_total = 1000.0
Δ_max = 1.0
Δ_min = 0.0
width = T_total / 3
s = 20 / width
n_points = 10000

times, energies, couplings, U_kato = evolve_projected_system(
    T_total=T_total,
    Δ_max=Δ_max,
    Δ_min=Δ_min,
    s=s,
    width=width,
    T_A=T_A,
    T_AB=T_AB,
    T_AC=T_AC,
    n_points=n_points,
)

plot_results(times, energies, couplings)

gamma_list = [γ0, γ1, γ2, γ3]
parity_projected = build_total_parity_projected(builder, V_ref)
ground_data = get_ground_manifold_data(T_total, Δ_max, Δ_min, s, width, T_A, T_AB, T_AC)

check_majorana_algebra(gamma_list)
check_path_properties(times, energies, parity_projected, T_total, Δ_max, Δ_min, s, width, T_A, T_AB, T_AC)
check_kato_transport(U_kato, ground_data["P0"], ground_data["PT"])
check_single_exchange(U_kato, gamma_list)
check_double_exchange(U_kato, gamma_list)
check_four_exchanges(U_kato, gamma_list)
check_parity_resolved_gate(U_kato, ground_data["V0"], parity_projected, γ2, γ3)
