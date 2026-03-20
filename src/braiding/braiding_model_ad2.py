from get_mzm_JW import get_full_gammas
import numpy as np
from scipy.linalg import expm
from tqdm import tqdm
from hamiltonian_builder import BraidingHamiltonianBuilder, default_config_path
from braiding_model import delta_pulse, plot_results, build_total_parity_projected
from full_system_hamiltonian import precompute_ops


def project_operator(operator_full, basis):
    return basis.conj().T @ operator_full @ basis


def normalize_trace(operator_sub):
    dim = operator_sub.shape[0]
    norm_factor = np.sqrt(np.trace(operator_sub @ operator_sub).real / dim)
    if np.isclose(norm_factor, 0.0):
        raise ValueError("Cannot normalize an operator with vanishing projected norm.")
    return operator_sub / norm_factor


def project_and_normalize(operator_full, basis):
    return normalize_trace(project_operator(operator_full, basis))


def hermitian_part(operator):
    return 0.5 * (operator + operator.conj().T)


def get_ground_manifold_data(T_total, Δ_max, Δ_min, s, width, T_A, T_B, T_C):
    H0, _ = build_hamiltonian(0, T_total, Δ_max, Δ_min, s, width, T_A, T_B, T_C)
    HT, _ = build_hamiltonian(T_total, T_total, Δ_max, Δ_min, s, width, T_A, T_B, T_C)

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



def build_hamiltonian(t, T_total, Δ_max, Δ_min, s, width, TA, TB, TC):
    """
    Constructs the time-dependent Hamiltonian H(t) = Σ Δ_j(t) iγ₀γ_j
    """
    
    # Time-dependent couplings
    Δ1 = delta_pulse(t, 0, width, s, Δ_max, Δ_min) + delta_pulse(t, T_total, width, s, Δ_max, Δ_min) - Δ_min
    Δ2 = delta_pulse(t, T_total/3, width, s, Δ_max, Δ_min)
    Δ3 =  delta_pulse(t, 2*T_total/3, width, s, Δ_max, Δ_min)


    # TA, TB, TC are already Hermitian projected operators.
    H = Δ1 * TA + Δ2 * TB + Δ3 * TC

    
    return H, (Δ1, Δ2, Δ3)




def evolve_system(T_total, Δ_max, Δ_min, s, width, TA, TB, TC, n_points=1000):
    times = np.linspace(0, T_total, n_points)
    dt = T_total/n_points

    energies = np.zeros((n_points, 8))  # Store all 8 eigenvalues
    couplings = np.zeros((n_points, 3))  # Store Δ1, Δ2, Δ3

    U_kato = np.eye(8, dtype=complex)


    H0, coupling0 = build_hamiltonian(0,T_total, Δ_max, Δ_min, s, width, TA, TB, TC)
    evals, evecs = np.linalg.eigh(H0)
    energies[0] = evals
    couplings[0] = coupling0

    V = evecs[:, :4]  # Initial degenerate subspace basis
    
    print("Analyzing spectrum over time...")
    for i in tqdm(range(1,len(times)), total=n_points):
        t = times[i]
        H, couplings[i] = build_hamiltonian(t, T_total, Δ_max, Δ_min, s, width, TA, TB, TC)
        
        evals, evecs = np.linalg.eigh(H)
        energies[i] = evals

        W = evecs[:, :4]  # Next degenerate subspace basis
        P = V @ V.conj().T
        P_next = W @ W.conj().T
        K = P @ ((P_next - P) / dt) - ((P_next - P) / dt) @ P
        U_kato = expm(-dt * K) @ U_kato
        V = W  # Move to next basis


    return times, energies, couplings, U_kato




def phase_aligned_error(U, target):
    overlap = np.trace(target.conj().T @ U)
    phase = 0.0 if np.isclose(np.abs(overlap), 0.0) else np.angle(overlap)
    return np.linalg.norm(U - np.exp(1j * phase) * target)


def get_single_majorana_components(term, gamma_labels, gamma_ops):
    dim = term.shape[0]
    rows = []

    for label, gamma in zip(gamma_labels, gamma_ops):
        coeff = np.trace(gamma.conj().T @ term).real / dim
        rows.append((abs(coeff), coeff, label))

    rows.sort(reverse=True, key=lambda item: item[0])
    return rows


def print_single_majorana_decomposition(rows, title, max_terms=None):
    print(f"\n{title}")
    if max_terms is None:
        max_terms = len(rows)

    for _, coeff, label in rows[:max_terms]:
        print(f"{label}: {coeff:.3e}")


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


def print_pairwise_operator_relation(label_a, op_a, label_b, op_b):
    same_error = np.linalg.norm(op_a - op_b)
    minus_error = np.linalg.norm(op_a + op_b)

    if same_error <= minus_error:
        relation = "+"
        error = same_error
    else:
        relation = "-"
        error = minus_error

    print(f"{label_a} ≈ {relation}{label_b}: {error:.2e}")


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


def check_path_properties(times, energies, parity_op, T_total, Δ_max, Δ_min, s, width, T_A, T_B, T_C):
    max_hermiticity_error = 0.0
    max_parity_commutator = 0.0

    for t in times:
        H, _ = build_hamiltonian(t, T_total, Δ_max, Δ_min, s, width, T_A, T_B, T_C)
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


if __name__ == "__main__":

    cre_ops, ann_ops, _ = precompute_ops(9)
    # Site indices A: 0,1,2; B: 3,4,5; C: 6,7,8
    # The microscopic junctions connect A2-B1 and A2-C1.
    c_dagB1 = cre_ops[3]
    cB1 = ann_ops[3]
    c_dagB2 = cre_ops[5]
    cB2 = ann_ops[5]
    c_dagC1 = cre_ops[6]
    cC1 = ann_ops[6]
    c_dagC2 = cre_ops[8]
    cC2 = ann_ops[8]



    (gamma_A1_full, gamma_A2_full), (gamma_B1_full, gamma_B2_full), (gamma_C1_full, gamma_C2_full) = get_full_gammas(levels_to_include=4, verbose=False)


    builder = BraidingHamiltonianBuilder(
        n_sites=3,
        dupes=3,
        specified_vals={"U": [0.1]},
        config_path=default_config_path(),
    )


    H_full = builder.full_system_hamiltonian()


    _, eigvecs = np.linalg.eigh(H_full)

    V_ref = eigvecs[:, :8]  # Reference basis (full low-energy basis at t=0)
    gamma_A1_sub = project_and_normalize(gamma_A1_full, V_ref)
    gamma_A2_sub = project_and_normalize(gamma_A2_full, V_ref)
    gamma_B1_sub = project_and_normalize(gamma_B1_full, V_ref)
    gamma_B2_sub = project_and_normalize(gamma_B2_full, V_ref)
    gamma_C1_sub = project_and_normalize(gamma_C1_full, V_ref)
    gamma_C2_sub = project_and_normalize(gamma_C2_full, V_ref)

    gamma_labels = ["γA1", "γA2", "γB1", "γB2", "γC1", "γC2"]
    gamma_ops = [
        gamma_A1_sub,
        gamma_A2_sub,
        gamma_B1_sub,
        gamma_B2_sub,
        gamma_C1_sub,
        gamma_C2_sub,
    ]

    local_majoranas = {
        "χ_B1x": c_dagB1 + cB1,
        "χ_B1y": 1j * (c_dagB1 - cB1),
        "χ_B2x": c_dagB2 + cB2,
        "χ_B2y": 1j * (c_dagB2 - cB2),
        "χ_C1x": c_dagC1 + cC1,
        "χ_C1y": 1j * (c_dagC1 - cC1),
        "χ_C2x": c_dagC2 + cC2,
        "χ_C2y": 1j * (c_dagC2 - cC2),
    }
    projected_local_majoranas = {
        label: project_and_normalize(operator, V_ref)
        for label, operator in local_majoranas.items()
    }

    for label, operator in projected_local_majoranas.items():
        rows = get_single_majorana_components(operator, gamma_labels, gamma_ops)
        print_single_majorana_decomposition(rows, f"{label} projected into the low-energy basis", max_terms=3)

    print("\nProjected local-operator relations")
    print_pairwise_operator_relation("χ_B1x", projected_local_majoranas["χ_B1x"], "χ_B2x", projected_local_majoranas["χ_B2x"])
    print_pairwise_operator_relation("χ_B1y", projected_local_majoranas["χ_B1y"], "χ_B2y", projected_local_majoranas["χ_B2y"])
    print_pairwise_operator_relation("χ_C1x", projected_local_majoranas["χ_C1x"], "χ_C2x", projected_local_majoranas["χ_C2x"])
    print_pairwise_operator_relation("χ_C1y", projected_local_majoranas["χ_C1y"], "χ_C2y", projected_local_majoranas["χ_C2y"])

    # The explicit junction analysis in braiding_model_ad.py shows that the
    # active braid pair is γB2/γC2. On the junction sites this channel is
    # selected by the local "y" Majorana, i(c† - c).
    chi_B_full = 1j * (c_dagB1 - cB1)
    chi_C_full = 1j * (c_dagC1 - cC1)

    γ0, γ1, γ2, γ3 = gamma_A1_sub, gamma_A2_sub, gamma_B2_sub, gamma_C2_sub
    T_A = 1j * γ0 @ γ1
    T_B = hermitian_part(project_operator(1j * gamma_A1_full @ chi_B_full, V_ref))
    T_C = hermitian_part(project_operator(1j * gamma_A1_full @ chi_C_full, V_ref))

    rows_T_B = get_bilinear_components(T_B, gamma_labels, gamma_ops)
    rows_T_C = get_bilinear_components(T_C, gamma_labels, gamma_ops)
    print_bilinear_decomposition(rows_T_B, "Projected decomposition of T_B = P iγA1 χ_B P", max_terms=6)
    print_bilinear_decomposition(rows_T_C, "Projected decomposition of T_C = P iγA1 χ_C P", max_terms=6)

    T_total = 1000.0
    Δ_max = 1.0
    Δ_min = 0
    width = T_total/3
    s = 20/width
    # Parameters
    params = {
        'T_total': T_total,
        'Δ_max': Δ_max,
        'Δ_min': Δ_min,
        's': s,
        'width': width,
        'TA': T_A,
        'TB': T_B,
        'TC': T_C,
        'n_points': 10000
    }


    times, energies, couplings, U_kato = evolve_system(**params)
    plot_results(times, energies, couplings)




    gamma_list = [γ0, γ1, γ2, γ3]
    parity_projected = build_total_parity_projected(builder, V_ref)
    ground_data = get_ground_manifold_data(T_total, Δ_max, Δ_min, s, width, T_A, T_B, T_C)

    check_majorana_algebra(gamma_list)
    check_path_properties(times, energies, parity_projected, T_total, Δ_max, Δ_min, s, width, T_A, T_B, T_C)
    check_kato_transport(U_kato, ground_data["P0"], ground_data["PT"])
    check_single_exchange(U_kato, gamma_list)
    check_double_exchange(U_kato, gamma_list)
    check_four_exchanges(U_kato, gamma_list)
    check_parity_resolved_gate(U_kato, ground_data["V0"], parity_projected, γ2, γ3)

"""

If i have 3 QD-SC-QD-SC-QD chains, I get 6 Majoranas: γ_A1, γ_A2, γ_B1, γ_B2, γ_C1, γ_C2, 2 from each chain. In the code i have called  them γ_A1-> γ0, γ_A2-> γ1, γ_B1-> γ2, γ_C1-> γ3.
Since my current function build_hamiltonian is actually a sweet spot hamiltonian that works as long as i have clean majoranas (as far as i understood my superisor), I need some way of checking wether any "unwanted" majoranas are present in the braiding protocol, or if they are left out as intended and we really do have a clean braid.
You have suggested a explicit Junction Hamiltonian that found out which majoranas are working and wether or not the junctions are connecting what we want, and in braiding_model_ad.py I think we found out that things are all right. The code suggested that they use γ_B2 and γ_C2 as the "braiding" Majoranas.
What I wanted to do in braiding_model_ad2.py is combine what we found in the braiding_model_ad.py file with a more realistic version my supervisor asked me to do. He told me that the Hamiltonian could be changed from:
H(t) = Δ1(t) iγ0γ1 + Δ2(t) iγ0γ2 + Δ3(t) iγ0γ3
to something like:
H(t) = Δ1(t) iγ0γ1 + Δ2(t) P γ0(c†_B + c_B)P + Δ3(t) P γ0 i(c†_C + c_C)P
Where P is my current V_ref (GS projector)
But it seems to me that when i do 
H(t) = Δ1(t) iγ0γ1 + Δ2(t) P γ0(c†_B + c_B)P + Δ3(t) P γ0 i(c†_C + c_C)P
It doesnt matter which c_B operator i use, if its the one for the first or the last dot. 
If i use:
c†_B1 + c_B1 with c†_C1 + c_C1 I get the same results as when i use c†_B2 + c_B2 with c†_C2 + c_C2.
Or even if i mix and match and use c†_B1 + c_B1 with c†_C2 + c_C2 or the other way around, i get the same results.
This is a bit surprising to me, because i thought that the specific choice of c_B and c_C would matter, since they correspond to different Majoranas. Like we found in braiding_model_ad.py. Is there something wrong with the way I am implementing the Hamiltonian or is there something else I am missing that explains this apparent insensitivity to the choice of c_B and c_C? 

"""
