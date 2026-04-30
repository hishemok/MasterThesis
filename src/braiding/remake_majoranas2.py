"""

Check: 
Tr[{γ, P(c^† + c)P}] = 2D
Tr[{γ, P i(c^† - c)P}] = 2D
*a little unsure of the last one*

by default:
γ = ∑_j |e_j⟩⟨o_j| + h.c.
\tilde{γ} = i∑_j |e_j⟩⟨o_j| - h.c.

Explore combinations of the two to get the desired result.

We need
HA, HB and HC. 
H_full to get P 
creation and annihilation operators

"""
from explore_hamiltonian_values import  get_parity_operator
from get_mzm_JW import build_JW_string, subsys_parity_oper, calculate_parities_optimized, subsystem_operators
from hamiltonian_builder import BraidingHamiltonianBuilder, default_config_path
import numpy as np

def get_stuff(specified_vals=None, include_full_system=True):
    
    if specified_vals is None:
        specified_vals = {"U": [0.1]}

    JW_A = np.eye(2**3)  # nothing before A
    JW_B = build_JW_string(3)            # sees all A sites
    JW_C = build_JW_string(6)            # sees all A+B sites

    builder = BraidingHamiltonianBuilder(
        n_sites=3,
        dupes=3,
        specified_vals=specified_vals,
        config_path=default_config_path(),
    )

    builder_sub = BraidingHamiltonianBuilder(
        n_sites=3,
        dupes=1,
        specified_vals=specified_vals,
        config_path=default_config_path(),
    )

    h_a = builder_sub.full_system_hamiltonian()
    h_a_eigvals, h_a_eigvecs = np.linalg.eigh(h_a)
    h_b = builder_sub.full_system_hamiltonian()
    h_b_eigvals, h_b_eigvecs = np.linalg.eigh(h_b)
    h_c = builder_sub.full_system_hamiltonian()
    h_c_eigvals, h_c_eigvecs = np.linalg.eigh(h_c)
    h_full = None
    h_full_eigvals = None
    h_full_eigvecs = None
    if include_full_system:
        h_full = builder.full_system_hamiltonian()
        h_full_eigvals, h_full_eigvecs = np.linalg.eigh(h_full)


    subsys_parity = subsys_parity_oper(sites=3)
    # print("Subsystem parity", subsys_parity.shape)
    even_energies_A, odd_energies_A, even_vec_A, odd_vec_A = calculate_parities_optimized(h_a_eigvecs, h_a_eigvals, subsys_parity)
    even_energies_B, odd_energies_B, even_vec_B, odd_vec_B = calculate_parities_optimized(h_b_eigvecs, h_b_eigvals, subsys_parity)
    even_energies_C, odd_energies_C, even_vec_C, odd_vec_C = calculate_parities_optimized(h_c_eigvecs, h_c_eigvals, subsys_parity)
    return JW_A, JW_B, JW_C, h_a, h_b, h_c, h_full, h_full_eigvals, h_full_eigvecs, even_vec_A, odd_vec_A, even_vec_B, odd_vec_B, even_vec_C, odd_vec_C

def get_operators():
    ops = subsystem_operators(3,3)
    cre = ops['cre']
    ann = ops['ann']
    
    return cre, ann

def get_projection_basis(eigenvectors, levels_to_include):
    basis = eigenvectors[:, :levels_to_include]
    overlap = basis.conj().T @ basis

    # The sliced eigenvectors form an orthonormal basis for the projected space.
    if not np.allclose(overlap, np.eye(levels_to_include, dtype=complex)):
        raise ValueError("Projection basis is not orthonormal: V†V != I")
    return basis


def calculate_trace_err(gamma, check, verbose=True):
    target = 2 * check.shape[0]
    anticomm = gamma @ check + check @ gamma
    if verbose:
        print("anticomm-", np.linalg.norm(anticomm))
    trace = np.trace(anticomm)

    err = np.abs(trace - target)
    return err

def calculate_operator_err(gamma, check):
    return np.linalg.norm(gamma - check)

def calculate_identity_anticommutator_err(gamma, check):
    dim = check.shape[0]
    target = 2 * np.eye(dim, dtype=complex)
    anticomm = gamma @ check + check @ gamma
    return np.linalg.norm(anticomm - target)

def tensor_prod_string(matrices):
    result = np.array([[1]], dtype=complex)
    for mat in matrices:
        result = np.kron(result, mat)
    return result

def jw_tensor(gamma, subsystem="A", JW_B=None, JW_C=None):
    I = np.eye(2**3, dtype=complex)

    if subsystem == "A":
        gamma = tensor_prod_string([gamma, I, I])
    elif subsystem == "B":
        gamma = tensor_prod_string([JW_B, gamma, I])
    elif subsystem == "C":
        gamma = tensor_prod_string([JW_C, gamma])
    else:
        raise ValueError("Subsystem must be 'A', 'B', or 'C'.")
    return gamma

def project_matrix(gamma, P):
    return P.conj().T @ gamma @ P

def projected_inner_product(left, right):
    return float(np.real(np.trace(left.conj().T @ right)))

def fit_projected_operator(projected_basis, target_projected):
    basis_count = len(projected_basis)
    gram = np.zeros((basis_count, basis_count), dtype=float)
    rhs = np.zeros(basis_count, dtype=float)

    for row in range(basis_count):
        rhs[row] = projected_inner_product(projected_basis[row], target_projected)
        for col in range(basis_count):
            gram[row, col] = projected_inner_product(projected_basis[row], projected_basis[col])

    coefficients = np.linalg.pinv(gram, rcond=1e-10) @ rhs
    fitted_projected = np.zeros_like(target_projected, dtype=complex)
    for coefficient, projected_operator in zip(coefficients, projected_basis):
        fitted_projected += coefficient * projected_operator

    return coefficients, fitted_projected

def construct_majoranas_w_check( even_vec, odd_vec, P, cre, ann, system_name="A", JW_B=None, JW_C=None, component_levels=None, verbose=True):
    if system_name == "A":
        cre = cre[2]
        ann = ann[2]
    elif system_name == "B":
        cre = cre[3]
        ann = ann[3]
    elif system_name == "C":
        cre = cre[6]
        ann = ann[6]
    
    check1 = P.conj().T @ (cre + ann) @ P
    check2 = P.conj().T @ (1j * (cre - ann)) @ P

    max_even = even_vec.shape[1]
    max_odd = odd_vec.shape[1]
    if component_levels is not None:
        max_even = min(max_even, component_levels)
        max_odd = min(max_odd, component_levels)

    local_basis = []
    full_basis = []
    projected_basis = []
    basis_labels = []

    max_pairs = min(max_even, max_odd)

    for level_index in range(max_pairs):
        even_state = even_vec[:, level_index]
        odd_state = odd_vec[:, level_index]

        plus_local = (
            np.outer(even_state, odd_state.conj())
            + np.outer(odd_state, even_state.conj())
        )
        minus_local = 1j * (
            np.outer(even_state, odd_state.conj())
            - np.outer(odd_state, even_state.conj())
        )
        for label, local_operator in (("plus", plus_local), ("minus", minus_local)):
            full_operator = jw_tensor(local_operator, subsystem=system_name, JW_B=JW_B, JW_C=JW_C)
            projected_operator = project_matrix(full_operator, P)

            local_basis.append(local_operator)
            full_basis.append(full_operator)
            projected_basis.append(projected_operator)
            basis_labels.append((level_index, level_index, label))

    coeffs_plus, gamma_plus_projected = fit_projected_operator(projected_basis, check1)
    coeffs_minus, gamma_minus_projected = fit_projected_operator(projected_basis, check2)

    gamma_plus_local = np.zeros_like(local_basis[0], dtype=complex)
    gamma_minus_local = np.zeros_like(local_basis[0], dtype=complex)
    gamma_plus_full = np.zeros_like(full_basis[0], dtype=complex)
    gamma_minus_full = np.zeros_like(full_basis[0], dtype=complex)

    for coefficient, local_operator, full_operator in zip(coeffs_plus, local_basis, full_basis):
        gamma_plus_local += coefficient * local_operator
        gamma_plus_full += coefficient * full_operator

    for coefficient, local_operator, full_operator in zip(coeffs_minus, local_basis, full_basis):
        gamma_minus_local += coefficient * local_operator
        gamma_minus_full += coefficient * full_operator

    plus_trace_err = calculate_trace_err(gamma_plus_projected, check1, verbose=verbose)
    minus_trace_err = calculate_trace_err(gamma_minus_projected, check2, verbose=verbose)
    plus_fit_err = calculate_operator_err(gamma_plus_projected, check1)
    minus_fit_err = calculate_operator_err(gamma_minus_projected, check2)
    plus_identity_err = calculate_identity_anticommutator_err(gamma_plus_projected, check1)
    minus_identity_err = calculate_identity_anticommutator_err(gamma_minus_projected, check2)

    if verbose:
        print(
            f"{system_name} plus fit: trace_err={plus_trace_err:.6e}, "
            f"matrix_err={plus_fit_err:.6e}, anticommutator_to_2I={plus_identity_err:.6e}"
        )
        print(
            f"{system_name} minus fit: trace_err={minus_trace_err:.6e}, "
            f"matrix_err={minus_fit_err:.6e}, anticommutator_to_2I={minus_identity_err:.6e}"
        )

    return {
        "subsystem": system_name,
        "basis_labels": basis_labels,
        "coefficients_plus": coeffs_plus,
        "coefficients_minus": coeffs_minus,
        "gamma_plus_local": gamma_plus_local,
        "gamma_minus_local": gamma_minus_local,
        "gamma_plus_full": gamma_plus_full,
        "gamma_minus_full": gamma_minus_full,
        "gamma_plus_projected": gamma_plus_projected,
        "gamma_minus_projected": gamma_minus_projected,
        "check_plus_projected": check1,
        "check_minus_projected": check2,
        "trace_error_plus": float(plus_trace_err),
        "trace_error_minus": float(minus_trace_err),
        "matrix_error_plus": float(plus_fit_err),
        "matrix_error_minus": float(minus_fit_err),
        "anticommutator_error_plus": float(plus_identity_err),
        "anticommutator_error_minus": float(minus_identity_err),
    }



def make_majoranas_for_B_and_C(levels_to_include=8, specified_vals=None, verbose=True):
    return make_majoranas_for_B_and_C_with_projection_dim(
        projection_dim=levels_to_include,
        specified_vals=specified_vals,
        verbose=verbose,
    )

def make_majoranas_for_B_and_C_with_projection_dim(
    projection_dim=8,
    specified_vals=None,
    verbose=True,
    projection_basis=None,
    component_levels=None,
):
    JW_A, JW_B, JW_C, h_a, h_b, h_c, h_full, h_full_eigvals, h_full_eigvecs, even_vec_A, odd_vec_A, even_vec_B, odd_vec_B, even_vec_C, odd_vec_C = get_stuff(
        specified_vals=specified_vals,
        include_full_system=projection_basis is None,
    )
    cre, ann = get_operators()
    if projection_basis is None:
        P = get_projection_basis(h_full_eigvecs, levels_to_include=projection_dim)
    else:
        P = projection_basis
    gb = construct_majoranas_w_check(
        even_vec_B,
        odd_vec_B,
        P,
        cre,
        ann,
        system_name="B",
        JW_B=JW_B,
        JW_C=JW_C,
        component_levels=component_levels,
        verbose=verbose,
    )
    gc = construct_majoranas_w_check(
        even_vec_C,
        odd_vec_C,
        P,
        cre,
        ann,
        system_name="C",
        JW_B=JW_B,
        JW_C=JW_C,
        component_levels=component_levels,
        verbose=verbose,
    )
    return gb, gc


if __name__ == "__main__":
    make_majoranas_for_B_and_C(levels_to_include=512)
