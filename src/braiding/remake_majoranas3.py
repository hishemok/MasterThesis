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

def fit_two_component_mixture(first_component, second_component, target):
    components = (first_component, second_component)
    gram = np.array(
        [
            [projected_inner_product(left, right) for right in components]
            for left in components
        ],
        dtype=float,
    )
    rhs = np.array(
        [projected_inner_product(component, target) for component in components],
        dtype=float,
    )
    coeffs = np.linalg.pinv(gram, rcond=1e-12) @ rhs
    mixture = coeffs[0] * first_component + coeffs[1] * second_component
    return coeffs, mixture


def construct_majoranas_w_check( even_vec, odd_vec, P, cre, ann, system_name="A", JW_B=None, JW_C=None, component_levels=None, verbose=True, tocheck="Minus"):
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

    if even_vec.shape[1] != odd_vec.shape[1]:
        raise ValueError("Problem detected mister:\nEven and odd vectors must have the same length.")
    

    if tocheck == "Plus":   
        check = check1
    elif tocheck == "Minus":
        check = check2
    elif tocheck == "Both":
        check = check1 + check2
    else:        
        raise ValueError("tocheck must be 'Plus', 'Minus', or 'Both'.")

    pairs = even_vec.shape[1]
    if component_levels is not None:
        pairs = min(pairs, component_levels)

    gamma_out = np.zeros_like(check, dtype=complex)
    gamma_plus_out = np.zeros_like(check, dtype=complex)
    gamma_minus_out = np.zeros_like(check, dtype=complex)
    for j in range(pairs):
        gp = np.outer(even_vec[:, j], odd_vec[:, j].conj()) + np.outer(odd_vec[:, j], even_vec[:, j].conj())
        gm = 1j * (np.outer(even_vec[:, j], odd_vec[:, j].conj()) - np.outer(odd_vec[:, j], even_vec[:, j].conj()))

        gp_full = jw_tensor(gp, subsystem=system_name, JW_B=JW_B, JW_C=JW_C)
        gm_full = jw_tensor(gm, subsystem=system_name, JW_B=JW_B, JW_C=JW_C)

        gp_projected = project_matrix(gp_full, P)
        gm_projected = project_matrix(gm_full, P)

        if tocheck == "Both":
            gamma_plus_out += gp_projected
            gamma_minus_out += gm_projected
            continue

        errormin = np.linalg.norm(gamma_out + gm_projected - check)
        errorplus = np.linalg.norm(gamma_out + gp_projected - check)

        if verbose:
            print(f"Pair {j}: error for γ = gp is {errorplus:.4e}, error for γ = gm is {errormin:.4e}")
            print(f"Adding pair {j} with {'+' if errorplus < errormin else '-'} sign to the Majorana operator.")
        if errorplus < errormin:
            gamma_out += gp_projected
        else:
            gamma_out += gm_projected

    if tocheck == "Both":
        coeffs, gamma_out = fit_two_component_mixture(
            gamma_plus_out,
            gamma_minus_out,
            check,
        )
        if verbose:
            print(
                "Best global mixed γ = "
                f"{coeffs[0]:.4e} γ_plus + {coeffs[1]:.4e} γ_minus, "
                f"error is {np.linalg.norm(gamma_out - check):.4e}"
            )
    
    return gamma_out, check


def _build_fit_result(even_vec, odd_vec, P, cre, ann, system_name, JW_B, JW_C, component_levels, verbose, tocheck):

    gamma, check = construct_majoranas_w_check(
        even_vec,
        odd_vec,
        P,
        cre,
        ann,
        system_name=system_name,
        JW_B=JW_B,
        JW_C=JW_C,
        component_levels=component_levels,
        verbose=verbose,
        tocheck=tocheck,
    )

    return {
        "subsystem": system_name,
        "gamma_projected": gamma,
        "check_projected": check,
        "matrix_error": float(np.linalg.norm(gamma - check)),
    }


def make_majoranas_for_B_and_C(levels_to_include=8, specified_vals=None, verbose=True):
    return make_majoranas_for_B_and_C_with_projection_dim(
        projection_dim=levels_to_include,
        specified_vals=specified_vals,
        verbose=verbose,
    )


def make_majoranas_for_B_and_C_with_projection_dim(projection_dim=8, specified_vals=None, projection_basis=None, component_levels=None, verbose=True, tocheck="Both"):

    (JW_A, JW_B, JW_C, h_a, h_b, h_c, h_full, h_full_eigvals, h_full_eigvecs, even_vec_A, odd_vec_A, even_vec_B, odd_vec_B, even_vec_C, odd_vec_C,) = get_stuff(
        specified_vals=specified_vals,
        include_full_system=projection_basis is None,
    )
    cre, ann = get_operators()
    if projection_basis is None:
        P = get_projection_basis(h_full_eigvecs, projection_dim)
    else:
        P = projection_basis

    b_result = _build_fit_result( even_vec_B, odd_vec_B, P, cre, ann, "B", JW_B, JW_C, component_levels, verbose, tocheck
    )
    c_result = _build_fit_result( even_vec_C, odd_vec_C, P, cre, ann, "C", JW_B, JW_C, component_levels, verbose, tocheck
    )
    return b_result, c_result

if __name__ == "__main__":
    JW_A, JW_B, JW_C, h_a, h_b, h_c, h_full, h_full_eigvals, h_full_eigvecs, even_vec_A, odd_vec_A, even_vec_B, odd_vec_B, even_vec_C, odd_vec_C = get_stuff()
    cre, ann = get_operators()
    P = get_projection_basis(h_full_eigvecs, 80)
   
    print("Constructing Majoranas for system B + check:")
    gamma_B_plus, check_B_plus = construct_majoranas_w_check(even_vec_B, odd_vec_B, P, cre, ann, system_name="B", JW_B=JW_B, JW_C=JW_C, verbose=True, tocheck="Plus")
 
    print("\nConstructing Majoranas for system B - check:")
    gamma_B_minus, check_B_minus = construct_majoranas_w_check(even_vec_B, odd_vec_B, P, cre, ann, system_name="B", JW_B=JW_B, JW_C=JW_C, verbose=True, tocheck="Minus")
    

    print("BOTH")
    gamma_B_both, check_B_both = construct_majoranas_w_check(even_vec_B, odd_vec_B, P, cre, ann, system_name="B", JW_B=JW_B, JW_C=JW_C, verbose=True, tocheck="Both")



    print("\nConstructing Majoranas for system B both check:")
    gamma_B_both, check_B_both = construct_majoranas_w_check(even_vec_B, odd_vec_B, P, cre, ann, system_name="B", JW_B=JW_B, JW_C=JW_C, verbose=True, tocheck="Both")

    print("\nConstructing Majoranas for system C + check:")
    gamma_C_plus, check_C_plus = construct_majoranas_w_check(even_vec_C, odd_vec_C, P, cre, ann, system_name="C", JW_B=JW_B, JW_C=JW_C, verbose=True, tocheck="Plus")
    print("\nConstructing Majoranas for system C - check:")
    gamma_C_minus, check_C_minus = construct_majoranas_w_check(even_vec_C, odd_vec_C, P, cre, ann, system_name="C", JW_B=JW_B, JW_C=JW_C, verbose=True, tocheck="Minus")
    print("\nConstructing Majoranas for system C both check:")
    gamma_C_both, check_C_both = construct_majoranas_w_check(even_vec_C, odd_vec_C, P, cre, ann, system_name="C", JW_B=JW_B, JW_C=JW_C, verbose=True, tocheck="Both")
    
