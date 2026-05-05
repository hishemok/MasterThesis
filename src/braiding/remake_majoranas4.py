from __future__ import annotations

import numpy as np

from remake_majoranas3 import (
    get_operators,
    get_projection_basis,
    get_stuff,
    jw_tensor,
    project_matrix,
)


def projected_inner_product(left, right):
    return float(np.real(np.trace(left.conj().T @ right)))


def fit_real_linear_combination(components, target):
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
    fit = np.zeros_like(target, dtype=complex)
    for coeff, component in zip(coeffs, components):
        fit = fit + coeff * component
    return coeffs, fit


def select_site_operators(cre, ann, system_name):
    if system_name == "A":
        site = 2
    elif system_name == "B":
        site = 3
    elif system_name == "C":
        site = 6
    else:
        raise ValueError("system_name must be 'A', 'B', or 'C'.")
    return cre[site], ann[site]


def target_operator(cdag, c, projection_basis, tocheck):
    plus = projection_basis.conj().T @ (cdag + c) @ projection_basis
    minus = projection_basis.conj().T @ (1j * (cdag - c)) @ projection_basis
    if tocheck == "Plus":
        return plus
    if tocheck == "Minus":
        return minus
    if tocheck == "Both":
        return plus + minus
    raise ValueError("tocheck must be 'Plus', 'Minus', or 'Both'.")


def build_transition_components(
    even_vecs,
    odd_vecs,
    projection_basis,
    system_name,
    JW_B,
    JW_C,
    component_levels=None,
    transition_mode="diagonal",
):
    n_even = even_vecs.shape[1]
    n_odd = odd_vecs.shape[1]
    if component_levels is not None:
        n_even = min(n_even, component_levels)
        n_odd = min(n_odd, component_levels)

    components = []
    labels = []
    for even_index in range(n_even):
        if transition_mode == "diagonal":
            odd_indices = [even_index] if even_index < n_odd else []
        elif transition_mode == "all":
            odd_indices = range(n_odd)
        else:
            raise ValueError("transition_mode must be 'all' or 'diagonal'.")

        even_vec = even_vecs[:, even_index]
        for odd_index in odd_indices:
            odd_vec = odd_vecs[:, odd_index]
            gamma_plus = (
                np.outer(even_vec, odd_vec.conj())
                + np.outer(odd_vec, even_vec.conj())
            )
            gamma_minus = 1j * (
                np.outer(even_vec, odd_vec.conj())
                - np.outer(odd_vec, even_vec.conj())
            )
            projected_plus = project_matrix(
                jw_tensor(gamma_plus, system_name, JW_B, JW_C),
                projection_basis,
            )
            projected_minus = project_matrix(
                jw_tensor(gamma_minus, system_name, JW_B, JW_C),
                projection_basis,
            )
            components.extend([projected_plus, projected_minus])
            labels.extend(
                [
                    (even_index, odd_index, "plus"),
                    (even_index, odd_index, "minus"),
                ]
            )
    return components, labels


def construct_majorana_fit(
    even_vecs,
    odd_vecs,
    projection_basis,
    cre,
    ann,
    system_name="A",
    JW_B=None,
    JW_C=None,
    component_levels=None,
    tocheck="Minus",
    transition_mode="diagonal",
):
    cdag, c = select_site_operators(cre, ann, system_name)
    check = target_operator(cdag, c, projection_basis, tocheck)
    components, labels = build_transition_components(
        even_vecs,
        odd_vecs,
        projection_basis,
        system_name,
        JW_B,
        JW_C,
        component_levels=component_levels,
        transition_mode=transition_mode,
    )
    coeffs, gamma = fit_real_linear_combination(components, check)
    error = float(np.linalg.norm(gamma - check))
    check_norm = float(np.linalg.norm(check))
    return {
        "subsystem": system_name,
        "gamma_projected": gamma,
        "check_projected": check,
        "matrix_error": error,
        "matrix_error_normalized": float(error / np.sqrt(check.shape[0])),
        "relative_error": float(error / check_norm) if check_norm else 0.0,
        "coefficients": coeffs,
        "component_labels": labels,
        "transition_mode": transition_mode,
        "tocheck": tocheck,
    }


def make_majoranas_for_B_and_C_with_projection_dim(
    projection_dim=8,
    specified_vals=None,
    projection_basis=None,
    component_levels=None,
    verbose=True,
    tocheck="Minus",
    transition_mode="diagonal",
):
    (
        JW_A,
        JW_B,
        JW_C,
        h_a,
        h_b,
        h_c,
        h_full,
        h_full_eigvals,
        h_full_eigvecs,
        even_vec_A,
        odd_vec_A,
        even_vec_B,
        odd_vec_B,
        even_vec_C,
        odd_vec_C,
    ) = get_stuff(
        specified_vals=specified_vals,
        include_full_system=projection_basis is None,
    )
    cre, ann = get_operators()
    if projection_basis is None:
        projection_basis = get_projection_basis(h_full_eigvecs, projection_dim)

    b_result = construct_majorana_fit(
        even_vec_B,
        odd_vec_B,
        projection_basis,
        cre,
        ann,
        system_name="B",
        JW_B=JW_B,
        JW_C=JW_C,
        component_levels=component_levels,
        tocheck=tocheck,
        transition_mode=transition_mode,
    )
    c_result = construct_majorana_fit(
        even_vec_C,
        odd_vec_C,
        projection_basis,
        cre,
        ann,
        system_name="C",
        JW_B=JW_B,
        JW_C=JW_C,
        component_levels=component_levels,
        tocheck=tocheck,
        transition_mode=transition_mode,
    )

    if verbose:
        for result in (b_result, c_result):
            print(
                f"{result['subsystem']} {tocheck} {transition_mode}: "
                f"error={result['matrix_error']:.4e}, "
                f"error/sqrt(dim)={result['matrix_error_normalized']:.4e}, "
                f"relative={result['relative_error']:.4e}"
            )
    return b_result, c_result


def make_majoranas_for_B_and_C(levels_to_include=8, specified_vals=None, verbose=True):
    return make_majoranas_for_B_and_C_with_projection_dim(
        projection_dim=levels_to_include,
        specified_vals=specified_vals,
        verbose=verbose,
    )


if __name__ == "__main__":
    for u_value in (0.0, 0.1, 2.0):
        print(f"\nU={u_value}, dim P=80")
        make_majoranas_for_B_and_C_with_projection_dim(
            projection_dim=80,
            specified_vals={"U": [u_value]},
            component_levels=4,
            verbose=True,
            tocheck="Minus",
            transition_mode="diagonal",
        )
