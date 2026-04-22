"""
##The point of this remake:

Up until now i ahve constructed majoranas as follows:
1. Diagonalize a subsystem Hamiltonian (e.g. just the A PMM)
2. Identify the even and odd parity eigenstates
3. Construct the Majorana operators as sums of outer products of these eigenstates:
- γ1 =∑  |o><e| + |e><o|
- γ2 =∑  i(|o><e| - |e><o|)

What I found out is:
It is not a given, that the majoranas must be constructed like this. If we take the equations above and specify energy levels:
- γ1 =∑_j  |o_j><e_j| + |e_j><o_j|
- γ2 =∑_j  i(|o_j><e_j| - |e_j><o_j|
this musnt be the only way to construct Majoranas. We can also do:
- γ = |o_0><e_0| + |e_0><o_0| +  i(|o_1><e_1| - |e_1><o_1| + ...
where it is alternating or changing which energy levels we are taking for the two Majoranas.
This could have effected the braiding results and couldve been the reason why the ideal and physical results from 'retry.py' didnt match.
In order to check which sequence of energy levels gives the correct Majoranas, we have this check:
for gamma 2
Tr[{γ_2, P_d (c†_{B_inner} + c_{B_inner})P_d}] = 2d
and for gamma 3
Tr[{γ_3, P_d (c†_{C_inner} + c_{C_inner})P_d}] = 0
where d is the dimensional choice in retry.py.

This code needs to replace "get_full_gammas" in 'retry.py', so here we fit
Majorana operators as linear combinations of the +/- building blocks from all
included levels, and use the strongest fitted operators for braiding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from explore_hamiltonian_values import calculate_parities_optimized
from full_system_hamiltonian import precompute_ops
from hamiltonian_builder import BraidingHamiltonianBuilder, default_config_path


n = 3
A_INNER = 0
B_INNER = 3
C_INNER = 6


@dataclass
class MajoranaFit:
    subsystem: str
    target_axis: str
    score: float
    overlap_x: float
    overlap_y: float
    coefficients_plus: np.ndarray
    coefficients_minus: np.ndarray
    gamma_local: np.ndarray = field(repr=False)


def tensorprod(mats=None):
    if mats is None:
        mats = []
    result = np.array([[1]], dtype=complex)
    for matrix in mats:
        result = np.kron(result, matrix)
    return result


def pauli_Z():
    return np.array([[1, 0], [0, -1]], dtype=complex)


def build_JW_string(subsys_before_sites):
    mat = np.array([[1]], dtype=complex)
    for _ in range(subsys_before_sites):
        mat = np.kron(mat, -pauli_Z())
    return mat


def pair_even_odd(even_energies, odd_energies):
    pairs = []
    for even_index, even_energy in enumerate(even_energies):
        odd_index = int(np.argmin(np.abs(odd_energies - even_energy)))
        pairs.append((even_index, odd_index))
    return pairs


def subsys_parity_oper(sites=3):
    number_ops = precompute_ops(sites)[2]
    dim = number_ops[0].shape[0]
    identity = np.eye(dim, dtype=complex)
    parity = identity.copy()
    for number_op in number_ops:
        parity = parity @ (identity - 2 * number_op)
    return parity


def get_projection_basis(eigenvectors, levels_to_include):
    basis = eigenvectors[:, :levels_to_include]
    overlap = basis.conj().T @ basis
    if not np.allclose(overlap, np.eye(levels_to_include, dtype=complex)):
        raise ValueError("Projection basis is not orthonormal: V†V != I")
    return basis


def project_operator(operator, basis):
    return basis.conj().T @ operator @ basis


def candidate_metadata(candidate):
    return {
        "subsystem": candidate.subsystem,
        "target_axis": candidate.target_axis,
        "score": float(candidate.score),
        "overlap_x": float(candidate.overlap_x),
        "overlap_y": float(candidate.overlap_y),
        "coefficients_plus": [float(value) for value in candidate.coefficients_plus],
        "coefficients_minus": [float(value) for value in candidate.coefficients_minus],
    }


def normalize_projected_operator(projected_operator, label):
    dim = projected_operator.shape[0]
    scale_squared = np.trace(projected_operator @ projected_operator).real / dim
    if scale_squared <= 0:
        raise ValueError(f"{label} has non-positive normalization scale {scale_squared:.3e}.")
    return projected_operator / np.sqrt(scale_squared)


def embedded_subsystem_operator(local_operator, subsystem):
    subsystem = subsystem.upper()
    if subsystem == "A":
        return tensorprod([local_operator, np.eye(2 ** (2 * n), dtype=complex)])
    if subsystem == "B":
        return tensorprod([build_JW_string(n), local_operator, np.eye(2**n, dtype=complex)])
    if subsystem == "C":
        return tensorprod([build_JW_string(2 * n), local_operator])
    raise ValueError(f"Unknown subsystem {subsystem!r}.")


def projected_inner_product(left, right):
    dim = left.shape[0]
    return float(np.real(np.trace(left @ right)) / dim)


def construct_majorana_components(even_vecs, odd_vecs, even_energies, odd_energies, levels_to_include):
    max_pairs = min(levels_to_include, even_vecs.shape[1], odd_vecs.shape[1])
    pairs = pair_even_odd(even_energies, odd_energies)
    components = []

    for level_index in range(max_pairs):
        even_index, odd_index = pairs[level_index]
        even_vec = even_vecs[:, even_index]
        odd_vec = odd_vecs[:, odd_index]

        plus_component = np.outer(odd_vec, even_vec.conj()) + np.outer(even_vec, odd_vec.conj())
        minus_component = 1j * (np.outer(odd_vec, even_vec.conj()) - np.outer(even_vec, odd_vec.conj()))
        components.append((plus_component, minus_component))

    return components


def construct_majorana_fit_components(even_vecs, odd_vecs, levels_to_include):
    """
    Build a full odd-even operator basis from the included levels.

    Restricting the fit to matched same-level pairs leaves out cross-level terms
    |o_j><e_k| +/- h.c. with j != k. Those missing components turn out to matter
    for the projected local operators at large projection dimension.
    """

    max_even = min(levels_to_include, even_vecs.shape[1])
    max_odd = min(levels_to_include, odd_vecs.shape[1])
    components = []

    for even_index in range(max_even):
        even_vec = even_vecs[:, even_index]
        for odd_index in range(max_odd):
            odd_vec = odd_vecs[:, odd_index]

            plus_component = np.outer(odd_vec, even_vec.conj()) + np.outer(even_vec, odd_vec.conj())
            minus_component = 1j * (np.outer(odd_vec, even_vec.conj()) - np.outer(even_vec, odd_vec.conj()))
            components.append((plus_component, minus_component))

    return components


def construct_majoranas_from_pattern(components, pattern):
    dim = components[0][0].shape[0]
    gamma_first = np.zeros((dim, dim), dtype=complex)
    gamma_second = np.zeros((dim, dim), dtype=complex)

    for use_swapped_component, (plus_component, minus_component) in zip(pattern, components):
        if use_swapped_component:
            gamma_first += minus_component
            gamma_second += plus_component
        else:
            gamma_first += plus_component
            gamma_second += minus_component

    return gamma_first, gamma_second


def construct_majoranas(even_vecs, odd_vecs, even_energies, odd_energies, n=1):
    components = construct_majorana_components(
        even_vecs,
        odd_vecs,
        even_energies,
        odd_energies,
        levels_to_include=n,
    )
    return construct_majoranas_from_pattern(components, (0,) * len(components))


def embed_subsystem_majoranas(gamma_first_local, gamma_second_local, subsystem):
    identity_sub = np.eye(2**n, dtype=complex)
    identity_two_subsystems = np.eye(2 ** (2 * n), dtype=complex)

    subsystem = subsystem.upper()
    if subsystem == "A":
        return (
            tensorprod([gamma_first_local, identity_two_subsystems]),
            tensorprod([gamma_second_local, identity_two_subsystems]),
        )
    if subsystem == "B":
        jw_b = build_JW_string(n)
        return (
            tensorprod([jw_b, gamma_first_local, identity_sub]),
            tensorprod([jw_b, gamma_second_local, identity_sub]),
        )
    if subsystem == "C":
        jw_c = build_JW_string(2 * n)
        return (
            tensorprod([jw_c, gamma_first_local]),
            tensorprod([jw_c, gamma_second_local]),
        )
    raise ValueError(f"Unknown subsystem {subsystem!r}.")


def fit_majorana_to_target(components, subsystem, basis, target_axis, target_x_full_operator, target_y_full_operator):
    local_basis = []
    full_basis = []
    for plus_component, minus_component in components:
        local_basis.append(plus_component)
        local_basis.append(minus_component)
        full_basis.append(embedded_subsystem_operator(plus_component, subsystem))
        full_basis.append(embedded_subsystem_operator(minus_component, subsystem))

    target_x_projected = normalize_projected_operator(
        project_operator(target_x_full_operator, basis),
        f"{subsystem}_{target_axis}_target_x",
    )
    target_y_projected = normalize_projected_operator(
        project_operator(target_y_full_operator, basis),
        f"{subsystem}_{target_axis}_target_y",
    )
    if target_axis == "x":
        target_projected = target_x_projected
    else:
        target_projected = target_y_projected

    projected_basis = [
        project_operator(operator_full, basis)
        for operator_full in full_basis
    ]
    basis_count = len(projected_basis)
    gram = np.zeros((basis_count, basis_count), dtype=float)
    rhs = np.zeros(basis_count, dtype=float)

    for row in range(basis_count):
        rhs[row] = projected_inner_product(projected_basis[row], target_projected)
        for col in range(basis_count):
            gram[row, col] = projected_inner_product(projected_basis[row], projected_basis[col])

    coefficients = np.linalg.pinv(gram, rcond=1e-10) @ rhs
    gamma_local = np.zeros_like(local_basis[0])
    for coefficient, local_operator in zip(coefficients, local_basis):
        gamma_local += coefficient * local_operator

    gamma_full = embedded_subsystem_operator(gamma_local, subsystem)
    gamma_projected = normalize_projected_operator(
        project_operator(gamma_full, basis),
        f"{subsystem}_{target_axis}_fit",
    )
    overlap_x = projected_inner_product(gamma_projected, target_x_projected)
    overlap_y = projected_inner_product(gamma_projected, target_y_projected)
    score = overlap_x if target_axis == "x" else overlap_y

    if score < 0:
        coefficients = -coefficients
        gamma_local = -gamma_local
        overlap_x = -overlap_x
        overlap_y = -overlap_y
        score = -score

    coefficients_plus = coefficients[0::2].copy()
    coefficients_minus = coefficients[1::2].copy()
    return MajoranaFit(
        subsystem=subsystem.upper(),
        target_axis=target_axis,
        score=float(score),
        overlap_x=float(overlap_x),
        overlap_y=float(overlap_y),
        coefficients_plus=coefficients_plus,
        coefficients_minus=coefficients_minus,
        gamma_local=gamma_local.copy(),
    )


def get_full_gammas_with_check(
    even_vecs,
    odd_vecs,
    even_energies,
    odd_energies,
    cdag,
    c,
    basis,
    dimension,
    *,
    levels_to_include=4,
    subsystem="B",
    verbose=True,
):
    """
    Fit a Majorana in the span of all plus/minus pieces from the included levels.

    We fit once to the projected x-like local operator and once to the projected y-like
    local operator, then keep whichever fit is stronger for braiding.
    """

    if basis.shape[1] != dimension:
        raise ValueError(
            f"dimension={dimension} does not match the basis width {basis.shape[1]}."
        )

    components = construct_majorana_fit_components(
        even_vecs,
        odd_vecs,
        levels_to_include=levels_to_include,
    )
    if not components:
        raise ValueError("No even/odd parity pairs were available for the Majorana fit.")

    target_x_full_operator = cdag + c
    target_y_full_operator = 1j * (cdag - c)
    fit_x = fit_majorana_to_target(
        components,
        subsystem,
        basis,
        "x",
        target_x_full_operator,
        target_y_full_operator,
    )
    fit_y = fit_majorana_to_target(
        components,
        subsystem,
        basis,
        "y",
        target_x_full_operator,
        target_y_full_operator,
    )
    selected_fit = fit_x if fit_x.score >= fit_y.score else fit_y
    companion_fit = fit_y if selected_fit is fit_x else fit_x

    if verbose:
        print(
            f"{subsystem} fit_x: score={fit_x.score:.6f}, "
            f"overlap_x={fit_x.overlap_x:.6f}, overlap_y={fit_x.overlap_y:.6f}"
        )
        print(
            f"{subsystem} fit_y: score={fit_y.score:.6f}, "
            f"overlap_x={fit_y.overlap_x:.6f}, overlap_y={fit_y.overlap_y:.6f}"
        )
        print(
            f"{subsystem} selected axis: {selected_fit.target_axis} "
            f"(score={selected_fit.score:.6f})"
        )

    return {
        "x": fit_x,
        "y": fit_y,
        "selected": selected_fit,
        "companion": companion_fit,
    }


def get_full_gammas(
    levels_to_include=4,
    verbose=False,
    specified_vals=None,
    projection_basis=None,
    projection_dim=None,
    return_candidates=False,
):
    if specified_vals is None:
        specified_vals = {"U": [0.1]}

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

    if projection_basis is None:
        if projection_dim is None:
            raise ValueError(
                "get_full_gammas needs either projection_basis or projection_dim "
                "to rank the candidate combinations."
            )
        h_full = builder.full_system_hamiltonian()
        _, full_eigvecs = np.linalg.eigh(h_full)
        projection_basis = get_projection_basis(full_eigvecs, projection_dim)
    else:
        projection_dim = projection_basis.shape[1]

    h_sub = builder_sub.full_system_hamiltonian()
    h_sub_eigvals, h_sub_eigvecs = np.linalg.eigh(h_sub)
    even_energies, odd_energies, even_vecs, odd_vecs = calculate_parities_optimized(
        h_sub_eigvecs,
        h_sub_eigvals,
        subsys_parity_oper(sites=n),
    )

    operators = builder.get_operators()
    a_fits = get_full_gammas_with_check(
        even_vecs,
        odd_vecs,
        even_energies,
        odd_energies,
        operators["cre"][A_INNER],
        operators["ann"][A_INNER],
        projection_basis,
        projection_dim,
        levels_to_include=levels_to_include,
        subsystem="A",
        verbose=verbose,
    )
    b_fits = get_full_gammas_with_check(
        even_vecs,
        odd_vecs,
        even_energies,
        odd_energies,
        operators["cre"][B_INNER],
        operators["ann"][B_INNER],
        projection_basis,
        projection_dim,
        levels_to_include=levels_to_include,
        subsystem="B",
        verbose=verbose,
    )
    c_fits = get_full_gammas_with_check(
        even_vecs,
        odd_vecs,
        even_energies,
        odd_energies,
        operators["cre"][C_INNER],
        operators["ann"][C_INNER],
        projection_basis,
        projection_dim,
        levels_to_include=levels_to_include,
        subsystem="C",
        verbose=verbose,
    )
    gamma_A1_full = embedded_subsystem_operator(a_fits["x"].gamma_local, "A")
    gamma_A2_full = embedded_subsystem_operator(a_fits["y"].gamma_local, "A")
    gamma_B1_full, gamma_B2_full = embed_subsystem_majoranas(
        b_fits["x"].gamma_local,
        b_fits["y"].gamma_local,
        "B",
    )
    gamma_C1_full, gamma_C2_full = embed_subsystem_majoranas(
        c_fits["x"].gamma_local,
        c_fits["y"].gamma_local,
        "C",
    )

    gamma_tuple = (
        (gamma_A1_full, gamma_A2_full),
        (gamma_B1_full, gamma_B2_full),
        (gamma_C1_full, gamma_C2_full),
    )
    candidate_data = {
        "A": a_fits,
        "B": b_fits,
        "C": c_fits,
        "metadata": {
            "projection_dim": int(projection_dim),
            "levels_to_include": int(levels_to_include),
        },
    }

    if return_candidates:
        return gamma_tuple, candidate_data
    return gamma_tuple


if __name__ == "__main__":
    get_full_gammas(levels_to_include=4, projection_dim=512, verbose=True)
