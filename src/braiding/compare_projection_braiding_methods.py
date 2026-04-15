"""Compare four projected braiding models on cumulative low-energy spaces.

The four models are:
  - ideal:             hand-picked four-Majorana bilinears
  - bilinear_fit:      bilinear projection of the physical junction operators
  - local_projection:  projected local-operator construction from braiding_model_ad2.py
  - physical:          projected microscopic junction operators

The script sweeps cumulative projection spaces P_gs, P_gs+P_excited1, ...
and keeps only those with cumulative dimension <= ``max_cum_dim``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from extended_projection_braiding import (
    build_projection_stack,
    build_total_parity_full,
    normalize_projected_majorana,
)
from extract_extended_projection_results import (
    build_junction_operator,
    build_total_parity_projected,
    check_parity_resolved_gate,
    choose_active_majoranas,
    flatten_site,
    format_single_exchange_errors,
    get_bilinear_components,
    get_ground_manifold_data,
    project_operator,
    reconstruct_bilinear_term,
    run_model_comparison,
)
from full_system_hamiltonian import precompute_ops
from get_mzm_JW import build_JW_string, construct_majoranas, subsys_parity_oper, tensorprod
from hamiltonian_builder import BraidingHamiltonianBuilder, default_config_path
from explore_hamiltonian_values import calculate_parities_optimized


def hermitian_part(operator):
    return 0.5 * (operator + operator.conj().T)


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Compare ideal, bilinear-fit, local-projection, and physical "
        "junction braids on cumulative projection spaces."
    )
    parser.add_argument(
        "--u-values",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 2.0],
        help="Interaction strengths U to evaluate.",
    )
    parser.add_argument(
        "--max-cum-dim",
        type=int,
        default=80,
        help="Only include cumulative projection spaces up to this dimension.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=300,
        help="Number of time steps used in each braid evolution.",
    )
    parser.add_argument(
        "--energy-tol",
        type=float,
        default=1e-2,
        help="Energy tolerance used to group approximately degenerate manifolds.",
    )
    parser.add_argument(
        "--levels-to-include",
        type=int,
        default=4,
        help="How many levels to include when constructing the full Majorana operators.",
    )
    parser.add_argument(
        "--b-site",
        type=int,
        default=3,
        help="Site index used for the local-projection B operator. Default is B1.",
    )
    parser.add_argument(
        "--c-site",
        type=int,
        default=6,
        help="Site index used for the local-projection C operator. Default is C1.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("savebraidvals4.txt"),
        help="Where to store the text summary.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra diagnostics while running.",
    )
    return parser


def project_and_normalize(operator_full, basis, label):
    operator_sub = project_operator(operator_full, basis)
    return normalize_projected_majorana(operator_sub, label)


def get_full_gammas_fast(levels_to_include=4, *, specified_vals=None):
    if specified_vals is None:
        specified_vals = {"U": [0.1]}

    builder_sub = BraidingHamiltonianBuilder(
        n_sites=3,
        dupes=1,
        specified_vals=specified_vals,
        config_path=default_config_path(),
    )

    h_sub = builder_sub.full_system_hamiltonian()
    h_sub_eigvals, h_sub_eigvecs = np.linalg.eigh(h_sub)
    subsys_parity = subsys_parity_oper(sites=3)
    even_energies, odd_energies, even_vecs, odd_vecs = calculate_parities_optimized(
        h_sub_eigvecs,
        h_sub_eigvals,
        subsys_parity,
    )
    gamma_1, gamma_2 = construct_majoranas(
        even_vecs,
        odd_vecs,
        even_energies,
        odd_energies,
        n=levels_to_include,
    )

    jw_b = build_JW_string(3)
    jw_c = build_JW_string(6)
    identity_sub = np.eye(2**3, dtype=complex)
    identity_two_subsystems = np.eye(2**6, dtype=complex)

    gamma_a1_full = tensorprod([gamma_1, identity_two_subsystems])
    gamma_a2_full = tensorprod([gamma_2, identity_two_subsystems])
    gamma_b1_full = tensorprod([jw_b, gamma_1, identity_sub])
    gamma_b2_full = tensorprod([jw_b, gamma_2, identity_sub])
    gamma_c1_full = tensorprod([jw_c, gamma_1])
    gamma_c2_full = tensorprod([jw_c, gamma_2])

    return (
        (gamma_a1_full, gamma_a2_full),
        (gamma_b1_full, gamma_b2_full),
        (gamma_c1_full, gamma_c2_full),
    )


def collect_projected_majoranas(basis, gamma_full_ops):
    gamma_labels = list(gamma_full_ops.keys())
    gamma_ops = [
        project_and_normalize(gamma_full_ops[label], basis, label)
        for label in gamma_labels
    ]
    return gamma_labels, gamma_ops


def build_physical_junctions(builder):
    operators = builder.get_operators()
    a_edge = flatten_site(0, 2)
    b_edge = flatten_site(1, 0)
    c_edge = flatten_site(2, 0)

    junction_ab_full = build_junction_operator(
        operators,
        a_edge,
        b_edge,
        t_couple=builder.t[0],
        delta_couple=builder.Delta[0],
    )
    junction_ac_full = build_junction_operator(
        operators,
        a_edge,
        c_edge,
        t_couple=builder.t[0],
        delta_couple=builder.Delta[0],
    )
    return junction_ab_full, junction_ac_full


def run_method(
    model_name,
    terms,
    *,
    gamma_list,
    parity_projected,
    t_total,
    delta_max,
    delta_min,
    steepness,
    width,
    n_points,
    transport_dim,
    verbose,
):
    term_a, term_b, term_c = terms
    result = run_model_comparison(
        model_name=model_name,
        T_total=t_total,
        Δ_max=delta_max,
        Δ_min=delta_min,
        s=steepness,
        width=width,
        T_A=term_a,
        T_AB=term_b,
        T_AC=term_c,
        n_points=n_points,
        transport_dim=transport_dim,
        gamma_list=gamma_list,
        verbose=verbose,
    )
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
    result["parity_gate"] = check_parity_resolved_gate(
        result["U_kato"],
        ground_data["V0"],
        parity_projected,
        gamma_list[2],
        gamma_list[3],
        verbose=verbose,
    )
    return result


def summarize_method(model_name, result, dim_sub):
    single_exchange = result["single_exchange"]
    parity_gate = result["parity_gate"]
    return [
        f" {model_name}: max_braid_error={single_exchange['max_error']:.2e} "
        f"| normalized={single_exchange['max_error'] / np.sqrt(dim_sub):.2e}",
        f"  braid_errors: {format_single_exchange_errors(single_exchange)}",
        (
            f"  off_block_leakage={parity_gate['off_block_leakage']:.2e} "
            f"| odd_target_error={parity_gate['odd_target_error']:.2e} "
            f"| even_target_error={parity_gate['even_target_error']:.2e}"
        ),
    ]


def compare_for_u(
    u_value,
    *,
    levels_to_include,
    max_cum_dim,
    n_points,
    energy_tol,
    b_site,
    c_site,
    verbose,
):
    specified_vals = {"U": [u_value]}

    (gamma_a1_full, gamma_a2_full), (gamma_b1_full, gamma_b2_full), (gamma_c1_full, gamma_c2_full) = get_full_gammas_fast(
        levels_to_include=levels_to_include,
        specified_vals=specified_vals,
    )
    gamma_full_ops = {
        "γA1": gamma_a1_full,
        "γA2": gamma_a2_full,
        "γB1": gamma_b1_full,
        "γB2": gamma_b2_full,
        "γC1": gamma_c1_full,
        "γC2": gamma_c2_full,
    }

    builder = BraidingHamiltonianBuilder(
        n_sites=3,
        dupes=3,
        specified_vals=specified_vals,
        config_path=default_config_path(),
    )

    h_full = builder.full_system_hamiltonian()
    eigvals, eigvecs = np.linalg.eigh(h_full)
    parity_full = build_total_parity_full(builder)
    projection_blocks = build_projection_stack(
        eigvals,
        eigvecs,
        parity_full,
        energy_tol=energy_tol,
    )
    basis_stack = [block["basis"] for block in projection_blocks]
    cumulative_stack = [np.hstack(basis_stack[: idx + 1]) for idx in range(len(basis_stack))]

    junction_ab_full, junction_ac_full = build_physical_junctions(builder)

    cre_ops, ann_ops, _ = precompute_ops(9)
    chi_b_full = 1j * (cre_ops[b_site] - ann_ops[b_site])
    chi_c_full = 1j * (cre_ops[c_site] - ann_ops[c_site])

    t_total = 1.0
    delta_max = 1.0
    delta_min = 0.0
    width = t_total / 3.0
    steepness = 20.0 / width

    lines = [f"U = {u_value:.1f}"]

    for block, basis in zip(projection_blocks, cumulative_stack):
        dim_sub = basis.shape[1]
        if dim_sub > max_cum_dim:
            break

        if dim_sub % 2 != 0:
            if verbose:
                print(f"Skipping odd-dimensional projection with dim={dim_sub}.")
            continue

        transport_dim = dim_sub // 2
        gamma_labels, gamma_ops = collect_projected_majoranas(basis, gamma_full_ops)
        gamma_map = dict(zip(gamma_labels, gamma_ops))

        t_ab = project_operator(junction_ab_full, basis)
        t_ac = project_operator(junction_ac_full, basis)
        rows_ab = get_bilinear_components(t_ab, gamma_labels, gamma_ops)
        rows_ac = get_bilinear_components(t_ac, gamma_labels, gamma_ops)
        active = choose_active_majoranas(rows_ab, rows_ac, gamma_labels, gamma_ops, verbose=verbose)

        gamma0 = active["γ0"]
        gamma1 = active["γ1"]
        gamma2 = active["γ2"]
        gamma3 = active["γ3"]
        gamma_list_physical = [gamma0, gamma1, gamma2, gamma3]
        active_labels = (
            gamma_labels[active["γ0_idx"]],
            gamma_labels[active["γ1_idx"]],
            gamma_labels[active["γ2_idx"]],
            gamma_labels[active["γ3_idx"]],
        )

        term_a = 1j * gamma0 @ gamma1
        t_ab_bilinear_fit = reconstruct_bilinear_term(rows_ab, gamma_ops)
        t_ac_bilinear_fit = reconstruct_bilinear_term(rows_ac, gamma_ops)
        t_ab_ideal = 1j * gamma0 @ gamma2
        t_ac_ideal = 1j * gamma0 @ gamma3
        bilinear_residual_ab = np.linalg.norm(t_ab - t_ab_bilinear_fit) / np.sqrt(dim_sub)
        bilinear_residual_ac = np.linalg.norm(t_ac - t_ac_bilinear_fit) / np.sqrt(dim_sub)

        local_projection_reference_labels = ("γA1", "γA2", "γB2", "γC2")
        gamma_list_local_projection = [
            gamma_map[local_projection_reference_labels[0]],
            gamma_map[local_projection_reference_labels[1]],
            gamma_map[local_projection_reference_labels[2]],
            gamma_map[local_projection_reference_labels[3]],
        ]
        term_a_local_projection = 1j * gamma_list_local_projection[0] @ gamma_list_local_projection[1]
        term_b_local_projection = hermitian_part(project_operator(1j * gamma_a1_full @ chi_b_full, basis))
        term_c_local_projection = hermitian_part(project_operator(1j * gamma_a1_full @ chi_c_full, basis))
        local_projection_matches_active = active_labels == local_projection_reference_labels

        parity_projected = build_total_parity_projected(builder, basis)
        model_definitions = {
            "ideal": {
                "terms": (term_a, t_ab_ideal, t_ac_ideal),
                "gamma_list": gamma_list_physical,
            },
            "bilinear_fit": {
                "terms": (term_a, t_ab_bilinear_fit, t_ac_bilinear_fit),
                "gamma_list": gamma_list_physical,
            },
            "local_projection": {
                "terms": (
                    term_a_local_projection,
                    term_b_local_projection,
                    term_c_local_projection,
                ),
                "gamma_list": gamma_list_local_projection,
            },
            "physical": {
                "terms": (term_a, t_ab, t_ac),
                "gamma_list": gamma_list_physical,
            },
        }
        model_results = {}

        for model_name, model_definition in model_definitions.items():
            model_results[model_name] = run_method(
                model_name,
                model_definition["terms"],
                gamma_list=model_definition["gamma_list"],
                parity_projected=parity_projected,
                t_total=t_total,
                delta_max=delta_max,
                delta_min=delta_min,
                steepness=steepness,
                width=width,
                n_points=n_points,
                transport_dim=transport_dim,
                verbose=verbose,
            )

        block_lines = [
            f"{block['name']} cum_dim={dim_sub}",
            (
                " active_majoranas: "
                f"γ0={active_labels[0]}, γ1={active_labels[1]}, "
                f"γ2={active_labels[2]}, γ3={active_labels[3]}"
            ),
            (
                " physical_bilinear_residuals: "
                f"AB={bilinear_residual_ab:.2e}, AC={bilinear_residual_ac:.2e}"
            ),
        ]
        if not local_projection_matches_active:
            block_lines.append(
                " local_projection_reference_channel_mismatch: "
                "local_projection uses "
                f"{local_projection_reference_labels}, active channel is {active_labels}"
            )

        for model_name in ("ideal", "bilinear_fit", "local_projection", "physical"):
            block_lines.extend(summarize_method(model_name, model_results[model_name], dim_sub))

        if len(lines) > 1:
            lines.append("")
        lines.append("\n".join(block_lines))

    return "\n".join(lines)


def main():
    args = build_argument_parser().parse_args()

    sections = []
    for index, u_value in enumerate(args.u_values):
        if index:
            sections.append("")
        sections.append(
            compare_for_u(
                u_value,
                levels_to_include=args.levels_to_include,
                max_cum_dim=args.max_cum_dim,
                n_points=args.n_points,
                energy_tol=args.energy_tol,
                b_site=args.b_site,
                c_site=args.c_site,
                verbose=args.verbose,
            )
        )

    output_text = "\n".join(sections) + "\n"
    args.output.write_text(output_text, encoding="utf-8")
    print(output_text, end="")
    print(f"\nSaved comparison results to {args.output}")


if __name__ == "__main__":
    main()
