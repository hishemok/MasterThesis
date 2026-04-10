from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.linalg import expm

REPO_ROOT = Path(__file__).resolve().parents[2]
MPLCONFIGDIR = REPO_ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .get_mzm_JW import get_full_gammas
    from .hamiltonian_builder import BraidingHamiltonianBuilder, default_config_path, precompute_ops
except ImportError:
    from get_mzm_JW import get_full_gammas
    from hamiltonian_builder import BraidingHamiltonianBuilder, default_config_path, precompute_ops


plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 200,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


def delta_pulse(t, t_peak, width, steepness, delta_max, delta_min):
    t_start = t_peak - width / 2.0
    t_end = t_peak + width / 2.0
    rise = 1.0 / (1.0 + np.exp(-steepness * (t - t_start)))
    fall = 1.0 / (1.0 + np.exp(steepness * (t - t_end)))
    return delta_min + (delta_max - delta_min) * rise * fall


def build_ideal_hamiltonian(t, t_total, delta_max, delta_min, steepness, width, gamma0, gamma1, gamma2, gamma3):
    delta_1 = (
        delta_pulse(t, 0.0, width, steepness, delta_max, delta_min)
        + delta_pulse(t, t_total, width, steepness, delta_max, delta_min)
        - delta_min
    )
    delta_2 = delta_pulse(t, t_total / 3.0, width, steepness, delta_max, delta_min)
    delta_3 = delta_pulse(t, 2.0 * t_total / 3.0, width, steepness, delta_max, delta_min)
    hamiltonian = (
        delta_1 * 1j * gamma0 @ gamma1
        + delta_2 * 1j * gamma0 @ gamma2
        + delta_3 * 1j * gamma0 @ gamma3
    )
    return hamiltonian, (delta_1, delta_2, delta_3)


def build_projected_hamiltonian(t, t_total, delta_max, delta_min, steepness, width, t_a, t_b, t_c):
    delta_1 = (
        delta_pulse(t, 0.0, width, steepness, delta_max, delta_min)
        + delta_pulse(t, t_total, width, steepness, delta_max, delta_min)
        - delta_min
    )
    delta_2 = delta_pulse(t, t_total / 3.0, width, steepness, delta_max, delta_min)
    delta_3 = delta_pulse(t, 2.0 * t_total / 3.0, width, steepness, delta_max, delta_min)
    hamiltonian = delta_1 * t_a + delta_2 * t_b + delta_3 * t_c
    return hamiltonian, (delta_1, delta_2, delta_3)


def evolve_protocol(hamiltonian_fn, t_total, dim=8, n_points=3000):
    times = np.linspace(0.0, t_total, n_points)
    dt = times[1] - times[0]
    energies = np.zeros((n_points, dim))
    couplings = np.zeros((n_points, 3))
    u_kato = np.eye(dim, dtype=complex)

    h0, couplings[0] = hamiltonian_fn(times[0])
    evals, evecs = np.linalg.eigh(h0)
    energies[0] = evals
    basis = evecs[:, :4]

    for idx in range(1, len(times)):
        hamiltonian, couplings[idx] = hamiltonian_fn(times[idx])
        evals, evecs = np.linalg.eigh(hamiltonian)
        energies[idx] = evals

        next_basis = evecs[:, :4]
        projector = basis @ basis.conj().T
        next_projector = next_basis @ next_basis.conj().T
        d_projector = (next_projector - projector) / dt
        kato_generator = projector @ d_projector - d_projector @ projector
        u_kato = expm(-dt * kato_generator) @ u_kato
        basis = next_basis

    return times, energies, couplings, u_kato


def project_operator(operator_full, basis):
    return basis.conj().T @ operator_full @ basis


def normalize_trace(operator_sub):
    dim = operator_sub.shape[0]
    norm = np.sqrt(np.trace(operator_sub @ operator_sub).real / dim)
    if np.isclose(norm, 0.0):
        raise ValueError("Cannot normalize an operator with vanishing projected norm.")
    return operator_sub / norm


def project_and_normalize(operator_full, basis):
    return normalize_trace(project_operator(operator_full, basis))


def hermitian_part(operator):
    return 0.5 * (operator + operator.conj().T)


def flatten_site(subsystem, site, n_sites=3):
    return subsystem * n_sites + site


def build_junction_operator(operators, site_a, site_b, t_couple=0.0, delta_couple=0.0):
    left = min(site_a, site_b)
    right = max(site_a, site_b)
    key = (left, right)

    dim = operators["num"][0].shape[0]
    junction = np.zeros((dim, dim), dtype=complex)

    if t_couple != 0.0:
        junction += -float(t_couple) * operators["hop"][key]
    if delta_couple != 0.0:
        junction += float(delta_couple) * operators["pair"][key]

    return hermitian_part(junction)


def build_total_parity_projected(builder, basis):
    operators = builder.get_operators()
    num_ops = operators["num"]
    dim_full = num_ops[0].shape[0]
    identity = np.eye(dim_full, dtype=complex)
    parity_full = identity.copy()

    for number_op in num_ops:
        parity_full = parity_full @ (identity - 2.0 * number_op)

    return basis.conj().T @ parity_full @ basis


def get_ground_manifold_data(hamiltonian_fn, t_total):
    h0, _ = hamiltonian_fn(0.0)
    h1, _ = hamiltonian_fn(t_total)
    evals_0, evecs_0 = np.linalg.eigh(h0)
    evals_1, evecs_1 = np.linalg.eigh(h1)
    v0 = evecs_0[:, :4]
    v1 = evecs_1[:, :4]
    p0 = v0 @ v0.conj().T
    p1 = v1 @ v1.conj().T
    return {"evals_0": evals_0, "evals_1": evals_1, "v0": v0, "v1": v1, "p0": p0, "p1": p1}


def phase_aligned_error(unitary, target):
    overlap = np.trace(target.conj().T @ unitary)
    phase = 0.0 if np.isclose(np.abs(overlap), 0.0) else np.angle(overlap)
    return np.linalg.norm(unitary - np.exp(1j * phase) * target)


def compute_path_metrics(times, energies, parity_op, hamiltonian_fn):
    hermiticity_error = 0.0
    parity_error = 0.0

    for time_value in times:
        hamiltonian, _ = hamiltonian_fn(time_value)
        hermiticity_error = max(hermiticity_error, np.linalg.norm(hamiltonian - hamiltonian.conj().T))
        parity_error = max(parity_error, np.linalg.norm(hamiltonian @ parity_op - parity_op @ hamiltonian))

    return {
        "max_hermiticity_error": float(hermiticity_error),
        "max_parity_commutator": float(parity_error),
        "max_ground_splitting": float(np.max(energies[:, 3] - energies[:, 0])),
        "min_gap": float(np.min(energies[:, 4] - energies[:, 3])),
    }


def compute_transport_metrics(u_kato, p0, p1):
    dim = u_kato.shape[0]
    identity = np.eye(dim, dtype=complex)
    return {
        "unitarity_error": float(np.linalg.norm(u_kato.conj().T @ u_kato - identity)),
        "transport_error": float(np.linalg.norm(u_kato @ p0 @ u_kato.conj().T - p1)),
        "loop_closure_error": float(np.linalg.norm(p1 - p0)),
    }


def compute_exchange_metrics(u_kato, gamma_list):
    single_expected = {
        "gamma2_to_minus_gamma3": (gamma_list[2], -gamma_list[3]),
        "gamma3_to_gamma2": (gamma_list[3], gamma_list[2]),
        "gamma1_to_gamma1": (gamma_list[1], gamma_list[1]),
        "gamma0_to_gamma0": (gamma_list[0], gamma_list[0]),
    }
    double_expected = {
        "gamma2_to_minus_gamma2": (gamma_list[2], -gamma_list[2]),
        "gamma3_to_minus_gamma3": (gamma_list[3], -gamma_list[3]),
        "gamma1_to_gamma1": (gamma_list[1], gamma_list[1]),
        "gamma0_to_gamma0": (gamma_list[0], gamma_list[0]),
    }

    single_errors = {}
    for label, (source, target) in single_expected.items():
        transformed = u_kato.conj().T @ source @ u_kato
        single_errors[label] = float(np.linalg.norm(transformed - target))

    u_double = u_kato @ u_kato
    double_errors = {}
    for label, (source, target) in double_expected.items():
        transformed = u_double.conj().T @ source @ u_double
        double_errors[label] = float(np.linalg.norm(transformed - target))

    u_four = u_double @ u_double
    four_errors = {}
    for index, gamma in enumerate(gamma_list):
        transformed = u_four.conj().T @ gamma @ u_four
        four_errors[f"gamma{index}_to_gamma{index}"] = float(np.linalg.norm(transformed - gamma))

    return {
        "single_exchange": single_errors,
        "double_exchange": double_errors,
        "four_exchanges": four_errors,
        "max_single_exchange_error": float(max(single_errors.values())),
        "max_double_exchange_error": float(max(double_errors.values())),
        "max_four_exchange_error": float(max(four_errors.values())),
    }


def compute_parity_gate_metrics(u_kato, v0, parity_op, gamma2, gamma3):
    u_ground = v0.conj().T @ u_kato @ v0
    parity_ground = v0.conj().T @ parity_op @ v0

    parity_vals, parity_vecs = np.linalg.eigh(parity_ground)
    u_parity = parity_vecs.conj().T @ u_ground @ parity_vecs

    off_block = np.linalg.norm(u_parity[:2, 2:]) + np.linalg.norm(u_parity[2:, :2])
    odd_block = u_parity[:2, :2]
    even_block = u_parity[2:, 2:]

    u_target = expm(-0.25 * np.pi * (gamma2 @ gamma3))
    u_target_ground = v0.conj().T @ u_target @ v0
    u_target_parity = parity_vecs.conj().T @ u_target_ground @ parity_vecs
    odd_target = u_target_parity[:2, :2]
    even_target = u_target_parity[2:, 2:]

    return {
        "parity_eigenvalues": [float(value) for value in np.real_if_close(parity_vals)],
        "off_block_leakage": float(off_block),
        "odd_block_target_error": float(phase_aligned_error(odd_block, odd_target)),
        "even_block_target_error": float(phase_aligned_error(even_block, even_target)),
        "odd_block_eigenvalues": [str(value) for value in np.linalg.eigvals(odd_block)],
        "even_block_eigenvalues": [str(value) for value in np.linalg.eigvals(even_block)],
    }


def get_bilinear_components(term, gamma_labels, gamma_ops):
    dim = term.shape[0]
    rows = []
    for left in range(len(gamma_ops)):
        for right in range(left + 1, len(gamma_ops)):
            basis_op = 1j * gamma_ops[left] @ gamma_ops[right]
            coeff = np.trace(basis_op.conj().T @ term).real / dim
            rows.append(
                {
                    "abs_coeff": float(abs(coeff)),
                    "coeff": float(coeff),
                    "left": int(left),
                    "right": int(right),
                    "label": f"i {gamma_labels[left]} {gamma_labels[right]}",
                }
            )
    rows.sort(key=lambda row: row["abs_coeff"], reverse=True)
    return rows


def get_single_majorana_components(term, gamma_labels, gamma_ops):
    dim = term.shape[0]
    rows = []
    for label, gamma in zip(gamma_labels, gamma_ops):
        coeff = np.trace(gamma.conj().T @ term).real / dim
        rows.append({"abs_coeff": float(abs(coeff)), "coeff": float(coeff), "label": label})
    rows.sort(key=lambda row: row["abs_coeff"], reverse=True)
    return rows


def get_partner_index(index):
    return {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}[index]


def choose_active_majoranas(rows_ab, rows_ac, gamma_labels, gamma_ops):
    dominant_ab = rows_ab[0]
    dominant_ac = rows_ac[0]
    pair_ab = {dominant_ab["left"], dominant_ab["right"]}
    pair_ac = {dominant_ac["left"], dominant_ac["right"]}
    common = pair_ab & pair_ac
    if len(common) != 1:
        raise ValueError("Could not identify a unique shared Majorana.")

    gamma0_idx = common.pop()
    gamma2_idx = dominant_ab["right"] if dominant_ab["left"] == gamma0_idx else dominant_ab["left"]
    gamma3_idx = dominant_ac["right"] if dominant_ac["left"] == gamma0_idx else dominant_ac["left"]
    gamma1_idx = get_partner_index(gamma0_idx)

    return {
        "gamma0_idx": gamma0_idx,
        "gamma1_idx": gamma1_idx,
        "gamma2_idx": gamma2_idx,
        "gamma3_idx": gamma3_idx,
        "gamma0_label": gamma_labels[gamma0_idx],
        "gamma1_label": gamma_labels[gamma1_idx],
        "gamma2_label": gamma_labels[gamma2_idx],
        "gamma3_label": gamma_labels[gamma3_idx],
        "gamma0": gamma_ops[gamma0_idx],
        "gamma1": gamma_ops[gamma1_idx],
        "gamma2": gamma_ops[gamma2_idx],
        "gamma3": gamma_ops[gamma3_idx],
    }


def relation_error(operator_a, operator_b):
    same_error = np.linalg.norm(operator_a - operator_b)
    minus_error = np.linalg.norm(operator_a + operator_b)
    if same_error <= minus_error:
        return {"relation": "+", "error": float(same_error)}
    return {"relation": "-", "error": float(minus_error)}


def format_metric(value, digits=3):
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e-2 and magnitude < 1e3:
        return f"{value:.{digits}f}"
    return f"{value:.{digits}e}"


def latex_gamma(label):
    name = label.replace("gamma", "")
    return rf"\gamma_{{{name}}}"


def latex_bilinear(label):
    _, left, right = label.split()
    return rf"$i\,{latex_gamma(left)}{latex_gamma(right)}$"


def plot_protocol_summary(times, energies, couplings, title, output_path):
    time_scaled = times / times[-1]
    shifted_energies = energies - energies[:, [0]]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )

    low_colors = ["#0b3c5d", "#1d6996", "#3b8bc2", "#6baed6"]
    high_colors = ["#8c8c8c", "#b0b0b0", "#d0d0d0", "#e0e0e0"]

    for idx in range(4):
        label = "Ground manifold" if idx == 0 else None
        axes[0].plot(time_scaled, shifted_energies[:, idx], color=low_colors[idx], lw=2.0, label=label)
    for idx in range(4, 8):
        label = "Excited states" if idx == 4 else None
        axes[0].plot(time_scaled, shifted_energies[:, idx], color=high_colors[idx - 4], lw=1.6, label=label)

    min_gap = np.min(shifted_energies[:, 4] - shifted_energies[:, 3])
    max_split = np.max(shifted_energies[:, 3] - shifted_energies[:, 0])
    axes[0].text(
        0.03,
        0.95,
        rf"$\max(E_3-E_0)={max_split:.2e}$" + "\n" + rf"$\min(E_4-E_3)={min_gap:.2f}$",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#cccccc", "alpha": 0.95},
    )
    axes[0].set_ylabel(r"$E - E_0$")
    axes[0].set_title(title)
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.2)

    coupling_labels = [r"$\Delta_1(t)$", r"$\Delta_2(t)$", r"$\Delta_3(t)$"]
    coupling_colors = ["#5c2a9d", "#00897b", "#c75b12"]
    for idx in range(3):
        axes[1].plot(time_scaled, couplings[:, idx], lw=2.2, color=coupling_colors[idx], label=coupling_labels[idx])
    axes[1].set_xlabel(r"Normalized time $t/T$")
    axes[1].set_ylabel("Coupling")
    axes[1].legend(loc="upper right", ncol=3, frameon=False)
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"))
    plt.close(fig)


def plot_decomposition_summary(rows_ab, rows_ac, output_path, n_terms=3):
    selected_ab = rows_ab[:n_terms]
    selected_ac = rows_ac[:n_terms]
    max_coeff = max(
        max(abs(row["coeff"]) for row in selected_ab),
        max(abs(row["coeff"]) for row in selected_ac),
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.3), sharex=True)
    for axis, title, rows in zip(
        axes,
        ["Projected AB Junction", "Projected AC Junction"],
        [selected_ab, selected_ac],
    ):
        labels = [latex_bilinear(row["label"]) for row in rows][::-1]
        values = [row["coeff"] for row in rows][::-1]
        colors = ["#0b3c5d" if value >= 0 else "#b34700" for value in values]
        axis.barh(labels, values, color=colors, alpha=0.9)
        axis.axvline(0.0, color="#444444", lw=1.0)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)

    axes[0].set_xlim(-1.08 * max_coeff, 1.08 * max_coeff)
    axes[0].set_xlabel("Projected coefficient")
    axes[1].set_xlabel("Projected coefficient")

    fig.tight_layout()
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"))
    plt.close(fig)


def make_protocol_summary(model_name, gamma_labels, times, energies, couplings, u_kato, parity_op, ground_data, gamma_list, hamiltonian_fn):
    path_metrics = compute_path_metrics(times, energies, parity_op, hamiltonian_fn)
    transport_metrics = compute_transport_metrics(u_kato, ground_data["p0"], ground_data["p1"])
    exchange_metrics = compute_exchange_metrics(u_kato, gamma_list)
    parity_gate_metrics = compute_parity_gate_metrics(u_kato, ground_data["v0"], parity_op, gamma_list[2], gamma_list[3])

    return {
        "model_name": model_name,
        "gamma_labels": gamma_labels,
        "path": path_metrics,
        "transport": transport_metrics,
        "exchange": exchange_metrics,
        "parity_gate": parity_gate_metrics,
        "spectrum": {
            "times": times.tolist(),
            "energies": energies.tolist(),
            "couplings": couplings.tolist(),
        },
    }


def prepare_reference_data(u_value=0.1, levels_to_include=4):
    builder = BraidingHamiltonianBuilder(
        n_sites=3,
        dupes=3,
        specified_vals={"U": [u_value]},
        config_path=default_config_path(),
    )
    h_full = builder.full_system_hamiltonian()
    _, eigvecs = np.linalg.eigh(h_full)
    basis_ref = eigvecs[:, :8]

    (gamma_a1_full, gamma_a2_full), (gamma_b1_full, gamma_b2_full), (gamma_c1_full, gamma_c2_full) = get_full_gammas(
        levels_to_include=levels_to_include,
        verbose=False,
    )

    gamma_full = {
        "gammaA1": gamma_a1_full,
        "gammaA2": gamma_a2_full,
        "gammaB1": gamma_b1_full,
        "gammaB2": gamma_b2_full,
        "gammaC1": gamma_c1_full,
        "gammaC2": gamma_c2_full,
    }
    gamma_sub = {label: project_and_normalize(operator, basis_ref) for label, operator in gamma_full.items()}

    return {"builder": builder, "basis_ref": basis_ref, "gamma_full": gamma_full, "gamma_sub": gamma_sub}


def run_ideal_reference(reference_data, t_total, delta_max, delta_min, width, steepness, n_points):
    gamma0 = reference_data["gamma_sub"]["gammaA1"]
    gamma1 = reference_data["gamma_sub"]["gammaA2"]
    gamma2 = reference_data["gamma_sub"]["gammaB1"]
    gamma3 = reference_data["gamma_sub"]["gammaC1"]

    hamiltonian_fn = lambda time_value: build_ideal_hamiltonian(
        time_value,
        t_total,
        delta_max,
        delta_min,
        steepness,
        width,
        gamma0,
        gamma1,
        gamma2,
        gamma3,
    )
    times, energies, couplings, u_kato = evolve_protocol(hamiltonian_fn, t_total=t_total, n_points=n_points)
    parity_op = build_total_parity_projected(reference_data["builder"], reference_data["basis_ref"])
    ground_data = get_ground_manifold_data(hamiltonian_fn, t_total=t_total)

    return make_protocol_summary(
        "Ideal effective model",
        ["gammaA1", "gammaA2", "gammaB1", "gammaC1"],
        times,
        energies,
        couplings,
        u_kato,
        parity_op,
        ground_data,
        [gamma0, gamma1, gamma2, gamma3],
        hamiltonian_fn,
    )


def run_projected_microscopic(reference_data, t_total, delta_max, delta_min, width, steepness, n_points):
    builder = reference_data["builder"]
    basis_ref = reference_data["basis_ref"]
    gamma_labels = ["gammaA1", "gammaA2", "gammaB1", "gammaB2", "gammaC1", "gammaC2"]
    gamma_ops = [reference_data["gamma_sub"][label] for label in gamma_labels]
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

    t_ab = project_operator(junction_ab_full, basis_ref)
    t_ac = project_operator(junction_ac_full, basis_ref)
    rows_ab = get_bilinear_components(t_ab, gamma_labels, gamma_ops)
    rows_ac = get_bilinear_components(t_ac, gamma_labels, gamma_ops)
    active = choose_active_majoranas(rows_ab, rows_ac, gamma_labels, gamma_ops)

    t_a = 1j * active["gamma0"] @ active["gamma1"]
    hamiltonian_fn = lambda time_value: build_projected_hamiltonian(
        time_value,
        t_total,
        delta_max,
        delta_min,
        steepness,
        width,
        t_a,
        t_ab,
        t_ac,
    )
    times, energies, couplings, u_kato = evolve_protocol(hamiltonian_fn, t_total=t_total, n_points=n_points)
    parity_op = build_total_parity_projected(builder, basis_ref)
    ground_data = get_ground_manifold_data(hamiltonian_fn, t_total=t_total)

    summary = make_protocol_summary(
        "Projected microscopic junction model",
        [
            active["gamma0_label"],
            active["gamma1_label"],
            active["gamma2_label"],
            active["gamma3_label"],
        ],
        times,
        energies,
        couplings,
        u_kato,
        parity_op,
        ground_data,
        [active["gamma0"], active["gamma1"], active["gamma2"], active["gamma3"]],
        hamiltonian_fn,
    )
    summary["decomposition"] = {
        "ab": rows_ab,
        "ac": rows_ac,
        "active_majoranas": {
            "gamma0": active["gamma0_label"],
            "gamma1": active["gamma1_label"],
            "gamma2": active["gamma2_label"],
            "gamma3": active["gamma3_label"],
        },
    }
    return summary


def run_local_projection_model(reference_data, t_total, delta_max, delta_min, width, steepness, n_points):
    basis_ref = reference_data["basis_ref"]
    builder = reference_data["builder"]
    gamma_labels = ["gammaA1", "gammaA2", "gammaB1", "gammaB2", "gammaC1", "gammaC2"]
    gamma_ops = [reference_data["gamma_sub"][label] for label in gamma_labels]

    creators, annihilators, _ = precompute_ops(9)
    cdag_b1, c_b1 = creators[3], annihilators[3]
    cdag_b2, c_b2 = creators[5], annihilators[5]
    cdag_c1, c_c1 = creators[6], annihilators[6]
    cdag_c2, c_c2 = creators[8], annihilators[8]

    local_majoranas = {
        "chi_B1x": cdag_b1 + c_b1,
        "chi_B1y": 1j * (cdag_b1 - c_b1),
        "chi_B2x": cdag_b2 + c_b2,
        "chi_B2y": 1j * (cdag_b2 - c_b2),
        "chi_C1x": cdag_c1 + c_c1,
        "chi_C1y": 1j * (cdag_c1 - c_c1),
        "chi_C2x": cdag_c2 + c_c2,
        "chi_C2y": 1j * (cdag_c2 - c_c2),
    }
    projected_local = {label: project_and_normalize(op, basis_ref) for label, op in local_majoranas.items()}

    local_components = {
        label: get_single_majorana_components(projected_local[label], gamma_labels, gamma_ops)
        for label in projected_local
    }
    local_relations = {
        "chi_B1x_vs_chi_B2x": relation_error(projected_local["chi_B1x"], projected_local["chi_B2x"]),
        "chi_B1y_vs_chi_B2y": relation_error(projected_local["chi_B1y"], projected_local["chi_B2y"]),
        "chi_C1x_vs_chi_C2x": relation_error(projected_local["chi_C1x"], projected_local["chi_C2x"]),
        "chi_C1y_vs_chi_C2y": relation_error(projected_local["chi_C1y"], projected_local["chi_C2y"]),
    }

    gamma0 = reference_data["gamma_sub"]["gammaA1"]
    gamma1 = reference_data["gamma_sub"]["gammaA2"]
    gamma2 = reference_data["gamma_sub"]["gammaB2"]
    gamma3 = reference_data["gamma_sub"]["gammaC2"]
    t_a = 1j * gamma0 @ gamma1
    t_b = hermitian_part(project_operator(1j * reference_data["gamma_full"]["gammaA1"] @ local_majoranas["chi_B1y"], basis_ref))
    t_c = hermitian_part(project_operator(1j * reference_data["gamma_full"]["gammaA1"] @ local_majoranas["chi_C1y"], basis_ref))

    hamiltonian_fn = lambda time_value: build_projected_hamiltonian(
        time_value,
        t_total,
        delta_max,
        delta_min,
        steepness,
        width,
        t_a,
        t_b,
        t_c,
    )
    times, energies, couplings, u_kato = evolve_protocol(hamiltonian_fn, t_total=t_total, n_points=n_points)
    parity_op = build_total_parity_projected(builder, basis_ref)
    ground_data = get_ground_manifold_data(hamiltonian_fn, t_total=t_total)

    summary = make_protocol_summary(
        "Projected local-operator model",
        ["gammaA1", "gammaA2", "gammaB2", "gammaC2"],
        times,
        energies,
        couplings,
        u_kato,
        parity_op,
        ground_data,
        [gamma0, gamma1, gamma2, gamma3],
        hamiltonian_fn,
    )
    summary["local_projection"] = {
        "components": local_components,
        "relations": local_relations,
    }
    return summary


def write_checks_table_tex(output_path, summaries):
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"Model & $\max(E_3-E_0)$ & $\min(E_4-E_3)$ & Max 1x err. & Leakage & Odd err. & Even err. \\",
        r"\hline",
    ]
    for summary in summaries:
        lines.append(
            " & ".join(
                [
                    summary["model_name"],
                    format_metric(summary["path"]["max_ground_splitting"]),
                    format_metric(summary["path"]["min_gap"]),
                    format_metric(summary["exchange"]["max_single_exchange_error"]),
                    format_metric(summary["parity_gate"]["off_block_leakage"]),
                    format_metric(summary["parity_gate"]["odd_block_target_error"]),
                    format_metric(summary["parity_gate"]["even_block_target_error"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decomposition_table_tex(output_path, microscopic_summary):
    rows_ab = microscopic_summary["decomposition"]["ab"][:3]
    rows_ac = microscopic_summary["decomposition"]["ac"][:3]

    lines = [
        r"\begin{tabular}{llc}",
        r"\hline",
        r"Junction & Term & Coefficient \\",
        r"\hline",
    ]
    for row in rows_ab:
        lines.append(f"AB & {latex_bilinear(row['label'])} & {format_metric(row['coeff'])} \\\\")
    lines.append(r"\hline")
    for row in rows_ac:
        lines.append(f"AC & {latex_bilinear(row['label'])} & {format_metric(row['coeff'])} \\\\")
    lines.extend([r"\hline", r"\end{tabular}"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_local_projection_table_tex(output_path, local_summary):
    interesting_labels = ["chi_B1y", "chi_B2y", "chi_C1y", "chi_C2y"]
    components = local_summary["local_projection"]["components"]
    relations = local_summary["local_projection"]["relations"]
    pretty_operator_labels = {
        "chi_B1y": r"$\chi_{B1,y}$",
        "chi_B2y": r"$\chi_{B2,y}$",
        "chi_C1y": r"$\chi_{C1,y}$",
        "chi_C2y": r"$\chi_{C2,y}$",
    }

    lines = [
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Projected operator & Dominant Majorana & Coefficient \\",
        r"\hline",
    ]
    for label in interesting_labels:
        dominant = components[label][0]
        lines.append(
            rf"{pretty_operator_labels[label]} & ${latex_gamma(dominant['label'])}$ & {format_metric(dominant['coeff'])} \\"
        )
    lines.append(r"\hline")
    lines.append(r"\multicolumn{3}{l}{Pairwise relation errors} \\")
    lines.append(r"\hline")
    relation_labels = {
        "chi_B1y_vs_chi_B2y": r"$\chi_{B1,y} \approx -\chi_{B2,y}$",
        "chi_C1y_vs_chi_C2y": r"$\chi_{C1,y} \approx -\chi_{C2,y}$",
    }
    for key, pretty_label in relation_labels.items():
        relation = relations[key]["relation"]
        error = relations[key]["error"]
        relation_text = "same sign" if relation == "+" else "opposite sign"
        lines.append(rf"{pretty_label} & {relation_text} & {format_metric(error)} \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Generate thesis-ready braiding figures and summaries.")
    parser.add_argument("--u", type=float, default=0.1, help="Interaction strength used to select the optimized 3-dot configuration.")
    parser.add_argument("--levels", type=int, default=4, help="Number of even-odd pairs included in the subsystem Majorana operators.")
    parser.add_argument("--n-points", type=int, default=3000, help="Number of time steps used in the adiabatic evolution.")
    parser.add_argument("--t-total", type=float, default=1000.0, help="Total braid time.")
    parser.add_argument("--delta-max", type=float, default=1.0, help="Maximum pulse strength.")
    parser.add_argument("--delta-min", type=float, default=0.0, help="Minimum pulse strength.")
    parser.add_argument("--fig-dir", type=Path, default=REPO_ROOT / "texmex" / "figs", help="Directory for saved figures.")
    parser.add_argument("--generated-dir", type=Path, default=REPO_ROOT / "texmex" / "generated", help="Directory for saved tables and JSON.")
    return parser


def main():
    args = build_argument_parser().parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.generated_dir.mkdir(parents=True, exist_ok=True)

    width = args.t_total / 3.0
    steepness = 20.0 / width

    reference = prepare_reference_data(u_value=args.u, levels_to_include=args.levels)
    ideal_summary = run_ideal_reference(
        reference,
        t_total=args.t_total,
        delta_max=args.delta_max,
        delta_min=args.delta_min,
        width=width,
        steepness=steepness,
        n_points=args.n_points,
    )
    projected_summary = run_projected_microscopic(
        reference,
        t_total=args.t_total,
        delta_max=args.delta_max,
        delta_min=args.delta_min,
        width=width,
        steepness=steepness,
        n_points=args.n_points,
    )
    local_summary = run_local_projection_model(
        reference,
        t_total=args.t_total,
        delta_max=args.delta_max,
        delta_min=args.delta_min,
        width=width,
        steepness=steepness,
        n_points=args.n_points,
    )

    plot_protocol_summary(
        np.array(ideal_summary["spectrum"]["times"]),
        np.array(ideal_summary["spectrum"]["energies"]),
        np.array(ideal_summary["spectrum"]["couplings"]),
        "Ideal four-Majorana braid",
        args.fig_dir / "braiding_ideal_protocol.pdf",
    )
    plot_protocol_summary(
        np.array(projected_summary["spectrum"]["times"]),
        np.array(projected_summary["spectrum"]["energies"]),
        np.array(projected_summary["spectrum"]["couplings"]),
        "Projected microscopic braid",
        args.fig_dir / "braiding_projected_protocol.pdf",
    )
    plot_decomposition_summary(
        projected_summary["decomposition"]["ab"],
        projected_summary["decomposition"]["ac"],
        args.fig_dir / "braiding_junction_decomposition.pdf",
    )

    write_checks_table_tex(
        args.generated_dir / "braiding_checks_comparison.tex",
        [ideal_summary, projected_summary, local_summary],
    )
    write_decomposition_table_tex(
        args.generated_dir / "braiding_junction_decomposition.tex",
        projected_summary,
    )
    write_local_projection_table_tex(
        args.generated_dir / "braiding_local_projection.tex",
        local_summary,
    )

    output = {
        "selection": reference["builder"].selection,
        "ideal_reference": ideal_summary,
        "projected_microscopic": projected_summary,
        "projected_local_operator": local_summary,
    }
    (args.generated_dir / "braiding_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("Saved figures:")
    print(f"  {args.fig_dir / 'braiding_ideal_protocol.pdf'}")
    print(f"  {args.fig_dir / 'braiding_projected_protocol.pdf'}")
    print(f"  {args.fig_dir / 'braiding_junction_decomposition.pdf'}")
    print("Saved tables and summary:")
    print(f"  {args.generated_dir / 'braiding_checks_comparison.tex'}")
    print(f"  {args.generated_dir / 'braiding_junction_decomposition.tex'}")
    print(f"  {args.generated_dir / 'braiding_local_projection.tex'}")
    print(f"  {args.generated_dir / 'braiding_summary.json'}")


if __name__ == "__main__":
    main()
