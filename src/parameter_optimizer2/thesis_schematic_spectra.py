import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

_MPLCONFIGDIR = Path(__file__).resolve().parents[2] / ".mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


LOSS_RE = re.compile(r"Loss:\s*([0-9.eE+-]+)")
N_RE = re.compile(r"configuration\s+(\d+)-\s*site system")


@dataclass
class OptimizedSpectrum:
    n: int
    u: float
    loss: float
    even: np.ndarray
    odd: np.ndarray
    t: np.ndarray
    interaction: np.ndarray
    eps: np.ndarray
    delta: np.ndarray


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_configurations(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of configurations in {path}.")
    return data


def parse_loss(entry: dict) -> float:
    match = LOSS_RE.search(entry.get("header", ""))
    return float(match.group(1)) if match else math.inf


def infer_n(entry: dict) -> int:
    match = N_RE.search(entry.get("header", ""))
    if match:
        return int(match.group(1))
    eps = entry.get("physical_parameters", {}).get("eps")
    if eps:
        return len(eps)
    raise ValueError(f"Could not infer system size from {entry.get('header', '')!r}.")


def infer_u(entry: dict) -> float:
    fixed_u = entry.get("parameter_configs", {}).get("U", {}).get("fixed")
    if isinstance(fixed_u, (int, float)):
        return float(fixed_u)
    physical_u = entry.get("physical_parameters", {}).get("U")
    if physical_u:
        return float(physical_u[0])
    raise ValueError(f"Could not infer U from {entry.get('header', '')!r}.")


def find_best_entry(entries: list[dict], n: int, u: float) -> dict:
    matching = [
        entry
        for entry in entries
        if infer_n(entry) == n and abs(infer_u(entry) - u) < 1e-9
    ]
    if not matching:
        raise ValueError(f"No saved configuration found for n={n}, U={u}.")
    return min(matching, key=parse_loss)


def expanded(values: list[float] | np.ndarray, expected_length: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 1 and expected_length > 1:
        array = np.repeat(array, expected_length)
    return array


def build_record(entry: dict) -> OptimizedSpectrum:
    n = infer_n(entry)
    u = infer_u(entry)
    phys = entry["physical_parameters"]

    t = expanded(phys["t"], n - 1)
    interaction = expanded(phys["U"], n - 1)
    eps = expanded(phys["eps"], n)
    delta = expanded(phys["Delta"], n - 1)

    hamiltonian = build_hamiltonian(n=n, t=t, interaction=interaction, eps=eps, delta=delta)
    evals, evecs = np.linalg.eigh(hamiltonian)
    parity = parity_operator(n)
    parities = np.real(np.sum(np.conj(evecs) * (parity @ evecs), axis=0))
    even = evals[parities >= 0]
    odd = evals[parities < 0]
    ground_shift = min(float(np.min(even)), float(np.min(odd)))

    return OptimizedSpectrum(
        n=n,
        u=u,
        loss=parse_loss(entry),
        even=even - ground_shift,
        odd=odd - ground_shift,
        t=t,
        interaction=interaction,
        eps=eps,
        delta=delta,
    )


def tensor_product(*matrices: np.ndarray) -> np.ndarray:
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result


def sigma_site(site: int, n: int, operator: np.ndarray) -> np.ndarray:
    identity = np.eye(2, dtype=complex)
    operators = [identity] * n
    operators[site] = operator
    return tensor_product(*operators)


def creation_annihilation(site: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    create_local = 0.5 * (sigma_site(site, n, sigma_x) + 1j * sigma_site(site, n, sigma_y))
    annihilate_local = 0.5 * (sigma_site(site, n, sigma_x) - 1j * sigma_site(site, n, sigma_y))

    jordan_wigner = np.eye(2**n, dtype=complex)
    for previous_site in range(site):
        jordan_wigner = jordan_wigner @ sigma_site(previous_site, n, -sigma_z)

    return jordan_wigner @ create_local, jordan_wigner @ annihilate_local


def parity_operator(n: int) -> np.ndarray:
    parity = np.eye(2**n, dtype=complex)
    for site in range(n):
        create, annihilate = creation_annihilation(site, n)
        number = create @ annihilate
        parity = parity @ (np.eye(2**n, dtype=complex) - 2 * number)
    return parity


def build_hamiltonian(
    n: int,
    t: np.ndarray,
    interaction: np.ndarray,
    eps: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    create = []
    annihilate = []
    number = []
    for site in range(n):
        create_site, annihilate_site = creation_annihilation(site, n)
        create.append(create_site)
        annihilate.append(annihilate_site)
        number.append(create_site @ annihilate_site)

    hamiltonian = np.zeros((2**n, 2**n), dtype=complex)
    for site in range(n - 1):
        hamiltonian += -t[site] * (
            annihilate[site] @ create[site + 1] + create[site] @ annihilate[site + 1]
        )
        hamiltonian += delta[site] * (
            create[site] @ create[site + 1] + annihilate[site + 1] @ annihilate[site]
        )
        hamiltonian += interaction[site] * (number[site] @ number[site + 1])

    for site in range(n):
        hamiltonian += eps[site] * number[site]

    return hamiltonian


def format_u(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 350,
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 18,
            "axes.titlesize": 22,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 17,
            "legend.fontsize": 16,
            "axes.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_record(record: OptimizedSpectrum, output_stem: Path) -> None:
    style_matplotlib()

    figure_width = 6.6 if record.n == 2 else 7.4
    fig, (ax_spectrum, ax_setup) = plt.subplots(
        2,
        1,
        figsize=(figure_width, 5.8),
        gridspec_kw={"height_ratios": [2.15, 1.15]},
        constrained_layout=True,
    )

    even_color = "#2F6690"
    odd_color = "#B23A48"
    connector_color = "#777777"

    ax_spectrum.hlines(record.even, -0.24, 0.24, color=even_color, linewidth=4.4)
    ax_spectrum.hlines(record.odd, 0.76, 1.24, color=odd_color, linewidth=4.4)

    for even, odd in zip(record.even, record.odd):
        if abs(float(even) - float(odd)) <= 1e-2:
            ax_spectrum.hlines(
                0.5 * (float(even) + float(odd)),
                0.24,
                0.76,
                color=connector_color,
                linewidth=2.2,
                linestyles="dashed",
                zorder=1,
            )

    ax_spectrum.set_xlim(-0.45, 1.45)
    ax_spectrum.set_xticks([0, 1])
    ax_spectrum.set_xticklabels(["Even", "Odd"])
    ax_spectrum.set_ylabel(r"$E-E_0$")
    ax_spectrum.set_title(fr"{dot_count_label(record.n)} spectrum, $U={format_u(record.u)}$")
    ax_spectrum.grid(axis="y", alpha=0.24, linewidth=0.9)

    legend_lines = [
        plt.Line2D([0], [0], color=even_color, linewidth=4.4, label="Even"),
        plt.Line2D([0], [0], color=odd_color, linewidth=4.4, label="Odd"),
        plt.Line2D([0], [0], color=connector_color, linewidth=2.2, linestyle="--", label="paired"),
    ]
    ax_spectrum.legend(handles=legend_lines, frameon=False, loc="upper right")

    draw_setup(ax_setup, record)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def dot_count_label(n: int) -> str:
    names = {2: "Two-dot", 3: "Three-dot", 4: "Four-dot"}
    return names.get(n, fr"$n={n}$ dot")


def draw_setup(axis: plt.Axes, record: OptimizedSpectrum) -> None:
    n = record.n
    axis.set_xlim(-0.55, n - 0.45)
    axis.set_ylim(-1.05, 1.05)
    axis.axis("off")

    dot_y = 0.0
    sc_width = 0.76
    sc_height = 0.64

    for site in range(n):
        axis.scatter(
            site,
            dot_y,
            s=1150,
            color="#2878B5",
            edgecolor="black",
            linewidth=1.3,
            zorder=3,
        )
        axis.text(
            site,
            dot_y,
            fr"$d_{site + 1}$",
            ha="center",
            va="center",
            fontsize=20,
            color="white",
            fontweight="bold",
            zorder=4,
        )
        axis.text(
            site,
            -0.75,
            fr"$\epsilon_{site + 1}={record.eps[site]:.3f}$",
            ha="center",
            va="center",
            fontsize=18,
        )

    for bond in range(n - 1):
        x_mid = bond + 0.5
        rect = plt.Rectangle(
            (x_mid - sc_width / 2, dot_y - sc_height / 2),
            sc_width,
            sc_height,
            facecolor="#E6E6E6",
            edgecolor="black",
            linewidth=1.1,
            zorder=2,
        )
        axis.add_patch(rect)
        axis.text(
            x_mid,
            dot_y,
            fr"$t_{bond + 1}={record.t[bond]:.2f}$" + "\n" + fr"$\Delta_{bond + 1}={record.delta[bond]:.2f}$",
            ha="center",
            va="center",
            fontsize=18,
            linespacing=1.0,
        )
        axis.text(
            x_mid,
            0.70,
            fr"$U_{bond + 1}={record.interaction[bond]:.2f}$",
            ha="center",
            va="center",
            fontsize=18,
            color="#6F3C97",
        )


def parity_eigenvectors(record: OptimizedSpectrum) -> tuple[np.ndarray, np.ndarray]:
    hamiltonian = build_hamiltonian(
        n=record.n,
        t=record.t,
        interaction=record.interaction,
        eps=record.eps,
        delta=record.delta,
    )
    _, evecs = np.linalg.eigh(hamiltonian)
    parity = parity_operator(record.n)
    parities = np.real(np.sum(np.conj(evecs) * (parity @ evecs), axis=0))
    return evecs[:, parities >= 0], evecs[:, parities < 0]


def majorana_polarization_lowest_pair(record: OptimizedSpectrum) -> np.ndarray:
    even_vecs, odd_vecs = parity_eigenvectors(record)
    even_ground = even_vecs[:, 0]
    odd_ground = odd_vecs[:, 0]

    profile = []
    for site in range(record.n):
        create, annihilate = creation_annihilation(site, record.n)
        gamma_x = create + annihilate
        gamma_y = 1j * (create - annihilate)
        matrix_element_x = np.vdot(odd_ground, gamma_x @ even_ground)
        matrix_element_y = np.vdot(odd_ground, gamma_y @ even_ground)
        profile.append(np.real(matrix_element_x**2 + matrix_element_y**2))

    return np.asarray(profile, dtype=float)


def plot_majorana_polarization(record: OptimizedSpectrum, output_stem: Path) -> None:
    style_matplotlib()

    profile = majorana_polarization_lowest_pair(record)
    sites = np.arange(1, record.n + 1)

    fig, axis = plt.subplots(figsize=(5.9, 4.5), constrained_layout=True)
    axis.axhline(0, color="#777777", linewidth=1.6, linestyle="--", zorder=1)
    axis.plot(
        sites,
        profile,
        color="#2F6690",
        marker="o",
        markersize=8,
        linewidth=3.0,
        zorder=3,
    )

    axis.set_title(
        fr"Majorana polarization, $n={record.n}$, $U={format_u(record.u)}$",
        fontsize=18,
        pad=10,
    )
    axis.set_xlabel("Dot index", fontsize=17)
    axis.set_ylabel(r"$M_i$", fontsize=18)
    axis.set_xticks(sites)
    axis.set_ylim(-1.18, 1.18)
    axis.set_xlim(0.75, record.n + 0.25)
    axis.tick_params(axis="both", labelsize=15)
    axis.grid(alpha=0.22, linewidth=0.9)

    for site, value in zip(sites, profile):
        if value > 0.75:
            vertical_offset = -0.15
        elif value < -0.75:
            vertical_offset = 0.15
        else:
            vertical_offset = -0.15
        axis.text(
            site,
            value + vertical_offset,
            f"{value:.3f}",
            ha="center",
            va="center",
            fontsize=12,
            color="#222222",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
        )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Regenerate thesis-ready two-dot spectrum and parameter schematic figures."
    )
    parser.add_argument("--config", type=Path, default=root / "configuration.json")
    parser.add_argument("--fig-dir", type=Path, default=root / "texmex" / "figs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = load_configurations(args.config)
    targets = {
        0.0: "2DotU0",
        0.1: "2DotU01",
        2.0: "2DotU2",
    }

    for u, filename in targets.items():
        entry = find_best_entry(entries, n=2, u=u)
        record = build_record(entry)
        output_stem = args.fig_dir / filename
        plot_record(record, output_stem)
        print(f"Saved {output_stem.with_suffix('.pdf')} and {output_stem.with_suffix('.png')}")

    three_dot_entry = find_best_entry(entries, n=3, u=0.1)
    three_dot_record = build_record(three_dot_entry)
    plot_record(three_dot_record, args.fig_dir / "3DotU01")
    plot_majorana_polarization(three_dot_record, args.fig_dir / "3DotMP01")
    print(f"Saved {(args.fig_dir / '3DotU01').with_suffix('.pdf')} and {(args.fig_dir / '3DotU01').with_suffix('.png')}")
    print(f"Saved {(args.fig_dir / '3DotMP01').with_suffix('.pdf')} and {(args.fig_dir / '3DotMP01').with_suffix('.png')}")


if __name__ == "__main__":
    main()
