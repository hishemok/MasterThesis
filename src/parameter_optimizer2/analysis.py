import numpy as np
import matplotlib.pyplot as plt
import torch

from operators import parity_operator_torch, from_torch
from measurements import calculate_parities, charge_difference_torch, Majorana_polarization_torch


def extract_physical_params(model, theta):
    """
    Convert raw θ into physically meaningful parameters.
    """
    theta = theta.detach().clone()

    adjusted = model.adjust_tensor(theta)
    dict_params = model.tensor_to_dict(adjusted)
    phys = model.get_physical_parameters(dict_params)

    return phys


def compute_spectrum(model, phys_params):
    """
    Build the Hamiltonian and compute the full spectrum and parity classification.
    """
    H_torch = model.build(phys_params)
    evals, evecs = torch.linalg.eigh(H_torch)

    n = phys_params["eps"].numel()

    P = parity_operator_torch(n)
    even_states, odd_states, even_vecs, odd_vecs = calculate_parities(evals, evecs, P)

    return evals, evecs, even_states, odd_states, even_vecs, odd_vecs


def plot_majorana_polarization(even_vecs, odd_vecs, n):
    """
    Plot the Majorana polarization for each state.
    """
    MP = Majorana_polarization_torch(even_vecs, odd_vecs, n)

    for i in range(MP.shape[0]):
        plt.plot(range(n), MP[i], 'o-', label=f'State {i}')

    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Site index')
    plt.ylabel('Majorana polarization')
    plt.title(f'Majorana polarization profile ({n}-site system)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_setup_schematic(phys_params, n):
    """
    Draw schematic QD–SC–QD setup with t, U, eps, Delta values.
    """
    t = phys_params["t"].cpu().numpy()
    U = phys_params["U"].cpu().numpy()
    eps = phys_params["eps"].cpu().numpy()
    Delta = phys_params["Delta"].cpu().numpy()
    
    
    if len(t) != n - 1:
        t = [t[0]] * (n-1)
    if len(U) != n - 1:
        U = [U[0]] * (n-1)
    if len(Delta) != n - 1:
        Delta = [Delta[0]] * (n-1)
    if len(eps) != n:
        eps = [eps[0]] *  n
    n = len(eps)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')

    dot_y = 0
    sc_h = 0.5
    sc_w = 0.9

    # Dots
    for i in range(n):
        ax.scatter(i, dot_y, s=800, color='tab:blue', edgecolor='black', zorder=3)
        ax.text(i, dot_y - 0.5, f"eps={eps[i]:.2f}", ha='center', fontsize=9)

    # SC links
    for i in range(n - 1):
        x_mid = (i + i + 1) / 2

        rect = plt.Rectangle(#type: ignore
            (x_mid - sc_w/2, dot_y - sc_h/2),
            sc_w,
            sc_h,
            color='lightgray',
            ec='black',
            zorder=2
        )
        ax.add_patch(rect)

        ax.text(x_mid, dot_y,
                f"t={t[i]:.2f}\nΔ={Delta[i]:.2f}",
                ha='center', va='center', fontsize=8)
        ax.text(x_mid, dot_y + 0.6,
                f"U={U[i]:.2f}",
                ha='center', fontsize=9, color='tab:purple')

    ax.set_title("QD–SC–QD optimized configuration")
    plt.tight_layout()
    plt.show()


def parity_spectrum_plot(n, evals, even_states, odd_states):
    """
    Plot the parity-resolved energy spectrum.
    """
    y_even = even_states.cpu().numpy()
    y_odd  = odd_states.cpu().numpy()

    degeneracies = [e for e, o in zip(y_even, y_odd) if abs(e - o) < 1e-2]

    plt.figure(figsize=(10, 4))

    plt.hlines(y_even, -0.2, 0.2, color='tab:blue', label='Even')
    plt.hlines(y_odd,  0.8, 1.2, color='tab:red', label='Odd')
    plt.hlines(degeneracies, 0.2, 0.8, color='gray', linestyles='--')

    plt.xticks([0, 1], ["Even", "Odd"])
    plt.ylabel("Energy")
    plt.title(f"Parity-resolved spectrum ({n}-site system)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def combined_analysis_plot(n, evals, even_states, odd_states, even_vecs, odd_vecs, phys):
    """
    Plot parity-resolved spectrum on top and schematic below in one figure.
    """

    # Split physical params
    t = phys["t"]
    U = phys["U"]
    eps = phys["eps"]
    Delta = phys["Delta"]

    # Compute degeneracy lines (optional)
    degeneracy_lines = []
    for Ee, Eo in zip(even_states, odd_states):
        if abs(Ee - Eo) < 1e-2:
            degeneracy_lines.append(Ee)

    # Figure with two rows
    fig, axs = plt.subplots(2, 1, figsize=(10,5), gridspec_kw={'height_ratios':[2,1]})
    ax1, ax2 = axs

    # -----------------------------
    # Top: Parity-resolved spectrum
    # -----------------------------
    ax1.hlines(even_states, -0.2, 0.2, color='tab:blue', label='Even')
    ax1.hlines(odd_states, 0.8, 1.2, color='tab:red', label='Odd')
    if degeneracy_lines:
        ax1.hlines(degeneracy_lines, 0.2, 0.8, color='tab:gray', linestyles='dashed', label='Degeneracies')
    ax1.set_xticks([0,1])
    ax1.set_xticklabels(['Even', 'Odd'])
    ax1.set_ylabel("Energy")
    ax1.set_title(f"Parity-resolved spectrum ({n}-dot)")
    ax1.legend(frameon=False)

    # -----------------------------
    # Bottom: QD–SC–QD schematic
    # -----------------------------
    ax2.set_xlim(-0.5, n-0.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.axis('off')

    dot_y = 0
    sc_height = 0.5
    sc_width = 0.8

    for i in range(n):
        ax2.scatter(i, dot_y, s=800, color='tab:blue', edgecolor='black', zorder=3)
        ax2.text(i, dot_y-0.5, f"$\\epsilon_{i}$={eps[i]:.2f}", ha='center', fontsize=9)

    for i in range(n-1):
        x = (i + (i+1))/2
        rect = plt.Rectangle((x - sc_width/2, dot_y - sc_height/2), #type: ignore
                             sc_width, sc_height,
                             color='lightgray', ec='black', lw=1.2, zorder=2)
        ax2.add_patch(rect)
        ax2.text(x, dot_y, f"t={float(t[i]):.2f}\nΔ={float(Delta[i]):.2f}", 
                 ha='center', va='center', fontsize=8)
        ax2.text(x, dot_y + 0.6, f"$U_{i}$={float(U[i]):.2f}", ha='center', fontsize=9, color='tab:purple')

    ax2.set_title("Optimized QD–SC–QD setup")

    plt.tight_layout()
    plt.show()



def analyze(model, θ):
    """
    Main analysis wrapper:
    - Extract phys params
    - Build Hamiltonian
    - Compute spectrum
    - Plot everything
    """
    phys = extract_physical_params(model, θ)
    n = phys["eps"].numel()

    evals, evecs, even_states, odd_states, even_vecs, odd_vecs = compute_spectrum(model, phys)

    # Print energy difference
    if len(even_states) == len(odd_states):
        total_delta = torch.sum(torch.abs(even_states - odd_states)).item()
        print(f"Total energy |Ee - Eo| difference = {total_delta:.6e}")
    else:
        print("Warning: uneven number of even/odd states.")

    # Charge difference
    charge_diff = charge_difference_torch(even_vecs, odd_vecs, n).item()
    print(f"Charge difference (even - odd) = {charge_diff:.6e}")

    # Plots
    plot_majorana_polarization(even_vecs, odd_vecs, n)
    combined_analysis_plot(n, evals, even_states, odd_states, even_vecs, odd_vecs, phys)
    # parity_spectrum_plot(n, evals, even_states, odd_states)
    # plot_setup_schematic(phys,n)


def write_configs_to_file(theta_dict, n, parameter_configs=None, filename="configuration.json", loss=None):
    """
    Append a new configuration block to the file.
    The file becomes a chronological log of configurations:
    
    ## configuration N- site system | Loss: ...
    { ... JSON ... }

    ## configuration N- site system | Loss: ...
    { ... JSON ... }
    """
    from hamiltonian import HamiltonianModel
    import os
    import json

    model = HamiltonianModel(n=n, param_configs=parameter_configs)

    # Convert dict → tensor
    theta_tensor = model.dict_to_tensor(theta_dict)

    # Get adjusted → physical parameters
    adjusted = model.adjust_tensor(theta_tensor)
    param_dict = model.tensor_to_dict(adjusted)
    physical_params = model.get_physical_parameters(param_dict)

    # Compute loss
    if loss is None:
        H = model.build(physical_params)
        P = parity_operator_torch(n)
        loss = model.loss(H, P).item()

    # Build output block
    entry = {
        "header": f"configuration {n}- site system | Loss: {loss:.6e}",
        "parameter_configs": parameter_configs,
        "raw_parameter_values": theta_dict,
        "physical_parameters": {
            key: [float(v) for v in value] for key, value in physical_params.items()
        }
    }

    # Load existing list or create new
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, "r") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("The JSON file must contain a list.")
    else:
        data = []

    # Check for duplicates
    duplicate_found = any(
        d.get("raw_parameter_values") == entry["raw_parameter_values"] and
        d.get("parameter_configs") == entry["parameter_configs"]
        for d in data
    )

    if duplicate_found:
        print("⚠ Configuration already exists — skipping save.")
        return

    # Append new entry and save
    data.append(entry)
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"✔ Configuration appended to valid JSON file: {filename}")