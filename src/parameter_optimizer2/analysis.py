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
    parity_spectrum_plot(n, evals, even_states, odd_states)
    plot_majorana_polarization(even_vecs, odd_vecs, n)
    plot_setup_schematic(phys,n)
