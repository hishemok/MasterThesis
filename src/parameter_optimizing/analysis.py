import numpy as np
import matplotlib.pyplot as plt
import torch

from operators import precompute_ops, parity_operator, from_torch
from measurements import calculate_parities, charge_difference, Majorana_polarization

def print_params(theta, n, fixed_params=None):
    fixed_params = fixed_params or {}
    idx = 0

    if 't' not in fixed_params:
        t = theta[idx]; idx += 1
    else:
        t = fixed_params['t']

    if 'U' not in fixed_params:
        U = theta[idx:idx+n-1]; idx += n-1
    else:
        U = fixed_params['U']

    if 'eps' not in fixed_params:
        eps = theta[idx:idx+n]; idx += n
    else:
        eps = fixed_params['eps']

    if 'Delta' not in fixed_params:
        Delta = theta[idx]
    else:
        Delta = fixed_params['Delta']

    print(f"t = {t}")
    print(f"U = {U}")
    print(f"ε = {eps}")
    print(f"Δ = {Delta}")


def plot_parity_spectrum(n, theta, model):
    parameters = {}
    parameters['t'] = theta[0]
    parameters['U'] = torch.tensor(theta[1:1+(n-1)], dtype=torch.float64)
    parameters['eps'] = torch.tensor(theta[1+(n-1):1+(n-1)+n], dtype=torch.float64)
    parameters['Delta'] = theta[-1] 

    H_torch = model.build(parameters)
    H = from_torch(H_torch)

    evals, evecs = np.linalg.eigh(H)

    P_np = parity_operator(n)
    parities = [np.real(np.vdot(ev, P_np @ ev)) for ev in evecs.T]
    even_vecs = []
    odd_vecs = []
    print(f"{n}-dot system eigenvalues and their parity sectors:")
    for i, (E,p) in enumerate(zip(evals, parities)):
        if abs(p) < .9:
            print(f"Warning: Eigenvalue {E:.4f} has ambiguous parity {p:.4f}")
        sector = "even" if p > .9 else "odd"
        if sector == "even":
            even_vecs.append(evecs[:, i])
        else:
            odd_vecs.append(evecs[:, i])
        print(f"Eigenvalue {E:.4f} belongs to {sector} parity sector, with parity {p:.4f}")
    print("")
    odd_vecs = np.array(odd_vecs).T
    even_vecs = np.array(even_vecs).T


    MP_np = Majorana_polarization(even_vecs, odd_vecs, n, model)
    Majorana_metric = MP_np


    for i in range(Majorana_metric.shape[0]):
        plt.plot(range(n), Majorana_metric[i], 'o-', label=f'|M̃|_{i}')
    plt.xlabel('Site index')
    plt.ylabel('Majorana localization')
    plt.title(f'Majorana localization profile for {n}-dot system')
    plt.axhline(0, color='k', linestyle='--')
    plt.legend()
    plt.show()

    evens = [(E, v) for E, v, p in zip(evals, evecs.T, parities) if p >= 0]
    odds  = [(E, v) for E, v, p in zip(evals, evecs.T, parities) if p <  0]
    y_even = [E for E,_ in evens]
    y_odd = [E for E,_ in odds]

    Total_energy_difference = sum([abs(Ee - Eo) for Ee, Eo in zip(y_even, y_odd)])
    print(f"Total energy difference between even and odd states: {Total_energy_difference:.6e}\n")
    degeneracy_lines = []
    for i, Ee in enumerate(y_even):
        Eo = y_odd[i]
        if abs(Ee - Eo) < 1e-2:
            degeneracy_lines.append(Ee)


    charge_diff = charge_difference(even_vecs, odd_vecs, n)


    print(f"Charge difference between even and odd states: {charge_diff:.6e}\n")


    t = theta[0]
    U = theta[1:1+(n-1)]
    eps = theta[1+(n-1):1+(n-1)+n]
    Delta = theta[-1]

    fig, axs = plt.subplots(2, 1, figsize=(10,4), gridspec_kw={'height_ratios':[2,1]})
    ax1, ax2 = axs

    # -----------------------------
    # Left panel: parity-resolved energy spectrum
    # -----------------------------
    ax1.hlines(y_even, -0.2, 0.2, color='tab:blue', label='Even')
    ax1.hlines(y_odd,  0.8, 1.2, color='tab:red', label='Odd')
    ax1.hlines(degeneracy_lines, 0.2, 0.8, color='tab:gray', linestyles='dashed', label='Degeneracies')
    ax1.set_xticks([0,1])
    ax1.set_xticklabels(['Even','Odd'])
    ax1.set_title(f"Parity-resolved spectrum ({n}-dot)")
    ax1.set_ylabel("Energy")
    ax1.legend(frameon=False)

    # -----------------------------
    # Right panel: schematic of QD–SC–QD chain
    # -----------------------------
    ax2.set_xlim(-0.5, n-0.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.axis('off')

    dot_y = 0
    sc_height = 0.5
    sc_width = 0.8

    # Draw dots (quantum dots)
    for i in range(n):
        ax2.scatter(i, dot_y, s=800, color='tab:blue', edgecolor='black', zorder=3)
        ax2.text(i, dot_y-0.5, f"$\\epsilon_{i}$={eps[i]:.2f}", ha='center', fontsize=9)

    # Draw superconductors (rectangles between dots)
    for i in range(n-1):
        x = (i + (i+1))/2
        rect = plt.Rectangle((x - sc_width/2, dot_y - sc_height/2), sc_width, sc_height, # type: ignore
                             color='lightgray', ec='black', lw=1.2, zorder=2)
        ax2.add_patch(rect)

        # Coupling t and Δ inside SC
        ax2.text(x, dot_y, f"t={t:.2f}\nΔ={Delta:.2f}", ha='center', va='center', fontsize=8)

        # U label above connection
        ax2.text(x, dot_y+0.6, f"$U_{i}$={U[i]:.2f}", ha='center', fontsize=9, color='tab:purple')

    ax2.set_title("Optimized QD–SC–QD setup")

    plt.tight_layout()
    plt.show()
