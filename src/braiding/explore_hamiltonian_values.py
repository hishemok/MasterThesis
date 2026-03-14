from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from hamiltonian_builder import BraidingHamiltonianBuilder, default_config_path
import numpy as np
import matplotlib.pyplot as plt

try:
    from .get_setup import pull_configurations
except ImportError:
    from get_setup import pull_configurations


def calculate_parities_optimized(eigvecs, eigvals, P, tol=1e-12):
    """
    Splits eigenstates into even/odd parity sectors.

    Parameters
    ----------
    eigvecs : (dim, dim) complex ndarray
        Eigenvectors (columns).
    eigvals : (dim,) ndarray
        Corresponding eigenvalues.
    P : (dim, dim) ndarray
        Parity operator.
    tol : float
        Numerical tolerance for parity classification.

    Returns
    -------
    even_energies, odd_energies,
    even_vecs, odd_vecs
    """

    # Compute P|ψ>
    Pv = P @ eigvecs

    # Compute ⟨ψ|P|ψ⟩ for each eigenvector
    parities = np.sum(eigvecs.conj() * Pv, axis=0).real

    # Robust classification (important if near zero numerically)
    even_mask = parities > tol
    odd_mask  = parities < -tol

    # Slice directly with boolean masks (fast, vectorized)
    even_energies = eigvals[even_mask]
    odd_energies  = eigvals[odd_mask]

    even_vecs = eigvecs[:, even_mask]
    odd_vecs  = eigvecs[:, odd_mask]

    return even_energies, odd_energies, even_vecs, odd_vecs

def get_parity_operator(system, operators):
    """
    Returns the parity operator for subsystem A, B, C, or Full.
    P = ∏_i (I - 2 n_i)
    """

    num = operators['num']
    dim = num[0].shape[0]
    I = np.eye(dim, dtype=complex)

    if system == 'A':
        sites = range(0, 3)
    elif system == 'B':
        sites = range(3, 6)
    elif system == 'C':
        sites = range(6, 9)
    elif system == 'Full':
        sites = range(9)
    else:
        raise ValueError("System must be 'A', 'B', 'C', or 'Full'.")

    P = I.copy()
    for i in sites:
        P = P @ (I - 2 * num[i])

    return P


if __name__ == "__main__":
    builder = BraidingHamiltonianBuilder(
    n_sites=3,
    dupes=3,
    specified_vals={"U": [0.1]},
    config_path=default_config_path(),
)

    h_a = builder.subsystem_hamiltonian("A")
    h_b = builder.subsystem_hamiltonian("B")
    h_c = builder.subsystem_hamiltonian("C")
    h_full = builder.full_system_hamiltonian()

    print("Loaded parameters:")
    for name, values in builder.parameters.items():
        print(f"  {name} = {values}")

    if builder.selection is not None:
        print(f"Selected configuration loss: {builder.selection['loss']:.6e}")

    print("H_A shape:", h_a.shape)
    print("H_full shape:", h_full.shape)
    print("||H_full - (H_A + H_B + H_C)|| =", np.linalg.norm(h_full - (h_a + h_b + h_c)))

    h_a_eigvals, h_a_eigvecs = np.linalg.eigh(h_a)
    h_b_eigvals, h_b_eigvecs = np.linalg.eigh(h_b)
    h_c_eigvals, h_c_eigvecs = np.linalg.eigh(h_c)
    h_full_eigvals, h_full_eigvecs = np.linalg.eigh(h_full)

    #Split parities
    P = get_parity_operator("Full", builder.get_operators())
    even_energies, odd_energies, even_vecs, odd_vecs = calculate_parities_optimized(h_full_eigvecs, h_full_eigvals, P)

    even_energies_A, odd_energies_A, _, _ = calculate_parities_optimized(h_a_eigvecs, h_a_eigvals, get_parity_operator("A", builder.get_operators()))
    even_energies_B, odd_energies_B, _, _ = calculate_parities_optimized(h_b_eigvecs, h_b_eigvals, get_parity_operator("B", builder.get_operators()))
    even_energies_C, odd_energies_C, _, _ = calculate_parities_optimized(h_c_eigvecs, h_c_eigvals, get_parity_operator("C", builder.get_operators()))


    plt.figure(figsize=(10,6))
    for i in range(len(even_energies_A)):
        plt.hlines(even_energies_A[i], -0.3, 0.3, color='tab:blue', label='Even A' if i==0 else "")
        plt.hlines(even_energies_A[i], -0.5, 5.5, color='gray', linestyles='--')
    for i in range(len(odd_energies_A)):
        plt.hlines(odd_energies_A[i], 0.7, 1.3, color='tab:red', label='Odd A' if i==0 else "")
    for i in range(len(even_energies_B)):

        plt.hlines(even_energies_B[i], 1.7, 2.3, color='tab:green', label='Even B' if i==0 else "")
    for i in range(len(odd_energies_B)):
        plt.hlines(odd_energies_B[i], 2.7, 3.3, color='tab:orange', label='Odd B' if i==0 else "")  
    for i in range(len(even_energies_C)):
        plt.hlines(even_energies_C[i], 3.7, 4.3, color='tab:purple', label='Even C' if i==0 else "")
    for i in range(len(odd_energies_C)):
        plt.hlines(odd_energies_C[i], 4.7, 5.3, color='tab:brown', label='Odd C' if i==0 else "")

    plt.ylim(min(even_energies_A.min(), even_energies_B.min(), even_energies_C.min()) - 0.1, max(odd_energies_A.max(), odd_energies_B.max(), odd_energies_C.max()) + 0.1)
    plt.xticks([0, 1, 2, 3, 4, 5], ["Even A", "Odd A", "Even B", "Odd B", "Even C", "Odd C"]) 
    plt.ylabel("Energy")
    plt.title(f"Parity-resolved spectrum of Subsystems A, B, C at T=0")    
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()