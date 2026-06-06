from dataclasses import dataclass

import numpy as np

from get_mzm_JW import precompute_operators
from hamiltonian_builder import BraidingHamiltonianBuilder, default_config_path
from step_projected_braiding_local import parity_op


@dataclass(frozen=True)
class PairFit:
    fit: np.ndarray
    coeffs: np.ndarray
    error: float


@dataclass(frozen=True)
class BlockwiseFit:
    fit: np.ndarray
    error: float
    offblock_error: float
    block_fits: list[PairFit]


def hermitian_part(matrix):
    return 0.5 * (matrix + matrix.conj().T)


def relative_error(actual, expected, eps=1e-14):
    return np.linalg.norm(actual - expected, ord="fro") / max(
        np.linalg.norm(expected, ord="fro"),
        eps,
    )


def build_single_hamiltonian(specified_vals):
    builder = BraidingHamiltonianBuilder(
        n_sites=3,
        dupes=1,
        specified_vals=specified_vals,
        config_path=default_config_path(),
    )
    return builder.full_system_hamiltonian()


def split_eigenstates_by_parity(eigvals, eigvecs, parity, parity_tol=1e-6):
    parities = np.sum(eigvecs.conj() * (parity @ eigvecs), axis=0).real
    even_mask = parities > parity_tol
    odd_mask = parities < -parity_tol

    if np.any(~(even_mask | odd_mask)):
        bad = parities[~(even_mask | odd_mask)]
        raise ValueError(f"Could not classify parity values {bad}.")

    return (
        eigvals[even_mask],
        eigvals[odd_mask],
        eigvecs[:, even_mask],
        eigvecs[:, odd_mask],
    )


def pair_even_odd_by_energy(even_energies, odd_energies):
    pairs = []
    used_odd = set()

    for even_idx, even_energy in enumerate(even_energies):
        available_odd = [idx for idx in range(len(odd_energies)) if idx not in used_odd]
        odd_idx = min(available_odd, key=lambda idx: abs(odd_energies[idx] - even_energy))
        used_odd.add(odd_idx)
        pairs.append((even_idx, odd_idx))

    return pairs


def create_single_label(specified_vals, levels):
    h_single = build_single_hamiltonian(specified_vals)
    eigvals, eigvecs = np.linalg.eigh(h_single)

    ops = precompute_operators(n=3, dup=1)
    parity = parity_op(ops, sites=3)
    even_energies, odd_energies, even_vecs, odd_vecs = split_eigenstates_by_parity(
        eigvals,
        eigvecs,
        parity,
    )

    pairs = pair_even_odd_by_energy(even_energies, odd_energies)[:levels]
    label_operator = np.zeros_like(h_single, dtype=complex)

    for label, (even_idx, odd_idx) in enumerate(pairs):
        even_vec = even_vecs[:, even_idx]
        odd_vec = odd_vecs[:, odd_idx]
        pair_projector = (
            np.outer(even_vec, even_vec.conj())
            + np.outer(odd_vec, odd_vec.conj())
        )
        label_operator += label * pair_projector

    return hermitian_part(label_operator), len(pairs)


def tensor_prod(matrices):
    result = np.asarray(matrices[0], dtype=complex)
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result


def create_full_label(n_subsystems, levels, specified_vals):
    single_label, n_labels = create_single_label(specified_vals, levels)
    dim = single_label.shape[0]
    identity = np.eye(dim, dtype=complex)

    # Base notation makes every tuple (A_label, B_label, C_label) unique.
    base = n_labels + 1
    full_label = np.zeros((dim**n_subsystems, dim**n_subsystems), dtype=complex)

    for subsystem in range(n_subsystems):
        factors = [identity] * n_subsystems
        factors[subsystem] = single_label
        full_label += base**subsystem * tensor_prod(factors)

    return hermitian_part(full_label), base


def project_full_label(full_label, basis):
    return hermitian_part(basis.conj().T @ full_label @ basis)


def decode_label(value, base, n_subsystems):
    value = int(round(value.real))
    labels = []

    for _ in range(n_subsystems):
        labels.append(value % base)
        value //= base

    return tuple(labels)


def find_blocks(basis, specified_vals, levels, n_subsystems=3, tol=1e-6):
    full_label, base = create_full_label(n_subsystems, levels, specified_vals)

    if basis.shape[0] != full_label.shape[0]:
        raise ValueError(
            f"Basis must have shape ({full_label.shape[0]}, group_dim), "
            f"got {basis.shape}."
        )

    eigenvalues, eigenvectors = np.linalg.eigh(project_full_label(full_label, basis))

    blocks = []
    start = 0
    for stop in range(1, len(eigenvalues) + 1):
        finished = stop == len(eigenvalues)
        separated = (
            not finished
            and abs(eigenvalues[stop] - eigenvalues[start]) > tol
        )
        if not (finished or separated):
            continue

        value = np.mean(eigenvalues[start:stop]).real
        label = decode_label(value, base, n_subsystems)
        vectors = eigenvectors[:, start:stop]
        blocks.append((label, vectors))
        start = stop

    return blocks


def block_diag(matrices):
    total_dim = sum(matrix.shape[0] for matrix in matrices)
    result = np.zeros((total_dim, total_dim), dtype=complex)

    start = 0
    for matrix in matrices:
        stop = start + matrix.shape[0]
        result[start:stop, start:stop] = matrix
        start = stop

    return result


def fit_to_majorana_pair(local_operator, gamma1, gamma2):
    design = np.column_stack([gamma1.reshape(-1), gamma2.reshape(-1)])
    coeffs = np.linalg.lstsq(design, local_operator.reshape(-1), rcond=None)[0]
    fit = hermitian_part(coeffs[0] * gamma1 + coeffs[1] * gamma2)
    return PairFit(
        fit=fit,
        coeffs=coeffs,
        error=relative_error(fit, local_operator),
    )


def fit_blockwise_to_majorana_pair(local_operator, gamma1, gamma2, blocks):
    rotation = np.column_stack([vectors for _, vectors in blocks])
    identity = np.eye(local_operator.shape[0], dtype=complex)

    if rotation.shape != local_operator.shape:
        raise ValueError("Blocks do not span the full operator space.")
    if np.linalg.norm(rotation @ rotation.conj().T - identity) > 1e-6:
        raise ValueError("Block vectors are not a complete orthonormal basis.")

    block_fits = []
    local_blocks = []
    for _, vectors in blocks:
        local_block = vectors.conj().T @ local_operator @ vectors
        gamma1_block = vectors.conj().T @ gamma1 @ vectors
        gamma2_block = vectors.conj().T @ gamma2 @ vectors

        local_blocks.append(local_block)
        block_fits.append(
            fit_to_majorana_pair(local_block, gamma1_block, gamma2_block)
        )

    fit = hermitian_part(
        rotation @ block_diag([result.fit for result in block_fits]) @ rotation.conj().T
    )
    block_part = hermitian_part(
        rotation @ block_diag(local_blocks) @ rotation.conj().T
    )

    return BlockwiseFit(
        fit=fit,
        error=relative_error(fit, local_operator),
        offblock_error=relative_error(block_part, local_operator),
        block_fits=block_fits,
    )


def format_blocks(blocks):
    return ", ".join(f"{label}:{vectors.shape[1]}" for label, vectors in blocks)
