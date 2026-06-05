from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LabelBlock:
    label: tuple[int, ...]
    vectors: np.ndarray
    label_value: float


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
    blocks: list[LabelBlock]
    block_fits: list[PairFit]


def hermitian_part(matrix):
    return 0.5 * (matrix + matrix.conj().T)


def relative_error(actual, expected, eps=1e-14):
    return np.linalg.norm(actual - expected, ord="fro") / max(np.linalg.norm(expected, ord="fro"), eps)


def tensor_product(*matrices):
    result = np.asarray(matrices[0], dtype=complex)
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result


def block_diag(matrices):
    total_dim = sum(matrix.shape[0] for matrix in matrices)
    result = np.zeros((total_dim, total_dim), dtype=complex)

    start = 0
    for matrix in matrices:
        stop = start + matrix.shape[0]
        result[start:stop, start:stop] = matrix
        start = stop

    return result


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


def build_pair_label_operator(single_hamiltonian, single_parity, levels_to_include=4):
    eigvals, eigvecs = np.linalg.eigh(single_hamiltonian)
    even_energies, odd_energies, even_vecs, odd_vecs = split_eigenstates_by_parity(
        eigvals,
        eigvecs,
        single_parity,
    )
    pairs = pair_even_odd_by_energy(even_energies, odd_energies)

    dim = single_hamiltonian.shape[0]
    label_operator = np.zeros((dim, dim), dtype=complex)
    n_labels = min(levels_to_include, len(pairs))

    for label, (even_idx, odd_idx) in enumerate(pairs[:n_labels]):
        even_vec = even_vecs[:, even_idx]
        odd_vec = odd_vecs[:, odd_idx]
        projector = np.outer(even_vec, even_vec.conj()) + np.outer(odd_vec, odd_vec.conj())
        label_operator += label * projector

    return hermitian_part(label_operator), n_labels


def build_full_label_operator(single_label_operator, n_labels, n_subsystems=3):
    identity = np.eye(single_label_operator.shape[0], dtype=complex)
    label_base = n_labels + 1
    full_label_operator = np.zeros(
        (single_label_operator.shape[0] ** n_subsystems, single_label_operator.shape[0] ** n_subsystems),
        dtype=complex,
    )

    for subsystem in range(n_subsystems):
        factors = [identity] * n_subsystems
        factors[subsystem] = single_label_operator
        full_label_operator += label_base**subsystem * tensor_product(*factors)

    return hermitian_part(full_label_operator), label_base


def decode_label(label_value, label_base, n_subsystems=3):
    label_int = int(round(float(np.real(label_value))))
    labels = []

    for _ in range(n_subsystems):
        labels.append(label_int % label_base)
        label_int //= label_base

    return tuple(labels)


def split_group_by_label_operator(group_basis, label_operator, label_base, n_subsystems=3, label_tol=1e-6):
    label_subspace = hermitian_part(group_basis.conj().T @ label_operator @ group_basis)
    label_eigvals, label_eigvecs = np.linalg.eigh(label_subspace)
    order = np.argsort(label_eigvals)
    label_eigvals = label_eigvals[order]
    label_eigvecs = label_eigvecs[:, order]

    blocks = []
    start = 0
    for stop in range(1, len(label_eigvals) + 1):
        finished = stop == len(label_eigvals)
        separated = not finished and abs(label_eigvals[stop] - label_eigvals[start]) > label_tol
        if not (finished or separated):
            continue

        value = float(np.mean(label_eigvals[start:stop]).real)
        blocks.append(
            LabelBlock(
                label=decode_label(value, label_base, n_subsystems=n_subsystems),
                vectors=label_eigvecs[:, start:stop],
                label_value=value,
            )
        )
        start = stop

    return blocks


def project_to_label_blocks(operator, blocks):
    block_vectors = [block.vectors for block in blocks]
    rotation = np.column_stack(block_vectors)
    projected_blocks = [block.vectors.conj().T @ operator @ block.vectors for block in blocks]
    return rotation @ block_diag(projected_blocks) @ rotation.conj().T


def fit_to_majorana_pair(local_operator, gamma1, gamma2):
    design = np.column_stack([gamma1.reshape(-1), gamma2.reshape(-1)])
    coeffs = np.linalg.lstsq(design, local_operator.reshape(-1), rcond=None)[0]
    fit = coeffs[0] * gamma1 + coeffs[1] * gamma2
    fit = hermitian_part(fit)
    return PairFit(fit=fit, coeffs=coeffs, error=relative_error(fit, local_operator))


def fit_blockwise_to_majorana_pair(local_operator, gamma1, gamma2, blocks):
    block_vectors = [block.vectors for block in blocks]
    rotation = np.column_stack(block_vectors)

    if rotation.shape != local_operator.shape:
        raise ValueError("Label blocks do not span the full operator space.")

    coverage_error = np.linalg.norm(rotation @ rotation.conj().T - np.eye(local_operator.shape[0], dtype=complex))
    if coverage_error > 1e-6:
        raise ValueError(f"Label block basis is not complete: coverage error {coverage_error:.3e}.")

    block_fits = []
    for block in blocks:
        vectors = block.vectors
        local_block = vectors.conj().T @ local_operator @ vectors
        gamma1_block = vectors.conj().T @ gamma1 @ vectors
        gamma2_block = vectors.conj().T @ gamma2 @ vectors
        block_fits.append(fit_to_majorana_pair(local_block, gamma1_block, gamma2_block))

    fit_in_block_basis = block_diag([fit.fit for fit in block_fits])
    fit = hermitian_part(rotation @ fit_in_block_basis @ rotation.conj().T)
    block_part = project_to_label_blocks(local_operator, blocks)

    return BlockwiseFit( fit=fit, error=relative_error(fit, local_operator), offblock_error=relative_error(block_part, local_operator), blocks=blocks, block_fits=block_fits,
    )


def format_blocks(blocks):
    return ", ".join(f"{block.label}:{block.vectors.shape[1]}" for block in blocks)
