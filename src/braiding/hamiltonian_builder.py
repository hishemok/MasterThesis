from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    from .get_setup import pull_configurations
except ImportError:
    from get_setup import pull_configurations


SubsystemRef = int | str
SiteRef = tuple[SubsystemRef, int]
Coupling = tuple[SiteRef | None, SiteRef | None, float, float]
OnsiteOverride = tuple[SubsystemRef, int, float]


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configuration.json"


def tensor_product(*matrices: np.ndarray) -> np.ndarray:
    result = np.asarray(matrices[0], dtype=np.complex128)
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result


def sigma_site(site: int, n_sites: int, operator: np.ndarray) -> np.ndarray:
    identity = np.eye(2, dtype=np.complex128)
    ops = [identity] * n_sites
    ops[site] = operator
    return tensor_product(*ops)


def creation_annihilation(site: int, n_sites: int) -> tuple[np.ndarray, np.ndarray]:
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    f_dag = 0.5 * (sigma_site(site, n_sites, sigma_x) + 1j * sigma_site(site, n_sites, sigma_y))
    f = 0.5 * (sigma_site(site, n_sites, sigma_x) - 1j * sigma_site(site, n_sites, sigma_y))

    jw_string = np.eye(2**n_sites, dtype=np.complex128)
    for idx in range(site):
        jw_string = jw_string @ sigma_site(idx, n_sites, -sigma_z)

    return jw_string @ f_dag, jw_string @ f


def precompute_ops(n_sites: int) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    creators: list[np.ndarray] = []
    annihilators: list[np.ndarray] = []
    numbers: list[np.ndarray] = []

    for site in range(n_sites):
        creator, annihilator = creation_annihilation(site, n_sites)
        creators.append(creator)
        annihilators.append(annihilator)
        numbers.append(creator @ annihilator)

    return creators, annihilators, numbers


def _normalize_parameter(values: Sequence[float] | np.ndarray, expected_length: int, name: str) -> np.ndarray:
    array = np.atleast_1d(np.asarray(values, dtype=float))
    if array.size == 1:
        array = np.repeat(array, expected_length)
    if array.size != expected_length:
        raise ValueError(f"{name} must have length 1 or {expected_length}, got {array.size}.")
    return array


@dataclass
class BraidingHamiltonianBuilder:
    n_sites: int = 3
    dupes: int = 3
    specified_vals: dict | None = field(default_factory=lambda: {"U": [0.1]})
    configs: dict | None = None
    config_path: str | Path = field(default_factory=default_config_path)
    t: Sequence[float] | np.ndarray | None = None
    U: Sequence[float] | np.ndarray | None = None
    eps: Sequence[float] | np.ndarray | None = None
    Delta: Sequence[float] | np.ndarray | None = None
    selection: dict | None = field(init=False, default=None)
    _operators: dict | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.n_sites < 2:
            raise ValueError("n_sites must be at least 2.")
        if self.dupes < 1:
            raise ValueError("dupes must be at least 1.")

        self.config_path = Path(self.config_path).expanduser()
        defaults = self._load_defaults() if any(value is None for value in (self.t, self.U, self.eps, self.Delta)) else {}

        self.t = _normalize_parameter(self.t if self.t is not None else defaults["t"], self.n_sites - 1, "t")
        self.U = _normalize_parameter(self.U if self.U is not None else defaults["U"], self.n_sites - 1, "U")
        self.eps = _normalize_parameter(self.eps if self.eps is not None else defaults["eps"], self.n_sites, "eps")
        self.Delta = _normalize_parameter(
            self.Delta if self.Delta is not None else defaults["Delta"],
            self.n_sites - 1,
            "Delta",
        )

    @property
    def parameters(self) -> dict[str, np.ndarray]:
        return {
            "t": self.t.copy(),
            "U": self.U.copy(),
            "eps": self.eps.copy(),
            "Delta": self.Delta.copy(),
        }

    def _load_defaults(self) -> dict[str, Sequence[float]]:
        match = pull_configurations(
            self.n_sites,
            configs=self.configs,
            specified_vals=self.specified_vals,
            path=str(self.config_path),
        )

        if not match:
            raise ValueError(
                "No matching configuration found for "
                f"n_sites={self.n_sites}, specified_vals={self.specified_vals}, configs={self.configs}."
            )

        loss, configuration, physical_params = match
        self.selection = {
            "loss": loss,
            "configuration": configuration,
            "physical_parameters": physical_params,
            "config_path": str(self.config_path),
        }
        return physical_params

    def _resolve_subsystem(self, which: SubsystemRef) -> int:
        if isinstance(which, int):
            if 0 <= which < self.dupes:
                return which
            raise ValueError(f"Subsystem index must be between 0 and {self.dupes - 1}, got {which}.")

        label = which.strip().upper()
        if len(label) != 1:
            raise ValueError(f"Invalid subsystem label {which!r}.")

        index = ord(label) - ord("A")
        if 0 <= index < self.dupes:
            return index
        raise ValueError(f"Subsystem label must be between 'A' and '{chr(ord('A') + self.dupes - 1)}'.")

    def _flatten_site(self, site_ref: SiteRef) -> int:
        subsystem, site = site_ref
        subsystem_index = self._resolve_subsystem(subsystem)
        if not 0 <= site < self.n_sites:
            raise ValueError(f"Site index must be between 0 and {self.n_sites - 1}, got {site}.")
        return subsystem_index * self.n_sites + site

    def _build_eps_full(self, onsite_overrides: Iterable[OnsiteOverride] | None) -> np.ndarray:
        eps_full = np.tile(self.eps, (self.dupes, 1))
        if onsite_overrides is None:
            return eps_full

        for subsystem, site, value in onsite_overrides:
            subsystem_index = self._resolve_subsystem(subsystem)
            if not 0 <= site < self.n_sites:
                raise ValueError(f"Site index must be between 0 and {self.n_sites - 1}, got {site}.")
            eps_full[subsystem_index, site] = value

        return eps_full

    def get_operators(self) -> dict[str, dict | list[np.ndarray]]:
        if self._operators is not None:
            return self._operators


        total_sites = self.n_sites * self.dupes
        creators, annihilators, numbers = precompute_ops(total_sites)
        hop_ops: dict[tuple[int, int], np.ndarray] = {}
        pair_ops: dict[tuple[int, int], np.ndarray] = {}
        density_ops: dict[tuple[int, int], np.ndarray] = {}

        for left in range(total_sites):
            for right in range(left + 1, total_sites):
                hop_ops[(left, right)] = creators[left] @ annihilators[right] + creators[right] @ annihilators[left]
                pair_ops[(left, right)] = creators[left] @ creators[right] + annihilators[right] @ annihilators[left]
                density_ops[(left, right)] = numbers[left] @ numbers[right]

        self._operators = {
            "cre": creators,
            "ann": annihilators,
            "num": numbers,
            "hop": hop_ops,
            "pair": pair_ops,
            "dens": density_ops,
        }
        return self._operators

    def _build_hamiltonian(
        self,
        active_subsystems: Iterable[SubsystemRef],
        couplings: Iterable[Coupling] = (),
        onsite_overrides: Iterable[OnsiteOverride] | None = None,
    ) -> np.ndarray:
        operators = self.get_operators()
        active = {self._resolve_subsystem(which) for which in active_subsystems}
        total_sites = self.n_sites * self.dupes
        dim = 2**total_sites
        hamiltonian = np.zeros((dim, dim), dtype=np.complex128)
        eps_full = self._build_eps_full(onsite_overrides)

        for subsystem in active:
            offset = subsystem * self.n_sites
            for bond in range(self.n_sites - 1):
                left = offset + bond
                right = offset + bond + 1
                key = (left, right)
                hamiltonian += -self.t[bond] * operators["hop"][key]
                hamiltonian += self.Delta[bond] * operators["pair"][key]
                hamiltonian += self.U[bond] * operators["dens"][key]

            for site in range(self.n_sites):
                hamiltonian += eps_full[subsystem, site] * operators["num"][offset + site]

        for site_a, site_b, t_couple, delta_couple in couplings:
            if site_a is None or site_b is None:
                continue

            left = self._flatten_site(site_a)
            right = self._flatten_site(site_b)
            if left == right:
                raise ValueError("Coupling endpoints must refer to different sites.")

            key = (min(left, right), max(left, right))
            if t_couple != 0:
                hamiltonian += -float(t_couple) * operators["hop"][key]
            if delta_couple != 0:
                hamiltonian += float(delta_couple) * operators["pair"][key]

        return 0.5 * (hamiltonian + hamiltonian.conj().T)

    def subsystem_hamiltonian(
        self,
        which: SubsystemRef,
        onsite_overrides: Iterable[OnsiteOverride] | None = None,
    ) -> np.ndarray:
        """
        Build one subsystem Hamiltonian embedded in the full Hilbert space.

        This reproduces the notebook's "Subsystem Hamiltonian" construction:
        only the selected subsystem contributes intra-chain terms, but the
        matrix still acts on the full `2 ** (n_sites * dupes)` Hilbert space.
        """
        return self._build_hamiltonian([which], onsite_overrides=onsite_overrides)

    def full_system_hamiltonian(
        self,
        couplings: Iterable[Coupling] = (),
        onsite_overrides: Iterable[OnsiteOverride] | None = None,
    ) -> np.ndarray:
        """
        Build the full system Hamiltonian.

        With the default arguments, this is just the sum of the uncoupled
        subsystem Hamiltonians. Pass `couplings` or `onsite_overrides` to add
        extra junction terms or detuning by hand.
        """
        return self._build_hamiltonian(
            range(self.dupes),
            couplings=couplings,
            onsite_overrides=onsite_overrides,
        )


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
