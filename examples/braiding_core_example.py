import numpy as np

from braiding_core import calculate_parities, evolve_system, hamiltonian, precompute_operators


def main() -> None:
    n_sites = 3
    dupes = 3

    t_vals = np.array([1.0, 0.9])
    u_vals = np.array([0.2, 0.2])
    eps_vals = np.array([0.0, 0.05, 0.0])
    delta_vals = np.array([0.7, 0.7])
    params = (t_vals, u_vals, eps_vals, delta_vals)

    operators = precompute_operators(n_sites, dupes)
    h0 = hamiltonian(n_sites, dupes, *params, operators=operators)
    eigvals, eigvecs = np.linalg.eigh(h0)
    even_energies, odd_energies, *_ = calculate_parities(eigvecs, eigvals, operators["num"])

    print("Imported braiding_core successfully")
    print(f"H0 shape: {h0.shape}")
    print(f"Parity split: even={len(even_energies)}, odd={len(odd_energies)}")
    print(f"Lowest even energy: {even_energies[0]:.6f}")
    print(f"Lowest odd energy: {odd_energies[0]:.6f}")

    total_time = 6.0
    time_steps = 3
    t_j = 0.8
    d_j = 0.6
    eps_detune_val = 1.1

    time_array, couplings, detunings, spectrum, states, gap = evolve_system(
        total_time,
        time_steps,
        params,
        t_j,
        d_j,
        eps_detune_val,
        n_sites,
        dupes,
        operators=operators,
        show_progress=False,
    )

    assert spectrum.shape == (time_steps, 2 ** (n_sites * dupes))
    assert states.shape == (time_steps, 2 ** (n_sites * dupes), 2 ** (n_sites * dupes))
    assert len(time_array) == time_steps
    assert len(couplings) == time_steps
    assert len(detunings) == time_steps
    assert np.isfinite(spectrum).all()

    print(f"Time grid: {time_array}")
    print(f"Spectrum shape: {spectrum.shape}")
    print(f"First AB pulse: {couplings[0][0][2]:.6f}")
    print(f"Last detuning value: {detunings[-1][0][2]:.6f}")
    print("Example completed successfully")


if __name__ == "__main__":
    main()
