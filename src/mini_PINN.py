import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from n_particles_degen import n_dot_Hamiltonian, parity_operator

# -------------------------------
# Helper: degeneracy scoring
# -------------------------------
def degeneracy_score(H_np, P):
    """
    Compute how 'degenerate' the system is:
    Lower score = closer to perfect degeneracy between parity sectors.
    """
    eigvals, eigvecs = np.linalg.eigh(H_np)
    parities = [np.real_if_close(np.vdot(vec, P @ vec)) for vec in eigvecs.T]

    # Group energies by parity
    even_energies = [E for E, p in zip(eigvals, parities) if np.isclose(p, 1)]
    odd_energies  = [E for E, p in zip(eigvals, parities) if np.isclose(p, -1)]

    n_levels = min(len(even_energies), len(odd_energies))
    diffs = [abs(even_energies[i] - odd_energies[i]) for i in range(n_levels)]
    return max(diffs)

# -------------------------------
# Convert parameters to Hamiltonian and score degeneracy
# -------------------------------
def degeneracy_loss(theta, n, eps_vals, Δ_val):
    """
    theta = [t, U0, U1,...]
    """
    t_val = theta[0]
    U_vals = theta[1:]
    
    H_sym = n_dot_Hamiltonian(n)
    ϵs = sp.symbols(f'ϵ0:{n}')
    Us = sp.symbols(f'U0:{n-1}')
    Δ, t = sp.symbols('Δ t', real=True)

    subs_dict = {Δ: Δ_val, t: t_val}
    for i, eps in enumerate(eps_vals):
        subs_dict[ϵs[i]] = eps
    for i, U in enumerate(U_vals):
        subs_dict[Us[i]] = U

    H_num = H_sym.subs(subs_dict)
    H_np = np.array(H_num).astype(np.complex128)

    P = parity_operator(n)
    return degeneracy_score(H_np, P)

# -------------------------------
# Neural network generator
# -------------------------------
class ParamNet(nn.Module):
    def __init__(self, n_U):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_U + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_U + 1),
            nn.Sigmoid()  # [0,1]
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------
# Main function to run search
# -------------------------------
def find_degenerate_params(n, eps_vals, Δ_val, steps=1000):
    n_U = n - 1
    net = ParamNet(n_U)
    optimizer = optim.Adam(net.parameters(), lr=1e-2)

    best_loss = np.inf
    best_theta = None

    for step in range(steps):
        noise = torch.rand(1, n)  # batch of 1
        params = net(noise)
        
        # scale to reasonable ranges
        t_val = 0.5 + params[0,0].item() * 2.0
        U_vals = 0.5 + params[0,1:].detach().numpy() * 2.0
        theta = [t_val] + list(U_vals)

        # Evaluate degeneracy
        loss = degeneracy_loss(theta, n, eps_vals, Δ_val)

        if loss < best_loss:
            best_loss = loss
            best_theta = theta

        if step % 50 == 0:
            print(f"Step {step}, degeneracy loss = {loss:.4e}")

        # Dummy backward to satisfy optimizer (gradient-free)
        optimizer.zero_grad()
        torch.tensor(loss, requires_grad=True).backward()
        optimizer.step()

    print("\nBest degeneracy loss:", best_loss)
    print("Best parameters:")
    print(f"t = {best_theta[0]}")
    for i,U in enumerate(best_theta[1:]):
        print(f"U{i} = {U}")

    # Plot parity-resolved spectrum for best parameters
    H_sym = n_dot_Hamiltonian(n)
    ϵs = sp.symbols(f'ϵ0:{n}')
    Us = sp.symbols(f'U0:{n-1}')
    Δ, t = sp.symbols('Δ t', real=True)

    subs_dict = {Δ: Δ_val, t: best_theta[0]}
    for i, eps in enumerate(eps_vals):
        subs_dict[ϵs[i]] = eps
    for i, U in enumerate(best_theta[1:]):
        subs_dict[Us[i]] = U

    H_np = np.array(H_sym.subs(subs_dict)).astype(np.complex128)
    eigvals, eigvecs = np.linalg.eigh(H_np)
    P = parity_operator(n)
    parities = [np.real_if_close(np.vdot(vec, P @ vec)) for vec in eigvecs.T]

    even_states = [(E,v) for E,v,p in zip(eigvals, eigvecs.T, parities) if np.isclose(p,1)]
    odd_states  = [(E,v,p) for E,v,p in zip(eigvals, eigvecs.T, parities) if np.isclose(p,-1)]

    plt.figure(figsize=(6,4))
    y_even = [E for E,_ in even_states]
    y_odd  = [E for E,_ in odd_states]
    plt.hlines(y_even, xmin=-0.2, xmax=0.2, colors='tab:blue', label='Even parity')
    plt.hlines(y_odd,  xmin=0.8, xmax=1.2, colors='tab:red',  label='Odd parity')
    plt.xlim(-0.5,1.5)
    plt.xticks([0,1],['Even','Odd'])
    plt.ylabel("Energy eigenvalues")
    plt.title(f"Parity-resolved spectrum for {n}-dot system")
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# -------------------------------
# Example run for 3-dot system
# -------------------------------
if __name__ == "__main__":
    n = 3
    U0_val = 1
    t_val = 1
    Δ_val = t_val + U0_val/2
    eps_vals = [-U0_val/2]*n

    find_degenerate_params(n, eps_vals, Δ_val, steps=500)
