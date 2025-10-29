import numpy as np
import matplotlib.pyplot as plt

# Define parameters and dispersion relation
t = 1
Delta = 0.5
a = 1
p = np.linspace(-np.pi, np.pi, 400)

def E(p, mu, Delta, t):
    E_plus = +np.sqrt((mu - 2*t*np.cos(p*a))**2 + 4*(Delta**2)*(np.sin(p*a)**2))
    E_minus = -E_plus
    return E_plus, E_minus

mus = [-3, -2, -1, 1, 2, 3]
titles = [
    r'Trivial phase ($\mu=-3$)',
    r'Critical point ($\mu=-2$)',
    r'Topological phase ($\mu=-1$)',
    r'Topological phase ($\mu=1$)',
    r'Critical point ($\mu=2$)',
    r'Trivial phase ($\mu=3$)'
]

# Create figure
fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=True, sharey=True)
axes = axes.flatten()

for ax, mu, title in zip(axes, mus, titles):
    E_plus, E_minus = E(p, mu, Delta, t)
    ax.plot(p, E_plus, label=r'$E_+(p)$', color='C0')
    ax.plot(p, E_minus, label=r'$E_-(p)$', color='C1')
    ax.set_title(title, fontsize=11, pad=5)
    ax.grid(True, alpha=0.4)

# Common labels
fig.text(0.5, 0.04, 'Momentum $p$', ha='center', fontsize=14)
fig.text(0.04, 0.5, 'Energy $E$', va='center', rotation='vertical', fontsize=14)

# Main title
fig.suptitle('Quasiparticle Band Structure for Different Chemical Potentials',
             fontsize=18, y=0.97, fontweight='bold')

plt.tight_layout(rect=(0.05, 0.05, 1.0, 0.94))
plt.savefig('Figures/band_structure.pdf', dpi=300)
plt.show()