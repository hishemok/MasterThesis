import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os

current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")
figpath = "texmex/figs"

savepath = os.path.join(current_dir, figpath, '3dparametersearch.pdf')

# ---------------------------
# Build a smooth "loss landscape"
# ---------------------------
def loss_landscape(x, y):
    # Broad bowl to keep the landscape bounded and nonnegative
    bowl = 0.025 * (x**2 + 0.8 * y**2)

    # Smooth oscillatory structure to create many local minima/maxima
    waves = (
        0.35 * np.sin(1.6 * x) * np.cos(1.3 * y)
        + 0.20 * np.cos(2.4 * x + 0.8 * y)
        + 0.12 * np.sin(2.8 * np.sqrt(x**2 + y**2))
    )

    # Several Gaussian wells to mimic local basins
    wells = (
        -0.90 * np.exp(-((x + 2.2)**2 / 1.8 + (y - 1.6)**2 / 1.2))
        -0.75 * np.exp(-((x - 1.8)**2 / 1.4 + (y + 2.1)**2 / 1.0))
        -1.25 * np.exp(-((x - 0.8)**2 / 0.9 + (y - 0.4)**2 / 0.7))  # deeper/global basin
        -0.60 * np.exp(-((x + 3.0)**2 / 1.1 + (y + 2.8)**2 / 1.5))
    )

    # A few positive bumps to create barriers
    bumps = (
        0.45 * np.exp(-((x + 0.5)**2 / 1.0 + (y + 1.2)**2 / 0.8))
        +0.35 * np.exp(-((x - 2.7)**2 / 0.8 + (y - 2.2)**2 / 1.1))
    )

    z = bowl + waves + wells + bumps

    # Shift so the minimum is exactly at/above zero
    z = z - np.min(z)
    return z

# ---------------------------
# Grid
# ---------------------------
n = 500
x = np.linspace(-5, 5, n)
y = np.linspace(-5, 5, n)
X, Y = np.meshgrid(x, y)
Z = loss_landscape(X, Y)

# ---------------------------
# Find global minimum for annotation
# ---------------------------
min_idx = np.unravel_index(np.argmin(Z), Z.shape)
x_min, y_min, z_min = X[min_idx], Y[min_idx], Z[min_idx]

# ---------------------------
# Plot
# ---------------------------
fig = plt.figure(figsize=(10, 7), dpi=180)
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    X, Y, Z,
    cmap=cm.viridis,
    linewidth=0,
    antialiased=True,
    alpha=0.97
)

# Optional contour projection onto the bottom plane
z_offset = -0.15 * Z.max()
ax.contour(
    X, Y, Z,
    levels=18,
    zdir='z',
    offset=z_offset,
    cmap=cm.viridis,
    linewidths=0.8
)

# # Mark the global minimum
# ax.scatter(
#     [x_min], [y_min], [z_min],
#     color='crimson',
#     s=60,
#     label='Global minimum',
#     depthshade=False
# )

# Labels
ax.set_xlabel(r'Reduced parameter coordinate $\phi_1$', labelpad=12)
ax.set_ylabel(r'Reduced parameter coordinate $\phi_2$', labelpad=12)
ax.set_zlabel(r'Loss $L(\theta)$', labelpad=10)

# Limits and view
ax.set_zlim(z_offset, Z.max())
ax.view_init(elev=30, azim=-55)

# Clean up panes/grid for a more thesis-like look
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)

# Colorbar
# cbar = fig.colorbar(surf, ax=ax, shrink=0.72, pad=0.08)
# cbar.set_label(r'Loss value $L(\theta)$')

# ax.legend(loc='upper left', frameon=False)
plt.title('Visualizing the Loss Landscape for Parameter Search', pad=20, fontsize=14)
fig.subplots_adjust(left=0.02, right=0.92, bottom=0.02, top=0.90)
plt.savefig(savepath, dpi=300)
plt.show()
