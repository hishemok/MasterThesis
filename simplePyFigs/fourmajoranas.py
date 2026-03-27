import matplotlib.pyplot as plt
import os

current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")
figpath = "texmex/figs"

savepath = os.path.join(current_dir, figpath, 'sixmajoranas.pdf')


## Plot 6 X's
x_ax = [0, 1, 4, 3, 2, 2]
y_ax = [0, 1, 0, 1, 2, 3.2]
markers = [[x_ax[0], y_ax[0]], [x_ax[1], y_ax[1]], [x_ax[2], y_ax[2]], [x_ax[3], y_ax[3]], [x_ax[4], y_ax[4]], [x_ax[5], y_ax[5]]]
lines_between = [[markers[0], markers[1]], [markers[2], markers[3]], [markers[4], markers[5]]]
majorana_labels = ["γ₅","γ₂", "γ₄", "γ₃", "γ₀", "γ₁"]
subsystem_markers = ["A", "B", "C"]
subsystem_positions = [[1, 0.4], [2.5, 2.6], [3, 0.4]]

fig, ax = plt.subplots()
ax.plot(x_ax, y_ax, 'x', markersize=30, markeredgewidth=3)
for i in range(len(lines_between)):
    line = lines_between[i]
    ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], '--', linewidth=1, color='gray')

for (x, y), label in zip(markers, majorana_labels):
    ax.text(x + 0.25, y , label, fontsize=16, ha='left', va='bottom')

for i in range(3):
    ax.text(subsystem_positions[i][0], subsystem_positions[i][1], subsystem_markers[i], fontsize=18, ha='center', va='center')

ax.set_xlim(-1, 5)
ax.set_ylim(-1, 4)
ax.set_aspect('equal', adjustable='box')
ax.set_title('Six Majorana Modes')
#Remove axes
ax.axis('off')

plt.savefig(savepath, dpi=300, bbox_inches='tight')
plt.show()