import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
from pathlib import Path

cwd = Path(__file__).parent

files_matched = [
    "braiding_results_matched_ops_U=0.0.txt",
    "braiding_results_matched_ops_U=0.1.txt",
    "braiding_results_matched_ops_U=0.5.txt",
    "braiding_results_matched_ops_U=1.0.txt",
    "braiding_results_matched_ops_U=2.0.txt"
]

files_local = [
    "braiding_results_step_projected_braiding_local_U=0.0.txt",
    "braiding_results_step_projected_braiding_local_U=0.1.txt",
    "braiding_results_step_projected_braiding_local_U=0.5.txt",
    "braiding_results_step_projected_braiding_local_U=1.0.txt",
    "braiding_results_step_projected_braiding_local_U=2.0.txt"
]

files_ideal = [
    "braiding_results_step_projected_braiding_U=0.0.txt",
    "braiding_results_step_projected_braiding_U=0.1.txt",
    "braiding_results_step_projected_braiding_U=0.5.txt",
    "braiding_results_step_projected_braiding_U=1.0.txt",
    "braiding_results_step_projected_braiding_U=2.0.txt"
]


linestyles = ["-", "--", "-.", "-.", "-"]
colors = ["navy", "indigo", "darkred", "darkgreen", "darkorange"]


local_dict={
    "0.0": {
            "groups": [],
            "errors": [],
            "mean": 0,
            "std": 0,
            "ground_state_overlap": 0},
    "0.1": {
            "groups": [],
            "errors": [],
            "mean": 0,
            "std": 0,
            "ground_state_overlap": 0},
    "0.5": {
            "groups": [],
            "errors": [],
            "mean": 0,
            "std": 0,
            "ground_state_overlap": 0},
    "1.0": {
            "groups": [],
            "errors": [],
            "mean": 0,
            "std": 0,
            "ground_state_overlap": 0},
    "2.0": {
            "groups": [],
            "errors": [],
            "mean": 0,
            "std": 0,
            "ground_state_overlap": 0}
}
matched_dict = local_dict.copy()
ideal_dict = local_dict.copy()

def read_groups_and_overlap_errors(filename):
    records = []
    current_record = None
    with open(cwd / filename, "r") as f:
        for line in f.readlines()[1:]:
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            try:
                float(parts[0])
                starts_new_record = True
            except ValueError:
                starts_new_record = False

            if starts_new_record:
                if current_record is not None:
                    records.append(current_record)
                current_record = stripped
            elif current_record is not None:
                current_record += " " + stripped

    if current_record is not None:
        records.append(current_record)

    groups = []
    errors = []
    for record in records:
        parts = record.split()
        groups.append(float(parts[0]))
        errors.append(abs(1 - float(parts[-1])))
    return groups, errors


def store_overlap_results(result_dict, filename):
    U_value = filename.split("U=")[-1].split(".txt")[0]
    groups, errors = read_groups_and_overlap_errors(filename)
    result_dict[U_value] = {
        "groups": groups,
        "errors": errors,
        "mean": np.mean(errors),
        "std": np.std(errors),
        "ground_state_overlap": errors[0],
    }


for filename in files_local:
    store_overlap_results(local_dict, filename)

for filename in files_ideal:
    store_overlap_results(ideal_dict, filename)

for filename in files_matched:
    store_overlap_results(matched_dict, filename)


display_floor = 5e-11
regime_y_limits = (display_floor, 2.0)


def plot_sector_series(axis, values, u_index, u_label):
    raw_values = np.abs(np.asarray(values))
    display_values = np.maximum(raw_values, display_floor)
    sector_position = np.linspace(0, 1, len(display_values))

    axis.plot(
        sector_position,
        display_values,
        linewidth=2.0,
        linestyle=linestyles[u_index],
        marker="o",
        markersize=3.5,
        label=fr"$U={u_label}$",
        color=colors[u_index],
    )

    unresolved = raw_values < display_floor
    if unresolved.any():
        axis.scatter(
            sector_position[unresolved],
            display_values[unresolved],
            marker="v",
            facecolors="none",
            edgecolors=colors[u_index],
            s=35,
            linewidths=1.0,
            zorder=3,
        )


def plot_sector_regimes(results, title, output_name, high_u, intermediate_u, low_u):
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=20)
    fig.supylabel(r"Absolute overlap deviation $|1-\mathcal{O}|$", fontsize=16)

    regimes = [
        (axes[0], high_u, "(a) High-deviation"),
        (axes[1], intermediate_u, "(b) Intermediate-deviation"),
        (axes[2], low_u, "(c) Low-deviation"),
    ]
    for axis, u_values, subtitle in regimes:
        for u_label in u_values:
            plot_sector_series(axis, results[u_label]["errors"], names.index(u_label), u_label)
        axis.set_title(subtitle, loc="left", fontsize=13, fontweight="bold")

    legend_placements = ["lower left", "upper left", "upper left"]
    for axis, legend_placement in zip(axes, legend_placements):
        axis.set_yscale("log")
        axis.set_ylim(*regime_y_limits)
        axis.grid(axis="y", which="major", alpha=0.25)
        axis.tick_params(axis="both", labelsize=11)
        axis.legend(
            loc=legend_placement,
            frameon=True,
            facecolor="white",
            framealpha=0.9,
            edgecolor="none",
            ncol=3,
            fontsize=10,
        )
    axes[2].set_xticks(
        [0, 0.5, 1],
        ["Ground-state sector", "Middle sectors", "Highest-energy sector"],
    )
    axes[2].set_xlabel("Normalized sector-group position", fontsize=14)
    fig.text(
        0.99,
        0.01,
        r"Exact printed zeros are unresolved and shown at $5\times10^{-11}$.",
        ha="right",
        fontsize=9,
        color="dimgray",
    )
    fig.tight_layout(rect=(0.02, 0.04, 1, 0.96))
    output_path = cwd.parents[1] / "texmex" / "figs" / output_name
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.08, dpi=220)
    plt.close(fig)
    # plt.show()

names = ["0.0", "0.1", "0.5", "1.0", "2.0"]

plot_sector_regimes(
    ideal_dict,
    "Energy-Level Majorana Baseline Braid Target Deviation",
    "ideal_operator_interaction_regimes",
    high_u=["1.0"],
    intermediate_u=["2.0"],
    low_u=["0.5", "0.1", "0.0"],
)
plot_sector_regimes(
    local_dict,
    "Projected Local-Operator Braid Target Deviation",
    "projected_operator_interaction_regimes",
    high_u=["1.0", "2.0"],
    intermediate_u=["0.5"],
    low_u=["0.1", "0.0"],
)
plot_sector_regimes(
    matched_dict,
    "Block-Matched Local-Operator Braid Target Deviation",
    "matched_operator_interaction_regimes",
    high_u=["1.0", "2.0"],
    intermediate_u=["0.5"],
    low_u=["0.1", "0.0"],
)




colors = ["navy", "indigo", "darkred", "darkgreen", "darkorange"]
bar_plot_floor = 7e-8


def plot_error_summary(axis, results, title):
    means = np.asarray([results[U]["mean"] for U in names])
    standard_deviations = np.asarray([results[U]["std"] for U in names])
    ground_sector_deviations = np.asarray(
        [results[U]["ground_state_overlap"] for U in names]
    )
    positions = np.arange(len(names))

    axis.bar(
        positions,
        means,
        width=0.68,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.92,
    )

    # A symmetric standard deviation may extend below zero and cannot be shown
    # on a logarithmic axis, so only the displayed lower whisker is clipped.
    lower_errors = np.minimum(standard_deviations, means - bar_plot_floor)
    axis.errorbar(
        positions,
        means,
        yerr=np.vstack((lower_errors, standard_deviations)),
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=3.5,
        label=r"Sector mean $\pm$ standard deviation",
        zorder=3,
    )
    axis.scatter(
        positions,
        ground_sector_deviations,
        marker="D",
        s=34,
        facecolors="white",
        edgecolors="black",
        linewidths=1.0,
        label="Ground-sector deviation",
        zorder=4,
    )

    axis.set_title(title, loc="left", fontsize=12, fontweight="bold")
    axis.set_yscale("log")
    axis.set_ylim(bar_plot_floor, 1.2)
    axis.grid(axis="y", which="major", color="0.8", linewidth=0.7, alpha=0.65)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", labelsize=10)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.2), sharex=True, sharey=True)
fig.suptitle("Braid overlap deviation by operator construction", fontsize=16)
fig.supylabel(r"Mean absolute overlap deviation $\langle D_{\mathcal{O}}\rangle$", fontsize=13)

plot_error_summary(axes[0], local_dict, "(a) Projected local operators")
plot_error_summary(axes[1], matched_dict, "(b) Block-matched local operators")
plot_error_summary(axes[2], ideal_dict, "(c) Energy-level Majorana baseline")

axes[2].set_xticks(np.arange(len(names)), names)
axes[2].set_xlabel(r"Interaction strength $U$", fontsize=12)

handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    legend_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.947),
    ncol=2,
    frameon=False,
    fontsize=10,
)
fig.tight_layout(rect=(0.02, 0.02, 1, 0.91), h_pad=1.1)

output_path = cwd.parents[1] / "texmex" / "figs" / "step_projection_mean_comparison"
fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.08, dpi=220)
plt.close(fig)

"""
plt.figure(figsize=(10, 6))


plt.title("Braiding Error vs Sector Groups for Different U Values (Local Projection)", fontsize=24)

length = 0


local_dict={
    "0.0": {"mean": 0,
            "std": 0,
            "ground_state_overlap": 0},
    "0.1": {"mean": 0,
            "std": 0,
            "ground_state_overlap": 0},
    "2.0": {"mean": 0,
            "std": 0,
            "ground_state_overlap": 0}
}



for i,filename in enumerate(filenames):
    with open(braiding_path / filename, "r") as f:
        lines = f.readlines()
        U_value = filename.split("U=")[-1].split(".txt")[0]
        times = []
        errors_ideal, errors_local = [], []
        for line in lines[1:]:
            time, fidelity_ideal, fidelity_local = line.split()
            times.append(float(time))
            errors_ideal.append(float(fidelity_ideal))
            errors_local.append(float(fidelity_local))
        
        # plt.plot(times, errors_ideal, label=f"U={U_value} (Ideal)")
        local_dict[U_value] = {
            "mean": 1- np.mean(errors_local),
            "std": np.std(errors_local),
            "ground_state_overlap": 1-errors_local[0]  # GS
        }
    plt.plot(times, np.ones(len(errors_local)) - errors_local, linewidth=2.5, linestyle=linestyles[i], label=f"U={U_value} (Local)")
    if len(errors_local) > length:
        length = len(errors_local)

ticks = np.arange(0, length, 1)
t1 = errors_local

plt.xlabel("Sector Groups", fontsize=18)
plt.ylabel("Fidelity", fontsize=18)
plt.xticks(ticks, fontsize=14)
plt.yticks(fontsize=14)
plt.yscale("log")
plt.legend()
plt.show()


filenames = ["braiding_results_step_projected_braiding_U=0.0.txt", "braiding_results_step_projected_braiding_U=0.1.txt", "braiding_results_step_projected_braiding_U=2.0.txt"]

linestyles = ["-", "--", "-."]



plt.figure(figsize=(10, 6))

plt.title("Braiding Error vs Sector Groups for Different U Values (Energy Projection)", fontsize=24)

length = 0
energy_dict={
    "0.0": {"mean": 0,
            "std": 0,
            "ground_state_overlap": 0},
    "0.1": {"mean": 0,
            "std": 0,
            "ground_state_overlap": 0},
    "2.0": {"mean": 0,
            "std": 0,
            "ground_state_overlap": 0}
}
for i,filename in enumerate(filenames):
    with open(braiding_path / filename, "r") as f:
        lines = f.readlines()
        U_value = filename.split("U=")[-1].split(".txt")[0]
        times = []
        errors = []
        for line in lines[1:]:
            time, fidelity = line.split()
            times.append(float(time))
            errors.append(float(fidelity))
        
        energy_dict[U_value] = {
            "mean": 1- np.mean(errors),
            "std": np.std(errors),
            "ground_state_overlap": 1-errors[0]  # GS
            }
    plt.plot(times, np.ones(len(errors)) - errors, linewidth=2.5, linestyle=linestyles[i], label=f"U={U_value} (Local)")
    if len(errors) > length:
        length = len(errors)

ticks = np.arange(0, length, 1)

plt.xlabel("Sector Groups", fontsize=18)
plt.ylabel("Fidelity", fontsize=18)
plt.xticks(ticks, fontsize=14)
plt.yticks(fontsize=14)
plt.yscale("log")
plt.legend()
plt.show()

#Bar plot comparing the mean braiding errors for local and energy projections
plt.figure(figsize=(12, 10))
plt.subplot(1, 2, 1)

names = ["0.0", "0.1", "2.0"]
colors = ["navy", "indigo", "darkred"]
labels = [f"U={name}" for name in names]
plt.bar(labels, [local_dict[U]["mean"] for U in names], color=colors)
plt.scatter(labels, [local_dict[U]["ground_state_overlap"] for U in names], color='magenta', label="Ground State Overlap", marker="x", s=220)
plt.errorbar(labels, [local_dict[U]["mean"] for U in names], yerr=[local_dict[U]["std"] for U in names], fmt='o', color='black', capsize=5,markersize=10, label="Error Bars")
plt.title("Mean Braiding Error for Local Projection", fontsize=18)
plt.ylabel("Mean Braiding Error", fontsize=14)
plt.legend()
plt.semilogy()
# plt.show()

plt.subplot(1, 2, 2)
names = ["0.0", "0.1", "2.0"]
colors = ["navy", "indigo", "darkred"]
labels = [f"U={name}" for name in names]
plt.bar(labels, [energy_dict[U]["mean"] for U in names], color=colors)
plt.scatter(labels, [energy_dict[U]["ground_state_overlap"] for U in names], color='magenta', label="Ground State Overlap", marker="x", s=220)
plt.errorbar(labels, [energy_dict[U]["mean"] for U in names], yerr=[energy_dict[U]["std"] for U in names], fmt='o', color='black', capsize=5,markersize=10, label="Error Bars")
plt.title("Mean Braiding Error for Energy Projection", fontsize=18)
plt.ylabel("Mean Braiding Error", fontsize=14)
plt.legend()
plt.semilogy()
plt.show()






cwd = Path(__file__).parent
braiding_path = cwd 
filenames = ["braiding_results_matched_ops_U=0.0.txt", "braiding_results_matched_ops_U=0.1.txt", "braiding_results_matched_ops_U=2.0.txt"]

linestyles = ["-", "--", "-."]



plt.figure(figsize=(10, 6))

plt.title("Braiding Error vs Sector Groups for Different U Values (Local Projection ~ Ideal Projection)", fontsize=24)

length = 0


local_dict={
    "0.0": {"mean": 0,
            "std": 0,
            "ground_state_overlap": 0},
    "0.1": {"mean": 0,
            "std": 0,
            "ground_state_overlap": 0},
    "2.0": {"mean": 0,
            "std": 0,
            "ground_state_overlap": 0}
}


for i,filename in enumerate(filenames):
    with open(braiding_path / filename, "r") as f:
        lines = f.readlines()
        U_value = filename.split("U=")[-1].split(".txt")[0]
        times = []
        errors_ideal = []
        for line in lines[1:]:
            time, dim, static, Bfit, Cfit, Boff_error, Coff_error, overlap = line.split()
            times.append(float(time))
            errors_ideal.append(float(overlap))
        #Group	Basis Dimension	Static Term Norm	B Fit Error	C Fit Error	B Offblock Error	C Offblock Error	Unitary Overlap
        # plt.plot(times, errors_ideal, label=f"U={U_value} (Ideal)")
        local_dict[U_value] = {
            "mean": 1- np.mean(errors_ideal),
            "std": np.std(errors_ideal),
            "ground_state_overlap": 1-errors_ideal[0]  # GS
        }

    plt.plot(times, np.ones(len(errors_ideal)) - errors_ideal, linewidth=2.5, linestyle=linestyles[i], label=f"U={U_value} (Local)")
    if len(errors_ideal) > length:
        length = len(errors_ideal)

ticks = np.arange(0, length, 1)
t2 = errors_ideal
plt.xlabel("Sector Groups", fontsize=18)
plt.ylabel("Fidelity", fontsize=18)
plt.xticks(ticks, fontsize=14)
plt.yticks(fontsize=14)
plt.yscale("log")
plt.legend()
plt.show()


plt.plot(np.array(t2)-np.array(t1), linewidth=2.5, label="Local - Ideal")
plt.xlabel("Sector Groups", fontsize=18)
plt.ylabel("Fidelity Difference", fontsize=18)
plt.xticks(ticks, fontsize=14)
plt.yticks(fontsize=14)
plt.yscale("log")
plt.legend()
plt.show()


"""
