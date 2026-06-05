import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

cwd = Path(__file__).parent
braiding_path = cwd 
filenames = ["braiding_results_step_projected_braiding_local_U=0.0.txt", "braiding_results_step_projected_braiding_local_U=0.1.txt", "braiding_results_step_projected_braiding_local_U=2.0.txt"]

linestyles = ["-", "--", "-."]



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
        fidelities_ideal, fidelities_local = [], []
        for line in lines[1:]:
            time, fidelity_ideal, fidelity_local = line.split()
            times.append(float(time))
            fidelities_ideal.append(float(fidelity_ideal))
            fidelities_local.append(float(fidelity_local))
        
        # plt.plot(times, fidelities_ideal, label=f"U={U_value} (Ideal)")
        local_dict[U_value] = {
            "mean": 1- np.mean(fidelities_local),
            "std": np.std(fidelities_local),
            "ground_state_overlap": 1-fidelities_local[0]  # GS
        }
    plt.plot(times, np.ones(len(fidelities_local)) - fidelities_local, linewidth=2.5, linestyle=linestyles[i], label=f"U={U_value} (Local)")
    if len(fidelities_local) > length:
        length = len(fidelities_local)

ticks = np.arange(0, length, 1)
t1 = fidelities_local

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
        fidelities = []
        for line in lines[1:]:
            time, fidelity = line.split()
            times.append(float(time))
            fidelities.append(float(fidelity))
        
        energy_dict[U_value] = {
            "mean": 1- np.mean(fidelities),
            "std": np.std(fidelities),
            "ground_state_overlap": 1-fidelities[0]  # GS
            }
    plt.plot(times, np.ones(len(fidelities)) - fidelities, linewidth=2.5, linestyle=linestyles[i], label=f"U={U_value} (Local)")
    if len(fidelities) > length:
        length = len(fidelities)

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
        fidelities_ideal = []
        for line in lines[1:]:
            time, dim, static, Bfit, Cfit, Boff_error, Coff_error, overlap = line.split()
            times.append(float(time))
            fidelities_ideal.append(float(overlap))
        #Group	Basis Dimension	Static Term Norm	B Fit Error	C Fit Error	B Offblock Error	C Offblock Error	Unitary Overlap
        # plt.plot(times, fidelities_ideal, label=f"U={U_value} (Ideal)")
        local_dict[U_value] = {
            "mean": 1- np.mean(fidelities_ideal),
            "std": np.std(fidelities_ideal),
            "ground_state_overlap": 1-fidelities_ideal[0]  # GS
        }

    plt.plot(times, np.ones(len(fidelities_ideal)) - fidelities_ideal, linewidth=2.5, linestyle=linestyles[i], label=f"U={U_value} (Local)")
    if len(fidelities_ideal) > length:
        length = len(fidelities_ideal)

ticks = np.arange(0, length, 1)
t2 = fidelities_ideal
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