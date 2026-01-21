import json
from hamiltonian import HamiltonianModel
from analysis import analyze 
from main import plot_known_parameters

def review_configurations(n_sites, filename="configuration.json"):
    """
    Reads all configurations from filename and plots the ones with n_sites.
    """
    # Load all configurations
    with open(filename, "r") as f:
        data = json.load(f)

    # Loop over configurations and pick ones matching n_sites
    count = 0
    for entry in data:
        header = entry.get("header", "")
        if f"configuration {n_sites}- site system" in header:
            theta = entry["raw_parameter_values"]
            print(f"\n>>> Plotting configuration: {header} <<<")
            plot_known_parameters(n_sites, theta)
            count += 1

    if count == 0:
        print(f"No configurations found for n={n_sites} in {filename}.")
    else:
        print(f"\nPlotted {count} configuration(s) for n={n_sites}.")

if __name__ == "__main__":
    n = 3
    """
    n=2, U=0.1:
    ϵ = [-0.05, -0.05]
    t, Δ = 1.00, 1.05
    """

    filename = "configuration.json"
    review_configurations(n, filename)
