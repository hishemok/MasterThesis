### this file reads configuration.json and extracts physical parameters for a given n_sites
import json

def get_configuration(filename="configuration.json"):
    """
    Reads configurations from a JSON file and returns them as a list of dictionaries.
    Each dictionary contains the header and physical parameter values.
    """
    with open(filename, "r") as f:
        data = json.load(f)
        n2, n3, n4 = [], [], []
        for entry in data:
            header = entry.get("header", "")
            #read n and loss from header
            n = header.split("configuration ")[1].split("- site system")[0].strip()
            loss = header.split("Loss: ")[1].split(" with")[0].strip()            
            theta = entry.get("physical_parameters", [])
            
            if n == "2":
                n2.append({"n": n, "loss": loss, "theta": theta})
            elif n == "3":
                n3.append({"n": n, "loss": loss, "theta": theta})
            elif n == "4":
                n4.append({"n": n, "loss": loss, "theta": theta})
        data = {"n2": n2, "n3": n3, "n4": n4}
    return data

def get_best_config(n, filename="configuration.json"):
    """
    Given n and the configurations dictionary, returns the configuration with the lowest loss.
    """
    configs = get_configuration(filename=filename)
    key = f"n{n}"
    if key not in configs:
        return None
    best_config = min(configs[key], key=lambda x: float(x["loss"]))
    return best_config


if __name__ == "__main__":
    configs = get_configuration()
    print("Configurations for n=2:", configs["n2"])

    best_config2 = get_best_config(2)
    print("Best configuration for n=2:", best_config2)
    best_config3 = get_best_config(3)
    print("Best configuration for n=3:", best_config3)
    best_config4 = get_best_config(4)
    print("Best configuration for n=4:", best_config4)
