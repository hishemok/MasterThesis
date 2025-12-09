import numpy as np
import matplotlib.pyplot as plt
import json


#def pull_configurations(n, configs = None, specified_vals = None, path = "configuration.json"):
  
#     #If headers contains n_sites, pull out the physical parameters
#     params_for_n = []
#     configs_for_n = []
#     losses_for_n = []
#     with open(path, "r") as f:
#         data = json.load(f)
#         for entry in data:
#             header = entry.get("header", "")
#             #read n and loss from header
#             # print(header)
#             num_dots = header.split("configuration ")[1].split("- site system")[0].strip()
#             loss = header.split("Loss: ")[1].strip() if "Loss: " in header else np.inf

#             if int(num_dots) == n:
#                 physical_params = entry.get("physical_parameters", {})
#                 params_for_n.append(physical_params)
#                 param_config = entry.get("parameter_configs", {})
#                 configs_for_n.append(param_config)
#                 losses_for_n.append(float(loss))

#     # def to_check(dict1, dict1_compare, dict2 = None, dict2_compare = None, loss = np.inf, saving_dict = {}):
#     #     dicts = [dict1, dict2] if dict2 is not None else [dict1]
#     #     dicts_compare = [dict1_compare, dict2_compare] if dict2_compare is not None else [dict1_compare]
#     #     i = 1
#     #     for d in dicts:
#     #         match = True
#     #         comparison_dict = dicts_compare[i-1]
#     #         for key, value in d.items():
#     #             if key in comparison_dict:
#     #                 if comparison_dict[key] != value:
#     #                     match = False
#     #                     break
#     #             else:
#     #                 match = False
#     #                 break
#     #         if match and i == len(dicts):
#     #             saving_dict["configs"].append(conf)
#     #             saving_dict["params"].append(params)
#     #             saving_dict["losses"].append(loss)
#     #         i += 1
                
    
#     matching_configs = {"n": n, "configs": [], "params": [], "losses": []}
#     # Compare physical_params to specified_vals, at the same time keep track of the corresponding parameter_configs
#     for i in range(len(params_for_n)):
#         params = params_for_n[i]
#         conf = configs_for_n[i]
#         loss = losses_for_n[i]
#         print(loss)
#         if specified_vals is None and configs is None:
#             matching_configs["configs"].append(conf)
#             matching_configs["params"].append(params)
#             matching_configs["losses"].append(float(loss))
#         elif specified_vals is not None and configs is None:
#             match = True
#             for key, value in specified_vals.items():
#                 if key in params:
#                     if params[key] != value:
#                         match = False
#                         break
#                 else:
#                     match = False
#                     break
#             if match:
#                 matching_configs["configs"].append(conf)
#                 matching_configs["params"].append(params)
#                 matching_configs["losses"].append(float(loss))
#         elif specified_vals is None and configs is not None:
#             match = True
#             for key, value in configs.items():
#                 if key in conf:
#                     if conf[key] != value:
#                         match = False
#                         break
#                 else:
#                     match = False
#                     break
#             if match:
#                 matching_configs["configs"].append(conf)
#                 matching_configs["params"].append(params)
#                 matching_configs["losses"].append(float(loss))
#         else:
#             match = True
#             for key, value in specified_vals.items():
#                 if key in params:
#                     if params[key] != value:
#                         match = False
#                         break
#                 else:
#                     match = False
#                     break
#             if match:
#                 for key, value in configs.items():
#                     if key in conf:
#                         if conf[key] != value:
#                             match = False
#                             break
#                     else:
#                         match = False
#                         break
#             if match:
#                 matching_configs["configs"].append(conf)
#                 matching_configs["params"].append(params)
#                 matching_configs["losses"].append(float(loss))

#     ## Take the best set of configurations
#     best_config = min(zip(matching_configs["losses"], matching_configs["configs"], matching_configs["params"]), key=lambda x: x[0])
#     return best_config#


def pull_configurations(n, configs=None, specified_vals=None, path="configuration.json") -> dict:
    params_for_n = []
    configs_for_n = []
    losses_for_n = []

    def matches_conditions(params, conf, specified_vals, configs):
        """Returns True if (params, conf) match the two user selection rules."""

        if specified_vals is not None:
            for key, val in specified_vals.items():
                if key not in params or params[key] != val:
                    return False

        if configs is not None:
            for key, val in configs.items():
                if key not in conf or conf[key] != val:
                    return False

        return True

    with open(path, "r") as f:
        data = json.load(f)

        for entry in data:
            header = entry.get("header", "")
            try:
                num_dots = int(header.split("configuration ")[1].split("- site system")[0].strip())
            except:
                continue

            if num_dots != n:
                continue

            loss = header.split("Loss: ")[1].strip() if "Loss: " in header else np.inf

            params_for_n.append(entry.get("physical_parameters", {}))
            configs_for_n.append(entry.get("parameter_configs", {}))
            losses_for_n.append(float(loss))

    matches = []

    for params, conf, loss in zip(params_for_n, configs_for_n, losses_for_n):
        if matches_conditions(params, conf, specified_vals, configs):
            matches.append((loss, conf, params))

    if not matches:
        print("No matching configurations found.")
        return {}

    best = min(matches, key=lambda x: x[0])   # (loss, conf, params)
    return best
 



def params_for_n_site_Hamiltonian(n, configs, specified_vals=None, path="configuration.json"):

    loss, configuration, physical_params = pull_configurations(n, configs, specified_vals, path)


    t = physical_params["t"]
    U = physical_params["U"]
    eps = physical_params["eps"]
    Delta = physical_params["Delta"]

    if len(t) != n - 1:
        if len(t) == 1:
            t = np.repeat(t, n - 1)
        else:
            raise ValueError(f"Length of t_vals {len(t)} does not match n_sites-1={n_sites-1}")
    if len(U) != n:
        if len(U) == 1:
            U = np.repeat(U, n-1)
        else:
            raise ValueError(f"Length of U_vals {len(U)} does not match n_sites={n_sites}")
    if len(eps) != n:
        if len(eps) == 1:
            eps = np.repeat(eps, n)
        else:
            raise ValueError(f"Length of epsilons {len(eps)} does not match n_sites={n_sites}")
    if len(Delta) != n - 1:
        if len(Delta) == 1:
            Delta = np.repeat(Delta, n - 1)
        else:
            raise ValueError(f"Length of delta_vals {len(Delta)} does not match n_sites-1={n_sites-1}")

    print(f"t: {t}")
    print(f"U: {U}")
    print(f"eps: {eps}")
    print(f"Delta: {Delta}")


    return (t, U, eps, Delta), (loss, configuration, specified_vals)
if __name__ == "__main__":
    n_sites = 3

    pars, extras = params_for_n_site_Hamiltonian(n_sites, configs=None, specified_vals={"U": [0.1]}, path="configuration.json")
  
    # configs = pull_configurations(n_sites, specified_vals={"U": [0.1]})
    # # print(configs)

    # loss    = configs[0]
    # config  = configs[1]
    # params  = configs[2]
    
    # H = create_hamiltonian(n_sites, params)

    # H = symbolic_hamiltonian(n_sites)
    # sp.pprint(H)
  
    # configurations = get_configuration()
    # print(f"Available configurations for n={n_sites}:")
    # # print(configurations["n3"][2])
    # for config in configurations[f"n{n_sites}"]:
    #     print(f"Loss: {config['loss']}, Theta: {config['theta']}")

    # # best_config3 = get_best_config(2)
    # # print("Best configuration for n=3:")
    # # print(best_config3["theta"])


    # Hnum = symbolic_hamiltonian_to_np(n_sites, configurations[f"n{n_sites}"][0])
    # eigvals, eigvecs = np.linalg.eigh(Hnum)