import numpy as np
import torch
from operators import to_torch, device, precompute_ops

class HamiltonianModel:
    def __init__(self, n, param_configs=None, device=None):
        """
        Initialize the Hamiltonian model with given parameters.
        n: Number of sites
        param_configs: Dictionary as follows:
            {
                "t": {"mode": "homogeneous", "fixed": None},
                "U": {"mode": "homogeneous", "fixed": None},
                "eps": {"mode": "inhomogeneous", "fixed": None},
                "Delta": {"mode": "inhomogeneous", "fixed": None}
            }
        param_configs: Dictionary defining parameter configurations.

        
        """
        self.n = n
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.param_configs = param_configs or self.default_param_configs()
        self.cre, self.ann, self.num = precompute_ops(n)
        self.cre_t = [to_torch(m, self.device) for m in self.cre]
        self.ann_t = [to_torch(m, self.device) for m in self.ann]
        self.num_t = [to_torch(m, self.device) for m in self.num]
    
    def default_param_configs(self):
        return {
            "t": {"mode": "inhomogeneous", "fixed": None},
            "U": {"mode": "inhomogeneous", "fixed": None},
            "eps": {"mode": "inhomogeneous", "fixed": None},
            "Delta": {"mode": "inhomogeneous", "fixed": None}
        }

    def get_params(self):
        """
        Creates a dictionary of parameter counts based on configurations.
        """
        params = {}
        n = self.n
        fixed_keys = []

        for key in ['t', 'U', 'eps', 'Delta']:
            mode = self.param_configs[key]['mode']
            fixed = self.param_configs[key]['fixed']
            
            if mode == 'homogeneous':
                params[key] = self.default_vals(key)[:1]  # single parameter
            elif mode == 'inhomogeneous':
                params[key] = self.default_vals(key)  # full parameter set

            if fixed is not None:
                fixed_keys.append(key)
        return params, fixed_keys

    def default_vals(self, key):
            n = self.n
            if key == 't':
                return torch.tensor([1.0] * (n - 1), dtype=torch.float64)
            elif key == 'U':
                return torch.tensor([1.0] * (n - 1), dtype=torch.float64)
            elif key == 'eps':
                return torch.tensor([-0.5] * n, dtype=torch.float64)
            elif key == 'Delta':
                return torch.tensor([1.0] * (n - 1), dtype=torch.float64)
            else:
                raise ValueError(f"Unknown parameter key: {key}")
            
    def build(self, parameters):
        """
        Build the Hamiltonian matrix given the parameters.
        parameters: Dictionary with keys 't', 'U', 'eps', 'Delta'
        """
        n = self.n
        H = torch.zeros((2**n, 2**n), dtype=torch.complex128, device=self.device)
        

        t = parameters['t']
        U = parameters['U']
        eps = parameters['eps']
        Delta = parameters['Delta']

        n_minus_1 = n - 1
        if len(t) == 1:
            t = t.repeat(n_minus_1)
        if len(U) == 1:
            U = U.repeat(n_minus_1)
        if len(Delta) == 1:
            Delta = Delta.repeat(n_minus_1)


        for j in range(n - 1):
            H += -t[j] * (self.ann_t[j] @ self.cre_t[j + 1] + self.cre_t[j] @ self.ann_t[j + 1])
            H += Delta[j] * (self.cre_t[j] @ self.cre_t[j + 1] + self.ann_t[j + 1] @ self.ann_t[j])

        for j in range(n):
            H += eps[j] * self.num_t[j]

        for j in range(n - 1):
            H += U[j] * self.num_t[j] @ self.num_t[j + 1]

        return H
    
    def get_tensor(self):
        """
        Get the tensor of the parameters for optimization.
        """
        params, fixed_keys = self.get_params()
        theta_list = []

        for key in ['t', 'U', 'eps', 'Delta']:
            if key not in fixed_keys:
                theta_list.extend(params[key].tolist())

        return  to_torch(np.array(theta_list), device=self.device, dtype=torch.float64)
    
    def set_tensor(self, theta):
        """
        Set the model parameters from the optimization tensor.
        """
        params_dict = self.tensor_to_dict(theta)
        self.parameters = params_dict


    def tensor_to_dict(self, theta):
        """
        Convert optimization tensor into a dictionary with keys 't', 'U', 'eps', 'Delta'.
        Homogeneous params: single value in list
        Inhomogeneous params: list of values
        Fixed params: left as fixed value
        """
        params_dict = {}
        n = self.n
        idx = 0

        for key in ['t', 'U', 'eps', 'Delta']:
            cfg = self.param_configs[key]
            fixed = cfg['fixed']
            mode = cfg['mode']

            if fixed is not None:
                if isinstance(fixed, (int, float)):
                    # convert scalar to appropriate list length
                    length = 1 if mode=="homogeneous" else (n-1 if key!="eps" else n)
                    params_dict[key] = torch.tensor([fixed]*length, dtype=torch.float64, device=self.device)
                else:
                    # fixed already provided as full tensor/list
                    params_dict[key] = torch.tensor(fixed, dtype=torch.float64, device=self.device)
                continue

            # determine how many entries to take from theta
            length = 1 if mode == "homogeneous" else (n-1 if key != "eps" else n)
            params_dict[key] = theta[idx:idx+length].to(dtype=torch.float64, device=self.device)
            idx += length

        return params_dict
  
    def dict_to_tensor(self, theta_dict):
        """
        Convert a dictionary {'t': [...], 'U': [...], 'eps': [...], 'Delta': [...]}
        into a single contiguous tensor that matches get_tensor().

        IMPORTANT: This function *omits* keys that are fixed in self.param_configs,
        so the returned tensor matches the same layout as model.get_tensor().
        """
        pieces = []
        n = self.n

        # require param_configs available
        if not hasattr(self, "param_configs") or self.param_configs is None:
            raise RuntimeError("HamiltonianModel.param_configs must be set to use dict_to_tensor")

        for key in ['t', 'U', 'eps', 'Delta']:
            cfg = self.param_configs.get(key, {"mode": "inhomogeneous", "fixed": None})
            fixed = cfg.get("fixed", None)

            # If this parameter is fixed, DO NOT include it in the optimization vector.
            # (This matches how get_tensor() builds the optimization tensor.)
            if fixed is not None:
                # warn if the user provided a value different from the fixed one
                if key in theta_dict:
                    provided = theta_dict[key]
                    # compare loosely
                    try:
                        provided_val = float(torch.tensor(provided).reshape(-1)[0])
                        fixed_val = float(fixed) if isinstance(fixed, (int, float)) else float(torch.tensor(fixed).reshape(-1)[0])
                        if abs(provided_val - fixed_val) > 1e-8:
                            print(f"Warning: provided value for fixed parameter '{key}' ({provided_val}) will be ignored in favor of fixed={fixed_val}.")
                    except Exception:
                        # ignore conversion errors, just warn
                        print(f"Warning: provided value for fixed parameter '{key}' will be ignored in favor of fixed={fixed}.")
                continue

            # not fixed -> include from the provided dict
            if key not in theta_dict:
                raise KeyError(f"dict_to_tensor: missing key '{key}' in provided theta_dict (required because '{key}' is not fixed).")

            vals = theta_dict[key]
            if not torch.is_tensor(vals):
                vals = torch.tensor(vals, dtype=torch.float64, device=self.device)

            # Homogeneous parameters should be length-1 tensors
            if cfg.get('mode', 'inhomogeneous') == "homogeneous":
                if vals.numel() != 1:
                    raise ValueError(f"Key '{key}' should have 1 value (homogeneous). Got {vals}.")
                pieces.append(vals.reshape(-1))
                continue

            # Inhomogeneous parameters
            expected_len = n if key == "eps" else (n - 1)
            if vals.numel() != expected_len:
                # auto-expand single scalar to expected length
                if vals.numel() == 1:
                    vals = vals.repeat(expected_len)
                else:
                    raise ValueError(f"Key '{key}' expected length {expected_len}, got {vals.numel()}.")

            pieces.append(vals.reshape(-1))

        if len(pieces) == 0:
            # nothing to optimize (all fixed)
            return torch.tensor([], dtype=torch.float64, device=self.device)

        return torch.cat(pieces)


    def get_physical_parameters(self, params_dict):
        """
        Returns physical parameters ready to build the Hamiltonian:
        - Fixed parameters are left unchanged
        - t, U, Delta are made positive via softplus if not fixed
        - Homogeneous parameters remain single value lists until build() needs expansion
        """
        phys_dict = {}
        for key in ['t','U','eps','Delta']:
            cfg = self.param_configs[key]
            fixed = cfg['fixed']
            vals = params_dict[key]

            # Keep the tensor and move to device/dtype safely
            vals = vals.to(dtype=torch.float64, device=self.device)

            if fixed is None:
                if key in ["t", "U", "Delta"]:
                    vals = vals.abs()#torch.nn.functional.softplus(vals)

            phys_dict[key] = vals

        return phys_dict
    
    def adjust_tensor(self, theta):
        """
        Inputs a tensor, returns a tensor within the physical parameter space,
        while preserving autograd.
        """
        tensor_dict = self.tensor_to_dict(theta)
        phys_params = self.get_physical_parameters(tensor_dict)
        _, fixed_keys = self.get_params()
        
        adjusted_list = []
        for key in ['t','U','eps','Delta']:
            if key not in fixed_keys:
                adjusted_list.append(phys_params[key])  # keep as tensor
        return torch.cat(adjusted_list) 

    def pretty_print_params(self, theta):
        """
        Print parameters in a readable format.
        """
        params_dict = self.tensor_to_dict(theta)
        phys_params = self.get_physical_parameters(params_dict)

        for key in ['t', 'U', 'eps', 'Delta']:
            print(f"{key} = {phys_params[key].cpu().numpy()}")
    
    def loss(self, H, P, weight_max=3, weight_min=0.1, gap_threshold=0.5):
        """
        Loss function
        """
        from measurements import calculate_parities, charge_difference_torch, MP_Penalty
        evals, evecs = torch.linalg.eigh(H)
        even_states, odd_states, even_vecs, odd_vecs = calculate_parities(evals, evecs, P)

        
        n_elements = min(len(even_states), len(odd_states))
        if n_elements == 0:
            raise ValueError("No states in one of the parity sectors \n Proceed to next restart")  # bad configuration Penalty
        if len(even_states) != len(odd_states):
            raise ValueError("Unequal number of even and odd states \n Bad configuration, proceed to next restart")
        
        weight_array = np.linspace(weight_max, weight_min, 2*n_elements-1)**2# First half for Degeneracy, second half for gaps
        weight_vec = torch.tensor(weight_array, device=self.device)

        penalty_array = torch.zeros(2*n_elements-1, device=self.device) # First half for Degeneracy, second half for gaps
        deg_terms = torch.abs(even_states[:n_elements] - odd_states[:n_elements]) # Cut out unequal lengths
        w = weight_vec.to(self.device)
        degeneracy_terms = w[:n_elements] * deg_terms
        penalty_array[:n_elements] = degeneracy_terms

        even_gaps = even_states[1:n_elements] - even_states[:n_elements-1]
        odd_gaps = odd_states[1:n_elements] - odd_states[:n_elements-1]
        worst_gaps = torch.min(torch.stack([even_gaps, odd_gaps]), dim=0).values

        gap_penalties = torch.nn.functional.softplus(gap_threshold - worst_gaps)
        penalty_array[n_elements:] = w[n_elements:] * gap_penalties

        charge_diff = charge_difference_torch(even_vecs, odd_vecs, self.n)

        MP_penalty = MP_Penalty(even_vecs, odd_vecs, self.n)

        total_loss = torch.sum(penalty_array) + charge_diff + MP_penalty

        mean_energy = evals.abs().mean().detach() + 1e-8
        normalized_loss = total_loss / mean_energy

        return normalized_loss.real

    def build_full_theta(self, theta):
        d = self.tensor_to_dict(theta)
        return (
            d["t"].tolist()
            + d["U"].tolist()
            + d["eps"].tolist()
            + d["Delta"].tolist()
        )


if __name__ == "__main__":
    # Example usage


    parameter_config = {
        "t": {"mode": "homogeneous", "fixed": 1.0},
        "U": {"mode": "homogeneous", "fixed": None},
        "eps": {"mode": "inhomogeneous", "fixed": None},
        "Delta": {"mode": "inhomogeneous", "fixed": None}
    }
    model = HamiltonianModel(n=4, param_configs=parameter_config)
    params, fixed = model.get_params()
    print("Parameters:", params)
    print("\nFixed keys:", fixed)
    tensor = model.get_tensor()
    print("\nOptimization tensor:", tensor)
    dictio = model.tensor_to_dict(tensor)
    print("\nParameter dictionary from tensor:", dictio)    
    phys_params = model.get_physical_parameters(dictio)
    print("\nPhysical parameters:", phys_params)    
    adjusted_tensor = model.adjust_tensor(tensor)
    print("\nAdjusted optimization tensor:", adjusted_tensor)
    model.pretty_print_params(adjusted_tensor)
    