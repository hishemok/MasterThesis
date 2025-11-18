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
                return torch.tensor([0.0] * n, dtype=torch.float64)
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

    
    def get_physical_parameters(self, params_dict):
        """
        Returns physical parameters ready to build the Hamiltonian:
        - Fixed parameters are left unchanged
        - t, U, Delta are made positive via softplus if not fixed
        - Homogeneous parameters remain single value lists until build() needs expansion
        """
        phys_dict = {}
        n = self.n

        for key in ['t','U','eps','Delta']:
            cfg = self.param_configs[key]
            fixed = cfg['fixed']
            mode = cfg['mode']
            vals = params_dict[key]

            # Convert to float tensor
            vals = torch.tensor(vals, dtype=torch.float64, device=self.device)

            if fixed is None:
                # Only apply transformations if not fixed
                if key in ["t", "U", "Delta"]:
                    vals = torch.nn.functional.softplus(vals)

            phys_dict[key] = vals

        return phys_dict

    def adjust_tensor(self, theta):
        """
        Inputs a tensor, returns a tensor within the physical parameter space.
        """
        tensor = self.tensor_to_dict(theta)
        phys_params = self.get_physical_parameters(tensor)
        _, fixed_keys = self.get_params() 
        adjusted_theta_list = []
        for key in ['t', 'U', 'eps', 'Delta']:
            if key not in fixed_keys:
                adjusted_theta_list.extend(phys_params[key].tolist())
        return to_torch(np.array(adjusted_theta_list), device=self.device)

    def pretty_print_params(self, theta):
        """
        Print parameters in a readable format.
        """
        params_dict = self.tensor_to_dict(theta)
        phys_params = self.get_physical_parameters(params_dict)

        for key in ['t', 'U', 'eps', 'Delta']:
            print(f"{key} = {phys_params[key].cpu().numpy()}")
                            
            

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
    