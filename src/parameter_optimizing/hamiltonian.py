import numpy as np
import torch
from operators import to_torch, device, precompute_ops

class HamiltonianModel:
    def __init__(self, n, fixed_params=None, device=None):
        self.n = n
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fixed_params = fixed_params or {}
        self.cre, self.ann, self.num = precompute_ops(n)
        self.cre_t = [to_torch(m, self.device) for m in self.cre]
        self.ann_t = [to_torch(m, self.device) for m in self.ann]
        self.num_t = [to_torch(m, self.device) for m in self.num]

    def map_theta(self, theta, keep_equal=None):
        """
        Map the optimization vector θ into model parameters.
        t, U, Delta  -> arrays of length n-1
        eps          -> array of length n
        """
        if keep_equal is None:
            keep_equal = []

        idx = 0
        n = self.n

        # ---- t ----
        if 't' in self.fixed_params:
            t = self.fixed_params['t']
        else:
            t_raw = theta[idx: idx + (n - 1)]
            idx += (n - 1)

            t_raw = torch.nn.functional.softplus(t_raw)
            if "t" in keep_equal:
                t_raw = torch.mean(t_raw) * torch.ones_like(t_raw)
            t = t_raw

        # ---- U ----
        if 'U' in self.fixed_params:
            U = self.fixed_params['U']
        else:
            U_raw = theta[idx: idx + (n - 1)]
            idx += (n - 1)

            U_raw = torch.nn.functional.softplus(U_raw)
            if "U" in keep_equal:
                U_raw = torch.mean(U_raw) * torch.ones_like(U_raw)
            U = U_raw

        # ---- eps ----
        if 'eps' in self.fixed_params:
            eps = self.fixed_params['eps']
        else:
            eps = theta[idx: idx + n]
            idx += n

            if "eps" in keep_equal:
                eps = torch.mean(eps) * torch.ones_like(eps)

        # ---- Delta ----
        if 'Delta' in self.fixed_params:
            Delta = self.fixed_params['Delta']
        else:
            D_raw = theta[idx: idx + (n - 1)]
            idx += (n - 1)

            D_raw = torch.nn.functional.softplus(D_raw)
            if "Delta" in keep_equal:
                D_raw = torch.mean(D_raw) * torch.ones_like(D_raw)
            Delta = D_raw
        
        # if "Delta" in keep_equal:
        #     print("Keeping Delta equal:", Delta)

        return {"t": t, "U": U, "eps": eps, "Delta": Delta}

        
    def dict_paras_to_theta(self, params):
        """
        Convert parameter dictionary to optimization vector θ.
        """
        theta_list = []

        if 't' not in self.fixed_params:
            theta_list.extend(params['t'].tolist())

        if 'U' not in self.fixed_params:
            theta_list.extend(params['U'].tolist())

        if 'eps' not in self.fixed_params:
            theta_list.extend(params['eps'].tolist())

        if 'Delta' not in self.fixed_params:
            theta_list.extend(params['Delta'].tolist())

        return to_torch(np.array(theta_list), device=self.device)

    def build(self, params):
        t = params.get('t', self.fixed_params.get('t'))
        U = params.get('U', self.fixed_params.get('U'))
        eps = params.get('eps', self.fixed_params.get('eps'))
        Delta = params.get('Delta', self.fixed_params.get('Delta'))


        dim = 2**self.n
        H = torch.zeros((dim,dim), dtype=torch.complex128, device=device)

        for i in range(self.n):
            # onsite 
            H += eps[i] * self.num_t[i]
            if i < self.n - 1:
                # hopping
                H += t[i] * (self.cre_t[i] @ self.ann_t[i+1] + self.cre_t[i+1] @ self.ann_t[i])
                # pairing
                H += Delta[i] * (self.cre_t[i] @ self.cre_t[i+1] + self.ann_t[i+1] @ self.ann_t[i])
                # Coulomb term
                H += U[i] * (self.num_t[i] @ self.num_t[i+1])

        return H