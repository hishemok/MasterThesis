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

    def map_theta(self, theta):
        """
        Map the optimization vector θ into model parameters.
        Handles both fixed and trainable parameters dynamically.
        """
        # parameter counter for theta
        idx = 0

        # hopping
        if 't' in self.fixed_params:
            t = self.fixed_params['t']
        else:
            t = theta[idx]; idx += 1

        # Coulomb interaction (U_i between sites)
        if 'U' in self.fixed_params:
            U = self.fixed_params['U']
        else:
            U = theta[idx: idx + self.n - 1]; idx += self.n - 1

        # onsite energies
        if 'eps' in self.fixed_params:
            eps = self.fixed_params['eps']
        else:
            eps = theta[idx: idx + self.n]; idx += self.n

        # pairing potential
        if 'Delta' in self.fixed_params:
            Delta = self.fixed_params['Delta']
        else:
            Delta = theta[idx]; idx += 1

        return {"t": t, "U": U, "eps": eps, "Delta": Delta}

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
                H += t * (self.cre_t[i] @ self.ann_t[i+1] + self.cre_t[i+1] @ self.ann_t[i])
                # pairing
                H += Delta * (self.cre_t[i] @ self.cre_t[i+1] + self.ann_t[i+1] @ self.ann_t[i])
                # Coulomb term
                H += U[i] * (self.num_t[i] @ self.num_t[i+1])

        return H