from full_system_hamiltonian import *
from get_setup import params_for_n_site_Hamiltonian
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import expm
from numba import njit, objmode
from scipy.sparse.linalg import eigsh
import time



class Hamiltonian:
    def __init__(self, n_sites, dupes, specified_vals={"U":[0.1]}):
        self.n_sites = n_sites
        self.dupes = dupes
        self.specified_vals = specified_vals
        self.pars, self.extras = params_for_n_site_Hamiltonian(n_sites, dupes, specified_vals,path="/home/Hishem/repos/MastersThesis/configuration.json")
    
    def precompute_operators(self, N=None):
        if N is None:
            N = self.n_sites * self.dupes
        self.operators = {}
        cre, ann, num = precompute_ops(N)

        self.operators['cre'] = cre
        self.operators['ann'] = ann
        self.operators['num'] = num
        hop_ops = {}
        pair_ops = {}
        dens_ops = {}
        for d in range(self.dupes):
            off = d * self.n_sites
            for i in range(self.n_sites-1):
                a, b = off+i, off+i+1
                hop_ops[(a,b)] = cre[a] @ ann[b] + cre[b] @ ann[a] 
                pair_ops[(a,b)] = cre[a] @ cre[b] + ann[b] @ ann[a]
                dens_ops[(a,b)] = num[a] @ num[b]
        # Inter PMM coupling terms
        # Inner dots only for now
        hop_ops[(0,self.n_sites)] = cre[0] @ ann[self.n_sites] + ann[0] @ cre[self.n_sites] #A0 - B0
        hop_ops[(0,2*self.n_sites)] = cre[0] @ ann[2*self.n_sites] + ann[0] @ cre[2*self.n_sites] #A0 - C0
        # hop_ops[(n,2*n)] = cre[n] @ ann[2*n] + ann[n] @ cre[2*n] #B0 - C0
        self.operators['hop'] = hop_ops
        self.operators['pair'] = pair_ops
        self.operators['dens'] = dens_ops
        return self.operators
    
    def simple_delta_pulse(t, T_peak, width, s, max_val,min_val):
        T_start = T_peak - width / 2
        T_end = T_peak + width / 2

        rise = 1/(1 + np.exp(-s*(t - T_start)))
        fall = 1/(1 + np.exp(s*(t - T_end)))

        return min_val + (max_val - min_val) * rise * fall
    
    def full_system_hamiltonian(self, couplings=(), eps_detune=None):
        
        cre = self.operators['cre']
        ann = self.operators['ann']
        num = self.operators['num']
        hop_ops = self.operators['hop']
        pair_ops = self.operators['pair']
        dens_ops = self.operators['dens']

        big_N = self.n_sites * self.dupes
        n = self.n_sites
        dup = self.dupes

        t_vals = self.pars["t"]
        delta_vals = self.pars["delta"]
        U_vals = self.pars["U"]
        eps_vals = self.pars["eps"]
        

        dim = 2**big_N
        H = np.zeros((dim, dim), dtype=complex)
       
        # cre, ann, num = precompute_ops(big_N)

        eps_full = np.tile(eps_vals, (dup,1))
        if eps_detune is not None:
            for i in range(len(eps_detune)):
                site, node, val = eps_detune[i]
                eps_full[int(site), int(node)] = val


        # Intra PMM terms
        for d in range(dup):
            off = d * n
            for j in range(n-1):
                i, k = off+j, off+j+1
                # print(f"Adding intra-PMM terms between sites {i} and {k}")
                H += -t_vals[j]   * hop_ops[(i,k)]
                H +=  delta_vals[j] * pair_ops[(i,k)]
                H +=  U_vals[j]   * dens_ops[(i,k)]

            for j in range(n):
                H += eps_full[d,j] * num[off+j]


        #  Inter or additional PMM couplings
        for cA, cB, t_c, d_c in couplings:
            if cA is None or cB is None:
                continue

            i = cA[0]*n + cA[1]
            j = cB[0]*n + cB[1]
            key = (min(i,j), max(i,j))
            if t_c != 0:
                H += -t_c * hop_ops[key]
            if d_c != 0 and cA[0]*n == cB[0]*n:
                H +=  d_c * pair_ops[key]
            if cA[0] != cB[0] and d_c != 0:
                raise RuntimeError("Inter-PMM pairing forbidden")
        return H

    def get_coupling_information(self, Total_time, time_steps, params, tJ, dJ, eps_detune_val):

    #Base parameters
    t_val, U_val, eps_val, Delta_val = self.pars["t"], self.pars["U"], self.pars["eps"], self.pars["delta"]
    width = Total_time / 3
    s = 20/width#

    dT, dD = 5, 5  # Add perturbations to intra PMM couplings

    #System setup
    A, B, C = 0, 1, 2
    outer, middle, inner = 2, 1, 0

    time_array = np.linspace(0, Total_time, time_steps)
    eps_detune_min = eps_val[0]
    eps_detune = []
    couplings = []
    for i in range(len(time_array)):
        t = time_array[i]

        #Tune chemical potentials for detuning
        detune_A = simple_delta_pulse(t, Total_time/2, 2*width, s, eps_detune_val, 0.0)
        
        tj_pulse_AB = simple_delta_pulse(t, Total_time/3, width, s, tJ, 0.0)
        tj_pulse_AC = simple_delta_pulse(t, 2*Total_time/3, width, s, tJ, 0.0)

        eps_detune.append([[A, outer, detune_A]])

        tj_pulse_A = (tj_pulse_AB + tj_pulse_AC) / np.linalg.norm(2*tJ) * (tJ + dT)  #+ t_val[0]
        dj_pulse_A = (tj_pulse_AB + tj_pulse_AC) / np.linalg.norm(2*tJ) * (dJ + dD)  #+ Delta_val[0]

        coups = [[(A, inner), (B, inner), tj_pulse_AB, 0.0],
                 [(C, inner), (A, inner), tj_pulse_AC, 0.0],
                 [(A, inner), (A, middle), tj_pulse_A, dj_pulse_A],
                 [(A, middle), (A, outer), tj_pulse_A, dj_pulse_A]
                 ]
        couplings.append(coups)

    return couplings, eps_detune, time_array