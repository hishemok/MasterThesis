from full_system_hamiltonian import *
from get_setup import params_for_n_site_Hamiltonian
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import expm
from numba import njit, objmode
from scipy.sparse.linalg import eigsh
import time

n_sites = 3
dupes = 3
big_N = n_sites * dupes


pars, extras = params_for_n_site_Hamiltonian(n_sites, configs=None, specified_vals={"U": [0.1]}, path="configuration.json")


t, U, eps, Delta = pars

print("t, U, eps, Delta =", t, U, eps, Delta)

t_couple = t 
delta_couple = Delta


operators = {}
cre, ann, num = precompute_ops(big_N)
operators['cre'] = cre
operators['ann'] = ann
operators['num'] = num
hop_ops = {}
pair_ops = {}
dens_ops = {}
for i in range(big_N):
    for j in range(i+1, big_N):
        hop_ops[(i,j)] = cre[i] @ ann[j] + ann[i] @ cre[j]
        pair_ops[(i,j)] = cre[i] @ cre[j] + ann[j] @ ann[i]
        dens_ops[(i,j)] = num[i] @ num[j]
operators['hop'] = hop_ops
operators['pair'] = pair_ops
operators['dens'] = dens_ops



def simple_delta_pulse(t, T_peak, width, s, max_val,min_val):
    T_start = T_peak - width / 2
    T_end = T_peak + width / 2

    rise = 1/(1 + np.exp(-s*(t - T_start)))
    fall = 1/(1 + np.exp(s*(t - T_end)))

    return min_val + (max_val - min_val) * rise * fall


AB_coupling = [(0,2), (1,0), 0.0, 0.0]
BC_coupling = [(1,0), (2,0), 0.0, 0.0]
CA_coupling = [(2,0), (0,2), 0.0, 0.0]

couplings = [
    AB_coupling,
    BC_coupling,
    CA_coupling
]

eps_detune = None#{1: 0.0}
H = big_H(n_sites, dupes, t, U, eps, Delta, couplings=couplings, eps_detune=eps_detune, operators=operators)


eigvals, eigvecs = np.linalg.eigh(H)
print(eigvals[:10])

ordered_couplings = {
    "AB": {
        "idx0": (0,2),
        "idx1": (1,0),
        "t_coup": None,
        "Delta_coup": None
    },
    "eps_detune": None,
    "BC": {
        "idx0": (1,0),
        "idx1": (2,0),
        "t_coup": None,
        "Delta_coup": None
    },
}


def time_evolve(Total_time, n_steps, params, t_coup_max, Delta_coup_max, eps_detune_max, ordered_couplings, operators, lower_bound=10):

    
    t_val, U_val, eps_val, Delta_val = params

    couplings = []
    eps_detune = {}
    OC = ordered_couplings
    for i in range(len(OC)): 
        key = list(OC.keys())[i]
        
        current = OC[key]
        print(current)
        if key == "eps_detune":
            if current is not None:
                eps_detune = current
            else: 
                eps_detune = {1: eps_detune_max}
        else:
            if current['t_coup'] is None:
                current['t_coup'] = t_coup_max
            if current['Delta_coup'] is None:
                current['Delta_coup'] = Delta_coup_max
            couplings.append([current['idx0'], current['idx1'], current['t_coup'], current['Delta_coup']])

    print("Couplings for time evolution:", couplings)
    print("Eps detune for time evolution:", eps_detune)


    #Create Coupling pulse arrays:
    time_array = np.linspace(0, Total_time, n_steps)
    dt = time_array[1] - time_array[0]
    ##Order of pulses AB peak at T=0 and T=Total_time, Eps Detune at T=Total_time/3, BC at T=2*Total_time/3
    AB_t_peaks = [0, Total_time]
    BC_t_peaks = [2*Total_time/3]
    eps_t_peaks = [Total_time/3]

    width = T_total / 3
    s = T_total * 6

    All_Couplings = []
    epsilons = []

    # Precompute the pulses
    for t in time_array:

        eps_detune_val = simple_delta_pulse(t, eps_t_peaks[0], width, s, eps_detune_max , 0)
        epsilons.append(eps_detune_val)
        
        current_couplings = []
        for i,coup in enumerate(couplings):
            # print(i, coup)
            if i == 0:
                curr_t = simple_delta_pulse(t, AB_t_peaks[0], width, s, coup[2] , 0) + simple_delta_pulse(t, AB_t_peaks[1], width, s, coup[2] , 0)
                curr_delta = simple_delta_pulse(t, AB_t_peaks[0], width, s, coup[3] , 0) + simple_delta_pulse(t, AB_t_peaks[1], width, s, coup[3] , 0)
                
                current_coupling = [coup[0], coup[1], curr_t, curr_delta]
                current_couplings.append(current_coupling)
            else:
                curr_t = simple_delta_pulse(t, BC_t_peaks[0], width, s, coup[2] , 0)
                curr_delta = simple_delta_pulse(t, BC_t_peaks[0], width, s, coup[3] , 0)
                
                current_coupling = [coup[0], coup[1], curr_t, curr_delta]
                current_couplings.append(current_coupling)

        All_Couplings.append(current_couplings)
    

    coupling_pulses = {
        "AB_coupling": [All_Couplings[i][0] for i in range(len(All_Couplings))],
        "BC_coupling": [All_Couplings[i][1] for i in range(len(All_Couplings))],
        "eps_detune": epsilons
    }

    eigvals = np.zeros((n_steps, lower_bound))
    eigvecs = np.zeros((n_steps, int(2**big_N), lower_bound), dtype=complex)

    # Time Evolution
    for i in tqdm(range(len(time_array))):
        H_t = big_H(n_sites, dupes, t_val, U_val, eps_val, Delta_val, couplings=All_Couplings[i], eps_detune={1: epsilons[i]}, operators=operators)
   
        vals, vecs = np.linalg.eigh(H_t)
        eigvals[i,:] = vals[:lower_bound]
        eigvecs[i,:,:] = vecs[:,:lower_bound]

    
    return eigvals, eigvecs, time_array, coupling_pulses


T_total = 300
n_steps = 500
t_coup_max = t_couple[0]
Delta_coup_max = delta_couple[0]
eps_detune_max = 1


eigvals, eigvecs, time_array, coupling_pulses = time_evolve(T_total, n_steps, pars, t_coup_max, Delta_coup_max, eps_detune_max, ordered_couplings, operators)

plt.figure(figsize=(10,6))
for i in range(eigvals.shape[1]):
    plt.plot(time_array, eigvals[:,i], label=f"Level {i}")
plt.xlabel("Time")
plt.ylabel("Energy")
plt.title("Energy Levels During Braiding Process")
plt.legend()
plt.grid()
plt.show()  

#plot coupling pulses
plt.figure(figsize=(10,6))
AB_coups = coupling_pulses['AB_coupling']
BC_coups = coupling_pulses['BC_coupling']
epsilons = coupling_pulses['eps_detune']
AB_t_vals = [coup[2] for coup in AB_coups]
AB_Delta_vals = [coup[3] for coup in AB_coups]
BC_t_vals = [coup[2] for coup in BC_coups]
BC_Delta_vals = [coup[3] for coup in BC_coups]
plt.plot(time_array, AB_t_vals, label="AB t coupling")
plt.plot(time_array, AB_Delta_vals, label="AB Delta coupling")
plt.plot(time_array, BC_t_vals, label="BC t coupling")
plt.plot(time_array, BC_Delta_vals, label="BC Delta coupling")
plt.plot(time_array, epsilons, label="Eps detune")
plt.xlabel("Time")
plt.ylabel("Coupling Strength")
plt.title("Coupling Pulses During Braiding Process")
plt.legend()
plt.grid()
plt.show()


def majorana_operators(n):
    create, annihilate, number = precompute_ops(n)
    majorana_ops = []
    for j in range(n):
        f_dag = create[j]
        f = annihilate[j]
        gamma_1 = f + f_dag
        gamma_2 = -1j * (f - f_dag)
        majorana_ops.append((gamma_1, gamma_2))
    return majorana_ops


K = 10    # e.g. 2, 4, 8 Lower bound
rho = sum(
    np.outer(eigvecs[i,:,k], eigvecs[i,:,k].conj())
    for k in range(K)
) / K

def majorana_correlation_matrix(rho, gamma_ops):
    """
    gamma_ops : flat list [γ1, γ2, ..., γ_{2N}]
    """
    n = len(gamma_ops)
    C = np.zeros((n, n))

    for a in range(n):
        for b in range(n):
            C[a, b] = np.trace(rho @ gamma_ops[a] @ gamma_ops[b]).real

    # Enforce antisymmetry
    C = 0.5 * (C - C.T)
    return C

C = majorana_correlation_matrix(rho, [g for pair in majorana_operators(big_N) for g in pair])
print(C)
plt.figure(figsize=(8,6))
plt.imshow(C, cmap='bwr', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
plt.title('Majorana Correlation Matrix')
plt.xlabel('Majorana Index')
plt.ylabel('Majorana Index')
plt.show()  

evals = np.linalg.eigvals(1j * C)
plt.figure(figsize=(8,6))
plt.plot(np.sort(evals), 'o-')
plt.title('Eigenvalues of iC')
plt.xlabel('Index')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()