"""Finding the parameter sweet spot for the many body PMM system"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from operators import dL, dL_dag, dR, dR_dag

epsL, epsR = sp.symbols('ϵ_L,ϵ_R') #Onsite energies
t, Delta = sp.symbols('t,Δ') #tunneling and pairing amplit
U = sp.symbols('U') #Coulomb interaction


def create_Hamiltonian():
    """Create the Hamiltonian for the system.
    H = H_onsite + H_tunnel + H_pairing + H_coulomb

    """
    H_onsite = epsL * dL_dag * dL + epsR * dR_dag * dR
    H_tunnel = t * (dL_dag * dR + dR_dag * dL)
    H_pairing = Delta * (dR * dL + dL_dag * dR_dag)
    H_coulomb = U * (dL_dag * dL) * (dR_dag * dR)

    H = H_onsite + H_tunnel + H_pairing + H_coulomb
    return H

### Create a function that takes in parameters and returns Hamiltonian as a numpy array
def H_matrix(epsL_val, epsR_val, t_val, Delta_val, U_val):
    H = create_Hamiltonian()
    H = H.subs({epsL:epsL_val, epsR:epsR_val, t:t_val, Delta:Delta_val, U:U_val})
    H_matrix = np.array(sp.Matrix(H).evalf(), dtype=complex)
    return H_matrix


### Use this for parameter sweeps and to keep track of eigenvalues WITH corresponding eigenvectors

### Fist Get analytical sweet spot for U
def Compute_U(epsL,epsR,Delta,t):
    p1 = (Delta**2-t**2 + epsL*epsR)
    p2 = (epsL + epsR + 2*np.sqrt(t**2 + (epsL - epsR)**2))
    p3 = 2*(t**2 - epsL*epsR)
    return p1*p2/p3

def number_op_exp_val(evals, evecs):
    nL_op = np.array(sp.Matrix(dL_dag * dL).evalf(), dtype=complex)
    nR_op = np.array(sp.Matrix(dR_dag * dR).evalf(), dtype=complex)
    P_op  = np.eye(4) - 2*(nL_op + nR_op) + 4*(nL_op @ nR_op)

    # classify eigenvectors by parity
    parities = []
    for k in range(len(evals)):
        vec = evecs[:, k]
        exp_parity = np.real(vec.conj().T @ P_op @ vec)
        parities.append(np.sign(exp_parity))

    even_idxs = [i for i, p in enumerate(parities) if p > 0]
    odd_idxs  = [i for i, p in enumerate(parities) if p < 0]

    gs_even_idx = min(even_idxs, key=lambda i: evals[i])
    gs_odd_idx  = min(odd_idxs,  key=lambda i: evals[i])

    gs_even = evecs[:, gs_even_idx]
    gs_odd  = evecs[:, gs_odd_idx]

    # expectation values
    exp_even_left  = gs_even.conj().T @ nL_op @ gs_even
    exp_even_right = gs_even.conj().T @ nR_op @ gs_even
    exp_odd_left   = gs_odd.conj().T  @ nL_op @ gs_odd
    exp_odd_right  = gs_odd.conj().T  @ nR_op @ gs_odd

    return exp_even_left - exp_odd_left, exp_even_right - exp_odd_right

t_val = 1
Delta_val = 1
epsL_vals = np.linspace(-3,3,50)
epsR_vals = np.linspace(-3,3,50)

eigenvalues = np.zeros((len(epsL_vals),len(epsR_vals),4))
eigenvectors = np.zeros((len(epsL_vals),len(epsR_vals),4,4), dtype=complex)

left_expvals = np.zeros((len(epsL_vals),len(epsR_vals)))
right_expvals = np.zeros((len(epsL_vals),len(epsR_vals)))

for i, eps_val in enumerate(epsL_vals):
    for j, epsR_val in enumerate(epsR_vals):
        U_val = 1#Compute_U(eps_val, epsR_val, Delta_val, t_val)
        H = H_matrix(eps_val, epsR_val, t_val, Delta_val, U_val)
        evals, evecs = np.linalg.eigh(H)

        eigenvalues[i,j,:] = evals
        eigenvectors[i,j,:,:] = evecs



# Plot the eigenvalues as a function of epsL and epsR
ground_state_degeneracy = eigenvalues[:,:,1] - eigenvalues[:,:,0]
gap = eigenvalues[:,:,2] - eigenvalues[:,:,1]  
# print((ground_state_degeneracy==).sum())

plt.figure(figsize=(10, 8))
plt.contourf(epsL_vals, epsR_vals, ground_state_degeneracy, levels=50, cmap='magma')
plt.colorbar(label='Ground State Degeneracy')
#Mark the line where degeneracy is zero
CS = plt.contour(epsL_vals, epsR_vals, np.abs(ground_state_degeneracy),
                 levels=[0.02], colors='cyan', linewidths=2)
plt.clabel(CS, inline=1, fontsize=10, fmt='Degeneracy Line: %.2f')
# plt.contour(epsL_vals, epsR_vals, abs(ground_state_degeneracy), levels=[0,0.01], label='Degeneracy Line', colors='red', linestyles='dashed', linewidths=10)
plt.xlabel('ϵ_L')
plt.ylabel('ϵ_R')
plt.title('Ground State Degeneracy and Excitation Gap')
# plt.axis('equal')
plt.legend()
plt.show()



plt.figure(figsize=(10, 8))
plt.contourf(epsL_vals, epsR_vals, gap, levels=50, cmap='magma')
plt.colorbar(label='Excitation Gap')
plt.xlabel('ϵ_L')
plt.ylabel('ϵ_R')
plt.title('Ground State Degeneracy and Excitation Gap')
plt.legend()
plt.show()




t_val = 1
Delta_val = 1
eps_vals = np.linspace(-3,3,50)

eigenvalues = np.zeros((len(eps_vals),4))
eigenvectors = np.zeros((len(eps_vals),4,4), dtype=complex)

U_val = 1
left_expvals = np.zeros((len(eps_vals)))
right_expvals = np.zeros((len(eps_vals)))

for i, eps_val in enumerate(eps_vals):
    # U_val = Compute_U(eps_val, eps_val, Delta_val, t_val)
    H = H_matrix(eps_val, eps_val, t_val, Delta_val, U_val)
    evals, evecs = np.linalg.eigh(H)

    eigenvalues[i,:] = evals
    eigenvectors[i,:,:] = evecs

    left_expvals[i], right_expvals[i] = number_op_exp_val(evals,evecs)

plt.plot(eps_vals, left_expvals, label=r'$\langle n_L \rangle_{even} - \langle n_L \rangle_{odd}$')
plt.plot(eps_vals, right_expvals, label=r'$\langle n_R \rangle_{even} - \langle n_R \rangle_{odd}$')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.xlabel('ϵ_L = ϵ_R')
plt.ylabel(r'$\Delta n$')
plt.title(r'Difference in Particle Number vs $ϵ$ at the Sweet Spot $(ϵ_R=ϵ_L, t=1, Δ=1)$')
plt.legend()
plt.grid()
plt.show()


t_val = 1
Delta_val = 1
epsL_vals = np.linspace(-3,3,50)
epsR_vals = np.linspace(-3,3,50)

U_val = 1

left_expvals = np.zeros((len(epsL_vals),len(epsR_vals)))
right_expvals = np.zeros((len(epsL_vals),len(epsR_vals)))

for i, eps_val in enumerate(epsL_vals):
    for j, epsR_val in enumerate(epsR_vals):
        #U_val = Compute_U(eps_val, epsR_val, Delta_val, t_val)
        H = H_matrix(eps_val, epsR_val, t_val, Delta_val, U_val)
        evals, evecs = np.linalg.eigh(H)

        left_expvals[i,j], right_expvals[i,j] = number_op_exp_val(evals,evecs)


plt.figure(figsize=(10, 8))
plt.contourf(epsL_vals, epsR_vals, left_expvals*right_expvals, levels=50, cmap='magma')
plt.colorbar(label=r'$⟨ n_L ⟩ ⋅ ⟨ n_R ⟩$')
plt.contour(epsL_vals, epsR_vals, left_expvals*right_expvals, levels=[0], colors='cyan', linewidths=2)        
plt.xlabel('ϵ_L')
plt.ylabel('ϵ_R')
plt.title(r'Difference in Particle Number $⟨ n_L ⟩ ⋅ ⟨ n_R ⟩$')
# plt.axis('equal')
plt.legend(["Zero Difference Line"])
plt.show()  

