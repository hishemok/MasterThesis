"""Finding the parameter sweet spot for the many body PMM system"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from operators import dL, dL_dag, dR, dR_dag

epsL, epsR = sp.symbols('ϵ_L,ϵ_R') #Onsite energies
t, Delta = sp.symbols('t,Δ') #tunneling and pairing amplit
U = sp.symbols('U') #Coulomb interaction

def check_eigenvalues():
    """Compare the analytical and numerical eigenvalues of the Hamiltonian."""
    H_onsite = epsL * dL_dag * dL + epsR * dR_dag * dR
    H_tunnel = t * (dL_dag * dR + dR_dag * dL)
    H_pairing = Delta * (dR * dL + dL_dag * dR_dag)
    H_coulomb = U * (dL_dag * dL) * (dR_dag * dR)

    H = H_onsite + H_tunnel + H_pairing + H_coulomb
    H = sp.simplify(H)
    eigvals = H.eigenvals()

    ### Analytical eigenvalues from block diagonalization
    S = epsL + epsR + U
    E_even = [(S + sp.sqrt(S**2 + 4*Delta**2))/2, (S - sp.sqrt(S**2 + 4*Delta**2))/2]
    E_odd  = [((epsL + epsR) + sp.sqrt((epsL - epsR)**2 + 4*t**2))/2, ((epsL + epsR) - sp.sqrt((epsL - epsR)**2 + 4*t**2))/2]

    ### Check equality
    all_eigvals = E_even + E_odd
    for val in all_eigvals:
        found = any(sp.simplify(val - ev) == 0 for ev in eigvals.keys())
        assert found, f"Eigenvalue {val} not found in numerical eigenvalues."
    
    return E_even, E_odd

Even, Odd = check_eigenvalues()


def number_operators():
    nL_op = dL_dag * dL
    nR_op = dR_dag * dR
    return nL_op, nR_op

def n_op_expval(even:np.ndarray, odd:np.ndarray):
    nL_op, nR_op = number_operators()
    #convert to numpy arrays
    nL_op = np.array(sp.Matrix(nL_op).evalf(), dtype=complex)
    nR_op = np.array(sp.Matrix(nR_op).evalf(), dtype=complex)

    e_bra = even.T.conj()
    e_ket = even

    o_bra = odd.T.conj()
    o_ket = odd

    exp_nL_even = np.zeros(even.shape[0], dtype=complex)
    exp_nR_even = np.zeros(even.shape[0], dtype=complex)
    exp_nL_odd = np.zeros(odd.shape[0], dtype=complex)
    exp_nR_odd = np.zeros(odd.shape[0], dtype=complex)

    for i in range(even.shape[0]):
        exp_nL_even[i] = e_bra[i,:] @ nL_op @ e_ket[:,i]
        exp_nR_even[i] = e_bra[i,:] @ nR_op @ e_ket[:,i]
        exp_nL_odd[i] = o_bra[i,:] @ nL_op @ o_ket[:,i]
        exp_nR_odd[i] = o_bra[i,:] @ nR_op @ o_ket[:,i]


    # exp_nL_even = e_bra @ nL_op @ e_ket
    # exp_nR_even = e_bra @ nR_op @ e_ket
    # exp_nL_odd = o_bra @ nL_op @ o_ket
    # exp_nR_odd = o_bra @ nR_op @ o_ket

    diff_nl = exp_nL_even - exp_nL_odd
    diff_nr = exp_nR_even - exp_nR_odd

    return diff_nl, diff_nr



#Check parameters for sweet spot
"""Sweet spot conditions:
[E_even]_min = [E_odd]_min
"""

def Compute_U(epsL,epsR,Delta,t):
    p1 = (Delta**2-t**2 + epsL*epsR)
    p2 = (epsL + epsR + 2*np.sqrt(t**2 + (epsL - epsR)**2))
    p3 = 2*(t**2 - epsL*epsR)
    return p1*p2/p3

eps_vals = np.linspace(0.01, 0.5, 100)
t_val = 1
Delta_val = 1

gs_even = []
ex_even = []
gs_odd = []
ex_odd = []

diff_nl = []
diff_nr = []

for i, eps in enumerate(eps_vals):
    U_val = Compute_U(eps,eps,Delta_val,t_val)


    even0 = float(Even[0].subs({epsL:eps, epsR:eps, U:U_val, Delta:Delta_val}).evalf())
    even1 = float(Even[1].subs({epsL:eps, epsR:eps, U:U_val, Delta:Delta_val}).evalf())
    odd0  = float(Odd[0].subs({epsL:eps, epsR:eps, t:t_val}).evalf())
    odd1  = float(Odd[1].subs({epsL:eps, epsR:eps, t:t_val}).evalf())

    gs_even.append(min(even0, even1))
    ex_even.append(max(even0, even1))
    gs_odd.append(min(odd0, odd1))
    ex_odd.append(max(odd0, odd1))
    

gs_even = np.array(gs_even)
gs_odd = np.array(gs_odd)
diff_nl, diff_nr = n_op_expval(gs_even, gs_odd)
plt.plot(eps_vals, diff_nl, label=r'$\Delta n_L$', linestyle='-')
plt.plot(eps_vals, diff_nr, label=r'$\Delta n_R$', linestyle='--')
plt.ylabel(r'$\Delta n$')
plt.title(r'Difference in Particle Number vs $ϵ$ at the Sweet Spot $(ϵ_R=ϵ_L, t=1, Δ=1)$')
plt.legend()
plt.grid()
plt.show()

exit()


plt.figure(figsize=(12, 12))
plt.subplot(3,1,1)
plt.plot(eps_vals, gs_even, label=r'$GS_{even}$', linestyle='-')
plt.plot(eps_vals, ex_even, label=r'$EX_{even}$', linestyle='--')
plt.plot(eps_vals, gs_odd, label=r'$GS_{odd}$', linestyle='-')
plt.plot(eps_vals, ex_odd, label=r'$EX_{odd}$', linestyle='--')
plt.ylabel('Eigenvalues')
plt.title(r'Eigenvalues vs $ϵ$ at the Sweet Spot $(ϵ_R=ϵ_L, t=1, Δ=1)$')
plt.legend()
plt.grid()

plt.subplot(3,1,2)
GS_gap = np.array(gs_even) - np.array(gs_odd)
plt.plot(eps_vals, GS_gap, label=r'$GS_{odd} - GS_{even}$', color='purple')  
plt.ylabel(r'$GS_{odd} - GS_{even}$')
plt.title(r'Gap between $GS_{odd}$ and $GS_{even}$ vs $ϵ$ at the Sweet Spot $(ϵ_R=ϵ_L, t=1, Δ=1)$')
plt.legend()
plt.grid()
plt.tight_layout()

plt.subplot(3,1,3)
#Lowest excited state - largest ground state
gs_even = np.array(gs_even)
gs_odd = np.array(gs_odd)
ex_even = np.array(ex_even)
ex_odd = np.array(ex_odd)
ex_low = np.minimum(ex_even, ex_odd)
gs_high = np.maximum(gs_even, gs_odd)
EX_gap = ex_low - gs_high
plt.plot(eps_vals, EX_gap, label=r'$min(EX) - max(GS)$', color='green')
plt.ylabel(r'$min(EX) - max(GS)$')
plt.xlabel(r'$ϵ$ (with $ϵ_L=ϵ_R, t=1, Δ=1$)')
plt.title(r'Gap between $min(EX)$ and $max(GS)$ vs $ϵ$ at the Sweet Spot $(ϵ_R=ϵ_L, t=1, Δ=1)$')
plt.legend()
plt.grid()
plt.tight_layout()
# plt.savefig('Figures/manybody_sweetspot_eps.png', dpi=300)
plt.show()



eps = 0.5
Delta_val = 1
t_vals = np.linspace(0.7, 1.3, 100)
gs_even = []
ex_even = []
gs_odd = []
ex_odd = []


for i, t_val in enumerate(t_vals):
    U_val = Compute_U(eps,eps,Delta_val,t_val)

    even0 = float(Even[0].subs({epsL:eps, epsR:eps, U:U_val, Delta:Delta_val}).evalf())
    even1 = float(Even[1].subs({epsL:eps, epsR:eps, U:U_val, Delta:Delta_val}).evalf())
    odd0  = float(Odd[0].subs({epsL:eps, epsR:eps, t:t_val}).evalf())
    odd1  = float(Odd[1].subs({epsL:eps, epsR:eps, t:t_val}).evalf())

    gs_even.append(min(even0, even1))
    ex_even.append(max(even0, even1))
    gs_odd.append(min(odd0, odd1))
    ex_odd.append(max(odd0, odd1))

plt.figure(figsize=(12, 12))
plt.subplot(3,1,1)
plt.plot(t_vals, gs_even, label=r'$GS_{even}$', linestyle='-')
plt.plot(t_vals, ex_even, label=r'$EX_{even}$', linestyle='--')
plt.plot(t_vals, gs_odd, label=r'$GS_{odd}$', linestyle='-')
plt.plot(t_vals, ex_odd, label=r'$EX_{odd}$', linestyle='--')
plt.ylabel('Eigenvalues')
plt.title(r'Eigenvalues vs $t$ at the Sweet Spot $(ϵ_R=ϵ_L=0.5, Δ=1)$')
plt.legend()
plt.grid()
plt.subplot(3,1,2)
GS_gap = np.array(gs_even) - np.array(gs_odd)
plt.plot(t_vals, GS_gap, label=r'$GS_{odd} - GS_{even}$', color='purple')  
plt.ylabel(r'$GS_{odd} - GS_{even}$')
plt.title(r'Gap between $GS_{odd}$ and $GS_{even}$ vs $t$ at the Sweet Spot $(ϵ_R=ϵ_L=0.5, Δ=1)$')
plt.legend()
plt.grid()
plt.tight_layout()  
plt.subplot(3,1,3)
#Lowest excited state - largest ground state
gs_even = np.array(gs_even)
gs_odd = np.array(gs_odd)
ex_even = np.array(ex_even)
ex_odd = np.array(ex_odd)
ex_low = np.minimum(ex_even, ex_odd)
gs_high = np.maximum(gs_even, gs_odd)
EX_gap = ex_low - gs_high   
plt.plot(t_vals, EX_gap, label=r'$min(EX) - max(GS)$', color='green')
plt.ylabel(r'$min(EX) - max(GS)$')
plt.xlabel(r'$t$ (with $ϵ_L=ϵ_R=0.5, Δ=1)$')
plt.title(r'Gap between $min(EX)$ and $max(GS)$ vs $t$ at the Sweet Spot $(ϵ_R=ϵ_L=0.5, Δ=1)$')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('Figures/manybody_sweetspot_t.png', dpi=300)
plt.show()

