"""This file is just for me to verify the operators I use in the Hamiltonian."""
import sympy as sp
import numpy as np

# Four dimentional Fock space basis
v00 = sp.Matrix([[1,0,0,0]]).T
v01 = sp.Matrix([[0,1,0,0]]).T
v10 = sp.Matrix([[0,0,1,0]]).T
v11 = sp.Matrix([[0,0,0,1]]).T

# Creation and annihilation operators for left dot
dL = sp.Matrix([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])
dL_dag =  dL.T
dR = sp.Matrix([[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]])
dR_dag = dR.T

def test_dL():
    """Test the left dot annihilation operator."""
    assert dL@v00 == sp.Matrix([[0],[0],[0],[0]])
    assert dL@v01 == sp.Matrix([[0],[0],[0],[0]])
    assert dL@v10 == v00
    assert dL@v11 == v01
    print("dL tests passed.")

def test_dL_dag():
    """Test the left dot creation operator."""
    assert dL_dag@v00 == v10
    assert dL_dag@v01 == v11
    assert dL_dag@v10 == sp.Matrix([[0],[0],[0],[0]])
    assert dL_dag@v11 == sp.Matrix([[0],[0],[0],[0]])
    print("dL_dag tests passed.")

def test_dR():
    """Test the right dot annihilation operator."""
    assert dR@v00 == sp.Matrix([[0],[0],[0],[0]])
    assert dR@v01 == v00
    assert dR@v10 == sp.Matrix([[0],[0],[0],[0]])
    assert dR@v11 == v10
    print("dR tests passed.")

def test_dR_dag():
    """Test the right dot creation operator."""
    assert dR_dag@v00 == v01
    assert dR_dag@v01 == sp.Matrix([[0],[0],[0],[0]])
    assert dR_dag@v10 == v11
    assert dR_dag@v11 == sp.Matrix([[0],[0],[0],[0]])
    print("dR_dag tests passed.")


"""Create the many-body Hamiltonian for the QD-SC-QD system."""
"""  
H = ∑_{α=L,R} ϵ_{α} d_{α}^{†}d_{α} + t (d_L^{†} d_R + d_R^{†} d_L) + Δ (d_R d_L + d_L^{†} d_R^{†})  + U_{LR} n_L n_R

"""

epsL, epsR = sp.symbols('ϵ_L,ϵ_R') #Onsite energies
t, Delta = sp.symbols('t,Δ') #tunneling and pairing amplitudes
U = sp.symbols('U') #Coulomb interaction

H_onsite = epsL * dL_dag * dL + epsR * dR_dag * dR
H_tunnel = t * (dL_dag * dR + dR_dag * dL)
H_pair = Delta * (dR * dL + dL_dag * dR_dag)
nL = dL_dag * dL
nR = dR_dag * dR
H_Coulomb = U * nL * nR

H_total = H_onsite + H_tunnel + H_pair + H_Coulomb
H_total = sp.simplify(H_total)


if __name__ == "__main__":
    test_dL()
    test_dL_dag()
    test_dR()
    test_dR_dag()
    print("Hamiltonian H_total:")
    sp.pprint(H_total)
