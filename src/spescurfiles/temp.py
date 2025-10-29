import sympy as sp


ϵ_L, ϵ_R, t, Δ, E = sp.symbols('ϵ_L ϵ_R t Δ E')
H = sp.Matrix([ [ϵ_L - E, t        , 0         ,          Δ],
                [t      , ϵ_R - E  , - Δ       ,          0],
                [0      , -Δ       , -ϵ_L - E  ,         -t],
                [Δ      , 0        , -t        ,   -ϵ_R - E]])

det = sp.simplify(H.det())
print(det)
a = -2*t**2 - Δ**2 - ϵ_L**2 - ϵ_R**2
b = (t**2 - ϵ_L*ϵ_R - Δ**2)**2
E_squared = sp.solve(sp.Eq(det, 0), E**2)
print("E²= ",E_squared)

# Set eps_L, eps_R to zero and t= Delta.
E_squared_sweet_spot = [expr.subs({ϵ_L:0, ϵ_R:0, t:Δ}) for expr in E_squared]
print("At the sweet spot (ϵ_L=ϵ_R=0, t=Δ): E²= ", E_squared_sweet_spot)
# find eigenvectors and eigenvalues
H_sweet_spot = H.subs({ϵ_L:0, ϵ_R:0, t:Δ})
eigen_data = H_sweet_spot.eigenvects()
for val, mult, vecs in eigen_data:
    print(f"Eigenvalue: {val}, Multiplicity: {mult}, Eigenvectors: {vecs}")
