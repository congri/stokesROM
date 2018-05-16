"""This should be an E-step module for Stokes/Darcy"""

import dolfin as df
import dolfin_adjoint as dfa
import numpy as np

# Create mesh and define function space
coarseMesh = df.UnitSquareMesh(2, 2)
V = df.FunctionSpace(coarseMesh, 'CG', 1)
Vcond = df.FunctionSpace(coarseMesh, 'DG', 0)


# Define Dirichlet boundary
class Dirichlet_bc(df.SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] < df.DOLFIN_EPS)


dirichlet_bc = Dirichlet_bc()

# Define essential boundary condition function
u0 = df.Expression('0.0', degree=2)
bc = df.DirichletBC(V, u0, dirichlet_bc, method='pointwise')

# Define natural boundary condition function
q_n = df.Expression('x[1]', degree=2)

# Source term
f = df.Expression('0.0', degree=2)

# Sample conductivity
cond = df.Function(Vcond)
cond.vector()[:] = 1.0 + np.random.lognormal(0, 1.0, Vcond.dim())

# Define variational problem
u = df.TrialFunction(V)
v = df.TestFunction(V)
a = cond * df.inner(df.grad(v), df.grad(u)) * df.dx
L = f * v * df.dx + q_n * v * df.ds

# Compute solution
u = df.Function(V)
dfa.solve(a == L, u, bc)

# Define simple functional. Remember to change finite difference functional!!!
J = dfa.Functional(df.inner(u, u) * df.dx)

# Compute gradient using dolfin adjoint
grad = dfa.compute_gradient(J, dfa.Control(cond), forget=False)

grad_array = np.asarray(grad.vector()[:])

print('true gradient: ', grad_array)


FDcheck = False
if FDcheck:
    # Define functional in form suitable for finite differences
    def fd_functional(u_func):
        return df.assemble(df.inner(u_func, u_func) * df.dx)

    # For repeated finite difference evaluation
    def forwardSolver(x):
        uu = df.TrialFunction(V)
        vv = df.TestFunction(V)
        cond_tmp = df.Function(Vcond)
        cond_tmp.vector()[:] = x
        aa = cond_tmp * df.inner(df.grad(uu), df.grad(vv)) * df.dx
        LL = f * vv * df.dx + q_n * vv * df.ds
        uu = df.Function(V)
        dfa.solve(aa == LL, uu, bc)
        return uu


    # Get finite difference gradient
    h = 1e-6
    cond_0 = cond.vector().get_local()
    grad_fd_array = np.zeros(Vcond.dim())
    u_0 = forwardSolver(cond_0)
    J_0 = fd_functional(u_0)
    print('J_0 = ', J_0)

    for i in range(Vcond.dim()):
        cond_fd = cond_0.copy()

        cond_fd[i] = cond_0[i] + h

        u_fd = forwardSolver(cond_fd)
        J_fd = fd_functional(u_fd)

        grad_fd_array[i] = (J_fd - J_0) / h

    print('fd gradient: ', grad_fd_array)
    print('relative gradient: ', np.divide(grad_array, grad_fd_array))
