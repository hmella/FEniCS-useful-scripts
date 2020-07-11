from dolfin import *
import time

# Mesh and function spaces
mesh = UnitSquareMesh(100, 100)
CG2 = FiniteElement('CG',mesh.ufl_cell(),2)
DG1 = VectorElement('DG',mesh.ufl_cell(),1)
V = FunctionSpace(mesh, CG2)
T = FunctionSpace(mesh, DG1)

# Build approximated projection operator
A = assemble(inner(TrialFunction(T), TestFunction(T))*dx)
b = assemble(inner(grad(Function(V)), TestFunction(T))*dx)
ones = Function(T)
ones.vector()[:] = 1
A_diag = A * ones.vector()
A_diag.set_local(1.0/A_diag.get_local())

# Test
u = interpolate(Expression('sin(x[1])*cos(x[0])',degree=2), V)
t0 = time.time()
grad_ref = project(grad(u), T)
t_proj = time.time() - t0

grad_e = Function(T)
t0 = time.time()
grad_e.vector()[:] = assemble(inner(grad(u),TestFunction(T))*dx) * A_diag
t_est = time.time() - t0
print('   Projection time:    {:f}'.format(t_proj))
print('   Approximation time: {:f}'.format(t_est))
print('   Speed-up:           {:.1f} X'.format(t_proj/t_est))
print(norm(grad_ref))
print(norm(grad_e))