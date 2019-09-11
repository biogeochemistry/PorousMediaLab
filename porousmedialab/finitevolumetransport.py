
#! here will try to implement the FiPy transport using more flexible finite volume
#! method: https://stackoverflow.com/questions/53977269/how-to-couple-advection-diffusion-reaction-pdes-with-fipy/54696332?noredirect=1#comment102137539_54696332
#! many different problems including unsaturated flow:
#! https://bitbucket.org/klkuhlm/fipy-sims/src/default/


from fipy import *

g = 0.66
L = 10.
nx = 1000
mu1 = 1.
mu2 = 1.
K = 1.
D1 = 1.
D2 = 1.

mesh = Grid1D(dx=L / 1000, nx=nx)

x = mesh.cellCenters[0]
convCoeff = g*(x-L/2) * [[1.]]

u1 = CellVariable(name="u1", mesh=mesh, value=0., hasOld=True)
u2 = CellVariable(name="u2", mesh=mesh, value=1., hasOld=True)
u1.setValue(1., where=(4*L/10 < x) & (x < 6*L/10))

## Neumann boundary conditions
u1.faceGrad.constrain(0., where=mesh.facesLeft)
u1.faceGrad.constrain(0., where=mesh.facesRight)
u2.faceGrad.constrain(0., where=mesh.facesLeft)
u2.faceGrad.constrain(0., where=mesh.facesRight)

sourceCoeff1 = mu1*u1/(K+u1)
sourceCoeff2 = mu2*u2/(K+u1)

eq1 = (TransientTerm(var=u1)
       == DiffusionTerm(coeff=convCoeff, var=u1)
       + ConvectionTerm(coeff=convCoeff, var=u1)
       - ImplicitSourceTerm(coeff=g, var=u1)
       - ImplicitSourceTerm(coeff=sourceCoeff1, var=u1))
eq2 = (TransientTerm(var=u2)
       == DiffusionTerm(coeff=D2, var=u2)
       + ConvectionTerm(coeff=convCoeff, var=u2)
       - ImplicitSourceTerm(coeff=g, var=u2)
       + ImplicitSourceTerm(coeff=sourceCoeff2, var=u2))

eqn = eq1 & eq2
vi = Viewer((u1, u2))

for t in range(100):
    u1.updateOld()
    u2.updateOld()
    eqn.solve(dt=1.e-3)
    vi.plot()
