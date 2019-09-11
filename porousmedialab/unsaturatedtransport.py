#! many different problems including unsaturated flow:
#! https://bitbucket.org/klkuhlm/fipy-sims/src/default/

from fipy import Grid1D, CellVariable, FaceVariable, TransientTerm, DiffusionTerm, Viewer
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# fipy script implementing Richards equation for unsaturated vertical flow.
# Gravity acts "down" is in direction of increasing x (like depth).

# Brooks-Corey model (1) sort of works, but has issues near saturation
# van Genuchten model (3) is more robust.

# capillary pressure models taken from:
# Warrick (2003) "Soil Water Dynamics", Table 2-1


L = 10.0
nx = 200
dx = L/nx

# since this is sort of like an explicit convection (of moisture by gravity)
# solution (i.e., .divergence() term in PDE), timeStep must be kept small for large K
timeStep = 0.01  # seconds
num_time_steps = 1000

mesh = Grid1D(dx=dx, nx=nx)
x_cc = mesh.cellCenters.value[0]
x_fc = mesh.faceCenters.value[0]

# saturated/common properties
# could make this heterogeneous by making it a FaceVariable
Ks = 1.0E-2     # saturated K [m/sec]

# rather dry "background"
h0 = -10.0
h = CellVariable(name="$\psi$", mesh=mesh, value=h0, hasOld=True) # [m] capillary head

# initial condition with middle section wetter than background (but not saturated)
h.setValue(-1.5, where=(x_cc > 3.0)&(x_cc < 5.0))

# to specify initial condition in terms of saturation, need h(Se) implemented for each model

C = CellVariable(name="$C(h)$", mesh=mesh, value=0.0) # moisture capacity
K = FaceVariable(name="$K(S_e)$", mesh=mesh, value=0.0) # hydraulic conductivity
Se = FaceVariable(name="$S_e(h)$", mesh=mesh,value=0.0) # saturation

# negative flux on left side means flow into domain
#h.faceGrad.constrain(-5.0*Ks, mesh.facesLeft)  # flux into top of domain
h.constrain(-1.0, mesh.facesRight) # water table condition at bottom

# define functional forms of capillary pressure models
# when a FaceVariable (nx+1) is needed from a CellVariable (nx)
# .arithmeticVaceValue is simpler
# .harmonicFaceValue is maybe better

alpha = 0.5
m = 0.75 # 0 < m < 1
n = 1.0/(1.0 - m) # typical
p = 0.5 # typical for Mualem-vG

def calc_C(x):
  # ((m-1) n (alpha (-x))^n ((alpha (-x))^n+1)^(m-2))/x
  return ((m-1)*n*(-alpha*x)**n * ((-alpha*x)**(n+1))**(m-2))/x

def calc_Se(x):
  return (1.0 + (-alpha*x)**n)**(-m)

def calc_K(x):
  return x**p*(1.0 - (1.0 - x**(1.0/m))**m)**2

C.setValue(calc_C(h))
Se.setValue(calc_Se(h.harmonicFaceValue))
K.setValue(calc_K(Se))

# Richards' equation
g = [1] # gravity "vector" (not really needed in 1D)
Req = (TransientTerm(coeff=C, var=h) == DiffusionTerm(coeff=K, var=h) - (K*g).divergence)

viewerh = Viewer(vars=h)
viewerh.plot()
viewerC = Viewer(vars=C)
viewerC.plot()


t = 0.0

for step in range(num_time_steps):

    t += timeStep
    h.updateOld()

    res = 100.0
    j = 0
    while res > 1.0E-5 and j < 20:
        res = Req.sweep(dt=timeStep,var=h)
        j += 1

    if j == 20:
        print('**WARNING: non-linear Picard iteration did not converge**')
    print(step,'(',j,res,')',t)


        # van Genuchten

    C.setValue(calc_C(h))
    Se.setValue(calc_Se(h.harmonicFaceValue))
    K.setValue(calc_K(Se))

    viewerC.plot()
