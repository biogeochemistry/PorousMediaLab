from sediment_class import *
import matplotlib.pyplot as plt

D = 40
w = 1
t = 1
dx = 0.1
L = 100
phi = 1
dt = 0.001

C = Sediment(L, dx, t, dt, phi, w)
C.add_solute_species('O2', D, 0.0, 1)
C.integrate()


x = np.linspace(0, L, L / dx + 1)
sol = 1 / 2 * (special.erfc((x - w * t) / 2 / np.sqrt(D * t)) + np.exp(w * x / D) * special.erfc((x + w * t) / 2 / np.sqrt(D * t)))

plt.plot(C.x, C.species['O2']['res'][:, -1], 'kx')
plt.plot(x, sol)
plt.show()
