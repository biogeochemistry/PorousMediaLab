from sediment import *
import matplotlib.pyplot as plt

D = 40
w = 1
t = 10

C = sediment(D, w, t)

L = 100
dx = 0.1
x = np.linspace(0, L, L / dx + 1)
sol = 1 / 2 * (special.erfc((x - w * t) / 2 / np.sqrt(D * t)) + np.exp(w * x / D) * special.erfc((x + w * t) / 2 / np.sqrt(D * t)))

plt.plot(x, C[:,-1])
plt.plot(x, sol)
plt.show()
