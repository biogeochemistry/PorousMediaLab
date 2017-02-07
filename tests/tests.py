from PorousMediaLab import *

def test_transport_equation():
    D = 40
    w = 0.2
    t = 0.1
    dx = 0.1
    L = 100
    phi = 1
    dt = 0.001
    C = PorousMediaLab(L, dx, t, dt, phi, w)
    C.add_solute_species('O2', D, 0.0, 1)
    C.dcdt.O2 = '0'
    C.solve()
    x = np.linspace(0, L, L / dx + 1)
    sol = 1 / 2 * (special.erfc((x - w * t) / 2 / np.sqrt(D * t)) + np.exp(w * x / D) * special.erfc((x + w * t) / 2 / np.sqrt(D * t)))

    assert max(C.O2.concentration[:, -1] - sol) < 1e-4
