import numpy as np
import scipy as sp
from scipy.sparse import spdiags
from scipy import special
# from scikits import umfpack


def sediment(D, w, years):
    BC1_top = 1
    F_bottom = 0
    L = 100
    tend = years
    C1_init = 0

    phi = 1
    dx = 0.1
    dt = 0.001
    x = np.linspace(0, L, L / dx + 1)
    N = x.size

    C1_init = C1_init * np.ones((N, 1))
    C1_init[0] = BC1_top

    [AL, AR] = AL_AR_dirichlet(D, w, phi, dt, dx, N)

    time = np.linspace(0, tend, tend / dt + 1)

    C1_res = np.zeros((N, time.size))
    C1_res[:, 0] = C1_init[:, 0]

    for i in np.arange(1, len(time)):
        C1_res[:, i - 1] = update_bc_dirichlet(C1_res[:, i - 1], BC1_top)
        B = AR.dot(C1_res[:, i - 1])
        C1_res[:, i] = linalg_solver(AL, B)
        C1_res[0, i] = BC1_top

    return C1_res


def linalg_solver(A, b):
    # # linalg_solver: x = A \ b
    # do not forget to install pip install scikit-umfpack
    # scikit-umfpack provides a wrapper of UMFPACK sparse direct solver to SciPy.
    # lu = umfpack.splu(A)
    # return umfpack.spsolve(A, b)
    return sp.sparse.linalg.spsolve(A, b, use_umfpack=True)


def update_bc_dirichlet(C, BC_top):
    # update_bc_dirichlet: function description
    C[0] = BC_top
    return C


def AL_AR_dirichlet(D, w, phi, dt, dx, N):
    # AL_AR_dirichlet: creates AL and AR matrices with Dirichlet BC
    s = phi * D * dt / dx / dx  #
    q = phi * w * dt / dx  #
    e1 = np.ones((N, 1))  #
    AL = spdiags(np.concatenate((e1 * (-s / 2 - q / 4), e1 * (1 + s), e1 * (-s / 2 + q / 4)), axis=1).T, [-1, 0, 1], N, N, format='csc')  # .toarray()
    AR = spdiags(np.concatenate((e1 * (s / 2 + q / 4), e1 * (1 - s), e1 * (s / 2 - q / 4)), axis=1).T, [-1, 0, 1], N, N, format='csc')  # .toarray()
    AL[0, 0] = 1
    AL[0, 1] = 0
    AL[N - 1, N - 1] = 1 + s
    AL[N - 1, N - 2] = -s
    AR[0, 0] = 1
    AR[0, 1] = 0
    AR[N - 1, N - 1] = 1 - s
    AR[N - 1, N - 2] = s
    return AL, AR


if __name__ == '__main__':
    #
    D = 0.5
    w = 3
    t = 10

    C = sediment(D, w, t)

    # L = 30
    # dx = 0.1
    # x = np.linspace(0, L, L / dx + 1)
    # sol = 1 / 2 * (special.erfc((x - w * t) / 2 / np.sqrt(D * t)) + np.exp(w * x / D) * special.erfc((x + w * t) / 2 / np.sqrt(D * t)))
