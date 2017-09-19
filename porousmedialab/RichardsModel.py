import numpy as np
from scipy.integrate import odeint
import porousmedialab.vg as vg


def thetaFun(psi, pars):
    if psi >= 0.:
        Se = 1.
    else:
        Se = (1 + abs(psi * pars['alpha'])**pars['n'])**(-pars['m'])
    return pars['thetaR'] + (pars['thetaS'] - pars['thetaR']) * Se


def CFun(psi, pars):
    if psi >= 0.:
        Se = 1.
    else:
        Se = (1 + abs(psi * pars['alpha'])**pars['n'])**(-pars['m'])
    dSedh = pars['alpha'] * pars['m'] / \
        (1 - pars['m']) * Se**(1 / pars['m']) * \
        (1 - Se**(1 / pars['m']))**pars['m']
    return Se * pars['Ss'] + (pars['thetaS'] - pars['thetaR']) * dSedh


def KFun(psi, pars):
    if psi >= 0.:
        Se = 1.
    else:
        Se = (1 + abs(psi * pars['alpha'])**pars['n'])**(-pars['m'])
    return pars['Ks'] * Se**pars['neta'] * (1 - (1 - Se**(1 / pars['m']))**pars['m'])**2


thetaFun = np.vectorize(thetaFun)
CFun = np.vectorize(CFun)
KFun = np.vectorize(KFun)


class RichardsModel:
    """Unsaturated transport model"""

    def __init__(self, z, t, psi0, qTop=-0.01, qBot=[], psiTop=[], psiBot=[]):
        # Boundary conditions
        self.qTop = -0.01
        self.qBot = []
        self.psiTop = []
        self.psiBot = []

        # soil type
        self.p = vg.HygieneSandstone()

        # Grid in space
        self.dz = 0.1
        self.ProfileDepth = 5
        self.z = z  # np.arange(self.dz / 2.0, self.ProfileDepth, self.dz)
        self.n = z.size

        # Grid in time
        self.t = np.linspace(0, t, 2)

        # Initial conditions
        self.psi0 = psi0

    def solve(self):
        self.psi = odeint(self.RichardsEquation, self.psi0, self.t, args=(
            self.dz, self.n, self.p, self.qTop, self.qBot, self.psiTop, self.psiBot), mxstep=500)
        self.psi0 = self.psi[-1, :]

    def RichardsEquation(self, psi, t, dz, n, p, qTop, qBot, psiTop, psiBot):

        # Basic properties:
        C = CFun(psi, p)

        # initialize vectors:
        q = np.zeros(n + 1)

        # Upper boundary
        if qTop == []:
            KTop = KFun(np.zeros(1) + psiTop, p)
            q[n] = -KTop * ((psiTop - psi[n - 1]) / dz * 2 + 1)
        else:
            q[n] = qTop

        # Lower boundary
        if qBot == []:
            if psiBot == []:
                # Free drainage
                KBot = KFun(np.zeros(1) + psi[0], p)
                q[0] = -KBot
            else:
                # Type 1 boundary
                KBot = KFun(np.zeros(1) + psiBot, p)
                q[0] = -KBot * ((psi[0] - psiBot) / dz * 2 + 1.0)
        else:
            # Type 2 boundary
            q[0] = qBot

        # Internal nodes
        i = np.arange(0, n - 1)
        Knodes = KFun(psi, p)
        Kmid = (Knodes[i + 1] + Knodes[i]) / 2.0

        j = np.arange(1, n)
        q[j] = -Kmid * ((psi[i + 1] - psi[i]) / dz + 1.0)

        # Continuity
        i = np.arange(0, n)
        dpsidt = (-(q[i + 1] - q[i]) / dz) / C

        return dpsidt
