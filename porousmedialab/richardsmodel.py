import numpy as np
from scipy.integrate import odeint
import porousmedialab.vg as vg


class RichardsModel:
    """Unsaturated transport model"""

    def __init__(self, z, t, psi0, qTop=-0.01, qBot=None, psiTop=None,
                 psiBot=None, soil_params=None):
        # Boundary conditions
        self.qTop = self._normalize_boundary(qTop)
        self.qBot = self._normalize_boundary(qBot)
        self.psiTop = self._normalize_boundary(psiTop)
        self.psiBot = self._normalize_boundary(psiBot)

        # soil type
        self.p = soil_params or vg.HygieneSandstone()

        # Grid in space
        self.z = z
        self.n = z.size
        self.dz = float(np.mean(np.diff(z))) if self.n > 1 else 0.1
        self.ProfileDepth = float(np.max(z) - np.min(z) + self.dz)

        # Grid in time
        self.t = np.linspace(0, t, 2)

        # Initial conditions
        self.psi0 = psi0

    @staticmethod
    def _normalize_boundary(value):
        if isinstance(value, list) and len(value) == 0:
            return None
        return value

    def solve(self):
        self.psi = odeint(self.RichardsEquation, self.psi0, self.t, args=(
            self.dz, self.n, self.p, self.qTop, self.qBot, self.psiTop, self.psiBot), mxstep=500)
        self.psi0 = self.psi[-1, :]

    def RichardsEquation(self, psi, t, dz, n, p, qTop, qBot, psiTop, psiBot):

        # Basic properties:
        C = vg.CFun(psi, p)

        # initialize vectors:
        q = np.zeros(n + 1)

        # Upper boundary
        if qTop is None:
            KTop = vg.KFun(np.zeros(1) + psiTop, p)
            q[n] = -KTop * ((psiTop - psi[n - 1]) / dz * 2 + 1)
        else:
            q[n] = qTop

        # Lower boundary
        if qBot is None:
            if psiBot is None:
                # Free drainage
                KBot = vg.KFun(np.zeros(1) + psi[0], p)
                q[0] = -KBot
            else:
                # Type 1 boundary
                KBot = vg.KFun(np.zeros(1) + psiBot, p)
                q[0] = -KBot * ((psi[0] - psiBot) / dz * 2 + 1.0)
        else:
            # Type 2 boundary
            q[0] = qBot

        # Internal nodes
        i = np.arange(0, n - 1)
        Knodes = vg.KFun(psi, p)
        Kmid = (Knodes[i + 1] + Knodes[i]) / 2.0

        j = np.arange(1, n)
        q[j] = -Kmid * ((psi[i + 1] - psi[i]) / dz + 1.0)

        # Continuity
        i = np.arange(0, n)
        dpsidt = (-(q[i + 1] - q[i]) / dz) / C

        return dpsidt
