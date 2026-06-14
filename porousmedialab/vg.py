# These are the van Genuchten (1980) equations
# The input is matric potential, psi and the hydraulic parameters.
# psi must be sent in as a numpy array.
import numpy as np


def _effective_saturation(psi, pars):
    """Effective saturation Se(psi), vectorized over a psi array or scalar.

    Se = 1 for psi >= 0 (saturated); the van Genuchten expression otherwise.
    ``np.where`` evaluates both branches, but the unsaturated expression is
    finite for psi >= 0 too (positive base), so no NaN/warning is produced.
    """
    psi = np.asarray(psi, dtype=float)
    return np.where(
        psi >= 0.0,
        1.0,
        (1 + np.abs(psi * pars['alpha'])**pars['n'])**(-pars['m']))


def thetaFun(psi, pars):
    Se = _effective_saturation(psi, pars)
    return pars['thetaR'] + (pars['thetaS'] - pars['thetaR']) * Se


def CFun(psi, pars):
    Se = _effective_saturation(psi, pars)
    # At Se == 1 (psi >= 0) this evaluates to 0, matching the saturated case.
    dSedh = pars['alpha'] * pars['m'] / \
        (1 - pars['m']) * Se**(1 / pars['m']) * \
        (1 - Se**(1 / pars['m']))**pars['m']
    return Se * pars['Ss'] + (pars['thetaS'] - pars['thetaR']) * dSedh


def KFun(psi, pars):
    Se = _effective_saturation(psi, pars)
    return pars['Ks'] * Se**pars['neta'] * (1 - (1 - Se**(1 / pars['m']))**pars['m'])**2


def setpars():
    pars = {}
    pars['thetaR'] = float(input("thetaR = "))
    pars['thetaS'] = float(input("thetaS = "))
    pars['alpha'] = float(input("alpha = "))
    pars['n'] = float(input("n = "))
    pars['m'] = 1 - 1 / pars['n']
    pars['Ks'] = float(input("Ks = "))
    pars['neta'] = float(input("neta = "))
    pars['Ss'] = float(input("Ss = "))
    return pars


def _create_soil_params(thetaR, thetaS, alpha, n, Ks, neta=0.5, Ss=1e-6):
    return {
        'thetaR': thetaR, 'thetaS': thetaS, 'alpha': alpha,
        'n': n, 'm': 1 - 1 / n, 'Ks': Ks, 'neta': neta, 'Ss': Ss
    }


def HygieneSandstone():
    return _create_soil_params(0.153, 0.25, 0.79, 10.4, 1.08)


def TouchetSiltLoam():
    return _create_soil_params(0.19, 0.469, 0.5, 7.09, 3.03)


def SiltLoamGE3():
    return _create_soil_params(0.131, 0.396, 0.423, 2.06, 0.0496)


def GuelphLoamDrying():
    return _create_soil_params(0.218, 0.520, 1.15, 2.03, 0.316)


def GuelphLoamWetting():
    return _create_soil_params(0.218, 0.434, 2.0, 2.76, 0.316)


def BeitNetofaClay():
    return _create_soil_params(0.0, 0.446, 0.152, 1.17, 0.00082)
