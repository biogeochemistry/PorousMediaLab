import math
import sys

sys.path.append('../../')

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import basinhopping

from porousmedialab.metrics import rmse
from porousmedialab.calibrator import find_indexes_of_intersections
from porousmedialab.column import Column
from porousmedialab import blackbox as bb
from thawmeasurements import (
    C1D9, C1D21, C1D33, F1C1D21, F1C1D33, F2C1D21, F2C1D33, F3C1D21, F3C1D33,
    SA, T1C1D9, T1C1D21, T1C1D33, T2C1D9, T2C1D21, T2C1D33, T3C1D9, T3C1D21,
    T3C1D33, C1h, Ci, D_column, D_SF6g, D_SF6w, F1C1h, F1Ci, F1T_frz, F1Ti,
    F1Tm, F2C1h, F2Ci, F2T_frz, F2Ti, F2Tm, F3C1h, F3Ci, F3T_frz, F3Ti, F3Tm,
    Kh, SF6_Hcc, T1C1h, T1Ci, T1T_thw, T1Ti, T1Tm, T2C1h, T2Ci, T2T_thw, T2Ti,
    T2Tm, T3C1h, T3Ci, T3Ti, T3Tm, T_frz, T_thw, Ti, Tm, Vh1, Vi, phi_m, z_phi,
    zm)

Tm = np.concatenate([
    0 + T1Tm, F1T_frz + F1Tm, T1T_thw + T2Tm, F2T_frz + F2Tm, T2T_thw + T3Tm,
    F3T_frz + F3Tm
])
Ti = np.array(np.array([16, 177, 350, 514, 681, 851]))
Tm = Tm - Ti[0]
Ti = Ti - Ti[0]

tend = 457
dt = 0.01
dx = 0.2    ## cm
L = 40    ## cm
x = np.linspace(0, L, L / dx + 1)
t = np.linspace(0, tend, round(tend / dt) + 1)
Chs = np.zeros(t.shape)    #
Fx = np.zeros(t.shape)
phi = (0.99 - 0.91) * np.exp(-x / 10) + 0.91

dT = T1Tm[1::2] - T1Tm[::2]

dC1h = (T1C1h[1::2] - T1C1h[::2])

Mi = T1Ci * Vi    # mass injected

#h_inj = Vi/SA/phi
h_inj = Vi / SA / 0.93

#Pores from the FTR experiment#

phi_w = phi * (0.875 / 0.97)
phi_g = phi * ((0.97 - 0.875) / 0.97)
phi_p = 1 - phi


def fun(k0):
    w, k_w_in, k_w_out, k_g_in, k_g_out = k0
    print(*k0)
    try:
        ftc1 = Column(L, dx, tend, dt)

        ftc1.add_species(
            theta=phi_g,
            element='SF6g',
            D=D_SF6g,
            init_C=0,
            bc_top=0,
            bc_top_type='constant',
            bc_bot=0,
            bc_bot_type='constant',
            w=-0.00)    #-0.055
        ftc1.add_species(
            theta=phi_w,
            element='SF6w',
            D=D_SF6w,
            init_C=0,
            bc_top=0,
            bc_top_type='constant',
            bc_bot=0,
            bc_bot_type='constant',
            w=w)

        # SF6mp stands for SF6 gas in micro pores, it is immobile and only collects SF6;
        ftc1.add_species(
            theta=phi_p,
            element='SF6mp',
            D=1e-18,
            init_C=0,
            bc_top=0,
            bc_top_type='flux',
            bc_bot=0,
            bc_bot_type='flux')

        #((phi_g**(10/3))/(phi**2))*
        #((phi_w**(10/3))/(phi**2))*

        # # Constants
        ftc1.constants['k_w_in'] = k_w_in    #from FTR w
        ftc1.constants['k_w_out'] = k_w_out
        #0.4

        ftc1.constants['k_g_in'] = k_g_in
        ftc1.constants['k_g_out'] = k_g_out

        ftc1.constants['phi_w'] = phi_w
        ftc1.constants['phi_g'] = phi_g
        ftc1.constants['phi_p'] = phi_p

        # # Rates of diffusion into pores and out
        ftc1.rates['R_w_in'] = 'k_w_in * SF6w'
        ftc1.rates['R_w_out'] = 'k_w_out * SF6mp'

        ftc1.rates['R_g_in'] = 'k_g_in * SF6w'
        ftc1.rates['R_g_out'] = 'k_g_out * SF6g'

        # # dcdt
        ftc1.dcdt[
            'SF6w'] = '-R_g_in + R_g_out * phi_g - R_w_in + R_w_out * phi_p'
        ftc1.dcdt['SF6g'] = 'R_g_in / phi_g - R_g_out'
        ftc1.dcdt['SF6mp'] = 'R_w_in / phi_p - R_w_out'

        for i in range(0, len(ftc1.time)):
            if (ftc1.time[i] > F1T_frz - 16 and ftc1.time[i] < T1T_thw - 16
               ) or (ftc1.time[i] > F2T_frz - 16 and ftc1.time[i] < T2T_thw - 16
                    ) or (ftc1.time[i] > F3T_frz - 16):
                ftc1.change_boundary_conditions(
                    'SF6g', i, bc_top=0, bc_top_type='flux')
                ftc1.change_boundary_conditions(
                    'SF6w', i, bc_top=0, bc_top_type='flux')
            else:
                ftc1.change_boundary_conditions(
                    'SF6g', i, bc_top=0, bc_top_type='constant')
                ftc1.change_boundary_conditions(
                    'SF6w', i, bc_top=0, bc_top_type='constant')
            if any([ftc1.time[i] == T_inj for T_inj in Ti]):
                SF6_add = np.zeros(x.size)
                SF6_add[x > 0] = 0
                SF6_add[x > 18 - (h_inj / 2)] = Ci[Ti == ftc1.time[i]]
                SF6_add[x > 18 + (h_inj / 2)] = 0
                new_profile = ftc1.profiles['SF6w'] + SF6_add    #
                ftc1.change_concentration_profile('SF6w', i, new_profile)

            ftc1.integrate_one_timestep(i)

        time_idxs = find_indexes_of_intersections(ftc1.time, Tm)

        zm = 9
        M1D9 = (
            ftc1.SF6w.concentration[ftc1.x == zm, time_idxs] *
            phi_w[ftc1.x == zm] + ftc1.SF6g.
            concentration[ftc1.x == zm, time_idxs] * phi_g[ftc1.x == zm]) / (
                phi_w[ftc1.x == zm] + phi_g[ftc1.x == zm])

        zm = 21
        M1D21 = (
            ftc1.SF6w.concentration[ftc1.x == zm, time_idxs] *
            phi_w[ftc1.x == zm] + ftc1.SF6g.
            concentration[ftc1.x == zm, time_idxs] * phi_g[ftc1.x == zm]) / (
                phi_w[ftc1.x == zm] + phi_g[ftc1.x == zm])

        zm = 33
        M1D33 = (
            ftc1.SF6w.concentration[ftc1.x == zm, time_idxs] *
            phi_w[ftc1.x == zm] + ftc1.SF6g.
            concentration[ftc1.x == zm, time_idxs] * phi_g[ftc1.x == zm]) / (
                phi_w[ftc1.x == zm] + phi_g[ftc1.x == zm])

        fx_time_idxs = find_indexes_of_intersections(ftc1.time, Tm[::2] + 1)
        F1 = ftc1.estimate_flux_at_top('SF6g')[fx_time_idxs]
        F2 = ftc1.estimate_flux_at_top('SF6w')[fx_time_idxs]
        F3 = ftc1.estimate_flux_at_top('SF6mp')[fx_time_idxs]
        fx_meas = dC1h*Vh1/SA/dT

        err = rmse(M1D9, C1D9[:len(M1D9) - len(C1D9)]) + rmse(
            M1D21, C1D21[:len(M1D21) - len(C1D21)]) + rmse(
                M1D33, C1D33[:len(M1D33) - len(C1D33)]) + 5 * rmse(F1+F2+F3,
                fx_meas[:len(F1) - len(fx_meas)])

    except:
        err = 1e8
    print(err)
    return err


w = -0.02
k_w_in = 0.06
k_w_out = 0.02
k_g_in = 0.9
k_g_out = 90

# bb.search(
#     f=fun,    # given function
#     box=[[-1., 0.], [0., 10.], [0., 10.], [0., 100.],
#          [0., 100.]],    # range of values for each parameter
#     n=20,    # number of function calls on initial stage (global search)
#     m=20,    # number of function calls on subsequent stage (local search)
#     batch=7,    # number of calls that will be evaluated in parallel
#     resfile='output.csv')    # text file where results will be saved

minimizer_kwargs = {"method": "Nelder-Mead"}
ret = basinhopping(
    fun,
    [-0.0196850617169, 0.0585636682011, 0.0242497929538, 1.09455525468, 54.991],
    minimizer_kwargs=minimizer_kwargs,
    niter=200)

# res_min = minimize(
#     fun, [w, k_w_in, k_w_out, k_g_in, k_g_out],
#     method='Nelder-Mead',
#     bounds=None,
#     options={
#         'xtol': 1e-6,
#         'disp': True
#     })
