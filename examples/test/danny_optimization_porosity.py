import math
import sys

sys.path.append('../../')

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import basinhopping

from porousmedialab.metrics import norm_rmse
from porousmedialab.calibrator import find_indexes_of_intersections
from porousmedialab.column import Column
from porousmedialab import blackbox as bb
from thawmeasurements import Vh2, Vh3, C2h, C3h, C3D33, C2D33, C3D21, C2D21, C3D9, C2D9, Tm_nz, C1h_nz, F3Tm, F3T_frz, T3Tm, T2T_thw, F2Tm, F2T_frz, T1T_thw, T2Tm, F1T_frz, T1Tm, C1h, T1C1h, F1C1h, T2C1h, F2C1h, T3C1h, F3C1h, C1D9, T1C1D9, T2C1D9, T3C1D9, C1D21, T1C1D21, F1C1D21, T2C1D21, F2C1D21, T3C1D21, F3C1D21, C1D33, T1C1D33, F1C1D33, T2C1D33, F2C1D33, T3C1D33, F3C1D33, Tm, T1Tm, F1Tm, T2Tm, F2Tm, T3Tm, F3Tm, zm, D_SF6w, D_SF6g, Kh, phi_m, z_phi, Vh1, SA, D_column, Vi, Ci, T1Ci, F1Ci, T2Ci, F2Ci, T3Ci, F3Ci, Ti, T1Ti, F1Ti, T2Ti, F2Ti, T3Ti, F3Ti, SF6_Hcc, T_frz, T_thw

dx = 0.2    ## cm
L = 40    ## cm
x = np.linspace(0, L, L / dx + 1)

Tm = np.concatenate([
    0 + T1Tm, F1T_frz + F1Tm, T1T_thw + T2Tm, F2T_frz + F2Tm, T2T_thw + T3Tm,
    F3T_frz + F3Tm
])
Ti = np.array(np.array([16, 177, 350, 514, 681, 851]))
t_shift = Ti[0]
Tm = Tm - t_shift
Ti_1 = Ti - t_shift

periods = np.concatenate([T_frz, T_thw])
periods.sort()
periods = periods - t_shift

dT = Tm_nz[1::2] - Tm_nz[::2]
dC1h = (C1h[1::2] - C1h[::2])
dC2h = (C2h[1::2] - C2h[::2])
dC3h = (C3h[1::2] - C3h[::2])
Fx_mean = (dC3h * Vh3 / SA / 2  + dC1h * Vh1 / SA / 2) / 2

CD33_mean = (C1D33  + C3D33) / 2
CD21_mean = (C1D21  + C3D21) / 2
CD9_mean = (C1D9 + C3D9) / 2

Mi = T1Ci * Vi    # mass injected
h_inj = Vi / SA / 0.93


def fun(k0):
    w, k_w_in, k_w_out, k_g_in, k_g_out, phi_1, phi_2, phi_3 = k0
    print(*k0)
    try:
        tend = periods[0]
        # tend = 457
        dt = 0.01
        dx = 0.2    ## cm
        L = 40    ## cm
        x = np.linspace(0, L, L / dx + 1)

        t = np.linspace(0, tend, round(tend / dt) + 1)
        #phi = 0.8
        Chs = np.zeros(t.shape)    #
        Fx = np.zeros(t.shape)

        phi = (phi_1 - 0.91) * np.exp(-x / 10) + 0.91
        phi_w = phi * (0.875 / 0.97)
        phi_g = phi * ((0.97 - 0.875) / 0.97)
        phi_p = 1 - phi


        ftc1 = Column(L, dx, tend, dt)

        ftc1.add_species(
            theta=((phi_g) / (1-np.log(phi))),
            name='SF6g',
            D=D_SF6g,
            init_C=0,
            bc_top=0,
            bc_top_type='constant',
            bc_bot=0,
            bc_bot_type='constant',
            w=-0.00)    #-0.055
        ftc1.add_species(
            theta=((phi_w) / (1-np.log(phi))),
            name='SF6w',
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
            name='SF6mp',
            D=1e-18,
            init_C=0,
            bc_top=0,
            bc_top_type='flux',
            bc_bot=0,
            bc_bot_type='flux')

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
        # ftc1.rates['R_w_in'] = '0'
        # ftc1.rates['R_w_out'] = '0'

        ftc1.rates['R_g_in'] = 'k_g_in * SF6w'
        ftc1.rates['R_g_out'] = 'k_g_out * SF6g'

        # # dcdt
        ftc1.dcdt[
            'SF6w'] = '-R_g_in + R_g_out * phi_g - R_w_in + R_w_out * phi_p'
        ftc1.dcdt['SF6g'] = 'R_g_in / phi_g - R_g_out'
        ftc1.dcdt['SF6mp'] = 'R_w_in / phi_p - R_w_out'

        Fx = np.zeros(t.size)

        for i in range(0, len(ftc1.time)):
            if (ftc1.time[i] > periods[0] and ftc1.time[i] < periods[1]) or (
                    ftc1.time[i] > periods[2] and ftc1.time[i] < periods[3]) or (
                        ftc1.time[i] > periods[4] and ftc1.time[i] < periods[5]):
                ftc1.change_boundary_conditions(
                    'SF6g', i, bc_top=0, bc_top_type='flux')
                ftc1.change_boundary_conditions(
                    'SF6w', i, bc_top=0, bc_top_type='flux')
                Fx[i] = 0
            else:
                ftc1.change_boundary_conditions(
                    'SF6g', i, bc_top=0, bc_top_type='constant')
                ftc1.change_boundary_conditions(
                    'SF6w', i, bc_top=0, bc_top_type='constant')
                F1 = ftc1.estimate_flux_at_top('SF6g', i)
                F2 = ftc1.estimate_flux_at_top('SF6w', i)
                F3 = ftc1.estimate_flux_at_top('SF6mp', i)
                Fx[i] = F1[i] + F2[i] + F3[i]
            if any([ftc1.time[i] == T_inj for T_inj in Ti_1]):
                SF6_add = np.zeros(x.size)
                SF6_add[x > 0] = 0
                SF6_add[x > 18 - (h_inj / 2)] = Ci[Ti_1 == ftc1.time[i]]
                SF6_add[x > 18 + (h_inj / 2)] = 0
                new_profile = ftc1.profiles['SF6w'] + SF6_add    #
                ftc1.change_concentration_profile('SF6w', i, new_profile)

            ftc1.integrate_one_timestep(i)

        Ti_2 = Ti - periods[0]
        tend = periods[2] - periods[0]
        dt = 0.01
        dx = 0.2    ## cm
        L = 40    ## cm
        x = np.linspace(0, L, L / dx + 1)
        t = np.linspace(0, tend, round(tend / dt) + 1)
        #phi = 0.8
        Chs = np.zeros(t.shape)    #
        Fx = np.zeros(t.shape)
        phi = (phi_2 - 0.91) * np.exp(-x / 10) + 0.91
        phi_w = phi * (0.875 / 0.97)
        phi_g = phi * ((0.97 - 0.875) / 0.97)
        phi_p = 1 - phi

        ftc2 = Column(L, dx, tend, dt)

        ftc2.add_species(
            theta=((phi_g) / (1-np.log(phi))),
            name='SF6g',
            D=D_SF6g,
            init_C=ftc1.profiles.SF6g,
            bc_top=0,
            bc_top_type='constant',
            bc_bot=0,
            bc_bot_type='constant',
            w=-0.00)    #-0.055
        ftc2.add_species(
            theta=((phi_w) / (1-np.log(phi))),
            name='SF6w',
            D=D_SF6w,
            init_C=ftc1.profiles.SF6w,
            bc_top=0,
            bc_top_type='constant',
            bc_bot=0,
            bc_bot_type='constant',
            w=w)

        # SF6mp stands for SF6 gas in micro pores, it is immobile and only collects SF6;
        ftc2.add_species(
            theta=phi_p,
            name='SF6mp',
            D=1e-18,
            init_C=ftc1.profiles.SF6mp,
            bc_top=0,
            bc_top_type='flux',
            bc_bot=0,
            bc_bot_type='flux')

        # # Constants
        ftc2.constants['k_w_in'] = k_w_in    #from FTR w
        ftc2.constants['k_w_out'] = k_w_out
        #0.4

        ftc2.constants['k_g_in'] = k_g_in
        ftc2.constants['k_g_out'] = k_g_out

        ftc2.constants['phi_w'] = phi_w
        ftc2.constants['phi_g'] = phi_g
        ftc2.constants['phi_p'] = phi_p

        # # Rates of diffusion into pores and out
        ftc2.rates['R_w_in'] = 'k_w_in * SF6w'
        ftc2.rates['R_w_out'] = 'k_w_out * SF6mp'
        # ftc2.rates['R_w_in'] = '0'
        # ftc2.rates['R_w_out'] = '0'

        ftc2.rates['R_g_in'] = 'k_g_in * SF6w'
        ftc2.rates['R_g_out'] = 'k_g_out * SF6g'

        # # dcdt
        ftc2.dcdt[
            'SF6w'] = '-R_g_in + R_g_out * phi_g - R_w_in + R_w_out * phi_p'
        ftc2.dcdt['SF6g'] = 'R_g_in / phi_g - R_g_out'
        ftc2.dcdt['SF6mp'] = 'R_w_in / phi_p - R_w_out'

        for i in range(0, len(ftc2.time)):
            if (ftc2.time[i] + periods[0] > periods[0]
                    and ftc2.time[i] + periods[0] < periods[1]) or (
                        ftc2.time[i] + periods[0] > periods[2]
                        and ftc2.time[i] + periods[0] < periods[3]) or (
                            ftc2.time[i] + periods[0] > periods[4]
                            and ftc2.time[i] + periods[0] < periods[5]):
                ftc2.change_boundary_conditions(
                    'SF6g', i, bc_top=0, bc_top_type='flux')
                ftc2.change_boundary_conditions(
                    'SF6w', i, bc_top=0, bc_top_type='flux')
                Fx[i] = 0
            else:
                ftc2.change_boundary_conditions(
                    'SF6g', i, bc_top=0, bc_top_type='constant')
                ftc2.change_boundary_conditions(
                    'SF6w', i, bc_top=0, bc_top_type='constant')
                F1 = ftc2.estimate_flux_at_top('SF6g', i)
                F2 = ftc2.estimate_flux_at_top('SF6w', i)
                F3 = ftc2.estimate_flux_at_top('SF6mp', i)
                Fx[i] = F1[i] + F2[i] + F3[i]
            if any([ftc2.time[i] == T_inj for T_inj in Ti_2]):
                SF6_add = np.zeros(x.size)
                SF6_add[x > 0] = 0
                SF6_add[x > 18 - (h_inj / 2)] = Ci[Ti_2 == ftc2.time[i]]
                SF6_add[x > 18 + (h_inj / 2)] = 0
                new_profile = ftc2.profiles['SF6w'] + SF6_add    #
                ftc2.change_concentration_profile('SF6w', i, new_profile)

            ftc2.integrate_one_timestep(i)

        Ti_3 = Ti - periods[2]
        tend = periods[4] - periods[2]
        dt = 0.01
        dx = 0.2    ## cm
        L = 40    ## cm
        x = np.linspace(0, L, L / dx + 1)
        t = np.linspace(0, tend, round(tend / dt) + 1)
        #phi = 0.8
        Chs = np.zeros(t.shape)    #
        Fx = np.zeros(t.shape)
        phi = (phi_3 - 0.91) * np.exp(-x / 10) + 0.91
        phi_w = phi * (0.875 / 0.97)
        phi_g = phi * ((0.97 - 0.875) / 0.97)
        phi_p = 1 - phi

        ftc3 = Column(L, dx, tend, dt)

        ftc3.add_species(
            theta=((phi_g) / (1-np.log(phi))),
            name='SF6g',
            D=D_SF6g,
            init_C=ftc2.profiles.SF6g,
            bc_top=0,
            bc_top_type='constant',
            bc_bot=0,
            bc_bot_type='constant',
            w=-0.00)    #-0.055
        ftc3.add_species(
            theta=((phi_w) / (1-np.log(phi))),
            name='SF6w',
            D=D_SF6w,
            init_C=ftc2.profiles.SF6w,
            bc_top=0,
            bc_top_type='constant',
            bc_bot=0,
            bc_bot_type='constant',
            w=w)

        # SF6mp stands for SF6 gas in micro pores, it is immobile and only collects SF6;
        ftc3.add_species(
            theta=phi_p,
            name='SF6mp',
            D=1e-18,
            init_C=ftc2.profiles.SF6mp,
            bc_top=0,
            bc_top_type='flux',
            bc_bot=0,
            bc_bot_type='flux')

        # # Constants
        ftc3.constants['k_w_in'] = k_w_in    #from FTR w
        ftc3.constants['k_w_out'] = k_w_out
        #0.4

        ftc3.constants['k_g_in'] = k_g_in
        ftc3.constants['k_g_out'] = k_g_out

        ftc3.constants['phi_w'] = phi_w
        ftc3.constants['phi_g'] = phi_g
        ftc3.constants['phi_p'] = phi_p

        # # Rates of diffusion into pores and out
        ftc3.rates['R_w_in'] = 'k_w_in * SF6w'
        ftc3.rates['R_w_out'] = 'k_w_out * SF6mp'
        # ftc3.rates['R_w_in'] = '0'
        # ftc3.rates['R_w_out'] = '0'

        ftc3.rates['R_g_in'] = 'k_g_in * SF6w'
        ftc3.rates['R_g_out'] = 'k_g_out * SF6g'

        # # dcdt
        ftc3.dcdt[
            'SF6w'] = '-R_g_in + R_g_out * phi_g - R_w_in + R_w_out * phi_p'
        ftc3.dcdt['SF6g'] = 'R_g_in / phi_g - R_g_out'
        ftc3.dcdt['SF6mp'] = 'R_w_in / phi_p - R_w_out'

        for i in range(0, len(ftc3.time)):
            if (ftc3.time[i] + periods[2] > periods[0]
                    and ftc3.time[i] + periods[2] < periods[1]) or (
                        ftc3.time[i] + periods[2] > periods[2]
                        and ftc3.time[i] + periods[2] < periods[3]) or (
                            ftc3.time[i] + periods[2] > periods[4]
                            and ftc3.time[i] + periods[2] < periods[5]):
                ftc3.change_boundary_conditions(
                    'SF6g', i, bc_top=0, bc_top_type='flux')
                ftc3.change_boundary_conditions(
                    'SF6w', i, bc_top=0, bc_top_type='flux')
                Fx[i] = 0
            else:
                ftc3.change_boundary_conditions(
                    'SF6g', i, bc_top=0, bc_top_type='constant')
                ftc3.change_boundary_conditions(
                    'SF6w', i, bc_top=0, bc_top_type='constant')
                F1 = ftc3.estimate_flux_at_top('SF6g', i)
                F2 = ftc3.estimate_flux_at_top('SF6w', i)
                F3 = ftc3.estimate_flux_at_top('SF6mp', i)
                Fx[i] = F1[i] + F2[i] + F3[i]
            if any([ftc3.time[i] == T_inj for T_inj in Ti_3]):
                SF6_add = np.zeros(x.size)
                SF6_add[x > 0] = 0
                SF6_add[x > 18 - (h_inj / 2)] = Ci[Ti_3 == ftc3.time[i]]
                SF6_add[x > 18 + (h_inj / 2)] = 0
                new_profile = ftc3.profiles['SF6w'] + SF6_add    #
                ftc3.change_concentration_profile('SF6w', i, new_profile)

            ftc3.integrate_one_timestep(i)

        zm = 9
        M1D9 = (ftc1.SF6w.concentration[ftc1.x == zm, :] * phi_w[ftc1.x == zm] +
                ftc1.SF6g.concentration[ftc1.x == zm, :] * phi_g[ftc1.x == zm]) / (
                    phi_w[ftc1.x == zm] + phi_g[ftc1.x == zm])

        M2D9 = (ftc2.SF6w.concentration[ftc2.x == zm, :] * phi_w[ftc2.x == zm] +
                ftc2.SF6g.concentration[ftc2.x == zm, :] * phi_g[ftc2.x == zm]) / (
                    phi_w[ftc2.x == zm] + phi_g[ftc2.x == zm])

        M3D9 = (ftc3.SF6w.concentration[ftc3.x == zm, :] * phi_w[ftc3.x == zm] +
                ftc3.SF6g.concentration[ftc3.x == zm, :] * phi_g[ftc3.x == zm]) / (
                    phi_w[ftc3.x == zm] + phi_g[ftc3.x == zm])

        MD9 = np.concatenate((M1D9[0], M2D9[0], M3D9[0]))

        zm = 21
        M1D21 = (ftc1.SF6w.concentration[ftc1.x == zm, :] * phi_w[ftc1.x == zm] +
                ftc1.SF6g.concentration[ftc1.x == zm, :] * phi_g[ftc1.x == zm]) / (
                    phi_w[ftc1.x == zm] + phi_g[ftc1.x == zm])

        M2D21 = (ftc2.SF6w.concentration[ftc2.x == zm, :] * phi_w[ftc2.x == zm] +
                ftc2.SF6g.concentration[ftc2.x == zm, :] * phi_g[ftc2.x == zm]) / (
                    phi_w[ftc2.x == zm] + phi_g[ftc2.x == zm])

        M3D21 = (ftc3.SF6w.concentration[ftc3.x == zm, :] * phi_w[ftc3.x == zm] +
                ftc3.SF6g.concentration[ftc3.x == zm, :] * phi_g[ftc3.x == zm]) / (
                    phi_w[ftc3.x == zm] + phi_g[ftc3.x == zm])

        MD21 = np.concatenate((M1D21[0], M2D21[0], M3D21[0]))

        zm = 33
        M1D33 = (ftc1.SF6w.concentration[ftc1.x == zm, :] * phi_w[ftc1.x == zm] +
                ftc1.SF6g.concentration[ftc1.x == zm, :] * phi_g[ftc1.x == zm]) / (
                    phi_w[ftc1.x == zm] + phi_g[ftc1.x == zm])

        M2D33 = (ftc2.SF6w.concentration[ftc2.x == zm, :] * phi_w[ftc2.x == zm] +
                ftc2.SF6g.concentration[ftc2.x == zm, :] * phi_g[ftc2.x == zm]) / (
                    phi_w[ftc2.x == zm] + phi_g[ftc2.x == zm])

        M3D33 = (ftc3.SF6w.concentration[ftc3.x == zm, :] * phi_w[ftc3.x == zm] +
                ftc3.SF6g.concentration[ftc3.x == zm, :] * phi_g[ftc3.x == zm]) / (
                    phi_w[ftc3.x == zm] + phi_g[ftc3.x == zm])

        MD33 = np.concatenate((M1D33[0], M2D33[0], M3D33[0]))

        MF1 = ftc1.estimate_flux_at_top('SF6g') + ftc1.estimate_flux_at_top(
            'SF6w') + ftc1.estimate_flux_at_top('SF6mp')
        MF2 = ftc2.estimate_flux_at_top('SF6g') + ftc2.estimate_flux_at_top(
            'SF6w') + ftc2.estimate_flux_at_top('SF6mp')
        MF3 = ftc3.estimate_flux_at_top('SF6g') + ftc3.estimate_flux_at_top(
            'SF6w') + ftc3.estimate_flux_at_top('SF6mp')

        MF = np.concatenate((MF1, MF2, MF3))

        MT = np.concatenate((ftc1.time, ftc2.time + ftc1.time[-1],
                            ftc3.time + ftc2.time[-1] + ftc1.time[-1]))

        idxs = find_indexes_of_intersections(MT, Tm, dt)
        idxs_f = find_indexes_of_intersections(MT, Tm[::2] + 1, dt)

        if np.isnan(MF).any() or np.isnan(MD9).any() or np.isnan(
                MD21).any() or np.isnan(MD33).any():
            err = 1e8
        else:
            err = norm_rmse(MD9[idxs], CD9_mean[:len(MD9[idxs])]) + norm_rmse(
                MD21[idxs], CD21_mean[:len(MD21[idxs])]) + norm_rmse(
                    MD33[idxs], CD33_mean[:len(MD33[idxs])]) + 10 * norm_rmse(
                        MF[idxs_f], Fx_mean[:len(MF[idxs_f])])

    except:
        err = 1e8
    print(":::::: {}".format(err))
    return err


# w = -0.02
# k_w_in = 0.06
# k_w_out = 0.02
# k_g_in = 0.9
# k_g_out = 90

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
    fun, [
        -0.00949713042005, 0.0664636050663, 0.0178208456813, 0.549858340604,
        116.065001772, 0.97, 0.97, 0.97
    ],
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
