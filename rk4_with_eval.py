# RK4
# k_1 = dt*dcdt(C0, dt)
# k_2 = dt*dcdt(C0+0.5*k_1, dt)
# k_3 = dt*dcdt(C0+0.5*k_2, dt)
# k_4 = dt*dcdt(C0+k_3, dt)
# C_new = C0 + (k_1+2*k_2+2*k_3+k_4)/6

import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt


def fun():
    dt = 1e-4

    coef = {'k': 1}
    rates = {'R1': 'k*y1'}
    dcdt = {'y1': '-R1', 'y2': '-R1'}
    C = {'y1': np.random.rand(400), 'y2': np.random.rand(400)}

    def k_loop(conc):
        rates_num = {}

        for k in rates:
            rates_num[k] = ne.evaluate(rates[k], {**coef, **conc})

        dcdt_num = {}

        for k in dcdt:
            dcdt_num[k] = ne.evaluate(dcdt[k], rates_num)

        Kn = {}

        for k in C:
            Kn[k] = dt * dcdt_num[k]
        return Kn

    def sum_k(A, B, b):
        C = {}
        for k in A:
            C[k] = A[k] + b * B[k] * dt
        return C

    def result(C_0, k_1, k_2, k_3, k_4):
        C_new = {}
        for k in C_0:
            C_new[k] = C_0[k] + (k_1[k] + 2 * k_2[k] + 2 * k_3[k] + k_4[k]) / 6
        return C_new

    # k1 = k_loop(C)
    # k2 = k_loop(sum_k(C, k1, 0.5))
    # k3 = k_loop(sum_k(C, k2, 0.5))
    # k4 = k_loop(sum_k(C, k3, 1))

    # res[0] = C[0]
    # C = result(C, k1, k2, k3, k4)
    # res = np.zeros(10000)
    # for i in range(10000):
    # res[i] = C['y1'][0]
    k1 = k_loop(C)
    k2 = k_loop(sum_k(C, k1, 0.5))
    k3 = k_loop(sum_k(C, k2, 0.5))
    k4 = k_loop(sum_k(C, k3, 1))
    C = result(C, k1, k2, k3, k4)
    # print(C_new)

    # return res

    # x = np.arange(0, 1, 0.1)
    # plt.plot(C_new['y1'])
    # plt.plot(np.exp(-x))
    # plt.show()


def fun_2():
    # gives very high error
    dt = 1e-4

    c = np.array([1])
    y = np.array([np.random.rand(400), np.random.rand(400)])
    R = np.array(['c[0]*y[0]*y[1]'])
    dcdt = np.array(['-4 * R[0]', '-R[0]'])

    def k_loop(y):
        rates_num = np.array([eval(k, {'c': c, 'y': y}) for k in R])
        dcdt_num = np.array([eval(k, {'R': rates_num}) for k in dcdt])
        return dt * dcdt_num

    def sum_k(A, B, b):
        return A + b * B * dt

    def result(y_0, k_1, k_2, k_3, k_4):
        return y_0 + (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6

    # res = np.zeros(10000)
    # for i in range(10000):
        # res[i] = y[0][0]
    k1 = k_loop(y)
    k2 = k_loop(sum_k(y, k1, 0.5))
    k3 = k_loop(sum_k(y, k2, 0.5))
    k4 = k_loop(sum_k(y, k3, 1))
    y = result(y, k1, k2, k3, k4)
    # print(C_new)

    return y
