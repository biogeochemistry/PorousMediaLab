# k_1 = dt*sediment_rates(C0, dt);
# k_2 = dt*sediment_rates(C0+0.5*k_1, dt);
# k_3 = dt*sediment_rates(C0+0.5*k_2, dt);
# k_4 = dt*sediment_rates(C0+k_3, dt);
# C_new = C0 + (k_1+2*k_2+2*k_3+k_4)/6;

import numpy as np

dt = 0.1

coef = {'k': 1}
rates = {'R1': 'k*y1*y2'}
dcdt = {'y1': '-4 * R1', 'y2': '-R1'}
C0 = {'y2': np.array([0, 0.1, 0.2, 0.3]), 'y1': np.array([0, 0.1, 0.2, 0.3])}


def k_loop(conc):
    rates_num = {}

    for k in rates:
        rates_num[k] = eval(rates[k], {**coef, **conc})

    dcdt_num = {}

    for k in dcdt:
        dcdt_num[k] = eval(dcdt[k], rates_num)

    Kn = {}

    for k in C0:
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


k1 = k_loop(C0)
k2 = k_loop(sum_k(C0, k1, 0.5))
k3 = k_loop(sum_k(C0, k2, 0.5))
k4 = k_loop(sum_k(C0, k3, 1))

C_new = result(C0, k1, k2, k3, k4)
print(C_new)
