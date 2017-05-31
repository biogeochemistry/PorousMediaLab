import numexpr as ne
from scipy.sparse import linalg


def linear_alg_solver(A, B):
    return linalg.spsolve(A, B, use_umfpack=True)


def ode_integrate(C0, dcdt, rates, coef, dt, solver='rk4'):
    """Integrates the reactions according to 4th Order Runge-Kutta method or Butcher 5th
    where the variables, rates, coef are passed as dictionaries
    """

    def implicit_solver(C_0):

        class Derivative:

            def __init__(self, f, h=1E-5):
                self.f = f
                self.h = float(h)

            def __call__(self, x):
                f, h = self.f, self.h
                return (f(x + h) - f(x - h)) / (2 * h)

        def Newton(f, x, dfdx, epsilon=1.0E-7, N=100, store=False):
            f_value = f(x)
            n = 0
            if store:
                info = [(x, f_value)]
            while abs(f_value) > epsilon and n <= N:
                dfdx_value = float(dfdx(x))
                if abs(dfdx_value) < 1E-14:
                    raise ValueError("Newton: f’(%g)=%g" % (x, dfdx_value))

                x = x - f_value / dfdx_value
                n += 1
                f_value = f(x)
                if store:
                    info.append((x, f_value))
            if store:
                return x, info
            else:
                return x, n, f_value

        def F(w):
            return w - k_loop(w) - C0

        dFdw = Derivative(F)

        C_new = {}
        k1 = k_loop(C_0)
        for element in C_0:
            w_start = C_0[element] + k1[element]
            C_new[element], _ = Newton(F, w_start, dFdw, N=30)

        raise NotImplemented

    def k_loop(conc, dt=dt):
        rates_num = {}
        for element, rate in rates.items():
            rates_num[element] = ne.evaluate(rate, {**coef, **conc})

        Kn = {}
        for element in dcdt:
            Kn[element] = dt * ne.evaluate(dcdt[element], {**coef, **rates_num})

        return Kn

    def sum_k(A, B, b):
        C_new = {}
        for k in A:
            C_new[k] = A[k] + b * B[k] * dt
        return C_new

    def rk4(C_0):
        """Integrates the reactions according to 4th Order Runge-Kutta method
            k_1 = dt*dcdt(C0, dt)
            k_2 = dt*dcdt(C0+0.5*k_1, dt)
            k_3 = dt*dcdt(C0+0.5*k_2, dt)
            k_4 = dt*dcdt(C0+k_3, dt)
            C_new = C0 + (k_1+2*k_2+2*k_3+k_4)/6
        """
        k1 = k_loop(C_0)
        k2 = k_loop(sum_k(C_0, k1, 0.5))
        k3 = k_loop(sum_k(C_0, k2, 0.5))
        k4 = k_loop(sum_k(C_0, k3, 1))
        C_new = {}
        num_rates = {}
        for element in C_0:
            num_rates[element] = (
                k1[element] + 2 * k2[element] + 2 * k3[element] + k4[element]) / 6
            C_new[element] = C_0[element] + num_rates[element]
        return C_new, num_rates

    def butcher5(C_0):
        """
        k_1 = dt*sediment_rates(C0, dt);
        k_2 = dt*sediment_rates(C0 + 1/4*k_1, dt);
        k_3 = dt*sediment_rates(C0 + 1/8*k_1 + 1/8*k_2, dt);
        k_4 = dt*sediment_rates(C0 - 1/2*k_2 + k_3, dt);
        k_5 = dt*sediment_rates(C0 + 3/16*k_1 + 9/16*k_4, dt);
        k_6 = dt*sediment_rates(C0 - 3/7*k_1 + 2/7*k_2 + 12/7*k_3 - 12/7*k_4 + 8/7*k_5, dt);
        C_new = C0 + (7*k_1 + 32*k_3 + 12*k_4 + 32*k_5 + 7*k_6)/90;
        """
        k1 = k_loop(C_0)
        k2 = k_loop(sum_k(C_0, k1, 1 / 4))
        k3 = k_loop(sum_k(sum_k(C_0, k1, 1 / 8), k2, 1 / 8))
        k4 = k_loop(sum_k(sum_k(C_0, k2, -0.5), k3, 1))
        k5 = k_loop(sum_k(sum_k(C_0, k1, 3 / 16), k4, 9 / 16))
        k6 = k_loop(sum_k(sum_k(sum_k(sum_k(sum_k(C_0, k1, -3 / 7),
                                            k2, 2 / 7), k3, 12 / 7), k4, -12 / 7), k5, 8 / 7))
        C_new = {}
        num_rates = {}
        for element in C_0:
            num_rates[element] = (7 * k1[element] + 32 * k3[element] +
                                  12 * k4[element] + 32 * k5[element] + 7 * k6[element]) / 90
            C_new[element] = C_0[element] + num_rates[element]
        return C_new, num_rates

    if solver == 'butcher5':
        return butcher5(C0)
    if solver == 'implicit':
        return implicit_solver(C0)

    return rk4(C0)
