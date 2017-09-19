import unittest
from spec import skip
import Column
import OdeSolver
import numpy as np
from scipy import special
# from unittest import TestCase


def create_lab():
    w = 0.2
    tend = 0.1
    dx = 0.1
    length = 100
    phi = 1
    dt = 0.001
    lab = Column.Column(length, dx, tend, dt, phi, w)
    return lab


class TestIntialization:
    """Test the initialization of the Column class with correct boundary conditions  and initial concentrations"""

    def initial_concentrations_test(self):
        """ Scalar initial condition assigned to the whole vector"""
        init_C = 12.32
        lab = create_lab()
        lab.add_species(True, 'O2', 40, init_C, bc_top=init_C, bc_top_type='dirichlet', bc_bot=0, bc_bot_type='neumann')
        assert np.array_equal(lab.O2.concentration[
                              :, 0], init_C * np.ones((lab.length / lab.dx + 1)))

    def boundary_conditions_test(self):
        """ Dirichlet BC at the interface always assigned"""
        lab = create_lab()
        init_C = 12.32
        bc = 0.2
        lab.add_species(True, 'O2', 40, init_C, bc_top=bc, bc_top_type='dirichlet', bc_bot=0, bc_bot_type='neumann')
        lab.solve()
        assert np.array_equal(lab.O2.concentration[
                              0, :], bc * np.ones((lab.tend / lab.dt + 1)))


class TestMathModel:
    """Testing the accuracy of numerical solutions of integrators"""

    def transport_equation_test(self):
        '''Check the transport equation integrator'''
        lab = create_lab()
        D = 40
        lab.add_species(True, 'O2', D, 0, bc_top=1, bc_top_type='dirichlet', bc_bot=0, bc_bot_type='neumann')
        lab.dcdt.O2 = '0'
        lab.solve()
        x = np.linspace(0, lab.length, lab.length / lab.dx + 1)
        sol = 1 / 2 * (special.erfc((x - lab.w * lab.tend) / 2 / np.sqrt(D * lab.tend)) + np.exp(
            lab.w * x / D) * special.erfc((x + lab.w * lab.tend) / 2 / np.sqrt(D * lab.tend)))

        assert max(lab.O2.concentration[:, -1] - sol) < 1e-4

    def rk4_ode_solver_test(self):
        """Testing 4th order Runge-Kutta ode solver"""
        # dCdt = -R, where R = kC, has a solution C(t)=C0*exp(-kt)
        # C0 = 1
        # k = 2
        C0 = {'C': 1}
        coef = {'k': 2}
        rates = {'R': 'k*C'}
        dcdt = {'C': '-R'}
        dt = 0.0001
        T = 0.01
        time = np.linspace(0, T, T / dt + 1)
        num_sol = np.array(C0['C'])
        for i in range(1, len(time)):
            C_new, _ = OdeSolver.ode_integrate(
                C0, dcdt, rates, coef, dt, solver=1)
            C0['C'] = C_new['C']
            num_sol = np.append(num_sol, C_new['C'])
        assert max(num_sol - np.exp(-coef['k'] * time)) < 1e-5

    def butcher5_ode_solver_test(self):
        """Testing 5th order Butcher ode solver"""
        # dCdt = -R, where R = kC, has a solution C(t)=C0*exp(-kt)
        # C0 = 1
        # k = 2
        C0 = {'C': 1}
        coef = {'k': 2}
        rates = {'R': 'k*C'}
        dcdt = {'C': '-R'}
        dt = 0.0001
        T = 0.01
        time = np.linspace(0, T, T / dt + 1)
        num_sol = np.array(C0['C'])
        for i in range(1, len(time)):
            C_new, _ = OdeSolver.ode_integrate(
                C0, dcdt, rates, coef, dt, solver=0)
            C0['C'] = C_new['C']
            num_sol = np.append(num_sol, C_new['C'])
        assert max(num_sol - np.exp(-coef['k'] * time)) < 1e-5

    def adjust_time_test(self):
        """adjusting time step"""
        skip()


class TestHandling(unittest.TestCase):
    """Test the exception handling with correct terminal messages"""

    def ode_solver_key_error_test(self):
        """Key error with incorrect rates and dcdt in the ode"""
        C0 = {'C': 1}
        coef = {'k': 2}
        rates = {'R': 'k*C'}
        dcdt = {'C1': '-R'}
        dt = 0.0001
        with self.assertRaises(KeyError):
            OdeSolver.ode_integrate(C0, dcdt, rates, coef, dt, solver=0)

    def bc_error_test(self):
        """boundary condition error """
        skip()
