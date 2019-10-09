""" Module for the calibration of the parameters with experimental results
"""
from collections import OrderedDict

import numpy as np
from scipy.optimize import minimize

from porousmedialab.dotdict import DotDict
from porousmedialab.metrics import norm_rmse


def find_indexes_of_intersections(s, o, eps):
    """
    find indexes of intersections simulated vs observed
    Args:
        s: simulated values (usually time)
        o: observed values (usually time/dates)
        eps: epsilon of time

    Returns:
        idxs: indexes of simulated values
    """

    idxs = np.array([])
    for o_i in o:
        idx, = np.where(abs(s - o_i) <= eps*1.01)
        if idx.size > 0:
            idxs = np.append(idxs, int(idx[0]))
    return idxs.astype(int)


class Calibrator:
    """ class setups and executes calibration routines
    """

    def __init__(self, lab):
        """ defines which parameters to calibrate, range of the
        parameters, optimization function etc.

        NOTE: self.parameters is ordered dictionary
        to ensure iteration over the same order and
        assigning correct x0 values

        """
        self.lab = lab
        self.parameters = OrderedDict({})
        self.measurements = DotDict({})
        self.error = np.nan
        self.verbose = False

    def add_parameter(self, name, lower_boundary, upper_boundary):
        """ add parameter to calibrate

        Arguments:
            name {str} -- name of the parameter matching name in model
            lb {float} -- lower boundary
            ub {float} -- upper boundary
        """
        self.parameters[name] = DotDict({})
        self.parameters[name]['lower_boundary'] = lower_boundary
        self.parameters[name]['upper_boundary'] = upper_boundary
        self.parameters[name]['value'] = self.lab.constants[name]

    def add_measurement(self, name, values, time, depth=0):
        """add measurment which will be used for calibration. Name
        of the measurement should match name variable in the model

        Arguments:
            name {str} -- name of the measurement
            values {np.array} -- values of measurement
            time {np.array} -- when measured (realative to model times)
            depth {float} -- depth of the measurement for column model
            (default: {0})
        """
        self.measurements[name] = DotDict({})
        self.measurements[name]['values'] = values
        self.measurements[name]['time'] = time
        self.measurements[name]['depth'] = depth

    def estimate_error(self, metric_fun=norm_rmse, disp=True):
        """estimates metrics of measured vs modeled

        Keyword Arguments:
            metric_fun {function} -- desired metric function (default: {rmse})
        """
        err = 0
        for m in self.measurements:
            idxs = find_indexes_of_intersections(
                self.lab.time, self.measurements[m]['time'], self.lab.dt / 2)
            err += metric_fun(self.lab.species[m]['concentration'][:, idxs],
                              self.measurements[m]['values'])
        if disp:
            print('::::: {} = {: .4e}'.format(metric_fun.__name__, err))
        self.error = err

    def min_function(self, x):
        """minimization function f(x) where x are
        parameters which are minimizaed and f is the
        metric

        NOTE: self.parameters is ordered dictionary
        to ensure iteration over the same order and
        assigning correct x values

        Arguments:
            x {[type]} -- [description]
        """
        # assign new values of parameters
        for i, key in enumerate(self.parameters):
            if self.verbose:
                print('{} = {: .4e}'.format(key, x[i]))
            self.lab.constants[key] = x[i]
        # solve with new params
        self.lab.solve(verbose=False)
        # estimate metrics
        self.estimate_error(disp=self.verbose)
        return self.error

    def iter_params(self):
        """creates initial x0

        NOTE: self.parameters is ordered dictionary
        to ensure iteration over the same order and
        assigning correct x0 values

        Returns:
            x0 -- x0 vector for minimization function
        """
        x0 = []
        bnds = []
        for p in self.parameters:
            x0.append(self.parameters[p]['value'])
            bnds.append((self.parameters[p]['lower_boundary'],
                         self.parameters[p]['upper_boundary']))
        return x0, bnds

    def print_final_results(self):
        """function plots final results
        """
        print(self.res.message)
        self.estimate_error()
        print('Calibrated parameters:')
        for param in self.parameters:
            print('\t{} = {: .4e}'.format(param, self.lab.constants[param]))

    def plot_final_results(self):
        """function plot graphs of measured vs modeled

        """
        raise NotImplementedError

    def run(self, verbose=False, method="TNC"):
        """ executes calibration of the model
        and prints final result
        """
        self.verbose = verbose
        x0, bnds = self.iter_params()

        self.res = minimize(
            self.min_function,
            x0,
            method=method,
            bounds=bnds,
            options={
                'maxiter': 100,
                'maxfun': 500
            })

        self.print_final_results()
