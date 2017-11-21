""" Module for the calibration of the parameters with experimental results
"""
import numpy as np


def find_indexes_of_intersections(s, o):
    """
    find indexes of intersections simulated vs observed
    Args:
        s: simulated values (usually time)
        o: observed values (usually time/dates)

    Returns:
        idxs: indexes of simulated values
    """

    idxs = np.array([])
    for o_i in o:
        idx, = np.where(s == o_i)
        if idx.size > 0:
            idxs = np.append(idxs, int(idx[0]))
    return idxs.astype(int)


class Measurement:
    """ class of measurement, stores measured information
    """

    def __init__(self, name, depth, time):
        self.name = name
        self.depth = depth
        self.time = time


class Calibrator:
    """ class setups and executes calibration routines
    """

    def __init__(self, params, lb, ub):
        """ defines which parameters to calibrate, range of the 
        parameters, optimization function etc.
        """
        self.parameter_list = params
        self.lower_boundary = lb
        self.upper_boundary = ub
        raise NotImplementedError

    def add_measurement(self, name, depth, time):
        """this method adds measured results,
        model needs to know what variables to compare with

        Parameters
        ----------
        name: name of the variable in the model
        depth: depth of measuremet
        time: vector of time stamps relative to init time of the model

        Returns
        -------
        """
        raise NotImplementedError

    def run(self):
        """ executes calibration of the model
        """
        raise NotImplementedError
