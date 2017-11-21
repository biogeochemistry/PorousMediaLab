""" Module for the calibration of the parameters with experimental results
"""


class Calibrator:
    """ class setups and executes calibration routines
    """

    def __init__(self):
        """ defines which parameters to calibrate, range of the 
        parameters, optimization function etc.
        """
        raise NotImplementedError

    def initiate_measurements(self, variable, depth, time):
        """ this method collects measured results,
        model needs to know what variables to compare with
        """
        raise NotImplementedError

    def run(self):
        """ executes calibration of the model
        """
        raise NotImplementedError
