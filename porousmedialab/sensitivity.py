""" Module for sensitivity test of the model
based on Salib frameword developed by....
"""


class Sensor:
    """ sensitivity test analysis of the PorousMediaLab results
    """

    def __init__(self):
        raise NotImplementedError

    def add_paramter(self, name, lower_boundary, upper_boundary):
        """add parameter to test

        Arguments:
            name {str} -- name of param, should match name in the model
            lower_boundary {float} -- lower boundary for test
            upper_boundary {float} -- upper boundary for test
        """
        raise NotImplementedError

    def add_reference(self, name):
        """add refernce for test, e.g. what we are testing as changing
        variable in the model (resulting concentration, or amount of biomass),
        it will be comared to itself with varius combination of parameters,
        which are under the test

        Arguments:
            name {str} -- name of the variable in the model
        Raises:
            NotImplementedError -- [description]
        """
        raise NotImplementedError

    def create_basis_for_analysis(self):
        """ runs model for first time and save the result into
        variable for later comparison as a basis
        """
        raise NotImplementedError

    def run_test(self):
        """ run sensetivity test
        """
        raise NotImplementedError


class FastFourierAnalysis(Sensor):
    """Fast Fourier analysis
    """

    def __init__(self, *args):
        super(FastFourierAnalysis, self).__init__(*args)


class SobolevAnalysis(Sensor):
    """Sobolev sensitivity analysis
    """

    def __init__(self, *args):
        super(SobolevAnalysis, self).__init__(*args)
