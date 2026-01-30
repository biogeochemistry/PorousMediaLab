"""Statistical metrics for Calibrator routines
"""

import numpy as np
from sklearn.metrics import r2_score


def filter_nan(s, o):
    """
    this functions removed the data  from simulated and observed data
    wherever the observed data contains nan
    this is used by all other functions, otherwise they will produce nan as
    output
    """
    data = np.array([s.flatten(), o.flatten()])
    data = np.transpose(data)
    data = data[~np.isnan(data).any(1)]
    return data[:, 0], data[:, 1]


def percentage_deviation(s, o):
    """
    Percent deviation
    input:
        s: simulated
        o: observed
    output:
        percent deviation
    """
    s, o = filter_nan(s, o)
    return sum(sum(abs(s - o) / abs(o)))


def pc_bias(s, o):
    """
    Percent Bias
    input:
        s: simulated
        o: observed
    output:
        pc_bias: percent bias
    """
    s, o = filter_nan(s, o)
    return 100.0 * sum(s - o) / sum(o)


def apb(s, o):
    """
    Absolute Percent Bias
    input:
        s: simulated
        o: observed
    output:
        apb_bias: absolute percent bias
    """
    s, o = filter_nan(s, o)
    return 100.0 * sum(abs(s - o)) / sum(o)


def rmse(s, o):
    """
    Root Mean Squared Error
    input:
        s: simulated
        o: observed
    output:
        rmse: root mean squared error
    """
    s, o = filter_nan(s, o)
    return np.sqrt(np.mean((s - o)**2))


def norm_rmse(s, o):
    """
    Normalized to stanard deviation Root Mean Squared Error
    input:
        s: simulated
        o: observed
    output:
        nrmse: normalized root mean squared error: RMSE / mean or SD
    """
    s, o = filter_nan(s, o)
    return rmse(s, o) / np.std(o)


def mae(s, o):
    """
    Mean Absolute Error
    input:
        s: simulated
        o: observed
    output:
        maes: mean absolute error
    """
    s, o = filter_nan(s, o)
    return np.mean(abs(s - o))


def bias(s, o):
    """
    Bias
    input:
        s: simulated
        o: observed
    output:
        bias: bias
    """
    s, o = filter_nan(s, o)
    return np.mean(s - o)


def NS(s, o):
    """
    Nash Sutcliffe efficiency coefficient (the same as r^2)
    input:
        s: simulated
        o: observed
    output:
        ns: Nash Sutcliffe efficient coefficient
    """
    s, o = filter_nan(s, o)
    return 1 - sum((s - o)**2) / sum((o - np.mean(o))**2)


def likelihood(s, o, N=5):
    """
    Likelihood
    input:
        s: simulated
        o: observed
    output:
        L: likelihood
    """
    s, o = filter_nan(s, o)
    return np.exp(-N * sum((s - o)**2) / sum((o - np.mean(o))**2))


def correlation(s, o):
    """
    correlation coefficient
    input:
        s: simulated
        o: observed
    output:
        corr: correlation coefficient
    """
    s, o = filter_nan(s, o)
    if s.size == 0:
        corr = np.nan
    else:
        corr = np.corrcoef(o, s)[0, 1]

    return corr


def index_agreement(s, o):
    """
    index of agreement
    input:
        s: simulated
        o: observed
    output:
        ia: index of agreement
    """
    s, o = filter_nan(s, o)
    ia = 1 - (np.sum((o - s)**2)) / (
        np.sum((np.abs(s - np.mean(o)) + np.abs(o - np.mean(o)))**2))
    return ia


def squared_error(s, o):
    """
    squared error
    input:
        s: simulated
        o: observed
    output:
        se: squared error
    """
    s, o = filter_nan(s, o)
    return sum((s - o)**2)


def coefficient_of_determination(s, o):
    """
    coefficient of determination (r-squared)
    input:
        s: simulated
        o: observed
    output:
        r2: coefficient of determination
    """
    s, o = filter_nan(s, o)
    o_mean = np.mean(o)
    se = squared_error(s, o)
    se_mean = squared_error(o, o_mean)
    r2 = 1 - (se / se_mean)
    return r2


def rsquared(s, o):
    """
    coefficient of determination (r-squared)
    using python sklern module
    input:
        s: simulated
        o: observed
    output:
        r2: coefficient of determination
    """
    s, o = filter_nan(s, o)
    return r2_score(o, s)
