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

    Raises:
        ValueError: if sum of observed values is zero
    """
    s, o = filter_nan(s, o)
    sum_o = sum(o)
    if sum_o == 0:
        raise ValueError("Cannot compute percent bias: sum of observed values is zero")
    return 100.0 * sum(s - o) / sum_o


def apb(s, o):
    """
    Absolute Percent Bias
    input:
        s: simulated
        o: observed
    output:
        apb_bias: absolute percent bias

    Raises:
        ValueError: if sum of observed values is zero
    """
    s, o = filter_nan(s, o)
    sum_o = sum(o)
    if sum_o == 0:
        raise ValueError("Cannot compute absolute percent bias: sum of observed values is zero")
    return 100.0 * sum(abs(s - o)) / sum_o


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

    Raises:
        ValueError: if observed values have zero standard deviation
    """
    s, o = filter_nan(s, o)
    std_o = np.std(o)
    if std_o == 0:
        raise ValueError("Cannot compute normalized RMSE: observed values have zero standard deviation")
    return rmse(s, o) / std_o


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

    Raises:
        ValueError: if observed values have zero variance
    """
    s, o = filter_nan(s, o)
    denom = sum((o - np.mean(o))**2)
    if denom == 0:
        raise ValueError("Cannot compute Nash-Sutcliffe coefficient: observed values have zero variance")
    return 1 - sum((s - o)**2) / denom


def likelihood(s, o, N=5):
    """
    Likelihood
    input:
        s: simulated
        o: observed
        N: scaling parameter (default 5)
    output:
        L: likelihood

    Raises:
        ValueError: if observed values have zero variance
    """
    s, o = filter_nan(s, o)
    denom = sum((o - np.mean(o))**2)
    if denom == 0:
        raise ValueError("Cannot compute likelihood: observed values have zero variance")
    return np.exp(-N * sum((s - o)**2) / denom)


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

    Raises:
        ValueError: if denominator is zero
    """
    s, o = filter_nan(s, o)
    denom = np.sum((np.abs(s - np.mean(o)) + np.abs(o - np.mean(o)))**2)
    if denom == 0:
        raise ValueError("Cannot compute index of agreement: denominator is zero")
    ia = 1 - (np.sum((o - s)**2)) / denom
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
