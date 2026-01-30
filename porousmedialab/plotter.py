import math
import numpy as np
import matplotlib.pyplot as plt
from porousmedialab.phcalc import Acid

import seaborn as sns
from matplotlib.colors import ListedColormap

sns.set_style("whitegrid")


def custom_plot(lab, x, y, ttl='', y_lbl='', x_lbl=''):
    plt.figure()
    ax = plt.subplot(111)
    plt.plot(x, y, lw=3)
    plt.title(ttl)
    plt.xlim(x[0], x[-1])
    plt.ylabel(y_lbl)
    plt.xlabel(x_lbl)
    ax.grid(linestyle='-', linewidth=0.2)
    return ax


def plot_batch_rates(batch, *args, **kwargs):
    for rate in sorted(batch.estimated_rates):
        plt.figure()
        plot_batch_rate(batch, rate, *args, **kwargs)


def plot_batch_rate(batch, rate, time_factor=1):
    plt.plot(batch.time * time_factor,
             batch.estimated_rates[rate][0] / time_factor, label=rate, lw=3)
    plt.ylabel(r'Rate, $[\Delta C/\Delta T]$')
    plt.xlabel('Time, [T]')
    plt.legend(frameon=1)
    plt.grid(linestyle='-', linewidth=0.2)


def plot_batch_deltas(batch, *args, **kwargs):
    for element in sorted(batch.species):
        plt.figure()
        plot_batch_delta(batch, element, *args, **kwargs)


def plot_batch_delta(batch, element, time_factor=1):
    plt.plot(batch.time[1:] * time_factor, batch.species[element]
             ['rates'][0] / time_factor, label=element, lw=3)
    plt.ylabel(r'Rate of change, $[\Delta C/ \Delta T]$')
    plt.xlabel('Time, [T]')
    plt.legend(frameon=1)
    plt.grid(linestyle='-', linewidth=0.2)


def saturation_index_countour(lab, elem1, elem2, Ks, labels=False):
    plt.figure()
    plt.title('Saturation index %s%s' % (elem1, elem2))
    resoluion = 100
    n = math.ceil(lab.time.size / resoluion)
    plt.xlabel('Time')
    z = np.log10((lab.species[elem1]['concentration'][:, ::n] + 1e-8) * (
        lab.species[elem2]['concentration'][:, ::n] + 1e-8) / lab.constants[Ks])
    lim = np.max(abs(z))
    lim = np.linspace(-lim - 0.1, +lim + 0.1, 51)
    X, Y = np.meshgrid(lab.time[::n], -lab.x)
    plt.xlabel('Time')
    CS = plt.contourf(X, Y, z, 20, cmap=ListedColormap(sns.color_palette(
        "RdBu_r", 101)), origin='lower', levels=lim, extend='both')
    if labels:
        plt.clabel(CS, inline=1, fontsize=10, colors='w')
    # cbar = plt.colorbar(CS)
    if labels:
        plt.clabel(CS, inline=1, fontsize=10, colors='w')
    cbar = plt.colorbar(CS)
    plt.ylabel('Depth')
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False)
    cbar.ax.set_ylabel('Saturation index %s%s' % (elem1, elem2))
    return ax


def plot_fractions(lab):
    for component in lab.acid_base_components:
        if isinstance(component['pH_object'], Acid):
            plt.figure()
            for idx in range(len(component['species'])):
                plt.plot(lab.time, lab.species[component['species'][idx]]
                         ['alpha'][0, :], label=component['species'][idx])
            plt.ylabel('Fraction')
            plt.xlabel('Time')
            plt.legend(frameon=1)
            plt.grid(linestyle='-', linewidth=0.2)


def all_plot_depth_index(lab, *args, **kwargs):
    for element in sorted(lab.species):
        plt.figure()
        plot_depth_index(lab, element, *args, **kwargs, ax=None)


def plot_depth_index(lab, element, idx=0, time_to_plot=False, time_factor=1, ax=None):
    if ax is None:
        ax = plt.subplot(111)
    if element == 'Temperature':
        ax.set_title('Temperature')
        ax.set_ylabel('Temperature, C')
    elif element == 'pH':
        ax.set_title('pH')
        ax.set_ylabel('pH')
    else:
        ax.set_ylabel('Concentration')
    if time_to_plot:
        num_of_elem = int(time_to_plot / lab.dt)
    else:
        num_of_elem = len(lab.time)
    t = lab.time[-num_of_elem:] * time_factor
    ax.set_xlabel('Time')
    if isinstance(element, str):
        ax.plot(t, lab.species[element]['concentration']
                [idx][-num_of_elem:], lw=3)
        ax.set_title(element + ' concentration')
    elif isinstance(element, (list, tuple)):
        for e in element:
            ax.plot(t, lab.species[e]['concentration']
                    [idx][-num_of_elem:], lw=3, label=e)
    ax.legend(frameon=1)
    ax.grid(linestyle='-', linewidth=0.2)
    return ax


def plot_depths(lab, element, depths=[0, 1, 2, 3, 4], time_to_plot=False):
    plt.figure()
    ax = plt.subplot(111)
    if element == 'Temperature':
        plt.title('Temperature at specific depths')
        plt.ylabel('Temperature, C')
    else:
        plt.title(element + ' concentration at specific depths')
        plt.ylabel('Concentration')
    if time_to_plot:
        num_of_elem = int(time_to_plot / lab.dt)
    else:
        num_of_elem = len(lab.time)
    t = lab.time[-num_of_elem:]
    plt.xlabel('Time')
    for depth in depths:
        lbl = str(depth)
        plt.plot(t, lab.species[element]['concentration'][int(
            depth / lab.dx)][-num_of_elem:], lw=3, label=lbl)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(linestyle='-', linewidth=0.2)
    return ax


def plot_times(lab, element, time_slices=[0, 1, 2, 3, 4]):
    plt.figure()
    ax = plt.subplot(111)
    if element == 'Temperature':
        plt.title('Temperature profile')
        plt.xlabel('Temperature, C')
    else:
        plt.title(element + ' concentration')
        plt.xlabel('Concentration')
    plt.ylabel('Depth, cm')
    for tms in time_slices:
        lbl = 'at time: %.2f ' % (tms)
        plt.plot(lab.species[element]['concentration'][
                 :, int(tms / lab.dt)], -lab.x, lw=3, label=lbl)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
    ax.grid(linestyle='-', linewidth=0.2)
    return ax


def plot_profiles(lab):
    for element in sorted(lab.species):
        plot_profile(lab, element)


def plot_profile(lab, element):
    plt.figure()
    plt.plot(lab.profiles[element], -lab.x,
             sns.xkcd_rgb["denim blue"], lw=3, label=element)
    if element == 'Temperature':
        plt.title('Temperature profile')
        plt.xlabel('Temperature, C')
    elif element == 'pH':
        plt.title('pH profile')
        plt.xlabel('pH')
    else:
        plt.title('%s concentration' % (element, ))
        plt.xlabel('Concentration')
    plt.ylabel('Depth')
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False)
    ax.grid(linestyle='-', linewidth=0.2)
    plt.legend()
    plt.tight_layout()
    return ax


def plot_contourplots(lab, **kwargs):
    for element in sorted(lab.species):
        contour_plot(lab, element, **kwargs)


def contour_plot(lab, element, labels=False, days=False, last_year=False):
    plt.figure()
    plt.title(element + ' concentration')
    resoluion = 100
    n = math.ceil(lab.time.size / resoluion)
    if last_year:
        k = n - int(1 / lab.dt)
    else:
        k = 1
    if days:
        X, Y = np.meshgrid(lab.time[k::n] * 365, -lab.x)
        plt.xlabel('Time')
    else:
        X, Y = np.meshgrid(lab.time[k::n], -lab.x)
        plt.xlabel('Time')
    z = lab.species[element]['concentration'][:, k - 1:-1:n]
    CS = plt.contourf(X, Y, z, 51, cmap=ListedColormap(
        sns.color_palette("Blues", 51)), origin='lower')
    if labels:
        plt.clabel(CS, inline=1, fontsize=10, colors='w')
    cbar = plt.colorbar(CS)
    plt.ylabel('Depth')
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False)
    cbar.ax.set_ylabel('%s [M/V]' % element)
    if element == 'Temperature':
        plt.title('Temperature contour plot')
        cbar.ax.set_ylabel('Temperature, C')
    if element == 'pH':
        plt.title('pH contour plot')
        cbar.ax.set_ylabel('pH')
    return ax


def plot_contourplots_of_rates(lab, **kwargs):
    rate = sorted(lab.estimated_rates)
    for r in rate:
        contour_plot_of_rates(lab, r, **kwargs)


def contour_plot_of_rates(lab, r, labels=False, last_year=False):
    plt.figure()
    plt.title('{}'.format(r))
    resoluion = 100
    n = math.ceil(lab.time.size / resoluion)
    if last_year:
        k = n - int(1 / lab.dt)
    else:
        k = 1
    z = lab.estimated_rates[r][:, k - 1:-1:n]
    # lim = np.max(np.abs(z))
    # lim = np.linspace(-lim - 0.1, +lim + 0.1, 51)
    X, Y = np.meshgrid(lab.time[k::n], -lab.x)
    plt.xlabel('Time')
    CS = plt.contourf(X, Y, z, 20, cmap=ListedColormap(
        sns.color_palette("Blues", 51)))
    if labels:
        plt.clabel(CS, inline=1, fontsize=10, colors='w')
    cbar = plt.colorbar(CS)
    plt.ylabel('Depth')
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False)
    cbar.ax.set_ylabel('Rate %s [M/V/T]' % r)
    return ax


def plot_contourplots_of_deltas(lab, **kwargs):
    elements = sorted(lab.species)
    if 'Temperature' in elements:
        elements.remove('Temperature')
    for element in elements:
        contour_plot_of_delta(lab, element, **kwargs)


def contour_plot_of_delta(lab, element, labels=False, last_year=False):
    plt.figure()
    plt.title('Rate of %s consumption/production' % element)
    resoluion = 100
    n = math.ceil(lab.time.size / resoluion)
    if last_year:
        k = n - int(1 / lab.dt)
    else:
        k = 1
    z = lab.species[element]['rates'][:, k - 1:-1:n]
    lim = np.max(np.abs(z))
    lim = np.linspace(-lim - 0.1, +lim + 0.1, 51)
    X, Y = np.meshgrid(lab.time[k:-1:n], -lab.x)
    plt.xlabel('Time')
    CS = plt.contourf(X, Y, z, 20, cmap=ListedColormap(sns.color_palette(
        "RdBu_r", 101)), origin='lower', levels=lim, extend='both')
    if labels:
        plt.clabel(CS, inline=1, fontsize=10, colors='w')
    cbar = plt.colorbar(CS)
    plt.ylabel('Depth')
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False)
    cbar.ax.set_ylabel(r'Rate of %s change $[\Delta/T]$' % element)
    return ax
