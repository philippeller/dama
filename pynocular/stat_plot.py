import numpy as np
from matplotlib import pyplot as plt

'''
Module to provide plotting convenience functions
to be used by data layer classes
'''
def plot_map(layer, var, cbar=False, fig=None, ax=None, **kwargs):
    '''
    plot a 2d color map
    '''
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()
    assert layer.grid.ndim == 2
    X, Y = layer.grid.edge_meshgrid

    pc = ax.pcolormesh(X, Y, layer[var].T, linewidth=0, rasterized=True, **kwargs)
    if cbar:
        fig.colorbar(pc, ax=ax, label=var)

    ax.set_xlabel(layer.grid.vars[0])
    ax.set_ylabel(layer.grid.vars[1])
    ax.set_xlim(layer.grid.edges[0][0], layer.grid.edges[0][-1])
    ax.set_ylim(layer.grid.edges[1][0], layer.grid.edges[1][-1])
    return pc

def plot_points_2d(layer, x, y, s=None, c=None, cbar=False, fig=None, ax=None, **kwargs):
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()
    if c is not None:
        c_label = c
        c = layer[c]
    else:
        assert not cbar
    if s is not None:
        if isinstance(s, basestring):
            s = layer[s]
    sc = ax.scatter(layer[x], layer[y], s=s, c=c, **kwargs)
    if cbar:
        fig.colorbar(sc, ax=ax, label=c_label)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return sc

def plot_contour(layer, var, fig=None, ax=None, **kwargs):
    '''
    contours from gird data
    '''
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()
    assert layer.grid.ndim == 2
    X, Y = layer.grid.point_meshgrid

    cs = ax.contour(X, Y, layer[var].T, **kwargs)
    return cs

def plot_step(layer, var, fig=None, ax=None, **kwargs):
    '''
    plot a step function, i.e. histogram
    '''
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()
    assert layer.grid.ndim == 1
    ax.hist(layer.grid[0].points, bins=layer.grid[0].edges, weights=layer[var], **kwargs)
    ax.set_xlabel(layer.grid[0].var)
    ax.set_ylabel(var)

def plot_band(layer, var1, var2, fig=None, ax=None, **kwargs):
    '''
    plot a band between two variables var1 and var2
    '''
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()
    assert layer.grid.ndim == 1
    ax.bar(layer.grid[0].points,
           layer[var2] - layer[var1],
           bottom=layer[var1],
           width = np.diff(layer.grid[0].edges),
           **kwargs)
    ax.set_xlabel(layer.grid[0].var)
    return var1, var2

def plot_errorband(layer, var, errors, fig=None, ax=None, **kwargs):
    '''
    plot a step histogram with errorbars around it as bands
    '''
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()
    if isinstance(errors, (tuple, list)):
        lower_error = layer[errors[0]]
        upper_error = layer[errors[1]]
    elif isinstance(errors, basestring):
        lower_error = layer[errors]
        upper_error = lower_error
    else:
        raise TypeError('errors must be tuple of variable names or a single variable name')
    assert layer.grid.ndim == 1

    ax.bar(layer.grid[0].points,
           lower_error + upper_error,
           bottom=layer[var] - lower_error,
           width = np.diff(layer.grid[0].edges),
           **kwargs)
    ax.set_xlabel(layer.grid[0].var)