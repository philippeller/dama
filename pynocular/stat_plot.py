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

def plot(layer, x, y, *args, **kwargs):
    fig = kwargs.pop('fig', plt.gcf())
    ax = kwargs.pop('fig', plt.gca())
    p = ax.plot(layer[x], layer[y], *args, **kwargs)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return p

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
        if isinstance(s, str):
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
    histtype = kwargs.pop('histtype', 'step')

    # let's only histogram finite values
    mask = np.isfinite(layer[var])
    ax.hist(layer.grid[0].points[mask], bins=layer.grid[0].edges, weights=layer[var][mask], histtype=histtype, **kwargs)
    ax.set_xlabel(layer.grid[0].var)
    ax.set_ylabel(var)

def plot_bands(layer, var, fig=None, ax=None, **kwargs):
    '''
    plot band between the variables values (expect each bin to have a 1d array)
    '''
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()
    assert layer.grid.ndim == 1
    
    cmap = kwargs.pop('cmap', 'Blues')
    cmap = plt.get_cmap(cmap)

    data = layer[var]
    n_points = data.shape[1]
    
    n_bands = (n_points+1)//2

    colors = cmap(np.linspace(0, 1, n_bands+1))[1:]
    
    colors = kwargs.pop('colors', colors)

    for i in range(n_bands):
        upper_idx = n_points - i - 1

        if not upper_idx == i:
            ax.bar(layer.grid[0].points,
                   data[:, upper_idx] - data[:,i],
                   bottom=data[:,i],
                   width = np.diff(layer.grid[0].edges),
                   color=colors[i],
                   **kwargs)
        else:
            ax.hist(layer.grid[0].points, bins=layer.grid[0].edges, weights=data[:,i], histtype='step', color=colors[i], **kwargs)
            #ax.bar(layer.grid[0].points,
            #       data[:, upper_idx] - data[:,i],
            #       bottom=data[:,i],
            #       width = np.diff(layer.grid[0].edges),
            #       edgecolor=colors[i+1],
            #       **kwargs)

    ax.set_xlabel(layer.grid[0].var)
    ax.set_ylabel(var)

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
    elif isinstance(errors, str):
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
