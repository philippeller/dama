import numpy as np

'''
Module to provide plotting convenience functions
to be used by data layer classes
'''
def plot_map(fig, ax, layer, var, cbar=False, **kwargs):
    '''
    plot a 2d color map
    '''
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

def plot_points_2d(fig, ax, layer, x, y, s=None, c=None, cbar=False, **kwargs):
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

def plot_contour(fig, ax, layer, var, **kwargs):
    '''
    contours from gird data
    '''
    assert layer.grid.ndim == 2
    X, Y = layer.grid.point_meshgrid

    cs = ax.contour(X, Y, layer[var].T, **kwargs)
    return cs

def plot_step(fig, ax, layer, var, **kwargs):
    '''
    plot a step function, i.e. histogram
    '''
    assert layer.grid.ndim == 1
    ax.hist(layer.grid[0].points, bins=layer.grid[0].edges, weights=layer[var], **kwargs)
    ax.set_xlabel(layer.grid[0].var)
    ax.set_ylabel(var)

def plot_band(fig, ax, layer, var1, var2, **kwargs):
    '''
    plot a band between two variables var1 and var2
    '''
    assert layer.grid.ndim == 1
    ax.bar(layer.grid[0].points,
           layer[var2] - layer[var1],
           bottom=layer[var1],
           width = np.diff(layer.grid[0].edges),
           **kwargs)
    ax.set_xlabel(layer.grid[0].var)

def plot_errorband(fig, ax, layer, var, errors, **kwargs):
    '''
    plot a step histogram with errorbars around it as bands
    '''
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
