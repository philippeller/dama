from __future__ import absolute_import
import numpy as np
import pynocular as pn
from matplotlib import pyplot as plt

'''Module to provide plotting convenience functions
to be used by data source classes
'''

__license__ = '''Copyright 2019 Philipp Eller

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

# === Modernized ===

def plot_bands(source, var=None, fig=None, ax=None, labels=None, **kwargs):
    '''
    plot band between the variable's values (expect each bin to have a 1d array)

    Parameters:
    -----------
    var : str
        Variable name ot be plotted (if source type is GridArry or GridData with
        a single variable, then that one is used by default)
    
    fig, ax : matplotlib figure and axis (optional)

    labels : iterable
        lables to add for the bands
    '''
    assert isinstance(source, (pn.GridData, pn.GridArray))
    assert source.grid.nax == 1

    if isinstance(source, pn.GridData):
        if var is None and len(source.data_vars) == 1:
            var = source.data_vars[0]
        data = np.array(source[var])

    else:
        data = np.array(source)

    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()
    
    cmap = kwargs.pop('cmap', 'Blues')
    cmap = plt.get_cmap(cmap)

    n_points = data.shape[1]
    
    n_bands = (n_points+1)//2

    colors = cmap(np.linspace(0, 1, n_bands+1))[1:]
    colors = kwargs.pop('colors', colors)

    grid_axis = source.grid.axes[0]

    for i in range(n_bands):
        upper_idx = n_points - i - 1
        if labels is not None and i < len(labels):
            label = labels[i]
        else:
            label = None

        if grid_axis.has_points:
            if not upper_idx == i:
                ax.fill_between(grid_axis.points,
                       data[:, i], 
                       data[:, upper_idx],
                       color=colors[i],
                       label=label,
                       **kwargs)
            else:
                ax.plot(grid_axis.points, data[:, i],
                        color=colors[i], 
                        label=label,
                        **kwargs)


        else:
            if not upper_idx == i:
                ax.bar(grid_axis.edges[:,0],
                       data[:, upper_idx] - data[:, i],
                       bottom=data[:, i],
                       width=grid_axis.edges.width,
                       color=colors[i],
                       align='edge',
                       label=label,
                       **kwargs)
            else:
                band_data = np.ma.asarray(data[:, i])
                band_data = np.ma.append(band_data, band_data[-1])
                ax.step(grid_axis.squeezed_edges, band_data,
                        where='post', 
                        label=label,
                        color=colors[i], **kwargs)

    ax.set_xlabel(source.grid.vars[0])
    ax.set_ylabel(var)

    if grid_axis.has_points:
        ax.set_xlim(grid_axis.points.min(), grid_axis.points.max())
    else:
        ax.set_xlim(grid_axis.edges.min(), grid_axis.edges.max())

def plot_map(source, var=None, cbar=False, fig=None, ax=None, **kwargs):
    '''
    plot a 2d color map

    Parameters:
    -----------

    var : str (optional)
        Variable name ot be plotted (if source type is GridArry or GridData with
        a single variable, then that one is used by default)
    cbar : bool (optional)
        Add colorbar to axis
    fig, ax : matplotlib figure and axis (optional)
    '''
    assert isinstance(source, (pn.GridData, pn.GridArray))
    assert source.grid.nax == 2

    if isinstance(source, pn.GridData):
        if var is None and len(source.data_vars) == 1:
            var = source.data_vars[0]
        data = np.array(source[var])

    else:
        data = np.array(source)
    
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    data = np.ma.asarray(data)

    if data.ndim == source.grid.nax + 1 and data.shape[-1] == 3:
        # plot as image
        pc = ax.imshow(data.swapaxes(0, 1)[::-1, :, :],
                       extent=(source.grid.edges[0].min(), 
                       source.grid.edges[0].max(),
                       source.grid.edges[1].min(),
                       source.grid.edges[1].max()),
                       **kwargs)
    else:
        X, Y = source.grid.edge_meshgrid
        pc = ax.pcolormesh(X, Y, data.T,
                           linewidth=0,
                           rasterized=True,
                           **kwargs)
        if cbar:
            fig.colorbar(pc, ax=ax, label=var)

    ax.set_xlabel(source.grid.vars[0])
    ax.set_ylabel(source.grid.vars[1])
    ax.set_xlim(source.grid.edges[0].min(), source.grid.edges[0].max())
    ax.set_ylim(source.grid.edges[1].min(), source.grid.edges[1].max())
    return pc

def plot_step(source, var=None, label=None, fig=None, ax=None, **kwargs):
    '''
    plot a step function, i.e. histogram
    var : str
        Variable name ot be plotted (if source type is GridArry or GridData with
        a single variable, then that one is used by default)
    label : str
    fig, ax : matplotlib figure and axis (optional)
    '''
    assert isinstance(source, (pn.GridData, pn.GridArray))
    assert source.grid.nax == 1

    if isinstance(source, pn.GridData):
        if var is None and len(source.data_vars) == 1:
            var = source.data_vars[0]
        data = np.array(source[var])

    else:
        data = np.array(source)

    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    data = np.ma.asarray(data)
    data = np.ma.append(data, data[-1])

    s = ax.step(source.grid.squeezed_edges[0], data, where='post', label=label, **kwargs)
    ax.set_xlabel(source.grid.vars[0])
    ax.set_ylabel(var)
    return s


# --- to be fixed ---

def plot1d(source, x, *args, **kwargs):
    '''1d plot'''
    fig = kwargs.pop('fig', plt.gcf())
    ax = kwargs.pop('ax', plt.gca())
    p = ax.plot(source[x], *args, **kwargs)
    ax.set_ylabel(x)
    return p

def plot(source, x, y, *args, **kwargs):
    '''2d plot'''
    fig = kwargs.pop('fig', plt.gcf())
    ax = kwargs.pop('ax', plt.gca())
    p = ax.plot(source[x], source[y], *args, **kwargs)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return p

def plot_points_2d(source, x, y, s=None, c=None, cbar=False, fig=None, ax=None, **kwargs):
    '''2d scatter plot'''
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()
    if c is not None:
        c_label = c
        c = source[c]
    else:
        assert not cbar
    if s is not None:
        if isinstance(s, str):
            s = source[s]
    sc = ax.scatter(np.array(source[x]), np.array(source[y]), s=np.array(s), c=np.array(c), **kwargs)
    if cbar:
        fig.colorbar(sc, ax=ax, label=c_label)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return sc

def plot_contour(source, var, fig=None, ax=None, **kwargs):
    '''
    contours from gird data
    '''
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()
    assert source.grid.nax == 2
    X, Y = source.grid.point_meshgrid

    labels = kwargs.pop('labels', None)
    inline = kwargs.pop('inline', True)

    cs = ax.contour(X, Y, source[var], **kwargs)

    if labels is not None:
        fmt = {}
        for l, s in zip(cs.levels, labels):
            fmt[l] = s

        ax.clabel(cs, cs.levels, inline=inline, fmt=fmt)

    return cs


def plot_errorband(source, var, errors, fig=None, ax=None, **kwargs):
    '''
    plot a step histogram with errorbars around it as bands
    '''
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()
    if isinstance(errors, (tuple, list)):
        lower_error = source[errors[0]]
        upper_error = source[errors[1]]
    elif isinstance(errors, str):
        lower_error = source[errors]
        upper_error = lower_error
    else:
        raise TypeError('errors must be tuple of variable names or a single variable name')
    assert source.grid.nax == 1

    ax.bar(source.grid[0].points,
           lower_error + upper_error,
           bottom=source[var] - lower_error,
           width=source.grid[0].edges.width,
           **kwargs)
    ax.set_xlabel(source.grid[0].var)
