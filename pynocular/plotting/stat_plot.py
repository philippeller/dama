from __future__ import absolute_import
import numpy as np
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

def plot_map(garray, label=None, cbar=False, fig=None, ax=None, **kwargs):
    '''
    plot a 2d color map
    '''
    
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()
    assert garray.grid.nax == 2

    data = np.ma.asarray(garray)

    if data.ndim == garray.grid.nax + 1 and data.shape[-1] == 3:
        # plot as image
        pc = ax.imshow(data.swapaxes(0, 1)[::-1, :, :], extent=(garray.grid.edges[0].min(), garray.grid.edges[0].max(), garray.grid.edges[1].min(), garray.grid.edges[1].max()), **kwargs)
    else:
        X, Y = garray.grid.edge_meshgrid
        pc = ax.pcolormesh(X, Y, data.T, linewidth=0, rasterized=True, **kwargs)
        if cbar:
            fig.colorbar(pc, ax=ax, label=label)

    ax.set_xlabel(garray.grid.vars[0])
    ax.set_ylabel(garray.grid.vars[1])
    ax.set_xlim(garray.grid.edges[0].min(), garray.grid.edges[0].max())
    ax.set_ylim(garray.grid.edges[1].min(), garray.grid.edges[1].max())
    return pc

def plot_step(garray, label=None, fig=None, ax=None, **kwargs):
    '''
    plot a step function, i.e. histogram
    '''
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()
    assert garray.grid.nax == 1

    data = np.ma.asarray(garray)
    data = np.ma.append(data, data[-1])

    s = ax.step(garray.grid.squeezed_edges[0], data, where='post', **kwargs)
    ax.set_xlabel(garray.grid.vars[0])
    ax.set_ylabel(label)
    return s


# --- to be fixed ---

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

    cs = ax.contour(X, Y, source[var], **kwargs)
    return cs

def plot_bands(source, var, fig=None, ax=None, **kwargs):
    '''
    plot band between the variables values (expect each bin to have a 1d array)
    '''
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()
    assert source.grid.nax == 1
    
    cmap = kwargs.pop('cmap', 'Blues')
    cmap = plt.get_cmap(cmap)

    data = np.array(source[var])
    n_points = data.shape[1]
    
    n_bands = (n_points+1)//2

    colors = cmap(np.linspace(0, 1, n_bands+1))[1:]
    
    colors = kwargs.pop('colors', colors)

    for i in range(n_bands):
        upper_idx = n_points - i - 1

        if not upper_idx == i:
            ax.bar(source.grid.points[0],
                   data[:, upper_idx] - data[:, i],
                   bottom=data[:, i],
                   width=source.grid.edges[0].width,
                   color=colors[i],
                   **kwargs)
        else:
            band_data = np.ma.asarray(data[:, i])
            band_data = np.ma.append(band_data, band_data[-1])
            ax.step(source.grid.squeezed_edges[0], band_data, where='post',  color=colors[i], **kwargs)

    ax.set_xlabel(source.grid.vars[0])
    ax.set_ylabel(var)

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
