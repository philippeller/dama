from __future__ import absolute_import
from collections.abc import Iterable
import copy
import numpy as np
import dragoman as dm
from dragoman import translations
from dragoman.utils.formatter import format_table
import dragoman.plotting

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


def wrap(func):
    '''wrapper function to translate named axis to int indices
    and to re-pack results into correct form'''
    def wrapped_func(*args, **kwargs):
        # find first instance of GridArray:
        first = None
        inputs = []
        grid = None
        for arg in args:
            if isinstance(arg, dm.GridArray):
                inputs.append(np.ma.asarray(arg))
                if first is None:
                    first = arg
                    grid = arg.grid
                else:
                    # make sure all grids are compatible
                    assert arg.grid == grid, 'Incompatible grids'
            else:
                inputs.append(arg)

        if first is None:
            raise ValueError()
        if 'axis' in kwargs:
            axis = kwargs.get('axis')
            if not isinstance(axis, tuple) and axis is not None:
                axis = (axis, )
            if axis is not None:
                new_axis = []
                for a in axis:
                    # translate them
                    if isinstance(a, str):
                        a = first.grid.vars.index(a)
                    if a < 0:
                        a += first.ndim
                    new_axis.append(a)
                if len(new_axis) == 1:
                    kwargs['axis'] = new_axis[0]
                else:
                    kwargs['axis'] = tuple(new_axis)
                axis = new_axis
        else:
            axis = None

        result = func(*inputs, **kwargs)
        if isinstance(result, (np.ma.masked_array, np.ndarray)):
            if result.ndim > 0:
                # new grid
                if axis is not None and any(
                    [a < first.grid.nax for a in axis]
                    ):
                    new_grid = copy.deepcopy(first.grid)
                    for a in sorted(axis)[::-1]:
                        # need to be careful, and start deleting from last element
                        if a < first.grid.nax:
                            del (new_grid.axes[a])
                else:
                    new_grid = first.grid

                new_obj = dm.GridArray(result, grid=new_grid)
                if new_obj.nax == 0:
                    return new_obj.data
                return new_obj
            if result.ndim == 0:
                return np.asscalar(result)
        return result

    return wrapped_func


class GridArray(np.ma.MaskedArray):
    '''Structure to hold a single GridArray

    Parameters
    ----------
    input_array : ndarray
    grid : dm.Grid (optional)
        if not specified, *args and **kwargs will be used to constrcut grid
        if those are also not specified, default grid is added
    '''
    def __new__(cls, input_array, *args, grid=None, **kwargs):
        # ToDo: sort out kwargs
        dtype = kwargs.pop('dtype', None)
        order = kwargs.pop('order', 'K')
        subok = kwargs.pop('subok', False)
        ndmin = kwargs.pop('ndmin', 0)

        super().__new__(
            cls,
            input_array,
            dtype=dtype,
            order=order,
            subok=subok,
            ndmin=ndmin
            )

        obj = np.ma.asarray(input_array).view(cls)
        if grid is not None:
            obj.grid = grid
        elif not len(args) == 0 or not len(kwargs) == 0:
            obj.grid = dm.Grid(*args, **kwargs)
        else:
            # make a default grid:
            if input_array.ndim <= 3:
                axes_names = ['x', 'y', 'z']
            else:
                axes_names = ['x%i' for i in range(input_array.ndim)]
            axes = {}
            for d, n in zip(axes_names, input_array.shape):
                axes[d] = np.arange(n)
            obj.grid = dm.Grid(**axes)
        return obj

    def __array_finalize__(self, obj, *args, **kwargs):
        super().__array_finalize__(obj)
        #print('finalize: args, kwargs', args, kwargs)
        if obj is None:
            print('obj none')
            return
        self.grid = getattr(obj, 'grid', None)
        return self

    def __repr__(self):
        return format_table(self, tablefmt='plain')

    def _repr_html_(self):
        '''for jupyter'''
        return format_table(self, tablefmt='html')

    def __str__(self):
        return format_table(self, tablefmt='plain')

    @property
    def nax(self):
        return self.grid.nax

    def __getitem__(self, item, *args):
        if isinstance(item, str):
            if item in self.grid.vars:
                data = self.get_array(item)
                new_data = dm.GridArray(data, grid=self.grid)
                return new_data

        if isinstance(item, dm.GridArray):
            if item.dtype == np.bool:
                mask = np.logical_and(~self.mask, ~np.asarray(item))
                new_item = dm.GridArray(np.ma.asarray(self), grid=self.grid)
                new_item.mask = mask
                return new_item
            raise NotImplementedError('get item %s' % item)
        if not isinstance(item, tuple):  # and not isinstance(item, slice):
            return self[(item, )]
        if isinstance(item, list):
            if all([isinstance(i, int) for i in item]):
                return self[(list, )]
            else:
                raise IndexError('Cannot process list of indices %s' % item)
        if isinstance(item, tuple):

            item = self.grid.convert_slice(item)

            new_grid = self.grid[item]
            if len(new_grid) == 0:
                # then we have a single element
                return np.ma.asarray(self)[item]
            return dm.GridArray(np.ma.asarray(self)[item], grid=new_grid)

    def get_array(self, var, flat=False):
        '''
        return bare array of data

        Parameters:
        -----------

        var : string
            variable to return of grid vars
        flat : bool (optional)
            if true return flattened (1d) array
        '''
        assert var in self.grid.vars
        array = self.grid.point_meshgrid[self.grid.vars.index(var)]
        if flat:
            if array.ndim == self.grid.nax:
                return array.ravel()
            return array.reshape(self.grid.size, -1)

        return array

    @property
    def array_shape(self):
        '''
        shape of array
        '''
        return self.shape

    def __setitem__(self, item, val):
        if isinstance(item, dm.GridArray):
            if item.dtype == np.bool:
                mask = np.logical_and(~self.mask, ~np.asarray(item))
                if np.isscalar(val):
                    self[item].data[mask] = val
                else:
                    self[item].data[mask] = val.data[mask]
                return
        if np.isscalar(self[item]):
            self.data[item] = val
            return
        if isinstance(item, list) and all([isinstance(i, int) for i in item]):
            self.data[item] = val
            return
        mask = ~self[item].mask
        if np.isscalar(val):
            self[item].data[mask] = val
        else:
            self[item].data[mask] = val.data[mask]

    @property
    def T(self):
        '''transpose'''
        if self.ndim > self.nax + 1:
            raise NotImplementedError()
        if self.nax == 1:
            return self
        if self.nax > 1:
            new_data = self.data.T
        if self.ndim == self.nax + 1:
            new_data = np.rollaxis(new_data, 0, self.ndim)
        return dm.GridArray(new_data, grid=self.grid.T)

    @wrap
    def __add__(self, other):
        return np.ma.add(self, other)

    @wrap
    def __radd__(self, other):
        return np.ma.add(other, self)

    @wrap
    def __sub__(self, other):
        return np.ma.subtract(self, other)

    @wrap
    def __rsub__(self, other):
        return np.ma.subtract(other, self)

    @wrap
    def __mul__(self, other):
        return np.ma.multiply(self, other)

    @wrap
    def __rmul__(self, other):
        return np.ma.multiply(other, self)

    @wrap
    def __truediv__(self, other):
        return np.ma.divide(self, other)

    @wrap
    def __rtruediv__(self, other):
        return np.ma.divide(other, self)

    @wrap
    def __pow__(self, other):
        return np.ma.power(self, other)

    @wrap
    def __rpow__(self, other):
        return np.ma.power(other, other)

    @wrap
    def __lt__(self, other):
        return np.ma.less(self, other)

    @wrap
    def __le__(self, other):
        return np.ma.less_equal(self, other)

    @wrap
    def __eq__(self, other):
        return np.ma.equal(self, other)

    @wrap
    def __ne__(self, other):
        return np.ma.not_equal(self, other)

    @wrap
    def __gt__(self, other):
        return np.ma.greater(self, other)

    @wrap
    def __ge__(self, other):
        return np.ma.greater_equal(self, other)

    @wrap
    def sum(self, **kwargs):
        return np.ma.sum(self, **kwargs)

    @wrap
    def mean(self, **kwargs):
        return np.ma.mean(self, **kwargs)

    @wrap
    def std(self, **kwargs):
        return np.ma.std(self, **kwargs)

    @wrap
    def average(self, **kwargs):
        return np.ma.average(self, **kwargs)

    @wrap
    def median(self, **kwargs):
        return np.ma.median(self, **kwargs)

    @wrap
    def min(self, **kwargs):
        return np.ma.min(self, **kwargs)

    @wrap
    def max(self, **kwargs):
        return np.ma.max(self, **kwargs)

    @wrap
    def cumsum(self, **kwargs):
        return np.ma.cumsum(self, **kwargs)

    @wrap
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        '''callable for numpy ufuncs'''
        #print('ufunc')
        return np.ma.asarray(self).__array_ufunc__(
            ufunc, method, *inputs, **kwargs
            )

    def __array__(self):
        print('array')

    def __array_prepare__(self, result, context=None):
        print('prepare')
        return result

    def __array_wrap__(self, out_arr, context=None):
        print('In __array_wrap__:')
        #print('   self is %s' % repr(self))
        #print('   arr is %s' % repr(out_arr))
        obj = np.ma.asarray(out_arr).view(dm.GridArray)
        obj.grid = self.grid
        return obj

    def flat(self):
        '''return values as flattened array'''
        if self.ndim == self.nax:
            return np.ma.asarray(self).ravel()
        return np.ma.asarray(self).reshape(self.grid.size, -1)

    # --- Tranlsation methods ---

    def interp(self, *args, method=None, fill_value=np.nan, **kwargs):
        return translations.Interpolation(
            self, *args, method=method, fill_value=fill_value, **kwargs
            ).run()

    interp.__doc__ = translations.Interpolation.__init__.__doc__

    def histogram(self, *args, density=False, **kwargs):
        return translations.Histogram(self, *args, density=density,
                                      **kwargs).run()

    histogram.__doc__ = translations.Histogram.__init__.__doc__

    def binwise(self, *args, **kwargs):
        return dm.BinnedData(data=self, *args, **kwargs)

    def lookup(self, *args, **kwargs):
        return translations.Lookup(self, *args, **kwargs).run()

    lookup.__doc__ = translations.Lookup.__init__.__doc__

    def kde(
        self,
        *args,
        bw='silverman',
        kernel='gaussian',
        density=True,
        **kwargs
        ):
        return translations.KDE(
            self, *args, bw=bw, kernel=kernel, density=density, **kwargs
            ).run()

    kde.__doc__ = translations.KDE.__init__.__doc__

    def resample(self, *args, **kwargs):
        return translations.Resample(self, *args, **kwargs).run()

    resample.__doc__ = translations.Resample.__init__.__doc__

    # --- Plotting ---

    def plot(self, **kwargs):
        if self.nax == 1:
            return self.plot_step(**kwargs)
        elif self.nax == 2:
            return self.plot_map(**kwargs)
        else:
            raise NotImplementedError()

    def plot_map(self, label=None, cbar=False, fig=None, ax=None, **kwargs):
        '''
        plot array as a map

        ax : pyplot axes object
        '''
        if self.nax == 2:
            return dm.plotting.plot_map(
                self, label=label, cbar=cbar, fig=fig, ax=ax, **kwargs
                )

        raise ValueError('Can only plot maps of 2d grids')

    def plot_contour(self, fig=None, ax=None, **kwargs):
        return dm.plotting.plot_contour(self, fig=fig, ax=ax, **kwargs)

    def plot_step(self, label=None, fig=None, ax=None, **kwargs):
        return dm.plotting.plot_step(
            self, label=label, fig=fig, ax=ax, **kwargs
            )

    plot_bands = dm.plotting.plot_bands
