from __future__ import absolute_import
from collections.abc import Iterable
import copy
import numpy as np
import pynocular as pn
from pynocular.utils.formatter import format_html
import pynocular.plotting

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
        for arg in args:
            if isinstance(arg, pn.GridArray):
                inputs.append(np.ma.asarray(arg))
                if first is None:
                    first = arg
            else:
                inputs.append(arg)

        if first is None:
            raise ValueError()
        if 'axis' in kwargs:
            axis = kwargs.get('axis')
            if not isinstance(axis, tuple) and axis is not None:
                axis = (axis,)
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
                if axis is not None and any([a < first.grid.nax for a in axis]):
                    new_grid = copy.deepcopy(first.grid)
                    for a in sorted(axis)[::-1]:
                        # need to be careful, and start deleting from last element
                        if a < first.grid.nax:
                            del(new_grid.axes[a])
                else:
                    new_grid = first.grid
                
                new_obj = pn.GridArray(result, grid=new_grid)
                if new_obj.nax == 0:
                    return new_obj.data
                return new_obj
            if result.ndim == 0:
                return np.asscalar(result)
        return result
    return wrapped_func


class GridArray(np.ma.MaskedArray):
    '''Structure to hold a single GridArray
    '''
    def __new__(cls, input_array, *args, grid=None, **kwargs):
        # ToDo: sort out kwargs
        dtype = kwargs.pop('dtype', None)
        order = kwargs.pop('order', 'K')
        subok = kwargs.pop('subok', False)
        ndmin = kwargs.pop('ndmin', 0)

        super().__new__(cls, input_array, dtype=dtype, order=order, subok=subok, ndmin=ndmin)

        obj = np.ma.asarray(input_array).view(cls)
        if grid is not None:
            obj.grid = grid
        else:
            obj.grid = pn.grid.Grid(*args, **kwargs)
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
        return 'GridArray(%s)'%(np.ma.asarray(self))

    def _repr_html_(self):
        '''for jupyter'''
        return format_html(self)
    
    def __str__(self):
        return '%s'%(np.ma.asarray(self))

    @property
    def nax(self):
        return self.grid.nax

    def __getitem__(self, item, *args):
        if isinstance(item, pn.GridArray):
            if item.dtype == np.bool:
                mask = np.logical_and(~self.mask, ~np.asarray(item))
                new_item = pn.GridArray(np.ma.asarray(self), grid=self.grid)
                new_item.mask = mask
                return new_item
            raise NotImplementedError('get item %s'%item)
        if not isinstance(item, tuple): # and not isinstance(item, slice):
            return self[(item,)]
        if isinstance(item, list):
            if all([isinstance(i, int) for i in item]):
                return self[(list,)]
            else:
                raise IndexError('Cannot process list of indices %s'%item)
        if isinstance(item, tuple):
            new_grid = self.grid[item]
            if len(new_grid) == 0:
                # then we have a single element
                return np.ma.asarray(self)[item]
            return pn.GridArray(np.ma.asarray(self)[item], grid=new_grid)

    def __setitem__(self, item, val):
        # ToDo: a[[1,3,5]] *= x does not assign
        #print(item, val)
        if np.isscalar(self[item]):
            self.data[item] = val
            return
        if isinstance(item, list) and all([isinstance(i, int) for i in item]):
            print('list')
            self.data[item] = val
            print(self.data)
            return
        if isinstance(self[item].data, np.ma.masked_array):
            mask = ~self[item].data.mask
        else:
            mask = slice(None)
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
        return pn.GridArray(new_data, grid=self.grid.T)
    
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
        return np.ma.asarray(self).__array_ufunc__(ufunc, method, *inputs, **kwargs)
        
    def __array__(self):
        print('array')


    def __array_prepare__(self, result, context=None):
        print('prepare')
        return result

    def __array_wrap__(self, out_arr, context=None):
        print('In __array_wrap__:')
        #print('   self is %s' % repr(self))
        #print('   arr is %s' % repr(out_arr))
        obj = np.ma.asarray(out_arr).view(pn.GridArray)
        obj.grid = self.grid
        return obj

    def flat(self):
        '''return values as flattened array'''
        if self.ndim == self.nax:
            return np.ma.asarray(self).ravel()
        return np.ma.asarray(self).reshape(self.grid.size, -1)


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
            return pn.plotting.plot_map(self, label=label, cbar=cbar, fig=fig, ax=ax, **kwargs)

        raise ValueError('Can only plot maps of 2d grids')

    def plot_step(self, label=None, fig=None, ax=None, **kwargs):
        return pn.plotting.plot_step(self, label=label, fig=fig, ax=ax, **kwargs)

