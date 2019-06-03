from __future__ import absolute_import
from collections import OrderedDict
from collections.abc import Iterable
import copy
import numpy as np
import pynocular as pn
from pynocular.utils.formatter import format_html
import pynocular.plotting
import tabulate

#__all__ = ['GridData']

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
                if axis is not None and any([a < first.grid.naxes for a in axis]):
                    new_grid = copy.deepcopy(first.grid)
                    for a in sorted(axis)[::-1]:
                        # need to be careful, and start deleting from last element
                        if a < first.grid.naxes:
                            del(new_grid.axes[a])
                else:
                    new_grid = first.grid
                
                new_obj = pn.GridArray(result, grid=new_grid)
                if new_obj.naxes == 0:
                    return new_obj.data
                return new_obj
            if result.ndim == 0:
                return np.asscalar(result)
        return result
    return wrapped_func


class GridArray(np.ma.MaskedArray):
    '''Structure to hold a single GridData item
    '''
    def __new__(cls, input_array, *args, grid=None, **kwargs):
        #print(type(input_array))

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
        self.name = getattr(obj, 'name', 'noname')
        return self

    def __repr__(self):
        return 'GridArray(%s : %s)'%(self.name, np.ma.asarray(self))

    def _repr_html_(self):
        '''for jupyter'''
        return format_html(self)
    
    def __str__(self):
        return '%s : %s'%(self.name, np.ma.asarray(self))

    @property
    def naxes(self):
        return self.grid.naxes

    def __getitem__(self, item, *args):
        #print('getitem')
        if isinstance(item, pn.GridArray):
            if item.dtype == np.bool:
                mask = np.logical_and(~self.mask, ~np.asarray(item))
                new_item = pn.GridArray(np.ma.asarray(self), grid=self.grid)
                new_item.mask = mask
                return new_item
            raise NotImplementedError('get item %s'%item)
        if not isinstance(item, tuple):
            return self[(item,)]
        elif isinstance(item, list):
            if all([isinstance(i, int) for i in item]):
                return self[(list,)]
            else:
                raise IndexError('Cannot process list of indices %s'%item)
        elif isinstance(item, tuple):
            new_grid = self.grid[item]
            if len(new_grid) == 0:
                # then we have a single element
                return np.ma.asarray(self)[item]
            return pn.GridArray(np.ma.asarray(self)[item], grid=new_grid)

    def __setitem__(self, var, val):
        # ToDo: a[[1,3,5]] *= x does not assign
        if np.isscalar(self[var]):
            self.data[var] = val
            return
        if isinstance(self[var]._data, np.ma.masked_array):
            mask = ~self[var]._data.mask
        else:
            mask = slice(None)
        if np.isscalar(val):
            self[var]._data[mask] = val
        else:
            self[var]._data[mask] = val._data[mask]

    @property
    def T(self):
        '''transpose'''
        if self.ndim > self.naxes + 1:
            raise NotImplementedError()
        if self.naxes == 1:
            return self
        if self.naxes > 1:
            new_data = self.data.T
        if self.ndim == self.naxes + 1:
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







class GridData(pn.data.Data):
    '''
    Class to hold grid data
    '''
    def __init__(self, *args, **kwargs):
        '''
        Set the grid
        '''
        self.data = []

        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], pn.GridArray):
            self.grid = args[0].grid
            self.add_data(args[0].name, args[0])
        elif len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], (pn.grid.Grid, pn.Grid)):
            self.grid = args[0]
        else:
            self.grid = pn.grid.Grid(*args, **kwargs)

    def __repr__(self):
        strs = []
        strs.append('GridData(%s'%self.grid.__repr__())
        strs.append(self.data.__repr__() + ')')
        return '\n'.join(strs)

    def __str__(self):
        strs = []
        strs.append(self.grid.__str__())
        strs.append(self.data.__str__())
        return '\n'.join(strs)

    def _repr_html_(self):
        '''for jupyter'''
        return format_html(self)

    def __setitem__(self, var, val):
        self.add_data(var, val)

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.vars:
                if item in self.data_vars:
                    idx = self.data_vars.index(item)
                    data = self.data[idx]
                else:
                    data = None
                new_data = pn.GridArray(self.grid, item, data)
                return new_data

        # mask
        if isinstance(item, pn.GridArray):
            if item.data.dtype == np.bool:
                # in this case it is a mask
                # ToDo: masked operations behave strangely, operations are applyed to all elements, even if masked
                new_data = pn.GridData(self.grid)
                for v in self.data_vars:
                    new_data[v] = self[v][item]
                return new_data
            raise NotImplementedError('get item %s'%item)

        # create new instance with only those vars
        if isinstance(item, Iterable) and all([isinstance(v, str) for v in item]):
            new_data = pn.GridData(self.grid)
            for v in item:
                if v in self.data_vars:
                    new_data[v] = self.data[v]
            return new_data

        # slice
        new_grid = self.grid[item]
        if len(new_grid) == 0:
            return {d.name : d[item] for d in self}
        new_data = pn.GridData(new_grid)
        for d in self:
            new_data[d.name] = d[item]
        return new_data
        

    @property
    def T(self):
        '''transpose'''
        new_obj = pn.GridData()
        new_obj.grid = self.grid.T
        new_obj.data = [d.T for d in self]
        return new_obj
        #raise NotImplementedError()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        '''interface to numpy unversal functions'''
        scalar_results = OrderedDict()
        array_result = pn.GridData()
        for var in inputs[0].data_vars:
            converted_inputs = [inp[var] for inp in inputs]
            result = converted_inputs[0].__array_ufunc__(ufunc, method, *converted_inputs, **kwargs)
            if isinstance(result, pn.GridArray):
                array_result[var] = result
            else:
                scalar_results[var] = result
        if len(array_result.data_vars) == 0:
            return scalar_results
        if len(scalar_results) == 0:
            return array_result
        return scalar_results, array_result
 
    @property
    def vars(self):
        '''
        Available variables
        '''
        return self.grid.vars + self.data_vars

    @property
    def data_vars(self):
        '''
        only data variables (no grid vars)
        '''
        return [d.name for d in self]

    @property
    def shape(self):
        return self.grid.shape

    @property
    def ndim(self):
        return self.grid.naxes

    @property
    def array_shape(self):
        '''
        shape of a single variable
        '''
        return self.shape

    def add_data(self, var, data):
        '''Add data

        Parameters
        ----------
        var : str
            name of data
        data : GridArray, GridData, Array
        '''
        if var in self.grid.vars:
            raise ValueError('Variable `%s` is already a grid dimension!'%var)

        if isinstance(data, pn.GridArray):
            if not self.grid.initialized:
                self.grid = data.grid
            else:
                assert self.grid == data.grid
            data.name = var
            #data = data.data

        elif isinstance(data, pn.GridData):
            # ToDo: needed?
            assert len(data.data_vars) == 1
            if self.grid.naxes == 0:
                self.grid == data.grid
            else:
                assert self.grid == data.grid
            data = data[0]

        elif self.ndim == 0:
            print('adding default grid')
            # make a default grid:
            if data.ndim <= 3 and var not in ['x', 'y', 'z']:
                axes_names = ['x', 'y', 'z']
            else:
                axes_names = ['x%i' for i in range(data.ndim+1)]
                axes_names.delete(var)
            axes = OrderedDict()
            for d, n in zip(axes_names, data.shape):
                axes[d] = np.arange(n)
            self.grid = pn.Grid(**axes)

        if data.ndim < self.ndim and self.shape[-1] == 1:
            # add new axis
            data = data[..., np.newaxis]

        data = pn.GridArray(self.grid, var, data)

        if not data.shape[:self.ndim] == self.shape:
            raise ValueError('Incompatible data of shape %s for grid of shape %s'%(data.shape, self.shape))

        if var in self.data_vars:
            idx = self.data_vars.index(var)
            self.data[idx] = data
        else:
            self.data.append(data)

    def get_array(self, var, flat=False):
        '''
        return array of data

        Parameters:
        -----------

        var : string
            variable to return
        flat : bool (optional)
            if true return flattened (1d) array
        '''
        if var in self.grid.vars:
            array = self.grid.point_mgrid[self.grid.vars.index(var)]
        else:
            array = np.asanyarray(self[var])
        if flat:
            if array.ndim == self.grid.naxes:
                return array.ravel()
            return array.reshape(self.grid.size, -1)

        return array

    def flat(self, var):
        '''return flatened-out array'''
        return self.get_array(var, flat=True)

    def __iter__(self):
        '''
        iterate over dimensions
        '''
        return iter(self.data)


    # --- Plotting methods ---

    def plot(self, var=None, **kwargs):
        if var is None and len(self.data_vars) == 1:
            var = self.data_vars[0]
        if self.ndim == 1:
            return self.plot_step(var, **kwargs)
        elif self.ndim == 2:
            return self.plot_map(var, **kwargs)

    def plot_map(self, var=None, cbar=False, fig=None, ax=None, **kwargs):
        '''
        plot a variable as a map

        ax : pyplot axes object
        var : str
        '''
        if var is None and len(self.data_vars) == 1:
            var = self.data_vars[0]
        if self.grid.naxes == 2:
            return pn.plotting.plot_map(self, var, cbar=cbar, fig=fig, ax=ax, **kwargs)

        raise ValueError('Can only plot maps of 2d grids')

    def plot_contour(self, var=None, fig=None, ax=None, **kwargs):
        if var is None and len(self.data_vars) == 1:
            var = self.data_vars[0]
        return pn.plotting.plot_contour(self, var, fig=fig, ax=ax, **kwargs)

    def plot_step(self, var=None, fig=None, ax=None, **kwargs):
        if var is None and len(self.data_vars) == 1:
            var = self.data_vars[0]
        return pn.plotting.plot_step(self, var, fig=fig, ax=ax, **kwargs)

    def plot_bands(self, var=None, fig=None, ax=None, **kwargs):
        if var is None and len(self.data_vars) == 1:
            var = self.data_vars[0]
        return pn.plotting.plot_bands(self, var, fig=fig, ax=ax, **kwargs)

    def plot_errorband(self, var, errors, fig=None, ax=None, **kwargs):
        if var is None and len(self.data_vars) == 1:
            var = self.data_vars[0]
        return pn.plotting.plot_errorband(self, var, errors, fig=fig, ax=ax, **kwargs)
