from __future__ import absolute_import
from collections import OrderedDict
from collections.abc import Iterable
import copy
import numpy as np
import pynocular as pn
from pynocular.utils.formatter import as_str, table_labels
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


class GridArray(object):
    '''Structure to hold a single GridData item
    '''
    def __init__(self, grid, name, input_array=None):
        self.grid = grid
        self.name = name
        
        if name in self.grid.vars:
            assert input_array is None, "Cannot add data with name same as grid variable"
        self._data = np.asanyarray(input_array)

        
    def __repr__(self):
        return 'GridArray(%s : %s)'%(self.name, self.data)

    def _repr_html_(self):
        '''for juopyter'''
        if self.naxes == 2:
            table_x = [0] * (self.shape[0] + 1)
            table = [copy.copy(table_x) for _ in range(self.shape[1] + 1)]
            
            table[0][0] = '<b>%s \\ %s</b>'%(self.grid.vars[1], self.grid.vars[0])
            
            x_labels = table_labels(self.grid, 0)
            y_labels = table_labels(self.grid, 1)
                        
            for i in range(self.shape[0]):
                table[0][i+1] = x_labels[i]
            for i in range(self.shape[1]):
                table[i+1][0] = y_labels[i]
                
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    table[j+1][i+1] = as_str(self.data[i, j])
                    
            return tabulate.tabulate(table, tablefmt='html')
        
        elif self.naxes == 1:
            table_x = [0] * (self.shape[0] + 1)
            table = [copy.copy(table_x) for _ in range(2)]
            table[0][0] = '<b>%s</b>'%self.grid.vars[0]
            table[1][0] = '<b>%s</b>'%self.name
            
            x_labels = table_labels(self.grid, 0)

            
            for i in range(self.shape[0]):
                table[0][i+1] = x_labels[i]
                table[1][i+1] = as_str(self.data[i])

            return tabulate.tabulate(table, tablefmt='html')
        
        else:
            return self.__repr__()
            
    
    def __str__(self):
        return '%s : %s'%(self.name, self.data)

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        return np.add(self, other)
    def __sub__(self, other):
        return np.subtract(self, other)
    def __mul__(self, other):
        return np.multiply(self, other)
    def __truediv__(self, other):
        return np.divide(self, other)
    def __pow__(self, other):
        return np.power(self, other)
    def __lt__(self, other):
        return np.less(self, other)
    def __le__(self, other):
        return np.less_equal(self, other)
    def __eq__(self, other):
        return np.equal(self, other)
    def __ne__(self, other):
        return np.not_equal(self, other)
    def __gt__(self, other):
        return np.greater(self, other)
    def __ge__(self, other): 
        return np.greater_equal(self, other)


    @property
    def ndim(self):
        return self.data.ndim

    @property
    def naxes(self):
        return self.grid.naxes

    def __getitem__(self, item):
        if isinstance(item, pn.GridArray):
            if item.data.dtype == np.bool:
                # in this case it is a mask
                # ToDo: masked operations behave strangely, operations are applyed to all elements, even if masked
                new_data = np.ma.MaskedArray(self.data, ~item.data)
                return pn.GridArray(self.grid, self.name, new_data)
            raise NotImplementedError('get item %s'%item)
        elif not isinstance(item, tuple):
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
                return self.data[item]
            return pn.GridArray(new_grid, self.name, self.data[item])

    def __setitem__(self, var, val):
        if isinstance(var, str):
            assert var not in self.grid.vars, "Cannot set grid dimension"
            assert val.shape == self.shape, "Incompatible dimensions %s and %s"%(self.shape, val.shape)
            self._data = np.asanyarray(val)
        else:
            self[var]._data[:] = val._data

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
        return pn.GridArray(self.grid.T, self.name, new_data)
    
    @property
    def shape(self):
        return self.data.shape
            
    @property
    def values(self):
        return self.get_array()

    @property
    def data(self):
        if self.name in self.grid.vars:
            return self.grid.point_mgrid[self.grid.vars.index(self.name)]
        else:
            return np.asanyarray(self._data)

    def get_array(self, flat=False):
        '''return array values

        Parameters
        ----------
        flat : bool
            flat-out the grid dimensions

        Retursn
        -------
        array
        '''
        array = self.data
        if flat:
            if array.ndim == self.grid.ndim:
                return array.ravel()
            return array.reshape(self.grid.size, -1)
        return array

    def __array__(self):
        return self.values

    def _get_axis(self, kwargs):
        '''translate axis to index'''
        axis = kwargs.get('axis', None)
        if isinstance(axis, str):
            axis = self.grid.vars.index(axis)
            kwargs['axis'] = axis
        return axis, kwargs

    def _pack_result(self, result, axis):
        '''repack result into correct type'''
        if isinstance(result, np.ndarray):
            if result.ndim > 0: 
                if self.name in self.grid.vars:
                    new_name = 'f(%s)'%self.name
                else:
                    new_name = self.name
                    
                # new grid
                if axis is not None:
                    new_grid = copy.deepcopy(self.grid)
                    del(new_grid.axes[axis])
                else:
                    new_grid = self.grid
                
                new_obj = pn.GridArray(new_grid, new_name, result)
                return new_obj
            if result.ndim == 0:
                return np.asscalar(result)
        return result


    def mean(self, **kwargs):
        '''necessary as it is not calling __array_ufunc__''' 
        axis, kwargs = self._get_axis(kwargs)
        result = np.mean(np.asanyarray(self), **kwargs)
        return self._pack_result(result, axis)
        
    def std(self, **kwargs):
        '''necessary as it is not calling __array_ufunc__''' 
        axis, kwargs = self._get_axis(kwargs)
        result = np.std(np.asanyarray(self), **kwargs)
        return self._pack_result(result, axis)
        
    def average(self, **kwargs):
        '''necessary as it is not calling __array_ufunc__''' 
        axis, kwargs = self._get_axis(kwargs)
        result = np.average(np.asanyarray(self), **kwargs)
        return self._pack_result(result, axis)
    
    def median(self, **kwargs):
        '''necessary as it is not calling __array_ufunc__''' 
        axis, kwargs = self._get_axis(kwargs)
        result = np.median(np.asanyarray(self), **kwargs)
        return self._pack_result(result, axis)
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        '''callable for numpy ufuncs'''
        array_inputs = [np.asanyarray(i) for i in inputs]

        axis, kwargs = self._get_axis(kwargs)
        # this is a bit complicated, but apprently necessary
        # otherwise masked element s of arrays get overwritten
        if any([isinstance(i, np.ma.masked_array) for i in array_inputs]):
            mask = np.full(self.shape, False)
            for i in array_inputs:
                if isinstance(i, np.ma.masked_array):
                    mask = mask | i.mask
            masked_inputs = [i[~mask] if i.shape == self.shape else i for i in array_inputs]
            mask_result = np.asanyarray(self).__array_ufunc__(ufunc, method, *masked_inputs, **kwargs)
            if mask_result.size == np.sum(~mask):
                result = np.empty_like(self.data)
                result[mask] = self.data[mask]
                result[~mask] = mask_result
            else:
                result = mask_result
        else:
            result = np.asanyarray(self).__array_ufunc__(ufunc, method, *array_inputs, **kwargs)

        return self._pack_result(result, axis)
        

    #def __array_prepare__(self, result, context=None):
    #    print('prepare')
    #    return result

    #def __array_finalize__(self, obj):
    #    print('finalize')

    #def __array_wrap__(self, out_arr, context=None):
    #    print('In __array_wrap__:')
    #    print('   self is %s' % repr(self))
    #    print('   arr is %s' % repr(out_arr))
    #    # then just call the parent
    #    return np.array(self).__array_wrap__(self, out_arr, context)




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
        if self.grid.naxes == 2:
            table_x = [0] * (self.grid.shape[0] + 1)
            table = [copy.copy(table_x) for _ in range(self.grid.shape[1] + 1)]
            
            table[0][0] = '<b>%s \\ %s</b>'%(self.grid.vars[1], self.grid.vars[0])
            
            x_labels = table_labels(self.grid, 0)
            y_labels = table_labels(self.grid, 1)
                        
            for i in range(self.shape[0]):
                table[0][i+1] = x_labels[i]
            for i in range(self.shape[1]):
                table[i+1][0] = y_labels[i]
                
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    all_data = []
                    #for var in self.data_vars:
                    for d in self:
                        all_data.append('%s = %s'%(d.name, as_str(d.data[i, j])))
                    table[j+1][i+1] = '<br>'.join(all_data)
                    
            return tabulate.tabulate(table, tablefmt='html')
        
        elif self.ndim == 1:
            table_x = [0] * (self.grid.shape[0] + 1)
            table = [copy.copy(table_x) for _ in range(len(self.data_vars)+1)]
            table[0][0] = '<b>%s</b>'%self.grid.vars[0]
            for i, var in enumerate(self.data_vars):
                table[i+1][0] = '<b>%s</b>'%var
            
            x_labels = table_labels(self.grid, 0)
            
            for i in range(self.shape[0]):
                table[0][i+1] = x_labels[i]
                for j, d in enumerate(self):
                    table[j+1][i+1] = as_str(d.data[i])

            return tabulate.tabulate(table, tablefmt='html')
        
        else:
            return self.__repr__()

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
        for var in inputs[0].data_vars:
            converted_inputs = [inp[var] for inp in inputs]
            result = converted_inputs[0].__array_ufunc__(ufunc, method, *converted_inputs, **kwargs)
            print('%s : %s'%(var, result))
        raise NotImplementedError()

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
            if self.grid.naxes == 0:
                self.grid == data.grid
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
                axes[d] = np.arange(n+1)
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
