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
        if self.ndim == 2:
            table_x = [0] * (self.shape[0] + 1)
            table = [copy.copy(table_x) for _ in range(self.shape[1] + 1)]
            
            table[0][0] = '<b>%s \ %s</b>'%(self.grid.vars[1], self.grid.vars[0])
            
            x_labels = table_labels(self.grid, 0)
            y_labels = table_labels(self.grid, 1)
                        
            for i in range(self.shape[0]):
                table[0][i+1] = x_labels[i]                    
            for i in range(self.shape[1]):
                table[i+1][0] = y_labels[i]
                
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    table[j+1][i+1] = as_str(self.data[i,j])
                    
            return tabulate.tabulate(table, tablefmt='html')
        
        elif self.ndim == 1:
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

    def __getitem__(self, item):
        if isinstance(item, pn.GridArray):
            if item.data.dtype == np.bool:
                # in this case it is a mask
                # ToDo: masked operations behave strangely, operations are applyed to all elements, even if masked
                new_data  = np.ma.MaskedArray(self.data, ~item.data)
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
            return pn.GridArray(self.grid[item], self.name, self.data[item])

    @property
    def T(self):
        '''transpose'''
        if self.ndim > 1:
            return pn.GridArray(self.grid.T, self.name, self.data.T)
        return self
    
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
                    del(new_grid.dims[axis])
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
        self.data = OrderedDict()
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
        if self.grid.ndim == 2:
            table_x = [0] * (self.grid.shape[0] + 1)
            table = [copy.copy(table_x) for _ in range(self.grid.shape[1] + 1)]
            
            table[0][0] = '<b>%s \ %s</b>'%(self.grid.vars[1], self.grid.vars[0])
            
            x_labels = table_labels(self.grid, 0)
            y_labels = table_labels(self.grid, 1)
                        
            for i in range(self.shape[0]):
                table[0][i+1] = x_labels[i]                    
            for i in range(self.shape[1]):
                table[i+1][0] = y_labels[i]
                
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    all_data = []
                    for var in self.data_vars:
                        all_data.append('%s = %s'%(var, as_str(self.data[var][i,j])))
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
                for j, var in enumerate(self.data_vars):
                    table[j+1][i+1] = as_str(self.data[var][i])

            return tabulate.tabulate(table, tablefmt='html')
        
        else:
            return self.__repr__()

    def __setitem__(self, var, val):
        self.add_data(var, val)

    def __getitem__(self, var):
        if isinstance(var, str):
            if var in self.vars:
                if var in self.data_vars:
                    data = self.data[var]
                else:
                    data = None
                new_data = pn.GridArray(self.grid, var, data)
                return new_data

        # create new instance with mask or slice applied
        new_data = GridData(self.grid)
        if isinstance(var, Iterable):
            for v in var:
                if v in self.data_vars:
                    new_data[v] = self.data[v]
            return new_data
        raise NotImplementedError('slicing not yet implemented for Grids')

    @property
    def T(self):
        '''transpose'''
        raise NotImplementedError()

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
        return self.grid.vars + list(self.data.keys())

    @property
    def data_vars(self):
        '''
        only data variables (no grid vars)
        '''
        return list(self.data.keys())

    def rename(self, old, new):
        '''rename one array

        Parameters
        ----------
        old : str
        new : str
        '''
        self.data[new] = self.data.pop(old)

    def update(self, new_data):
        '''update

        Parameters
        ----------
        new_data : GridData
        '''
        if not self.grid.initialized:
            self.grid = new_data.grid
        assert self.grid == new_data.grid
        self.data.update(new_data.data)

    @property
    def shape(self):
        return self.grid.shape

    @property
    def ndim(self):
        return self.grid.ndim

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
            if self.grid.ndim == 0:
                self.grid == data.grid
            else:
                assert self.grid == data.grid
            data = data.data

        elif isinstance(data, pn.GridData):
            assert len(data.data_vars) == 1
            if self.grid.ndim == 0:
                self.grid == data.grid
            else:
                assert self.grid == data.grid
            data = data.get_array(data.data_vars[0])

        elif self.ndim == 0:
            print('adding default grid')
            # make a default grid:
            if data.ndim <= 3 and var not in ['x', 'y', 'z']:
                dim_names = ['x', 'y', 'z']
            else:
                dim_names = ['x%i' for i in range(data.ndim+1)]
                dim_names.delete(var)
            dims = OrderedDict()
            for d,n in zip(dim_names, data.shape):
                dims[d] = np.arange(n+1)
            self.grid = pn.Grid(**dims)

        if not data.shape[:self.ndim] == self.shape:
            raise ValueError('Incompatible data of shape %s for grid of shape %s'%(data.shape, self.shape))

        self.data[var] = data

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
            array = self.data[var]
        if flat:
            if array.ndim == self.grid.ndim:
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
        return iter([self[n] for n in self.data_vars])


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
        if self.grid.ndim == 2:
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
