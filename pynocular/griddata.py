from __future__ import absolute_import
from collections import OrderedDict
from collections.abc import Iterable
import six
import numpy as np
import pynocular as pn
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


import copy

def table_labels(grid, dim):
    if grid[grid.vars[dim]]._edges is not None:
        return ['[%.2f, %.2f]'%(grid[grid.vars[dim]].edges[i], grid[grid.vars[dim]].edges[i+1]) for i in range(grid.shape[dim])]
    else:
        return ['%.2f, %.2f'%(grid[grid.vars[dim]].points[i]) for i in range(grid.shape[dim])]


class GridArray(object):
    '''Structure to hold a single GridData item
    '''
    def __init__(self, grid, name, input_array=None):
        self.grid = grid
        self.name = name
        
        if name in self.grid.vars:
            assert input_array is None, "Cannot add data with name same as grid variable"
        self._data = input_array

        
    def __repr__(self):
        return 'GridArray(%s : %s)'%(self.name, self.data)

    def _repr_html_(self):
        if self.ndim == 2:
            table_x = [0] * (self.shape[0] + 1)
            table = [copy.copy(table_x) for _ in range(self.shape[1] + 1)]
            
            table[0][0] = '%s \ %s'%(self.grid.vars[1], self.grid.vars[0])
            
            x_labels = table_labels(self.grid, 0)
            y_labels = table_labels(self.grid, 1)
                        
            for i in range(self.shape[0]):
                table[0][i+1] = x_labels[i]                    
            for i in range(self.shape[1]):
                table[i+1][0] = y_labels[i]
                
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    table[j+1][i+1] = '%.2f'%self.data[i,j]
                    
            return tabulate.tabulate(table, tablefmt='html')
        
        elif self.ndim == 1:
            table_x = [0] * (self.shape[0] + 1)
            table = [copy.copy(table_x) for _ in range(2)]
            table[0][0] = self.grid.vars[0]
            table[1][0] = self.name
            
            x_labels = table_labels(self.grid, 0)

            
            for i in range(self.shape[0]):
                table[0][i+1] = x_labels[i]
                table[1][i+1] = '%.2f'%self.data[i]

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

    def __getitem__(self, m):
        print('get item %s'%m)
        raise NotImplementedError()

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
            return np.array(self._data)

    def get_array(self, flat=False):
        array = self.data
        if flat:
            if array.ndim == self.grid.ndim:
                return array.ravel()
            return array.reshape(self.grid.size, -1)
        return array

    def __array__(self):
        #print('array')
        return self.values
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        #print('ufunc')
        array_inputs = [np.array(i) for i in inputs]
        #print(ufunc.nout)
        #print(kwargs)
        axis = kwargs.get('axis', None)
        #print(array_inputs)
        
        if isinstance(axis, str):
            kwargs['axis'] = self.grid.vars.index(axis)
        
        result = np.array(self).__array_ufunc__(ufunc, method, *array_inputs, **kwargs)
        
        if isinstance(result, np.ndarray):
            if result.ndim > 0 and result.shape[0] == len(self):  
                if self.name in self.grid.vars:
                    new_name = 'f(%s)'%self.name
                else:
                    new_name = self.name
                    
                # new grid
                if axis is not None:
                    new_grid = copy.deepcopy(self.grid)
                    if not isinstance(axis, str):
                        axis = list(grid.dims.keys())[axis]
                    new_grid.dims.pop(axis)
                else:
                    new_grid = self.grid
                
                new_obj = pn.GridArray(new_grid, new_name, result)
                return new_obj
            if result.ndim == 0:
                return np.asscalar(result)
        return result

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
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], pn.grid.Grid):
            self.grid = args[0]
        else:
            self.grid = pn.grid.Grid(*args, **kwargs)
        self.data = OrderedDict()

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
    def function_args(self):
        return self.grid.vars

    @property
    def vars(self):
        '''
        Available variables in this layer
        '''
        return self.grid.vars + list(self.data.keys())

    @property
    def data_vars(self):
        '''
        only data variables (no grid vars)
        '''
        return list(self.data.keys())

    def rename(self, old, new):
        self.data[new] = self.data.pop(old)

    def update(self, new_data):
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
        flat : bool
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
        return self.get_array(var, flat=True)

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
            return pn.stat_plot.plot_map(self, var, cbar=cbar, fig=fig, ax=ax, **kwargs)

        raise ValueError('Can only plot maps of 2d grids')

    def plot_contour(self, var=None, fig=None, ax=None, **kwargs):
        if var is None and len(self.data_vars) == 1:
            var = self.data_vars[0]
        return pn.stat_plot.plot_contour(self, var, fig=fig, ax=ax, **kwargs)

    def plot_step(self, var=None, fig=None, ax=None, **kwargs):
        if var is None and len(self.data_vars) == 1:
            var = self.data_vars[0]
        return pn.stat_plot.plot_step(self, var, fig=fig, ax=ax, **kwargs)

    def plot_bands(self, var=None, fig=None, ax=None, **kwargs):
        if var is None and len(self.data_vars) == 1:
            var = self.data_vars[0]
        return pn.stat_plot.plot_bands(self, var, fig=fig, ax=ax, **kwargs)

    def plot_errorband(self, var, errors, fig=None, ax=None, **kwargs):
        if var is None and len(self.data_vars) == 1:
            var = self.data_vars[0]
        return pn.stat_plot.plot_errorband(self, var, errors, fig=fig, ax=ax, **kwargs)
