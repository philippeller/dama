from __future__ import absolute_import
from collections import OrderedDict
from collections.abc import Iterable
import copy
import numpy as np
import pynocular as pn
from pynocular.utils.formatter import format_html
import pynocular.plotting
import tabulate

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


class GridData(pn.data.Data):
    '''
    Class to hold grid data
    '''
    def __init__(self, *args, **kwargs):
        '''
        Set the grid
        '''
        self.data = {}

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
                    data = self.data[item]
                else:
                    raise NotImplementedError()
                new_data = pn.GridArray(data, grid=self.grid)
                return new_data

        # mask
        if isinstance(item, pn.GridArray):
            if item.dtype == np.bool:
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
        return list(self.data.keys())

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

        if isinstance(data, (pn.GridArray, GridData)):
            if not self.grid.initialized:
                self.grid = data.grid
            else:
                assert self.grid == data.grid

        if isinstance(data, pn.GridData):
            assert len(data.data_vars) == 1
            data = data[0]

        if self.ndim == 0:
            print('adding default grid')
            # make a default grid:
            if data.ndim <= 3 and var not in ['x', 'y', 'z']:
                axes_names = ['x', 'y', 'z']
            else:
                axes_names = ['x%i' for i in range(data.ndim+1)]
                axes_names.delete(var)
            axes = {}
            for d, n in zip(axes_names, data.shape):
                axes[d] = np.arange(n)
            self.grid = pn.Grid(**axes)

        if data.ndim < self.ndim and self.shape[-1] == 1:
            # add new axis
            data = data[..., np.newaxis]

        data = np.ma.asarray(data)

        if not data.shape[:self.ndim] == self.shape:
            raise ValueError('Incompatible data of shape %s for grid of shape %s'%(data.shape, self.shape))

        self.data[var] = data

    def get_array(self, var, flat=False):
        '''
        return bare array of data

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
        return iter(self.values())

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()



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
