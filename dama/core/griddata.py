from __future__ import absolute_import
from collections import OrderedDict
from collections.abc import Iterable
import numpy as np
import dama as dm
from dama import translations
from dama.utils.formatter import format_table
from dama.utils.bind import bind
import dama.plotting

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


class GridData:
    '''
    Class to hold grid data
    '''
    __slots__ = [
            '_data',
            '_grid',
            'plot',
            ]

    def __init__(self, *args, **kwargs):
        '''
        Set the grid
        '''
        self._data = {}

        self._grid = dm.Grid()

        if len(args) == 0 and len(kwargs) > 0 and all(
            [isinstance(v, dm.GridArray) for v in kwargs.values()]
            ):
            for n, d in kwargs.items():
                self.add_data(n, d)
        elif len(args) == 1 and len(kwargs
                                    ) == 0 and isinstance(args[0], dm.Grid):
            self._grid = args[0]
        else:
            self._grid = dm.Grid(*args, **kwargs)

        self._setup_plotting_methods()


    def _setup_plotting_methods(self):
        '''dynamically set up plotting methods,
        depending on the number of axes'''

        if self._grid.nax == 1:
            #self.plot = 
            bind(self, dm.plotting.plot_step, 'plot')
        if self._grid.nax == 2:
            #self.plot =
            bind(self, dm.plotting.plot_map, 'plot')

    def __repr__(self):
        return format_table(self, tablefmt='plain')

    def __str__(self):
        return format_table(self, tablefmt='plain')

    def _repr_html_(self):
        '''for jupyter'''
        return format_table(self, tablefmt='html')

    def __setitem__(self, var, val):
        self.add_data(var, val)

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.vars:
                if item in self.data_vars:
                    data = self._data[item]
                    if callable(data):
                        self[item] = data()
                        data = self._data[item]
                else:
                    data = self.get_array(item)
                new_data = dm.GridArray(data, grid=self._grid)
                return new_data
            else:
                raise IndexError('No variable %s in DataSet' % item)

        # mask
        if isinstance(item, dm.GridArray):
            if item.dtype == np.bool:
                # in this case it is a mask
                # ToDo: masked operations behave strangely, operations are applyed to all elements, even if masked
                new_data = dm.GridData(self._grid)
                for v in self.data_vars:
                    new_data[v] = self[v][item]
                return new_data
            raise NotImplementedError('get item %s' % item)

        # create new instance with only those vars
        if isinstance(item,
                      Iterable) and all([isinstance(v, str) for v in item]):
            new_data = dm.GridData(self._grid)
            for v in item:
                if v in self.data_vars:
                    new_data[v] = self[v]
            return new_data

        # slice
        new_grid = self._grid[item]
        if len(new_grid) == 0:
            return {n: d[item] for n, d in self.items()}
        new_data = dm.GridData(new_grid)
        for n, d in self.items():
            new_data[n] = d[item]
        return new_data

    def __getattr__(self, item):
        try:
            return self[item]
        except Exception as e:
            raise AttributeError from e

    def __setattr__(self, item, value):
        if item in self.__slots__:
            object.__setattr__(self, item, value)
        else:
            self[item] = value

    @property
    def T(self):
        '''transpose'''
        new_obj = dm.GridData()
        new_obj._grid = self._grid.T
        for n, d in self.items():
            new_obj[n] = d.T
        return new_obj
        #raise NotImplementedError()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        '''interface to numpy unversal functions'''
        scalar_results = OrderedDict()
        array_result = dm.GridData()
        for var in inputs[0].data_vars:
            converted_inputs = [inp[var] for inp in inputs]
            result = converted_inputs[0].__array_ufunc__(
                ufunc, method, *converted_inputs, **kwargs
                )
            if isinstance(result, dm.GridArray):
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
        return self._grid.vars + self.data_vars

    def __dir__(self):
        """for tab completion"""
        return self.vars + object.__dir__(self)

    @property
    def data_vars(self):
        '''
        only data variables (no grid vars)
        '''
        return list(self._data.keys())

    @property
    def shape(self):
        return self._grid.shape

    @property
    def ndim(self):
        return self._grid.nax

    @property
    def array_shape(self):
        '''
        shape of a single variable
        '''
        return self.shape

    @property
    def grid(self):
        return self._grid

    @property
    def data(self):
        return self._data

    def add_data(self, var, data):
        '''Add data

        Parameters
        ----------
        var : str
            name of data
        data : GridArray, GridData, Array
        '''
        if callable(data):
            self._data[var] = data
            return

        if isinstance(data, (dm.GridArray, GridData)):
            if self._grid is None or not self._grid.initialized:
                self._grid = data.grid
                self._setup_plotting_methods()
            else:
                assert self._grid == data.grid

        if var in self._grid.vars:
            raise ValueError(
                'Variable `%s` is already a grid dimension!' % var
                )

        if isinstance(data, dm.GridData):
            assert len(data.data_vars) == 1
            data = data[0]

        if self.ndim == 0:
            print('adding default grid')
            # make a default grid:
            if data.ndim <= 3 and var not in ['x', 'y', 'z']:
                axes_names = ['x', 'y', 'z']
            else:
                axes_names = ['x%i' for i in range(data.ndim + 1)]
                axes_names.delete(var)
            axes = {}
            for d, n in zip(axes_names, data.shape):
                axes[d] = np.arange(n)
            self._grid = dm.Grid(**axes)

        if data.ndim < self.ndim and self.shape[-1] == 1:
            # add new axis
            data = data[..., np.newaxis]

        data = np.ma.asarray(data)

        if not data.shape[:self.ndim] == self.shape:
            raise ValueError(
                'Incompatible data of shape %s for grid of shape %s' %
                (data.shape, self.shape)
                )

        self._data[var] = data

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
        if var in self._grid.vars:
            array = self._grid.point_meshgrid[self._grid.vars.index(var)]
        else:
            array = self._data[var]
        if flat:
            if array.ndim == self._grid.nax:
                return array.ravel()
            return array.reshape(self._grid.size, -1)

        return array

    def flat(self, var):
        '''return flatened-out array'''
        return self.get_array(var, flat=True)

    def __iter__(self):
        '''
        iterate over dimensions
        '''
        return iter([self[v] for v in self.data_vars])

    def items(self):
        return [
            (v, self[v]) for v in self.data_vars
            ]

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

    def kde(self, *args, **kwargs):
        return translations.KDE(self, *args, **kwargs).run()

    kde.__doc__ = translations.KDE.__init__.__doc__

    def resample(self, *args, **kwargs):
        return translations.Resample(self, *args, **kwargs).run()

    resample.__doc__ = translations.Resample.__init__.__doc__

    # --- Plotting methods ---

    #def plot(self, var=None, **kwargs):
    #    if var is None and len(self.data_vars) == 1:
    #        var = self.data_vars[0]
    #    if self.ndim == 1:
    #        return self.plot_step(var, **kwargs)
    #    elif self.ndim == 2:
    #        return self.plot_map(var, **kwargs)

    def plot_map(self, var=None, cbar=False, fig=None, ax=None, **kwargs):
        '''
        plot a variable as a map

        ax : pyplot axes object
        var : str
        '''
        if var is None:
            if len(self.data_vars) == 1:
                var = self.data_vars[0]
            else:
                raise ValueError('Need to specify variable to plot')
        if self._grid.nax == 2:
            return dm.plotting.plot_map(
                self[var], label=var, cbar=cbar, fig=fig, ax=ax, **kwargs
                )

        raise ValueError('Can only plot maps of 2d grids')

    def plot_contour(self, var=None, fig=None, ax=None, **kwargs):
        if var is None and len(self.data_vars) == 1:
            var = self.data_vars[0]
        return dm.plotting.plot_contour(self, var, fig=fig, ax=ax, **kwargs)

    def plot_step(self, var=None, fig=None, ax=None, **kwargs):
        if var is None and len(self.data_vars) == 1:
            var = self.data_vars[0]
        return dm.plotting.plot_step(
            self[var], label=var, fig=fig, ax=ax, **kwargs
            )

    plot_bands = dm.plotting.plot_bands

    def plot_errorband(self, var, errors, fig=None, ax=None, **kwargs):
        if var is None and len(self.data_vars) == 1:
            var = self.data_vars[0]
        return dm.plotting.plot_errorband(
            self, var, errors, fig=fig, ax=ax, **kwargs
            )
