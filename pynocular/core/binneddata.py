from __future__ import absolute_import
from collections.abc import Iterable
import numpy as np
import numpy_indexed as npi
import pynocular as pn
from pynocular import translations
from pynocular.utils.formatter import format_table
from pynocular.utils.bind import bind
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

class BinnedData:
    '''
    Class to hold binned data
    '''

    # ToDo fill_value kwarg

    def __init__(self, grid=None, data=None, *args, **kwargs):
        '''
        grid : pn.Grid

        data : pn.PointData, pn.GridData or pn.GridArray
            -> if PointArray, sample is needed

        '''
        # set up grid
        if grid is None:
            if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], pn.Grid):
                self.grid = args[0]
            else:
                self.grid = pn.Grid(*args, **kwargs)


        #assert set(self.grid.vars) <= set(data.vars)

        if not self.grid.initialized:
            self.grid.initialize(data)

        # create sample (grid variables) and remove from data
        self.sample = [data.get_array(var, flat=True) for var in self.grid.vars]

        # The following is a bit wonky, ToDo
        if isinstance(data, pn.GridArray):
            self.single = True
            self.data = pn.PointData(test=data.flat())

        else:
            self.single = False
            data_vars = data.vars
            for var in self.grid.vars:
                data_vars.remove(var)

            if isinstance(data, pn.PointData):
                self.data = data[data_vars]

            elif isinstance(data, pn.GridData):
                self.data = pn.PointData()
                for var in data_vars:
                    self.data[var] = data.get_array(var)

            else:
                raise ValueError()

        self.indices = None
        self.group = None
 
    def compute_indices(self):
        self.indices = self.grid.compute_indices(self.sample)
        self.group = npi.group_by(self.indices)

    def __setitem__(self, var, val):
        self.add_data(var, val)

    def __getitem__(self, item):

        if isinstance(item, str):
            item = [item]
            
        item += self.grid.vars

        return pn.BinnedData(grid=self.grid, data=self.data[item])
        
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
        return self.grid.nax

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
        data : PointArray, PointData, Array
        '''
        if self.single:
            raise Exception('Cannot add data to a singular variable structure')

        if isinstance(data, pn.PointData):
            assert len(data.vars) == 1
            data = data[data.vars[0]]

        self.data[var] = data

    def run_np_indexed(self, method, fill_value=np.nan, **kwargs):
        '''run the numpy indexed methods
        Parameters:
        -----------

        method : str
            choice of ['sum', 'mean', 'min', 'max', 'std', 'var', 'argmin', 'argmax', 'median', 'mode', 'prod']
        '''
        axis = kwargs.pop('axis', None)
        assert axis is None, 'Axis kw not supported for BinnedData'

        if self.indices is None:
            self.compute_indices()
        outputs = {}
        output_maps = {}
        for var in self.data.vars:
            source_data = self.data[var]
            if source_data.ndim > 1: 
                output_maps[var] = np.full(self.grid.shape + source_data.shape[1:], fill_value=fill_value, dtype=self.data[var].dtype)
            else:
                output_maps[var] = np.full(self.grid.shape, fill_value=fill_value, dtype=self.data[var].dtype)
            source_data = self.data[var]
            indices, outputs[var] =  self.group.__getattribute__(method)(source_data)

        for i, idx in enumerate(indices):
            if idx < 0:
                continue
            out_idx = np.unravel_index(idx, self.grid.shape)
            for var in self.data.vars:
                output_maps[var][out_idx] = result = outputs[var][i]

        # Pack into GridArray
        if self.single:
            out_data = pn.GridArray(output_maps['test'], grid=self.grid)

        else:
        # Pack into GridData
            out_data = pn.GridData(self.grid)
            for var, output_map in output_maps.items():
                out_data[var] = output_map

        return out_data


    def apply_function(self, function, *args, fill_value=np.nan, return_len=None, **kwargs):
        '''apply function per bin'''

        if self.indices is None:
            self.compute_indices()

        outputs = {}
        output_maps = {}
        for var in self.data.vars:
            source_data = self.data[var]

            if return_len is None:
                # try to figure out return length of function

                if source_data.ndim > 1:
                    test_value = function(source_data[:3, [0]*(source_data.ndim-1)], *args, **kwargs)
                else:
                    test_value = function(source_data[:3], *args, **kwargs)
                if np.isscalar(test_value):
                    return_len = 1
                else:
                    return_len = len(test_value)

            if source_data.ndim > 1:
                if return_len > 1:
                    output_maps[var] = np.full(self.grid.shape + source_data.shape[1:] + (return_len, ), fill_value=fill_value, dtype=source_data.dtype)
                else:
                    output_maps[var] = np.full(self.grid.shape + source_data.shape[1:], fill_value=fill_value, dtype=source_data.dtype)
            else:
                if return_len > 1:
                    output_maps[var] = np.full(self.grid.shape + (return_len, ), fill_value=fill_value, dtype=source_data.dtype)
                else:
                    output_maps[var] = np.full(self.grid.shape, fill_value=fill_value, dtype=source_data.dtype)

        for i in range(self.grid.size):
            mask = self.indices == i
            if np.any(mask):
                out_idx = np.unravel_index(i, self.grid.shape)
                for var in self.data.vars:
                    source_data = self.data[var]
                    if source_data.ndim > 1:
                        for idx in np.ndindex(*source_data.shape[1:]):
                            output_maps[var][out_idx + (idx,)] = function(self.data[var][:, idx][mask], *args, **kwargs)
                    else:
                        output_maps[var][out_idx] = function(self.data[var][mask], *args, **kwargs)

        # Pack into GridArray
        if self.single:
            out_data = pn.GridArray(output_maps['test'], grid=self.grid)

        # Pack into GridData
        else:
            out_data = pn.GridData(self.grid)
            for var, output_map in output_maps.items():
                out_data[var] = output_map

        return out_data

    def sum(self, fill_value=np.nan, **kwargs):
        return self.run_np_indexed('sum', fill_value=fill_value, **kwargs)

    def mean(self, fill_value=np.nan, **kwargs):
        return self.run_np_indexed('mean', fill_value=fill_value, **kwargs)

    def min(self, fill_value=np.nan, **kwargs):
        return self.run_np_indexed('min', fill_value=fill_value, **kwargs)

    def max(self, fill_value=np.nan, **kwargs):
        return self.run_np_indexed('max', fill_value=fill_value, **kwargs)

    def std(self, fill_value=np.nan, **kwargs):
        return self.run_np_indexed('std', fill_value=fill_value, **kwargs)

    def var(self, fill_value=np.nan, **kwargs):
        return self.run_np_indexed('var', fill_value=fill_value, **kwargs)

    def argmin(self, fill_value=np.nan, **kwargs):
        return self.run_np_indexed('argmin', fill_value=fill_value, **kwargs)

    def argmax(self, fill_value=np.nan, **kwargs):
        return self.run_np_indexed('argmax', fill_value=fill_value, **kwargs)

    def median(self, fill_value=np.nan, **kwargs):
        return self.run_np_indexed('median', fill_value=fill_value, **kwargs)

    def mode(self, fill_value=np.nan, **kwargs):
        return self.run_np_indexed('mode', fill_value=fill_value, **kwargs)

    def prod(self, fill_value=np.nan, **kwargs):
        return self.run_np_indexed('prod', fill_value=fill_value, **kwargs)

    def quantile(self, q, fill_value=np.nan, **kwargs):
        if not isinstance(q, Iterable):
            return_len = 1
        else:
            return_len = len(q)
        return self.apply_function(np.quantile, q, return_len=return_len, fill_value=fill_value, **kwargs)

