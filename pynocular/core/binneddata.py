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

def wrap(func):
    def wrapped_func(self, *args, **kwargs):
        axis = kwargs.pop('axis')
        assert axis is None, 'Axis kw not supported for BinnedData'
        #print(args, kwargs)
        if self.indices is None:
            self.compute_indices()
        outputs = {}
        output_maps = {}
        for var in self.data.vars:
            if var in self.grid.vars:
                continue
            output_maps[var] = np.full(self.grid.shape, fill_value=np.nan)
            source_data = self.data[var]
            indices, outputs[var] =  func(self, source_data)

        for i, idx in enumerate(indices):
            if idx < 0:
                continue
            out_idx = np.unravel_index(idx, self.grid.shape)
            for var in self.data.vars:
                if var in self.grid.vars:
                    continue
                output_maps[var][out_idx] = result = outputs[var][i]

        # Pack into GridData
        out_data = pn.GridData(self.grid)
        for var, output_map in output_maps.items():
            out_data[var] = output_map

        return out_data
    return wrapped_func

class BinnedData:
    '''
    Class to hold binned data
    '''
    def __init__(self, data=None, *args, **kwargs):
        '''
        Set the grid
        '''
        if data is None:
            self.data = pn.PointData()
        else:
            self.data = data
        self.indices = None
        self.group = None
        self.sample = None

        # ToDo protect self.grid as private self._grid
        self.grid = None

        if len(args) == 0 and len(kwargs) > 0 and all([isinstance(v, pn.GridArray) for v in kwargs.values()]):
            for n,d in kwargs.items():
                self.add_data(n, d)
        elif len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], pn.Grid):
            self.grid = args[0]
        else:
            self.grid = pn.Grid(*args, **kwargs)

    def compute_indices(self):
        self.sample = [self.data.get_array(var, flat=True) for var in self.grid.vars]
        self.indices = self.grid.compute_indices(self.sample)
        self.group = npi.group_by(self.indices)

    def __setitem__(self, var, val):
        self.add_data(var, val)

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.vars:
                if item in self.data_vars:
                    data = self.data[item]
                else:
                    data = self.get_array(item)
                new_data = pn.GridArray(data, grid=self.grid)
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
            return {n : d[item] for n,d in self.items()}
        new_data = pn.GridData(new_grid)
        for n,d in self.items():
            new_data[n] = d[item]
        return new_data
        
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
        if isinstance(data, pn.PointData):
            assert len(data.vars) == 1
            data = data[data.vars[0]]

        self.data[var] = data

    @wrap
    def sum(self, source_data):
        return self.group.sum(source_data)
    @wrap
    def mean(self, source_data):
        return self.group.mean(source_data)
    @wrap
    def min(self, source_data):
        return self.group.min(source_data)
    @wrap
    def max(self, source_data):
        return self.group.max(source_data)
    @wrap
    def std(self, source_data):
        return self.group.std(source_data)
    @wrap
    def var(self, source_data):
        return self.group.var(source_data)
    @wrap
    def argmin(self, source_data):
        return self.group.argmin(source_data)
    @wrap
    def argmax(self, source_data):
        return self.group.argmax(source_data)
    @wrap
    def median(self, source_data):
        return self.group.median(source_data)
    @wrap
    def mode(self, source_data):
        return self.group.mode(source_data)
    @wrap
    def prod(self, source_data):
        return self.group.prod(source_data)
