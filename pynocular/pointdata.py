from __future__ import absolute_import
import numpy as np
import pandas
import pynocular as pn
import pynocular.plotting
from pynocular.data import Data
from pynocular.utils.formatter import format_html

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


class PointData(Data):
    '''
    Data Layer to hold point-type data structures (Pandas DataFrame, Dict, )
    '''
    def __init__(self, *args, **kwargs):
        self.data = {}

        if len(args) == 0 and len(kwargs) == 0:
            pass
        elif len(args) == 1 and len(kwargs) == 0:
            if isinstance(args[0], pandas.core.series.Series):
                self.data = pandas.DataFrame(args[0])
            elif isinstance(args[0], pandas.core.frame.DataFrame):
                self.data = args[0]
            elif isinstance(args[0], dict):
                args = args[0]
            else:
                raise ValueError("Did not understand input arguments")
        if len(kwargs) > 0:
            for k, v in kwargs.items():
                self[k] = v
        #else:
        #    raise ValueError("Did not understand input arguments")

    def __repr__(self):
        return 'PointData(%s)'%self.data.__repr__()

    def __str__(self):
        return self.data.__str__()

    def _repr_html_(self):
        '''for jupyter'''
        if self.type == 'df':
            return self.data._repr_html_()
        else:
            return format_html(self)


    @property
    def vars(self):
        '''
        Available variables
        '''
        if self.type == 'df':
            return list(self.data.columns)
        elif self.type == 'simple':
            return list(self.data.keys())
        else:
            return []

    @property
    def data_vars(self):
        '''Available variables'''
        return self.vars

    @property
    def type(self):
        '''type of stored data

        Returns
        -------
        type : str
            if data is a pandas.DataFrame: "df"
            else: "simple"
        '''
        if isinstance(self.data, pandas.core.frame.DataFrame):
            return 'df'
        return 'simple'

    def __len__(self):
        return len(self.data)

    @property
    def array_shape(self):
        '''the shape (first dimesnion only) of a single variable

        Returns
        -------
        shape : tuple
        '''
        if self.type == 'df':
            return (len(self.data),)
        elif self.type == 'simple':
            return (len(self[self.vars[0]]),)

    def __setitem__(self, var, val):
        if len(self) > 0:
            assert len(val) == self.array_shape[0], 'Incompatible dimensions'

        if isinstance(val, pn.PointArray):
            self.data[var] = val
        #elif isinstance(val, pn.PointData):
        #    # ToDo: is this fair enough?
        #    self.data[var] = val.data[val.vars[-1]]
        elif isinstance(val, np.ndarray):
            if self.type == 'df':
                self.data[var] = val
            else:
                val = pn.PointArray(val)
                self[var] = val
        else:
            raise ValueError()

    def __getitem__(self, var):
        #print(var)
        if self.type == 'df':
            result = self.data[var]
            if isinstance(result, pandas.core.frame.DataFrame):
                return PointData(result)
            elif isinstance(result, pandas.core.series.Series):
                return PointArray(result)

        if isinstance(var, str):
            if var in self.vars:
                return self.data[var]
            else:
                raise IndexError('No variable %s in DataSet'%var)

        # create new instance with mask or slice applied
        new_data = {}

        if isinstance(var, (tuple, list)) and all([isinstance(v, str) for v in var]):
            for v in var:
                new_data[v] = self[v]
        else:
            for n,d in self.items():
                new_data[n] = d[var]
        return PointData(new_data)
        
    def get_array(self, var, flat=False):
        return np.asarray(self[var])

    def __iter__(self):
        '''
        iterate over dimensions
        '''
        if self.type == 'df':
            return iter([self.data[var] for var in self.vars])
        return iter(self.data)

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()


    # --- Plotting functions ---

    def plot_scatter(self, x, y, c=None, s=None, cbar=False, fig=None, ax=None, **kwargs):
        pn.plotting.plot_points_2d(self, x, y, c=c, s=s, cbar=cbar, fig=fig, ax=ax, **kwargs)

    def plot(self, *args, **kwargs):
        if len(args) > 1:
            pn.plotting.plot(self, args[0], args[1], *args[2:], **kwargs)
        elif len(self) == 2:
            pn.plotting.plot(self, *self.vars, *args, **kwargs)
        else:
            raise ValueError('Need to specify variables to plot')


