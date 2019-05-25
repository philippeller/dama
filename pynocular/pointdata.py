from __future__ import absolute_import
import six
import numpy as np
import pandas
from collections import OrderedDict
from collections.abc import Iterable
from scipy.interpolate import griddata

import pynocular as pn
from pynocular.data import Data
from pynocular.stat_plot import *
from pynocular.utils.formatter import as_str, table_labels
import tabulate


#__all__ = ['PointData']

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


class PointDataDim(object):
    '''Structure to hold a single PointData item
    '''
    def __init__(self, *args, **kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            self.data = None
            self.name = None
        elif len(args) == 1 and len(kwargs) == 0:
            if isinstance(args[0], pandas.core.series.Series):
                self.data = args[0]
                self.name = args[0].name
            elif isinstance(args[0], OrderedDict) and len(args[0]) == 1:
                self.data = args[0].values()[0]
                self.name = args[0].keys()[0]
            else:
                raise ValueError()
        elif len(args) == 0 and len(kwargs) > 1:
            self.data = kwargs[0].values()[0]
            self.name = kwargs[0].keys()[0]
        else:
            raise ValueError("Did not understand input arguments")

    @property
    def type(self):
        if isinstance(self.data, pandas.core.series.Series):
            return 'df'
        elif isinstance(self.data, OrderedDict):
            return 'dict'

    def __repr__(self):
        return 'PointDataDim(%s : %s)'%(self.name, self.data)

    def _repr_html_(self):
        if self.type == 'df':
            return self.data._repr_html_()
        else:
            table = [[self.name] + [as_str(v) for v in np.array(self)]]
            return tabulate.tabulate(table, tablefmt='html')

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

    def __array__(self):
        return self.values

    @property
    def values(self):
        if self.type == 'df':
            return self.data.values
        return self.data

    def __array_wrap__(self, result, context=None):
        if isinstance(result, np.ndarray):
            if result.ndim > 0 and result.shape[0] == len(self):
                if self.type == 'df':
                    new_data = pandas.core.series.Series(result)
                    new_data.name = self.name
                    new_obj = pn.PointDataDim(new_data)
                else:
                    new_obj = pn.PointDataDim()
                    new_obj.data = result
                    new_obj.name = self.name
                return new_obj
            if result.ndim == 0:
                return np.asscalar(result)
        return result

class PointData(Data):
    '''
    Data Layer to hold point-type data structures (Pandas DataFrame, Dict, )
    '''
    def __init__(self, *args, **kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            self.data = OrderedDict()
        elif len(args) == 1 and len(kwargs) == 0:
            if isinstance(args[0], pandas.core.series.Series):
                self.data = pandas.DataFrame(args[0])
            elif isinstance(args[0], pandas.core.frame.DataFrame):
                self.data = args[0]
            elif isinstance(args[0], OrderedDict):
                self.data = args[0]
            else:
                self.data = OrderedDict(args[0])
        elif len(args) == 0 and len(kwargs) > 0:
            self.data = OrderedDict(kwargs)
        else:
            raise ValueError("Did not understand input arguments")

    def __repr__(self):
        return 'PointData(%s)'%self.data.__repr__()

    def __str__(self):
        return self.data.__str__()

    def _repr_html_(self):
        if self.type == 'df':
            return self.data._repr_html_()
        else:
            table = [[var] + [as_str(v) for v in np.array(self[var])] for var in self.vars]
            return tabulate.tabulate(table, tablefmt='html')


    @property
    def vars(self):
        '''
        Available variables in this layer
        '''
        if self.type == 'df':
            return list(self.data.columns)
        elif self.type == 'dict':
            return list(self.data.keys())
        else:
            return []

    @property
    def ndim(self):
        return len(self.vars)

    @property
    def type(self):
        if isinstance(self.data, pandas.core.frame.DataFrame):
            return 'df'
        elif isinstance(self.data, OrderedDict):
            return 'dict'

    @property
    def data_vars(self):
        return self.vars

    def rename(self, old, new):
        if self.type == 'df':
            self.data.rename(columns={old:new}, inplace=True)
        elif self.type == 'dict':
            self.data[new] = self.data.pop(old)

    def update(self, new_data):
        self.data.update(new_data.data)

    def __len__(self):
        if self.type == 'df':
            return len(self.data)
        elif self.type == 'dict':
            return len(self.data[list(self.data.keys())[0]])

    @property
    def array_shape(self):
        '''
        the shape of a single variable
        '''
        return (len(self),)

    def __setitem__(self, var, val):
        if self.ndim > 0:
            assert len(val) == len(self), 'Incompatible dimensions'
        if isinstance(val, pn.PointDataDim):
            self.data[var] = val.data
        elif isinstance(val, pn.PointData):
            # is this fair enough?
            self.data[var] = val.data[val.vars[-1]]
        else:
            self.data[var] = val

    def __getitem__(self, var):
        if self.type == 'df':
            result = self.data[var]
            if isinstance(result, pandas.core.frame.DataFrame):
                return PointData(result)
            elif isinstance(result, pandas.core.series.Series):
                return PointDataDim(result)

        if isinstance(var, str):
            if var in self.vars:
                new_data = PointDataDim()
                new_data.data = self.data[var]
                new_data.name = var
                return new_data

        # create new instance with mask or slice applied
        new_data = OrderedDict()

        if isinstance(var, Iterable):
            for v in var:
                new_data[v] = self.data[v]
        else:
            for key, val in self.data.items():
                new_data[key] = val[var]
        return PointData(new_data)
	
    def get_array(self, var, flat=False):
        return np.asarray(self[var])

    def plot_scatter(self, x, y, c=None, s=None, cbar=False, fig=None, ax=None, **kwargs):
        plot_points_2d(self, x, y, c=c, s=s, cbar=cbar, fig=fig, ax=ax, **kwargs)

    def plot(self, *args, **kwargs):
        if len(args) > 1:
            plot(self, args[0], args[1], *args[2:], **kwargs)
        elif self.ndim == 2:
            plot(self, *self.vars, *args, **kwargs)
        else:
            raise ValueError('Need to specify variables to plot')


