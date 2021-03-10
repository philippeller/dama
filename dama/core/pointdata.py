from __future__ import absolute_import
import numpy as np

try:
    import pandas
except ImportError:
    pandas = None

import dama as dm
from dama import translations
import dama.plotting
from dama.utils.formatter import format_table

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


class PointData(object):
    '''
    Data Layer to hold point-type data structures (Pandas DataFrame, Dict, )
    '''
    __slots__ = [
            '_data',
            ]

    def __init__(self, *args, **kwargs):
        self._data = {}

        if len(args) == 0 and len(kwargs) == 0:
            pass
        elif len(args) == 1 and len(kwargs) == 0:
            if pandas and isinstance(args[0], pandas.core.series.Series):
                self._data = pandas.DataFrame(args[0])
            elif pandas and isinstance(args[0], pandas.core.frame.DataFrame):
                self._data = args[0]
            elif isinstance(args[0], dict):
                kwargs = args[0]
            else:
                raise ValueError("Did not understand input arguments")
        if len(kwargs) > 0:
            for k, v in kwargs.items():
                self[k] = v
        #else:
        #    raise ValueError("Did not understand input arguments")

    def __repr__(self):
        return format_table(self, tablefmt='plain')

    def __str__(self):
        return format_table(self, tablefmt='plain')

    def _repr_html_(self):
        '''for jupyter'''
        if self.type == 'df':
            return self._data._repr_html_()
        return format_table(self, tablefmt='html')

    @property
    def data(self):
        return self._data

    @property
    def vars(self):
        '''
        Available variables
        '''
        if self.type == 'df':
            return list(self._data.columns)
        elif self.type == 'simple':
            return list(self._data.keys())
        return []

    def __dir__(self):
        """for tab completion"""
        return self.vars + object.__dir__(self)

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
        if pandas is not None:
            if isinstance(self._data, pandas.core.frame.DataFrame):
                return 'df'
        return 'simple'

    def __len__(self):
        return len(self._data)

    @property
    def size(self):
        return self.array_shape[0]

    @property
    def array_shape(self):
        '''the shape (first dimesnion only) of a single variable

        Returns
        -------
        shape : tuple
        '''
        if self.type == 'df':
            return (len(self._data), )
        elif self.type == 'simple':
            return (len(self[self.vars[0]]), )

    def __setitem__(self, var, val):
        if callable(val):
            self._data[var] = val
            return
        val = np.asanyarray(val)
        if val.ndim == 0:
            val = val[np.newaxis]

        if len(self) > 0:
            assert len(val) == self.array_shape[0], 'Incompatible dimensions'

        if isinstance(val, dm.PointArray):
            self._data[var] = val
        elif isinstance(val, np.ndarray):
            if self.type == 'df':
                self._data[var] = val
            else:
                val = dm.PointArray(val)
                self[var] = val
        else:
            raise ValueError()

    def __getitem__(self, var):
        if isinstance(var, str):
            if var in self.vars:
                data = self._data[var]
            else:
                raise IndexError('No variable %s in DataSet' % var)
            if callable(data):
                self[var] = data()
                data = self._data[var]
            if self.type == 'df':
                if isinstance(data, pandas.core.frame.DataFrame):
                    return dm.PointData(data)
                elif isinstance(data, pandas.core.series.Series):
                    return dm.PointArray(data)
            return data

        # create new instance with mask or slice applied
        new_data = {}

        if isinstance(var, (tuple,
                            list)) and all([isinstance(v, str) for v in var]):
            for v in var:
                new_data[v] = self[v]
        else:
            for n, d in self.items():
                new_data[n] = d[var]
        return dm.PointData(new_data)

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

    def get_array(self, var, flat=False):
        return np.asarray(self[var])

    def __iter__(self):
        '''
        iterate over dimensions
        '''
        if self.type == 'df':
            return iter([self._data[var] for var in self.vars])
        return iter(self._data)

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

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

    def kde(self, *args, **kwargs):
        return translations.KDE(self, *args, **kwargs).run()

    kde.__doc__ = translations.KDE.__init__.__doc__

    # --- Plotting functions ---

    plot = dm.plotting.plot

    def plot_2d(self, *args, labels=None, **kwargs):
        if len(args) == 2:
            dm.plotting.plot(self, *args, labels=labels, **kwargs)
        elif len(self) == 2:
            dm.plotting.plot(self, *self.vars, *args, **kwargs)
        else:
            raise ValueError('Need to specify 2 variables to plot')

    def plot_scatter(
        self, x, y, c=None, s=None, cbar=False, fig=None, ax=None, **kwargs
        ):
        dm.plotting.plot_points_2d(
            self, x, y, c=c, s=s, cbar=cbar, fig=fig, ax=ax, **kwargs
            )
