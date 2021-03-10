from __future__ import absolute_import
import numpy as np
import dama as dm
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


class PointArray(np.ndarray):
    '''Structure to hold a single PointData item
    '''
    def __new__(cls, input_array, *args, **kwargs):
        obj = np.asarray(input_array).view(cls)
        return obj

    def _repr_html_(self):
        '''for jupyter'''
        return format_table(self, tablefmt='html')

    def __repr__(self):
        return format_table(self, tablefmt='plain')

    def __str__(self):
        return format_table(self, tablefmt='plain')

    def __array_wrap__(self, obj):
        if obj.shape == ():
            return obj[()]  # if ufunc output is scalar, return it
        else:
            return np.ndarray.__array_wrap__(self, obj)

    def flat(self):
        return self

    def plot(self, *args, **kwargs):
        return dm.plotting.plot1d_all(self, *args, **kwargs)
