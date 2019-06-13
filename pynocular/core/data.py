from __future__ import absolute_import
'''Module providing a data base class and translation methods'''
import numpy as np
import pynocular as pn
from pynocular import translations

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


class Data(object):
    '''
    Data base class
    '''

    def interp(self, *args, method=None, fill_value=np.nan, **kwargs):
        '''interpolation from any data fromat to another

        Parameters:
        -----------
        method : string
            "nearest" = nearest neightbour interpolation
            "linear" = linear interpolation
            "cubic" = cubic interpolation (only for ndim < 3)
        fill_value : optional
            value for invalid points
        '''
        translator = translations.Interpolation(self, *args, method=method, fill_value=fill_value, **kwargs)
        return translator.run()

    def histogram(self, *args, density=False, **kwargs):
        '''Method for histograms'''
        translator = translations.Histogram(self, *args, density=density, **kwargs)
        return translator.run()

    def binwise(self, *args, method=None, function=None, fill_value=np.nan, density=False, **kwargs):
        '''translation from array data into binned form

        Parameters:
        -----------
        function : callable or str
            if str, choice of ['count', 'sum', 'mean', 'min', 'max', 'std', 'var', 'argmin', 'argmax', 'median', 'mode', 'prod']
        fill_value : optional
            value for invalid points
        '''
        translator = translations.Binwise(self, *args, function=function, fill_value=fill_value, **kwargs)
        return translator.run()

    def lookup(self, *args, **kwargs):
        '''lookup the bin content at given points

        Parameters:
        -----------
        '''
        translator = translations.Lookup(self, *args, **kwargs)
        return translator.run()

    def kde(self, *args, bw='silverman', kernel='gaussian', density=True, **kwargs):
        '''run KDE on regular grid

        Parameters:
        -----------

        bw : str or float
            coices of 'silverman', 'scott', 'ISJ' for 1d data
            float specifies fixed bandwidth
        kernel : str
        density : bool (optional)
            if false, multiply output by sum of data
        '''
        translator = translations.KDE(self, *args, bw=bw, kernel=kernel, density=density, **kwargs)
        return translator.run()

    def resample(self, *args, **kwargs):
        '''resample from binned data into other binned data
        ToDo: this is super inefficient
        '''
        translator = translations.Resample(self, *args, **kwargs)
        return translator.run()
