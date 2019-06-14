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


class Data:
    '''Data base class
    '''

    def interp(self, *args, method=None, fill_value=np.nan, **kwargs):
        return translations.Interpolation(self, *args, method=method, fill_value=fill_value, **kwargs).run()
    interp.__doc__ = translations.Interpolation.__init__.__doc__

    def histogram(self, *args, density=False, **kwargs):
        return translations.Histogram(self, *args, density=density, **kwargs).run()
    histogram.__doc__ = translations.Histogram.__init__.__doc__

    def binwise(self, *args, method=None, function=None, fill_value=np.nan, density=False, **kwargs):
        return translations.Binwise(self, *args, function=function, fill_value=fill_value, **kwargs).run()
    binwise.__doc__ = translations.Binwise.__init__.__doc__

    def lookup(self, *args, **kwargs):
        return translations.Lookup(self, *args, **kwargs).run()
    lookup.__doc__ = translations.Lookup.__init__.__doc__

    def kde(self, *args, bw='silverman', kernel='gaussian', density=True, **kwargs):
        return translations.KDE(self, *args, bw=bw, kernel=kernel, density=density, **kwargs).run()
    kde.__doc__ = translations.KDE.__init__.__doc__

    def resample(self, *args, **kwargs):
        return translations.Resample(self, *args, **kwargs).run()
    resample.__doc__ = translations.Resample.__init__.__doc__
