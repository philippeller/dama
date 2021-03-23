from __future__ import absolute_import
'''Module providing a data translation methods'''
import numpy as np
from scipy import ndimage

from dama.translations import Translation

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


class gaussian_filter(Translation):
    def __init__(
        self, source, *args, **kwargs
        ):

        self.args = args
        self.kwargs = kwargs

        if not source.grid.regular:
            print("Warning: this method assumes regular grids!")

        super().__init__(source,
                         source,
                         source_needs_grid=True,
                         dest_needs_grid=True,
                         )

    def setup(self):
        self.prepare_source_sample()
        self.prepare_dest_sample(flat=False)

    def eval(self, source_data):
        return ndimage.gaussian_filter(source_data, *self.args, **self.kwargs)

gaussian_filter.__init__.__doc__ = ndimage.gaussian_filter.__doc__
