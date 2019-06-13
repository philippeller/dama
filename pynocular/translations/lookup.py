from __future__ import absolute_import
'''Module providing a data translation methods'''
import numpy as np
import pynocular as pn
from pynocular.translations import Translation

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


class Lookup(Translation):

    def __init__(self, source, *args, **kwargs):
        '''lookup the bin content at given points

        Parameters:
        -----------

        ToDo: use ndimage.map_coordinates for regular grids
        '''
        super().__init__(source, *args,
                         source_needs_grid=True,
                         **kwargs)

    def setup(self):
        self.prepare_dest_sample()
        self.indices = self.source.grid.compute_indices(self.dest_sample)

    def eval(self, source_data):
        output_array = self.get_empty_output_array(source_data.shape[source_data.nax:])

        #TODO: make this better
        for i in range(len(output_array)):
            # check we're inside grid:
            ind = np.unravel_index(self.indices[i], [d+2 for d in self.source.grid.shape])
            ind = tuple([idx - 1 for idx in ind])
            inside = True
            for j in range(len(ind)):
                inside = inside and not ind[j] < 0 and not ind[j] >= self.source.grid.shape[j]
            if inside:
                idx = tuple(ind)
                output_array[i] = source_data[idx]
        return output_array


