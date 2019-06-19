from __future__ import absolute_import
'''Module providing a data translation methods'''
import numpy as np
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


class Resample(Translation):
    def __init__(self, source, *args, **kwargs):
        '''resample from binned data into other binned data
        ToDo: this is super inefficient
        '''
        super().__init__(source,
                         *args,
                         source_needs_grid=True,
                         dest_needs_grid=True,
                         **kwargs)

        assert self.dest.grid.vars == self.source.grid.vars, 'grid variables of source and destination must be identical'


    def setup(self):

        # we need a super sample of points, i.e. meshgrids of all combinations of source and dest
        # so first create for every dest.grid.var a vector of both, src and dest points
        lookup_sample = [np.concatenate([self.dest.grid[var].points, self.source.grid[var].points]) for var in self.wrt]
        mesh = np.meshgrid(*lookup_sample)
        self.lookup_sample = [m.flatten() for m in mesh]
        self.indices = self.source.grid.compute_indices(self.lookup_sample)
        

    def eval(self, source_data):

        lookup_array = np.full(self.lookup_sample[0].shape[0], np.nan)
        for i in range(len(lookup_array)):
            # check we're inside grid:
            idx = self.indices[i]
            if idx >= 0:
                ind = np.unravel_index(idx, self.source.grid.shape)
                lookup_array[i] = source_data[idx]

        # now bin both these points into destination
        bins = self.dest.grid.squeezed_edges
        lu_hist, _ = np.histogramdd(sample=self.lookup_sample, bins=bins, weights=lookup_array)
        lu_counts, _ = np.histogramdd(sample=self.lookup_sample, bins=bins)
        lu_hist /= lu_counts

        return lu_hist
