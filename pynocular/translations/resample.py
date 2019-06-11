from __future__ import absolute_import
'''Module providing a data base class for translation methods'''
from numbers import Number
import numpy as np
import pynocular as pn

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


def resample(source, *args, **kwargs):
    '''resample from binned data into other binned data
    ToDo: this is super inefficient
    '''

    if not hasattr(source, 'grid'):
        raise TypeError('source must have a grid defined')

    dest = generate_destination(source, *args, **kwargs)

    assert dest.grid.vars == source.grid.vars, 'grid variables of source and destination must be identical'

    # we need a super sample of points, i.e. meshgrids of all combinations of source and dest
    # so first create for every dest.grid.var a vector of both, src and dest points
    lookup_sample = [np.concatenate([dest.grid[var].points, source.grid[var].points]) for var in dest.grid.vars]
    mesh = np.meshgrid(*lookup_sample)
    lookup_sample = [m.flatten() for m in mesh]
    
    for source_var in source.vars:
        if source_var in dest.grid.vars:
            continue

        # lookup values
        source_data = source.get_array(source_var)
        indices = source.grid.compute_indices(lookup_sample)
        lookup_array = np.full(lookup_sample[0].shape[0], np.nan)
        for i in range(len(lookup_array)):
            # check we're inside grid:
            ind = np.unravel_index(indices[i], [d+2 for d in source.grid.shape])
            ind = tuple([idx - 1 for idx in ind])
            inside = True
            for j in range(len(ind)):
                inside = inside and not ind[j] < 0 and not ind[j] >= source.grid.shape[j]
            if inside:
                idx = tuple(ind)
                lookup_array[i] = source_data[idx]

        # now bin both these points into destination
        bins = dest.grid.squeezed_edges
        lu_hist, _ = np.histogramdd(sample=lookup_sample, bins=bins, weights=lookup_array)
        lu_counts, _ = np.histogramdd(sample=lookup_sample, bins=bins)
        lu_hist /= lu_counts

        dest[source_var] = lu_hist

    return dest
