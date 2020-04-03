from __future__ import absolute_import
'''Module providing a data translation methods'''
from KDEpy import FFTKDE
from collections.abc import Iterable 
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


class KDE(Translation):

    def __init__(self, source, *args, bw=None, kernel='gaussian', density=True, **kwargs):
        '''run KDE on regular grid

        Parameters:
        -----------

        source : GridData or PointData
        bw : str or float or iterable
            Will default to 'silverman; for 1d data and 1 otherwise
            coices of 'silverman', 'scott', 'ISJ' for 1d data
            float specifies fixed bandwidth, in case of iterable a separate, fixed bandwith per dimension
        kernel : str
            choices of 'gaussian', 'exponential', 'box', 'tri', 'epa', 'biweight', 'triweight', 'tricube', 'cosine'
        density : bool (optional)
            if false, multiply output by sum of data
        '''
        super().__init__(source,
                         *args,
                         dest_needs_grid=True,
                         **kwargs)

        if bw is None:
            bw = 'silverman' if self.dest.grid.nax == 1 else 1.
        self.bw = bw
        self.kernel = kernel
        self.density = density

        if density:
            self.additional_runs = {'density' : None}
        else:
            self.additional_runs = {'counts' : None}

        if not self.dest.grid.regular:
            raise TypeError('dest must have regular grid')


    def setup(self):
        self.prepare_source_sample(stacked=False)
        # every point must be inside output grid (requirement of KDEpy)
        masks = [np.logical_and(self.source_sample[i] > dim.points[0], self.source_sample[i] < dim.points[-1]) for i, dim in enumerate(self.dest.grid)]
        self.mask = np.all(masks, axis=0)
        #n_masked = np.sum(~mask)
        #if n_masked > 0:
        #    warnings.warn('Excluding %i points that are outside grid'%n_masked, Warning, stacklevel=0)
        sample = [s[self.mask] for s in self.source_sample]


        self.source_sample = np.stack(sample).T
        self.prepare_dest_sample(transposed=True)

        if isinstance(self.bw, (np.ndarray, list, tuple)):
            for i in range(self.dest.grid.nax):
                self.source_sample[:,i] /= self.bw[i]
                self.dest_sample[:,i] /= self.bw[i]
            bw = 1
        else:
            bw = self.bw

        self.kde = FFTKDE(bw=bw, kernel=self.kernel)

    def eval(self, source_data):

        if not self.density:
            volume = np.prod([ax.edges.width[0] for ax in self.dest.grid])

        if source_data is None:
            out_array = self.kde.fit(self.source_sample).evaluate(self.dest_sample)
            out_shape = self.dest.shape
            if not self.density:
                out_array *= self.source_sample.size / np.sum(out_array)

        else:
            source_data = source_data.flat()

            if source_data.ndim > 1:
                out_array = self.get_empty_output_array(source_data.shape[1:], flat=True)
                for idx in np.ndindex(*source_data.shape[1:]):
                    out_array[(Ellipsis,) + idx] = self.kde.fit(self.source_sample, weights=source_data[(Ellipsis,) + idx][self.mask]).evaluate(self.dest_sample)
                    if not self.density:
                        out_array[(Ellipsis,) + idx] *= np.sum(source_data[(Ellipsis,) + idx][self.mask]) / np.sum(out_array[(Ellipsis,) + idx])
                out_shape = (self.dest.shape) + (-1,)

            else:
                out_array = self.kde.fit(self.source_sample, weights=source_data[self.mask]).evaluate(self.dest_sample)
                out_shape = self.dest.shape
                if not self.density:
                    out_array *= np.sum(source_data[self.mask]) / np.sum(out_array)

        #if isinstance(self.bw, (np.ndarray, list, tuple)):
        #    out_array *= np.product(self.bw)

        return out_array.reshape(out_shape)
