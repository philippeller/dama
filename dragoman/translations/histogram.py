from __future__ import absolute_import
'''Module providing a data base class for translation methods'''

import numpy as np

from dragoman.translations import Translation

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


class Histogram(Translation):

    def __init__(self, source, *args, density=False, **kwargs):
        '''translation into historgram form

        Parameters:
        -----------
        density : bool (optional)
            if True return histograms as densities
        '''
        super().__init__(source,
                         *args,
                         dest_needs_grid=True,
                         **kwargs)

        self.density = density

        if density:
            self.additional_runs = {'density' : None}
        else:
            self.additional_runs = {'counts' : None}

    def setup(self):
        self.prepare_source_sample(stacked=False)


    def eval(self, source_data):

        if source_data is None:
            output_array, _ = np.histogramdd(sample=self.source_sample, bins=self.dest.grid.squeezed_edges, density=self.density)
         
        else:
            source_data = source_data.flat()

            if source_data.ndim > 1:
                output_array = self.get_empty_output_array(source_data.shape[1:])
                for idx in np.ndindex(*source_data.shape[1:]):
                    output_array[(Ellipsis,) + idx], _ = np.histogramdd(sample=self.source_sample, bins=self.dest.grid.squeezed_edges, weights=source_data[(Ellipsis,) + idx], density=self.density)
            else:
                output_array, _ = np.histogramdd(sample=self.source_sample, bins=self.dest.grid.squeezed_edges, weights=source_data, density=self.density)
            
        return output_array
