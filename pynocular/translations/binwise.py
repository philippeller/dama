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

class Binwise(Translation):
    
    def __init__(self, source, *args, function=None, fill_value=np.nan, **kwargs):
        '''translation from array data into binned form

        Parameters:
        -----------
        function : callable
        fill_value : optional
            value for invalid points
        '''
        super().__init__(source,
                         *args,
                         dest_needs_grid=True,
                         **kwargs)

        self.function = function
        self.fill_value = fill_value


    def setup(self):
        self.prepare_source_sample()
        self.indices = self.dest.grid.compute_indices(self.source_sample)


    def fill_single_map(self, output_map, source_data, return_len):
        '''fill a single map with a function applied to values according to indices
        '''
        for i in range(np.prod([d+2 for d in self.dest.grid.shape])):
            bin_source_data = source_data[self.indices == i]
            if len(bin_source_data) > 0:
                result = self.function(bin_source_data) 
                out_idx = np.unravel_index(i, [d+2 for d in self.dest.grid.shape])
                out_idx = tuple([idx - 1 for idx in out_idx])
                output_map[out_idx] = result

    
    def eval(self, source_data):
        source_data = source_data.flat()

        # find out what does the function return:
        if source_data.ndim > 1:
            test_value = self.function(source_data[:3, [0]*(source_data.ndim-1)])
        else:
            test_value = self.function(source_data[:3])
        if np.isscalar(test_value):
            return_len = 1
        else:
            return_len = len(test_value)

        if source_data.ndim > 1:
            if return_len > 1:
                output_array = self.get_empty_output_array(source_data.shape[1:] + (return_len, ))
                for idx in np.ndindex(*source_data.shape[1:]):
                    self.fill_single_map(output_array[(Ellipsis,) + idx + (slice(None),)], source_data[(Ellipsis,) + idx], return_len)
            else:
                output_array = self.get_empty_output_array(source_data.shape[1:])
                for idx in np.ndindex(*source_data.shape[1:]):
                    self.fill_single_map(output_array[(Ellipsis,) + idx], source_data[(Ellipsis,) + idx], return_len)
        else:
            if return_len > 1:
                output_array = self.get_empty_output_array((return_len, ))
            else:
                output_array = self.get_empty_output_array()
            self.fill_single_map(output_array, source_data, return_len)

        output_array[np.isnan(output_array)] = self.fill_value

        return output_array
