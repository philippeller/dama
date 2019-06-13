from __future__ import absolute_import
'''Module providing a data translation methods'''
import numpy as np
from pynocular.translations import Translation
import numpy_indexed as npi

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
        function : callable or str
            if str, choice of ['count', 'sum', 'mean', 'min', 'max', 'std', 'var', 'argmin', 'argmax', 'median', 'mode', 'prod']
        fill_value : optional
            value for invalid points
        '''
        super().__init__(source,
                         *args,
                         dest_needs_grid=True,
                         **kwargs)

        assert function is not None, 'Need to specify function'
        self.function = function
        self.fill_value = fill_value


    def setup(self):
        self.prepare_source_sample()
        self.indices = self.dest.grid.compute_indices(self.source_sample)
        self.group = npi.group_by(self.indices)


    def fill_single_map(self, output_map, source_data, return_len):
        '''fill a single map with a function applied to values according to indices
        '''

        if self.function in ['count', 'sum', 'mean', 'min', 'max', 'std', 'var', 'argmin', 'argmax', 'median', 'mode', 'prod']:
            indices, out =  self.group.__getattribute__(self.function)(source_data)
            for idx, result in zip(indices, out):
                out_idx = np.unravel_index(idx, [d+2 for d in self.dest.grid.shape])
                out_idx = tuple([idx - 1 for idx in out_idx])
                output_map[out_idx] = result
        
        else:
            for idx, data in self.group.split_iterable_as_unordered_iterable(source_data):
                result = self.function(data) 
                out_idx = np.unravel_index(idx, [d+2 for d in self.dest.grid.shape])
                out_idx = tuple([idx - 1 for idx in out_idx])
                output_map[out_idx] = result
    
    def eval(self, source_data):
        source_data = source_data.flat()

        if isinstance(self.function, str):
            return_len = 1
        else:
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
