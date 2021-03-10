from __future__ import absolute_import
'''Module providing a data translation methods'''
import numpy as np
from scipy import interpolate

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


class Interpolation(Translation):
    def __init__(
        self, source, *args, method=None, fill_value=np.nan, **kwargs
        ):
        '''interpolation from any data to another

        Parameters:
        -----------
        method : string
            "nearest" = nearest neightbour interpolation
            "linear" = linear interpolation
            "cubic" = cubic interpolation (only for ndim < 3)
        fill_value : Number (optional)
            value for invalid points
        '''
        super().__init__(source, *args, **kwargs)

        assert method in [
            None, 'nearest', 'linear', 'cubic'
            ], 'Illegal method %s' % method

        if method is None:
            if len(self.wrt) > 2:
                method = 'linear'
            else:
                method = 'cubic'
        self.method = method
        self.fill_value = fill_value

    def setup(self):
        self.prepare_source_sample()
        self.prepare_dest_sample(flat=False)

    def eval(self, source_data):

        if self.dest_has_grid:
            source_data = source_data.flat()
            mask = np.isfinite(source_data)
            if mask.ndim > 1:
                dim_mask = np.all(mask, axis=1)
            else:
                dim_mask = mask

            sample = self.source_sample[..., dim_mask]
            if source_data.ndim > 1:
                output_array = self.get_empty_output_array(
                    source_data.shape[1:]
                    )
                for idx in np.ndindex(*source_data.shape[1:]):
                    output_array[(Ellipsis, ) + idx] = interpolate.griddata(
                        points=sample.T,
                        values=source_data[(Ellipsis, ) + idx][dim_mask],
                        xi=self.dest_sample.T,
                        method=self.method,
                        fill_value=self.fill_value
                        ).T
            else:
                output_array = interpolate.griddata(
                    points=sample.T,
                    values=source_data[mask],
                    xi=self.dest_sample.T,
                    method=self.method,
                    fill_value=self.fill_value
                    ).T
                output_array = np.squeeze(output_array)
            return output_array

        else:
            if len(self.wrt) == 1:
                f = interpolate.interp1d(
                    self.source_sample[0],
                    source_data,
                    kind=self.method,
                    fill_value=self.fill_value,
                    bounds_error=False
                    )
                return f(self.dest_sample[0])

            elif len(self.wrt) == 2:
                f = interpolate.interp2d(
                    self.source_sample[0],
                    self.source_sample[1],
                    source_data,
                    kind=self.method,
                    fill_value=self.fill_value,
                    bounds_error=False
                    )
                return np.array(
                    [
                        f(x, y)[0] for x, y in
                        zip(self.dest_sample[0], self.dest_sample[1])
                        ]
                    )

            else:
                if self.method == 'nearest':
                    f = interpolate.NearestNDInterpolator(
                        self.source_sample.T, source_data
                        )
                else:
                    f = interpolate.LinearNDInterpolator(
                        self.source_sample.T,
                        source_data,
                        fill_value=self.fill_value
                        )
                return f(self.dest_sample.T)
