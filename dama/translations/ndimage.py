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


class convolve(Translation):
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
        return ndimage.convolve(source_data, *self.args, **self.kwargs)

convolve.__init__.__doc__ = ndimage.convolve.__doc__



class correlate(Translation):
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
        return ndimage.correlate(source_data, *self.args, **self.kwargs)

correlate.__init__.__doc__ = ndimage.correlate.__doc__


class gaussian_laplace(Translation):
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
        return ndimage.gaussian_laplace(source_data, *self.args, **self.kwargs)

gaussian_laplace.__init__.__doc__ = ndimage.gaussian_laplace.__doc__



class generic_filter(Translation):
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
        return ndimage.generic_filter(source_data, *self.args, **self.kwargs)

generic_filter.__init__.__doc__ = ndimage.generic_filter.__doc__



class laplace(Translation):
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
        return ndimage.laplace(source_data, *self.args, **self.kwargs)

laplace.__init__.__doc__ = ndimage.laplace.__doc__



class maximum_filter(Translation):
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
        return ndimage.maximum_filter(source_data, *self.args, **self.kwargs)

maximum_filter.__init__.__doc__ = ndimage.maximum_filter.__doc__



class median_filter(Translation):
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
        return ndimage.median_filter(source_data, *self.args, **self.kwargs)

median_filter.__init__.__doc__ = ndimage.median_filter.__doc__



class minimum_filter(Translation):
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
        return ndimage.minimum_filter(source_data, *self.args, **self.kwargs)

minimum_filter.__init__.__doc__ = ndimage.minimum_filter.__doc__



class percentile_filter(Translation):
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
        return ndimage.percentile_filter(source_data, *self.args, **self.kwargs)

percentile_filter.__init__.__doc__ = ndimage.percentile_filter.__doc__



class prewitt(Translation):
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
        return ndimage.prewitt(source_data, *self.args, **self.kwargs)

prewitt.__init__.__doc__ = ndimage.prewitt.__doc__



class rank_filter(Translation):
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
        return ndimage.rank_filter(source_data, *self.args, **self.kwargs)

rank_filter.__init__.__doc__ = ndimage.rank_filter.__doc__



class sobel(Translation):
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
        return ndimage.sobel(source_data, *self.args, **self.kwargs)

sobel.__init__.__doc__ = ndimage.sobel.__doc__



class uniform_filter(Translation):
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
        return ndimage.uniform_filter(source_data, *self.args, **self.kwargs)

uniform_filter.__init__.__doc__ = ndimage.uniform_filter.__doc__


