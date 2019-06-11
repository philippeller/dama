from __future__ import absolute_import
'''Module providing a data base class for translation methods'''
from numbers import Number
import numpy as np
import pynocular as pn
from scipy import interpolate

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


class Translator():
    '''Base class for translation methods'''

    def __init__(self,
                 source, 
                 *args,
                 source_needs_grid=False,
                 dest_needs_grid=False,
                 **kwargs):

        self.source = source
        self.source_has_grid = isinstance(self.source, (pn.GridData, pn.GridArray, pn.Grid))
        if source_needs_grid:
            assert self.source_has_grid, 'Source must provide grid'
        self.dest = self.generate_destination(*args, **kwargs)
        self.dest_has_grid = isinstance(self.dest, (pn.GridData, pn.GridArray, pn.Grid))
        if dest_needs_grid:
            assert self.dest_has_grid, 'Destination must provide grid'

        if self.dest_has_grid:
            self.wrt = self.dest.grid.vars
        elif self.source_has_grid:
            self.wrt = self.source.grid.vars
        else:
            self.wrt = self.dest.vars

        # checks
        if not set(self.wrt) <= set(source.vars):
            raise TypeError('the following variables are missing in the source: %s'%', '.join(set(self.vars) - (set(self.vars) & set(source.vars))))

        self.source_sample = None
        self.dest_sample = None

    def generate_destination(self, *args, **kwargs):
        '''Correctly set up a destination data format
        depending on the supplied input
        
        Parameters
        ----------
        args, kwargs

        '''
        if len(args) == 1 and len(kwargs) == 0:
            dest = args[0]
            if isinstance(dest, pn.GridData):
                grid = dest.grid
                grid.initialize(self.source)
                return pn.GridData(grid)
            if isinstance(dest, pn.Grid):
                grid = dest
                grid.initialize(self.source)
                return pn.GridData(grid)
            if isinstance(dest, pn.PointData):
                return dest

        # check if source has a grid and if any args are in there
        if isinstance(self.source, pn.GridData):
            dims = []
            for arg in args:
                # in thsio case the source may have a grid, get those edges
                if isinstance(arg, str):
                    if arg in self.source.grid.vars:
                        dims.append(self.source.grid[arg])
                        continue
                dims.append(arg)
            args = dims

        # instantiate
        grid = pn.grid.Grid(*args, **kwargs)
        grid.initialize(self.source)

        return pn.GridData(grid)

    def prepare_source_sample(self, flat=True, array=True):
        self.source_sample = [self.source.get_array(var, flat=flat) for var in self.wrt]
        if array:
            self.source_sample = np.stack(self.source_sample)

    def prepare_dest_sample(self, flat=True, array=True):
        self.dest_sample = [self.dest.get_array(var, flat=flat) for var in self.wrt]
        if array:
            self.dest_sample = np.stack(self.dest_sample)

    def setup(self):
        pass

    def run(self):

        self.setup()

        for var in self.source.vars:
            if var in self.wrt:
                continue
            source_data = self.source[var]
            result = self.eval(source_data)
            self.dest[var] = result
        return self.dest

    def eval(self, data):
        raise NotImplementedError()

    def get_empty_output_array(self, element_shape=tuple(), fill_value=np.nan):
        '''make empty array in shape of destinaion
    
        element_shape : tuple
            additional dimensions of the output
        fill_value : value
            fill value
        '''
        if fill_value is None:
            return np.empty(self.dest.array_shape + element_shape)
        return np.full(self.dest.array_shape + element_shape, fill_value)


class Lookup(Translator):

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


class Interpolator(Translator):
    def __init__(self, source, *args, method=None, fill_value=np.nan, **kwargs):
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

        assert method in [None, 'nearest', 'linear', 'cubic'], 'Illegal method %s'%method

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

            sample = self.source_sample[:,dim_mask] 
            if source_data.ndim > 1:
                output_array = self.get_empty_output_array(source_data.shape[1:])
                for idx in np.ndindex(*source_data.shape[1:]):
                    output_array[(Ellipsis,) + idx] = interpolate.griddata(points=sample.T, values=source_data[(Ellipsis,) + idx][dim_mask], xi=self.dest_sample.T, method=self.method, fill_value=self.fill_value).T
            else:
                output_array = interpolate.griddata(points=sample.T, values=source_data[mask], xi=self.dest_sample.T, method=self.method, fill_value=self.fill_value).T

            return output_array

        else:
            if len(self.wrt) == 1:
                f = interpolate.interp1d(self.source_sample[0], source_data, kind=self.method, fill_value=self.fill_value, bounds_error=False)
                return f(self.dest_sample[0])

            elif len(self.wrt) == 2:
                f = interpolate.interp2d(self.source_sample[0], self.source_sample[1], source_data, kind=self.method, fill_value=self.fill_value, bounds_error=False)
                return np.array([f(x, y)[0] for x, y in zip(self.dest_sample[0], self.dest_sample[1])])

            else:
                if method == 'nearest':
                    f = interpolate.NearestNDInterpolator(self.source_sample.T, source_data)
                else:
                    f = interpolate.LinearNDInterpolator(self.source_sample.T, source_data, fill_value=self.fill_value)
                return f(self.dest_sample.T)


"""

def binwise(source, *args, method=None, function=None, fill_value=np.nan, density=False, **kwargs):
    '''translation from array data into binned form

    Parameters:
    -----------
    method : string
        "sum" = weighted historgam
        "mean" = weighted histogram / histogram
    function : callable
    fill_value : optional
        value for invalid points
    '''
    if method is None and function is None:
        method = "sum"

    dest = generate_destination(source, *args, **kwargs)

    # check source has grid variables
    for var in dest.grid.vars:
        assert(var in source.vars), '%s not in %s'%(var, source.vars)

    # prepare arrays
    sample = [source.get_array(var, flat=True) for var in dest.grid.vars]

    if method is not None:
        for source_var in source.vars:
            if source_var in dest.grid.vars:
                continue

            source_data = source.get_array(source_var, flat=True)

            if source_data.ndim > 1:
                output_map = np.empty(shape=(*dest.grid.shape, *source_data.shape[1:]))
                for idx in np.ndindex(*source_data.shape[1:]):
                    output_map[(Ellipsis,) + idx] = get_single_hist(sample=sample, grid=dest.grid, weights=source_data[(Ellipsis,) + idx], method=method, density=density)
            else:
                output_map = get_single_hist(sample=sample, grid=dest.grid, weights=source_data, method=method, density=density)
            
            output_map[np.isnan(output_map)] = fill_value
            dest[source_var] = output_map

        if method == 'sum':
            # add counts
            output_map = get_single_hist(sample=sample, grid=dest.grid, weights=None, method=method, density=density)
            if density:
                dest['density'] = output_map
            else:
                dest['counts'] = output_map

        return dest

    # ------- function ---------

    elif function is not None:

        assert not density, 'Density is not available together with functions'

        indices = dest.grid.compute_indices(sample)
        
        # ToDo: compute all vars in one loop
        for source_var in source.vars:
            if source_var in dest.grid.vars:
                continue

            source_data = source.get_array(source_var, flat=True)

            # find out what does the function return:
            if source_data.ndim > 1:
                test_value = function(source_data[:3, [0]*(source_data.ndim-1)])
            else:
                test_value = function(source_data[:3])
            if isinstance(test_value, Number):
                return_len = 1
            else:
                return_len = len(test_value)

            if source_data.ndim > 1:
                if return_len > 1:
                    output_map = np.full(shape=(*dest.grid.shape, *source_data.shape[1:], return_len), fill_value=np.nan)
                    for idx in np.ndindex(*source_data.shape[1:]):
                        fill_single_map(output_map[(Ellipsis,) + idx + (slice(None),)], dest.grid, indices, source_data[(Ellipsis,) + idx], function, return_len)
                else:
                    output_map = np.full(shape=(*dest.grid.shape, *source_data.shape[1:]), fill_value=np.nan)
                    for idx in np.ndindex(*source_data.shape[1:]):
                        fill_single_map(output_map[(Ellipsis,) + idx], dest.grid, indices, source_data[(Ellipsis,) + idx], function, return_len)

            else:
                if return_len > 1:
                    output_map = np.full(dest.grid.shape + (return_len,), fill_value=np.nan)
                else:
                    output_map = np.full(dest.grid.shape, fill_value=np.nan)
                fill_single_map(output_map, dest.grid, indices, source_data, function, return_len)

            output_map[np.isnan(output_map)] = fill_value
            dest[source_var] = output_map

        return dest

    else:
        raise ValueError('need at least a method or a function specified')




def kde(source, *args, bw='silverman', kernel='gaussian', density=True, **kwargs):
    '''run KDE on regular grid

    Parameters:
    -----------

    source : GridData or PointData
    bw : str or float
        coices of 'silverman', 'scott', 'ISJ' for 1d data
        float specifies fixed bandwidth
    kernel : str
    density : bool (optional)
        if false, multiply output by sum of data
    '''
    dest = generate_destination(source, *args, **kwargs)

    if not isinstance(dest, pn.GridData):
        raise TypeError('dest must have GridData')

    if not dest.grid.regular:
        raise TypeError('dest must have regular grid')

    # check source has grid variables
    for var in dest.grid.vars:
        assert(var in source.vars), '%s not in %s'%(var, source.vars)

    # prepare arrays
    sample = [source.get_array(var, flat=True) for var in dest.grid.vars]

    # every point must be inside output grid (requirement of KDEpy)
    masks = [np.logical_and(sample[i] > dim.points[0], sample[i] < dim.points[-1]) for i, dim in enumerate(dest.grid)]
    mask = np.all(masks, axis=0)
    #n_masked = np.sum(~mask)
    #if n_masked > 0:
    #    warnings.warn('Excluding %i points that are outside grid'%n_masked, Warning, stacklevel=0)

    sample = [s[mask] for s in sample]

    sample = np.stack(sample).T
    #print(sample.shape)

    kde = FFTKDE(bw=bw, kernel=kernel)
    
    eval_grid = np.stack([dest.flat(var) for var in dest.grid.vars]).T

    for source_var in source.vars:
        if source_var in dest.vars:
            continue

        source_data = source.get_array(source_var, flat=True)

        if source_data.ndim > 1:
            out = np.empty(shape=(dest.grid.size, *source_data.shape[1:]))
            for idx in np.ndindex(*source_data.shape[1:]):
                out[(Ellipsis,) + idx] = kde.fit(sample, weights=source_data[(Ellipsis,) + idx][mask]).evaluate(eval_grid)
                if not density:
                    out[(Ellipsis,) + idx] *= np.sum(source_data[(Ellipsis,) + idx][mask])
            out_shape = (dest.shape) + (-1,)
        else:
            out = kde.fit(sample, weights=source_data[mask]).evaluate(eval_grid)
            out_shape = dest.shape
            if not density:
                out *= np.sum(source_data[mask])
        dest[source_var] = out.reshape(out_shape)
    out = kde.fit(sample).evaluate(eval_grid)
    if density:
        dest['density'] = out.reshape(dest.shape)
    else:
        dest['counts'] = out.reshape(dest.shape) * np.sum(mask)

    return dest


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

"""
