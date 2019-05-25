from __future__ import absolute_import
from numbers import Number
import numpy as np
from scipy import interpolate
from KDEpy import FFTKDE
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


def initialize_grid(grid, source):
    '''Method to initialize a grid if grid is not fully set up
    it derive information from source
    
    Parameters
    ----------
    grid : pn.Grid
    source : pn.GridData, pn.PointData

    Returns
    -------
    grid : pn.Grid
    '''
    # check dest grid is set up, otherwise do so
    for var in grid.vars:
        if grid[var].edges is None:
            # check if it might be from a grid
            if isinstance(source, pn.GridData):
                if var in source.grid.vars:
                    if isinstance(grid[var].nbins, float):
                        # this measn we want to multiply the old nbins
                        new_nbins = int(source.grid[var].nbins * grid[var].nbins)
                    else:
                        new_nbins = grid[var].nbins
                    grid[var].edges = np.linspace(source.grid[var].edges[0], source.grid[var].edges[-1], new_nbins+1)
                    continue
            # in this case it's pointdata
            print(var)
            grid[var].edges = np.linspace(np.nanmin(source[var]), np.nanmax(source[var]), grid[var].nbins+1)
    return grid

def generate_destination(source, *args, **kwargs):
    '''Correctly set up a destination data format
    depending on the supplied input
    
    Parameters
    ----------
    source : pn.GridData, pn.PointData
    args, kwargs

    Returns
    -------
    destination : pn.GridData, pn.PointData
    '''
    if len(args) == 1 and len(kwargs) == 0:
        dest = args[0]
        if isinstance(dest, pn.GridData):
            grid = initialize_grid(dest.grid, source)
            return pn.GridData(grid)
        if isinstance(dest, pn.grid.Grid):
            grid = initialize_grid(dest, source)
            return pn.GridData(grid)
        if isinstance(dest, pn.PointData):
            return pn.PointData(dest.data)

    # check if source has a grid and if any args are in there
    if isinstance(source, pn.GridData):
        dims = []
        for arg in args:
            # in thsio case the source may have a grid, get those edges
            if isinstance(arg, str):
                if arg in source.grid.vars:
                    dims.append(source.grid[arg])
                    continue
            dims.append(arg)
        args = dims

    # instantiate
    grid = pn.grid.Grid(*args, **kwargs)
    grid = initialize_grid(grid, source)

    return pn.GridData(grid)



def interp(source, *args, **kwargs):
    '''interpolation from array data into grids

    Parameters:
    -----------
    method : string
        "nearest" = nearest neightbour interpolation
        "linear" = linear interpolation
        "cubic" = cubic interpolation (only for ndim < 3)
    fill_value : Number (optional)
        value for invalid points
    '''
    method = kwargs.pop('method', None)

    assert method in [None, 'nearest', 'linear', 'cubic'], 'Illegal method %s'%method

    fill_value = kwargs.pop('fill_value', np.nan)

    dest = generate_destination(source, *args, **kwargs)

    if isinstance(dest, pn.PointData):
        xi_vars = dest.vars

        if not set(xi_vars) <= set(source.vars):
            raise TypeError('the following variable are not present in the source: %s'%', '.join(set(xi_vars) - (set(xi_vars) & set(source.vars))))

        if method is None:
            if len(xi_vars) > 2:
                method = 'linear'
            else:
                method = 'cubic'

        if method == 'cubic' and len(xi_vars) > 2:
            raise NotImplementedError('cubic interpolation only supported for 1 or 2 dimensions')

        for source_var in source.vars:
            if source_var in xi_vars:
                continue

            if len(xi_vars) == 1:
                f = interpolate.interp1d(source[xi_vars[0]], source[source_var], kind=method, fill_value=fill_value, bounds_error=False)
                output_array = f(dest[xi_vars[0]])

            elif len(xi_vars) == 2:
                f = interpolate.interp2d(source[xi_vars[0]], source[xi_vars[1]], source[source_var], kind=method, fill_value=fill_value, bounds_error=False)
                output_array = np.array([f(x, y)[0] for x, y in zip(np.array(dest[xi_vars[0]]), np.array(dest[xi_vars[1]]))])

            else:
                sample = [source.get_array(var, flat=True) for var in xi_vars]
                sample = np.vstack(sample).T
                if method == 'nearest':
                    f = interpolate.NearestNDInterpolator(sample, source[source_var])
                else:
                    f = interpolate.LinearNDInterpolator(sample, source[source_var], fill_value=fill_value)
                out_sample = [dest.get_array(var, flat=True) for var in xi_vars]
                out_sample = np.vstack(out_sample).T
                output_array = f(out_sample)

            dest[source_var] = output_array

        return dest
    
    if method is None:
        if dest.grid.ndim > 2:
            method = 'linear'
        else:
            method = 'cubic'

    if method == 'cubic' and dest.grid.ndim > 2:
        raise NotImplementedError('cubic interpolation only supported for 1 or 2 dimensions')
    

    # check source has grid variables
    for var in dest.grid.vars:
        assert(var in source.vars), '%s not in %s'%(var, source.vars)

    
    xi = dest.grid.point_mgrid

    for source_var in source.vars:
        if source_var in dest.grid.vars:
            continue

        source_data = source.get_array(source_var, flat=True)

        mask = np.isfinite(source_data)
        if mask.ndim > 1:
            dim_mask = np.all(mask, axis=1)
        else:
            dim_mask = mask

        # prepare arrays
        sample = [source.get_array(var, flat=True)[dim_mask] for var in dest.grid.vars]
        sample = np.vstack(sample)

        if source_data.ndim > 1:
            dest_map = np.full(shape=(*dest.grid.shape, *source_data.shape[1:]), fill_value=np.nan)
            for idx in np.ndindex(*source_data.shape[1:]):
                dest_map[(Ellipsis,) + idx] = interpolate.griddata(points=sample.T, values=source_data[(Ellipsis,) + idx][dim_mask], xi=tuple(xi), method=method, fill_value=fill_value)
        else:
            dest_map = interpolate.griddata(points=sample.T, values=source_data[mask], xi=tuple(xi), method=method, fill_value=fill_value)

        dest[source_var] = dest_map

    return dest

def binwise(source, *args, **kwargs):
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
    method = kwargs.pop('method', None)
    function = kwargs.pop('function', None)
    fill_value = kwargs.pop('fill_value', np.nan)

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
                    output_map[(Ellipsis,) + idx] = get_single_hist(sample=sample, grid=dest.grid, weights=source_data[(Ellipsis,) + idx], method=method)
            else:
                output_map = get_single_hist(sample=sample, grid=dest.grid, weights=source_data, method=method)
            
            output_map[np.isnan(output_map)] = fill_value
            dest[source_var] = output_map

        if method == 'sum':
            # add counts
            dest['counts'] = get_single_hist(sample=sample, grid=dest.grid, weights=None, method=method)

        return dest

    # ------- function ---------

    elif function is not None:
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



def lookup(source, *args, **kwargs):
    '''lookup the bin content at given points

    Parameters:
    -----------

    source_var : string
    '''
    if not hasattr(source, 'grid'):
        raise TypeError('source must have a grid defined')

    dest = generate_destination(source, *args, **kwargs)

    # check dest has grid variables
    for var in source.grid.vars:
        assert(var in dest.vars), '%s not in %s'%(var, dest.vars)

    # prepare arrays
    sample = [dest.get_array(var, flat=True) for var in source.grid.vars]

    indices = source.grid.compute_indices(sample)

    for source_var in source.vars:
        if source_var in dest.vars:
            continue

        source_data = source.get_array(source_var)

        if source_data.ndim > source.grid.ndim:
            output_array = np.full((np.product(dest.array_shape),)+source_data.shape[source.grid.ndim:], np.nan)
        else:
            output_array = np.full(np.product(dest.array_shape), np.nan)


        #TODO: make this better
        for i in range(len(output_array)):
            # check we're inside grid:
            ind = np.unravel_index(indices[i], [d+2 for d in source.grid.shape])
            ind = tuple([idx - 1 for idx in ind])
            inside = True
            for j in range(len(ind)):
                inside = inside and not ind[j] < 0 and not ind[j] >= source.grid.shape[j]
            if inside:
                idx = tuple(ind)
                output_array[i] = source_data[idx]

        dest[source_var] = output_array
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
        bins = dest.grid.edges
        lu_hist, _ = np.histogramdd(sample=lookup_sample, bins=bins, weights=lookup_array)
        lu_counts, _ = np.histogramdd(sample=lookup_sample, bins=bins)
        lu_hist /= lu_counts

        dest[source_var] = lu_hist

    return dest






class Data(object):
    '''
    Data base class to hold any form of data representation
    '''

    # ToDo: make this wrapping of functions into a decorator

    def interp(self, *args, **kwargs):
        '''interpolation from array data into grids

        Parameters:
        -----------
        method : string
            "nearest" = nearest neightbour interpolation
            "linear" = linear interpolation
            "cubic" = cubic interpolation (only for ndim < 3)
        fill_value : optional
            value for invalid points
        '''
        return interp(self, *args, **kwargs)

    def histogram(self, *args, **kwargs):
        '''Convenience method for histograms'''
        return binwise(self, *args, method='sum', **kwargs)

    def binwise(self, *args, **kwargs):
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
        return binwise(self, *args, **kwargs)

    def lookup(self, *args, **kwargs):
        '''lookup the bin content at given points

        Parameters:
        -----------

        source_var : string
        '''
        return lookup(self, *args, **kwargs)

    def resample(self, *args, **kwargs):
        '''resample from binned data into other binned data
        ToDo: this is super inefficient
        '''
        return resample(self, *args, **kwargs)




def get_single_hist(sample, grid, weights, method):
    '''Generate a single histogram
    '''

    # generate hists
    if method in ['sum', 'mean']:
        weighted_hist, _ = np.histogramdd(sample=sample, bins=grid.edges, weights=weights)

    if method in ['count', 'mean']:
        hist, _ = np.histogramdd(sample=sample, bins=grid.edges)

    # make outputs
    if method == 'count':
        return hist
    if method == 'mean':
        mask = (hist > 0.)
        weighted_hist[mask] /= hist[mask]
    return weighted_hist


def fill_single_map(output_map, grid, indices, source_data, function, return_len):
    '''fill a single map with a function applied to values according to indices
    '''
    for i in range(np.prod([d+2 for d in grid.shape])):
        bin_source_data = source_data[indices == i]
        if len(bin_source_data) > 0:
            result = function(bin_source_data) 
            out_idx = np.unravel_index(i, [d+2 for d in grid.shape])
            out_idx = tuple([idx - 1 for idx in out_idx])
            output_map[out_idx] = result
