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
    '''
    if grid is not fully set up, derive grid from source
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
            grid[var].edges = np.linspace(np.nanmin(source[var]), np.nanmax(source[var]), grid[var].nbins+1)
    return grid

def generate_destination(source, *args, **kwargs):
    '''
    Return correctly set up grid, depending on the supplied input
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
            # ToDo: only wrt variables
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

class Data(object):
    '''
    Data base class to hold any form of data representation
    '''
    def __init__(self, data):
        self.data = None
        self.set_data(data)

    def set_data(self, data):
        pass

    def add_data(self, var, data):
        pass

    def get_array(self, var):
        pass

    def __getitem__(self, var):
        return self.get_array(var)

    def __setitem__(self, var, data):
        if callable(data):
            new_data = data(self)
        else:
            new_data = data
        if isinstance(new_data, type(self)):
            # rename to desired var name
            # last variable added is the new one
            new_data.rename(new_data.data_vars[-1], var)
            self.update(new_data)
            return

        self.add_data(var, new_data)

    def __len__(self):
        return 0

    def __repr__(self):
        return 'Data(%s)'%self.data.__repr__()

    def __str__(self):
        return self.data.__str__()



    # ToDo: make this wrapping of functions into a decorator

    def interpolate(self, source_var=None, method=None, wrt=None, fill_value=np.nan):
        '''
        interpolation from array data into grids

        Parameters:
        -----------
        source_var : string
            input variable
        method : string
            "nearest" = nearest neightbour interpolation
            "linear" = linear interpolation
            "cubic" = cubic interpolation (only for ndim < 3)
        wrt : tuple
            specifying the variable with respect to which the interpolation is done
            None for griddata (will be wrt the r=destination grid)
        fill_value : optional
            value for invalid points
        '''
        source = self
        if isinstance(wrt, str):
            wrt = [wrt]

        def fun(*args, **kwargs):
            output = generate_destination(source, *args, **kwargs)

            if isinstance(output, pn.PointData):
                dest = args[0]
                if wrt is None:
                    # need to reassign variable because of scope
                    this_wrt = list(set(source.vars) & set(dest.vars) - set(source_var))
                    print('Automatic interpolation with respect to %s'%', '.join(this_wrt))
                else:
                    this_wrt = wrt

                if not set(this_wrt) <= set(dest.vars):
                    raise TypeError('the following variable are not present in the destination: %s'%', '.join(set(this_wrt) - (set(this_wrt) & set(dest.vars))))

                if len(this_wrt) == 1:
                    f = interpolate.interp1d(source[this_wrt[0]], source[source_var], kind=method, fill_value=fill_value, bounds_error=False)
                    output_array = f(dest[this_wrt[0]])

                elif len(this_wrt) == 2 and method in ['linear', 'cubic']:
                    f = interpolate.interp2d(source[this_wrt[0]], source[this_wrt[1]], source[source_var], kind=method, fill_value=fill_value, bounds_error=False)
                    output_array = np.array([f(x, y)[0] for x, y in zip(dest[this_wrt[0]], dest[this_wrt[1]])])

                elif method in ['nearest', 'linear']:
                    sample = [source.get_array(var, flat=True) for var in this_wrt]
                    sample = np.vstack(sample).T
                    if method == 'nearest':
                        f = interpolate.NearestNDInterpolator(sample, source[source_var])
                    else:
                        f = interpolate.LinearNDInterpolator(sample, source[source_var], fill_value=fill_value)
                    out_sample = [dest.get_array(var, flat=True) for var in this_wrt]
                    out_sample = np.vstack(out_sample).T
                    output_array = f(out_sample)

                else:
                    raise NotImplementedError('method %s not available for %i dimensional interpolation'%(method, len(this_wrt)))

                output[source_var] = output_array

                return output

            if wrt is not None:
                raise TypeError('wrt cannot be defined for a grid destination')

            if method == 'cubic' and output.grid.ndim > 2:
                raise NotImplementedError('cubic interpolation only supported for 1 or 2 dimensions')
            source_data = source.get_array(source_var, flat=True)

            mask = np.isfinite(source_data)

            # check source has grid variables
            for var in output.grid.vars:
                assert(var in source.vars), '%s not in %s'%(var, source.vars)

            # prepare arrays
            sample = [source.get_array(var, flat=True)[mask] for var in output.grid.vars]
            sample = np.vstack(sample)
            
            xi = output.grid.point_mgrid

            output_map = interpolate.griddata(points=sample.T, values=source_data[mask], xi=tuple(xi), method=method, fill_value=fill_value)

            output[source_var] = output_map

            return output

        return fun


    def histogram(self, source_var=None):
        if source_var is None:
            method = "count"
        else:
            method= "sum"

        return self.binned(source_var=source_var, method=method)

    def binned(self, source_var=None, method=None, function=None, fill_value=np.nan, **kwargs):
        '''
        translation from array data into binned form

        Parameters:
        -----------

        source_var : string
            input variable
        method : string
            "sum" = weighted historgam
            "mean" = weighted histogram / histogram
            "count" = histogram
        function : callable
        '''
        source = self

        if method is None and function is None and source_var is None:
            method = "count"

        if source_var is None:
            assert method == 'count'

        if method is None and function is None:
            method = "sum"

        def fun(*args, **kwargs):
            output = generate_destination(source, *args, **kwargs)

            if source_var is None:
                source_data = None
            else:
                source_data = source.get_array(source_var, flat=True)

            # check source has grid variables
            for var in output.grid.vars:
                assert(var in source.vars), '%s not in %s'%(var, source.vars)

            # prepare arrays
            sample = [source.get_array(var, flat=True) for var in output.grid.vars]

            if method is not None:

                if source_data is not None and source_data.ndim > 1:
                    output_map = np.empty(shape=(*output.grid.shape, *source_data.shape[1:]))

                    for idx in np.ndindex(*source_data.shape[1:]):
                        output_map[(Ellipsis,) + idx] = get_single_hist(sample=sample, grid=output.grid, weights=source_data[(Ellipsis,) + idx], method=method)
                else:
                    output_map = get_single_hist(sample=sample, grid=output.grid, weights=source_data, method=method)

            elif function is not None:
                indices = output.grid.compute_indices(sample)

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
                        output_map = np.full(shape=(*output.grid.shape, *source_data.shape[1:], return_len), fill_value=np.nan)
                        for idx in np.ndindex(*source_data.shape[1:]):
                            fill_single_map(output_map[(Ellipsis,) + idx + (slice(None),)], indices, source_data[(Ellipsis,) + idx], function, return_len)
                    else:
                        output_map = np.full(shape=(*output.grid.shape, *source_data.shape[1:]), fill_value=np.nan)
                        for idx in np.ndindex(*source_data.shape[1:]):
                            fill_single_map(output_map[(Ellipsis,) + idx], indices, source_data[(Ellipsis,) + idx], function, return_len)

                else:
                    if return_len > 1:
                        output_map = np.full(output.grid.shape + (return_len,), fill_value=np.nan)
                    else:
                        output_map = np.full(output.grid.shape, fill_value=np.nan)
                    fill_single_map(output_map, indices, source_data, function, return_len)
    
                output_map[np.isnan(output_map)] = fill_value

            else:
                raise ValueError('need at least a method or a function specified')
        
            if source_var is None:
                out_name = 'counts'
            else:
                out_name = source_var

            output[out_name] = output_map

            return output

        return fun

    def lookup(self, source_var=None, **kwargs):
        '''
        lookup the bin content at given points

        Parameters:
        -----------

        source_var : string
        '''
        source = self
        if not hasattr(source, 'grid'):
            raise TypeError('source must have a grid defined')

        def fun(*args, **kwargs):
            dest = generate_destination(source, *args, **kwargs)

            source_data = source.get_array(source_var)

            # check dest has grid variables
            for var in source.grid.vars:
                assert(var in dest.vars), '%s not in %s'%(var, dest.vars)


            # prepare arrays
            sample = [dest.get_array(var, flat=True) for var in source.grid.vars]

            indices = source.grid.compute_indices(sample)
            output_array = np.full(np.product(dest.array_shape), np.nan)

            #TODO: make this better
            for i in range(len(output_array)):
                # check we're inside grid:
                ind = indices[:, i]
                inside = True
                for j in range(len(ind)):
                    inside = inside and not ind[j] < 0 and not ind[j] >= source.grid.shape[j]
                if inside:
                    idx = tuple(ind)
                    output_array[i] = source_data[idx]

            dest[source_var] = output_array
            return dest

        return fun

    def resample(self, source_var, method='simple', **kwargs):
        '''
        resample from binned data into other binned data

        ToDo: this is super inefficient for grid->grid

        Parameters:
        -----------

        source_var : string

        method : str
            only method "simple" right now
        '''
        source = self

        if not hasattr(source, 'grid'):
            raise TypeError('source must have a grid defined')

        def fun(*args, **kwargs):
            output = generate_destination(source, *args, **kwargs)

            assert output.grid.vars == source.grid.vars, 'grid variables of source and destination must be identical'

            if method == 'simple':

                # we need a super sample of points, i.e. meshgrids of all combinations of source and dest
                # so first create for every dest.grid.var a vector of both, src and dest points
                lookup_sample = [np.concatenate([output.grid[var].points, source.grid[var].points]) for var in output.grid.vars]
                mesh = np.meshgrid(*lookup_sample)
                lookup_sample = [m.flatten() for m in mesh]
                
                # lookup values
                source_data = source.get_array(source_var)
                indices = source.grid.compute_indices(lookup_sample)
                lookup_array = np.full(lookup_sample[0].shape[0], np.nan)
                for i in range(len(lookup_array)):
                    # check we're inside grid:
                    ind = indices[:, i]
                    inside = True
                    for j in range(len(ind)):
                        inside = inside and not ind[j] < 0 and not ind[j] >= source.grid.shape[j]
                    if inside:
                        idx = tuple(ind)
                        lookup_array[i] = source_data[idx]

                # now bin both these points into destination
                bins = output.grid.edges
                lu_hist, _ = np.histogramdd(sample=lookup_sample, bins=bins, weights=lookup_array)
                lu_counts, _ = np.histogramdd(sample=lookup_sample, bins=bins)
                lu_hist /= lu_counts

                output[source_var] = lu_hist

                return output
            else:
                raise NotImplementedError('method %s unknown'%method)
        return fun

    #def window(self, source_var=None, method=None, function=None, window=[(-1, 1)], wrt=None, **kwargs):
    #    '''
    #    sliding window

    #    Parameters:
    #    -----------
    #    source_var : string
    #        input variable
    #    method : string
    #        "mean" = mean
    #    function : callable
    #        function to use on window
    #    window : list of tuples
    #        lower und upper extend of window in each dimension
    #    wrt : tuple
    #        specifying the variable with respect to which the interpolation is done
    #        None for griddata (will be wrt the r=destination grid)
    #    fill_value : optional
    #        value for invalid points
    #    kwargs : optional
    #        additional keyword argumnts to function
    #    '''
    #    source = self
    #    if isinstance(wrt, str):
    #        wrt = [wrt]


    #    if function is None:
    #        if method is None:
    #            raise ValueError('must provide method or function')
    #        if method == 'mean':
    #            function = np.average


    #    def fun(dest):
    #        if hasattr(dest, 'grid'):
    #            if wrt is not None:
    #                raise TypeError('wrt cannot be defined for a grid destination')

    #            grid = dest.grid
    #            this_wrt = dest.grid.vars
    #        else:

    #            grid = None
    #            if wrt is None:
    #                # need to reassign variable because of scope
    #                this_wrt = list(set(source.vars) & set(dest.vars) - set(source_var))
    #                print('Automatic with respect to %s'%', '.join(this_wrt))
    #            else:
    #                this_wrt = wrt

    #            if not set(this_wrt) <= set(dest.vars):
    #                raise TypeError('the following variable are not present in the destination: %s'%', '.join(set(this_wrt) - (set(this_wrt) & set(dest.vars))))

    #        source_data = source.get_array(source_var, flat=True)

    #        # prepare arrays
    #        source_sample = [source.get_array(var, flat=True) for var in this_wrt]
    #        source_sample = np.vstack(source_sample)
    #        dest_sample = [dest.get_array(var, flat=True) for var in this_wrt]
    #        dest_sample = np.vstack(dest_sample)

    #        output = np.zeros(dest_sample.shape[1])
    #        print(dest_sample.shape)

    #        for i in range(dest_sample.shape[1]):
    #            mask = np.ones(source_sample.shape[1]).astype(np.bool)
    #            for j in range(dest_sample.shape[0]):
    #                mask = np.logical_and(mask, source_sample[j] >= dest_sample[j, i] + window[j][0])
    #                mask = np.logical_and(mask, source_sample[j] <= dest_sample[j, i] + window[j][1])
    #            output[i] = function(source_data[mask], **kwargs)
    #
    #        if grid is None:
    #            return output
    #        return output, grid


    #    return fun


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


def fill_single_map(output_map, indices, source_data, function, return_len):
    '''
    fill a single map with a function applied to values according to indices
    '''

    if return_len > 1:
        iterator = np.nditer(output_map[...,0], flags=['multi_index'])
    else:
        iterator = np.nditer(output_map, flags=['multi_index'])

    while not iterator.finished:
        out_idx = iterator.multi_index
        mask = True
        for i, idx in enumerate(out_idx):
            mask = np.logical_and(indices[i] == idx, mask)
        bin_source_data = source_data[mask]
        if len(bin_source_data) > 0:
            result = function(bin_source_data) 
            output_map[out_idx] = result
        iterator.iternext()

