from __future__ import absolute_import
import numpy as np
from scipy import interpolate
from KDEpy import FFTKDE
import pynocular as pn


def get_grid(source, *args, **kwargs):
    '''
    Return correctly set up grid, depending on the supplied input
    '''
    if len(args) == 1 and len(kwargs) == 0:
        dest = args[0]
        if isinstance(dest, pn.GridData):
            return dest.grid
        if isinstance(dest, pn.grid.Grid):
            return dest

    # check if source has a grid and if any args are in there
    if isinstance(source, pn.GridData):
        dims = []
        for arg in args:
            # in thsio case the source may have a grid, get those edges
            if isinstance(arg, str):
                if arg in source.grid.vars:
                    dims.append(source.grid[arg])
                    continue
            dims.append[arg]
        args = dims

    # instantiate
    grid = pn.grid.Grid(*args, **kwargs)

    # check dest grid is set up, otherwise do so
    for var in grid.vars:
        if grid[var].edges is None:
            # check if it might be from a grid
            if isinstance(source, pn.GridData):
                if var in source.grid.vars:
                    grid[var].edges = np.linspace(source.grid[var].edges[0], source.grid[var].edges[-1], grid[var].nbins+1)
                    continue
            grid[var].edges = np.linspace(np.nanmin(source[var]), np.nanmax(source[var]), grid[var].nbins+1)

    return grid

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
            new_data.rename(new_data.data_vars[0], var)
            self.update(new_data)
            return

            #if len(new_data) == 2:
            #    # we have a (data, grid) as return
            #    assert hasattr(self, 'grid')
            #    if self.grid.initialized:
            #        assert self.grid == new_data[1]
            #    else:
            #        self.grid = new_data[1]
            #    new_data = new_data[0]
        self.add_data(var, new_data)

    def __len__(self):
        return 0

    def __repr__(self):
        return 'Data(%s)'%self.data.__repr__()

    def __str__(self):
        return self.data.__str__()

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
            if len(args) == 1 and len(kwargs) == 0:
                if isinstance(args[0], pn.PointData):
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
                        output = f(dest[this_wrt[0]])

                    elif len(this_wrt) == 2 and method in ['linear', 'cubic']:
                        f = interpolate.interp2d(source[this_wrt[0]], source[this_wrt[1]], source[source_var], kind=method, fill_value=fill_value, bounds_error=False)
                        output = np.array([f(x, y)[0] for x, y in zip(dest[this_wrt[0]], dest[this_wrt[1]])])

                    elif method in ['nearest', 'linear']:
                        sample = [source.get_array(var, flat=True) for var in this_wrt]
                        sample = np.vstack(sample).T
                        if method == 'nearest':
                            f = interpolate.NearestNDInterpolator(sample, source[source_var])
                        else:
                            f = interpolate.LinearNDInterpolator(sample, source[source_var], fill_value=fill_value)
                        out_sample = [dest.get_array(var, flat=True) for var in this_wrt]
                        out_sample = np.vstack(out_sample).T
                        output = f(out_sample)

                    else:
                        raise NotImplementedError('method %s not available for %i dimensional interpolation'%(method, len(this_wrt)))

                    return output

            grid = get_grid(source, *args, **kwargs)
            output = pn.GridData(grid)

            if wrt is not None:
                raise TypeError('wrt cannot be defined for a grid destination')

            if method == 'cubic' and grid.ndim > 2:
                raise NotImplementedError('cubic interpolation only supported for 1 or 2 dimensions')
            source_data = source.get_array(source_var, flat=True)

            mask = np.isfinite(source_data)

            # check source has grid variables
            for var in grid.vars:
                assert(var in source.vars), '%s not in %s'%(var, source.vars)

            # prepare arrays
            sample = [source.get_array(var, flat=True)[mask] for var in grid.vars]
            sample = np.vstack(sample)
            
            xi = grid.point_mgrid

            output_map = interpolate.griddata(points=sample.T, values=source_data[mask], xi=tuple(xi), method=method, fill_value=fill_value)

            output[source_var] = output_map

            return output

        return fun


    def histogram(self, source_var=None, method=None, function=None, fill_value=np.nan, **kwargs):
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
        kwargs : additional keyword arguments
            will be passed to `function`
        '''
        source = self

        if method is None and function is None and source_var is None:
            method = "count"

        if source_var is None:
            assert method == 'count'

        if method is None and function is None:
            method = "sum"

        def fun(*args, **kwargs):
            grid = get_grid(source, *args, **kwargs)
            output = pn.GridData(grid)

            if source_var is None:
                source_data = None
            else:
                source_data = source.get_array(source_var, flat=True)

            # check source has grid variables
            for var in grid.vars:
                assert(var in source.vars), '%s not in %s'%(var, source.vars)

            # prepare arrays
            sample = [source.get_array(var, flat=True) for var in grid.vars]

            if method is not None:
                bins = grid.edges
                # generate hists
                if method in ['sum', 'mean']:
                    weighted_hist, _ = np.histogramdd(sample=sample, bins=bins, weights=source_data)

                if method in ['count', 'mean']:
                    hist, _ = np.histogramdd(sample=sample, bins=bins)

                # make outputs
                if method == 'count':
                    output_map = hist
                elif method == 'sum':
                    output_map = weighted_hist
                elif method == 'mean':
                    mask = (hist > 0.)
                    weighted_hist[mask] /= hist[mask]
                    output_map = weighted_hist

            elif function is not None:
                indices = grid.compute_indices(sample)
                output_map = np.zeros(grid.shape) * np.nan

                grid_shape = grid.shape

                it = np.nditer(output_map, flags=['multi_index'])

                while not it.finished:
                    out_idx = it.multi_index
                    mask = True
                    for i, idx in enumerate(out_idx):
                        mask = np.logical_and(indices[i] == idx, mask)
                    bin_source_data = source_data[mask]
                    if len(bin_source_data) > 0:
                        result = function(bin_source_data) #, **kwargs)
                        output_map[out_idx] = result
                    it.iternext()

                output_map[np.isnan(output_map)] = fill_value
        
        
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
            grid = get_grid(source, *args, **kwargs)

            source_data = source.get_array(source_var)

            # check dest has grid variables
            for var in source.grid.vars:
                assert(var in dest.vars), '%s not in %s'%(var, dest.vars)


            # prepare arrays
            sample = [dest.get_array(var, flat=True) for var in source.grid.vars]

            indices = source.grid.compute_indices(sample)
            grid_shape = source.grid.shape
            #output_array = np.ones(dest.array_shape) * np.nan
            output_array = np.ones(np.product(dest.array_shape)) * np.nan

            #TODO: make this better
            for i in range(len(output_array)):
                # check we're inside grid:
                ind = indices[:, i]
                inside = True
                for j in range(len(ind)):
                    inside = inside and not ind[j] < 0 and not ind[j] >= grid_shape[j]
                if inside:
                    idx = tuple(ind)
                    output_array[i] = source_data[idx]

            return output_array.reshape(dest.array_shape)

        return fun

    def resample(self, source_var, method='simple', **kwargs):
        '''
        resample from binned data into other binned data

        Parameters:
        -----------

        source_var : string
        '''
        source = self
        if not hasattr(source, 'grid'):
            raise TypeError('source must have a grid defined')

        def fun(*args, **kwargs):
            grid = get_grid(source, *args, **kwargs)

            assert grid.vars == source.grid.vars, 'grid variables of source and destination must be identical'

            if method == 'simple':
                # first histogram points of source into destination
                source_data = source.get_array(source_var, flat=True)
                # prepare arrays
                sample = [source.get_array(var, flat=True) for var in grid.vars]

                bins = grid.edges
                # generate hists
                hist, _ = np.histogramdd(sample=sample, bins=bins, weights=source_data)
                counts, _ = np.histogramdd(sample=sample, bins=bins)
                mask = counts > 0
                hist[mask] /= counts[mask]

                # then lookup destination points in source
                # prepare arrays

                # fixme

                sample = [dest.get_array(var, flat=True) for var in source.grid.vars]

                source_data = source.get_array(source_var)
                indices = source.grid.compute_indices(sample)
                grid_shape = source.grid.shape
                lookup_array = np.ones(np.product(dest.array_shape)) * np.nan
                #TODO: make this better
                for i in range(len(lookup_array)):
                    # check we're inside grid:
                    ind = indices[:, i]
                    inside = True
                    for j in range(len(ind)):
                        inside = inside and not ind[j] < 0 and not ind[j] >= grid_shape[j]
                    if inside:
                        idx = tuple(ind)
                        lookup_array[i] = source_data[idx]

                lookup_array =  lookup_array.reshape(dest.array_shape)

                # where counts is <=1, replace hist by lookups
                mask  = counts <= 1
                hist[mask] = lookup_array[mask]
                return hist, grid
            else:
                raise NotImplementedError('method %s unknown'%method)
        return fun

    def window(self, source_var=None, method=None, function=None, window=[(-1, 1)], wrt=None, **kwargs):
        '''
        sliding window

        Parameters:
        -----------
        source_var : string
            input variable
        method : string
            "mean" = mean
        function : callable
            function to use on window
        window : list of tuples
            lower und upper extend of window in each dimension
        wrt : tuple
            specifying the variable with respect to which the interpolation is done
            None for griddata (will be wrt the r=destination grid)
        fill_value : optional
            value for invalid points
        kwargs : optional
            additional keyword argumnts to function
        '''
        source = self
        if isinstance(wrt, str):
            wrt = [wrt]


        if function is None:
            if method is None:
                raise ValueError('must provide method or function')
            if method == 'mean':
                function = np.average


        def fun(dest):
            if hasattr(dest, 'grid'):
                if wrt is not None:
                    raise TypeError('wrt cannot be defined for a grid destination')

                grid = dest.grid
                this_wrt = dest.grid.vars
            else:

                grid = None
                if wrt is None:
                    # need to reassign variable because of scope
                    this_wrt = list(set(source.vars) & set(dest.vars) - set(source_var))
                    print('Automatic with respect to %s'%', '.join(this_wrt))
                else:
                    this_wrt = wrt

                if not set(this_wrt) <= set(dest.vars):
                    raise TypeError('the following variable are not present in the destination: %s'%', '.join(set(this_wrt) - (set(this_wrt) & set(dest.vars))))

            source_data = source.get_array(source_var, flat=True)

            # prepare arrays
            source_sample = [source.get_array(var, flat=True) for var in this_wrt]
            source_sample = np.vstack(source_sample)
            dest_sample = [dest.get_array(var, flat=True) for var in this_wrt]
            dest_sample = np.vstack(dest_sample)

            output = np.zeros(dest_sample.shape[1])
            print(dest_sample.shape)

            for i in range(dest_sample.shape[1]):
                mask = np.ones(source_sample.shape[1]).astype(np.bool)
                for j in range(dest_sample.shape[0]):
                    mask = np.logical_and(mask, source_sample[j] >= dest_sample[j, i] + window[j][0])
                    mask = np.logical_and(mask, source_sample[j] <= dest_sample[j, i] + window[j][1])
                output[i] = function(source_data[mask], **kwargs)
    
            if grid is None:
                return output
            return output, grid


        return fun
