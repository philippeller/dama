import numpy as np
from scipy.interpolate import griddata

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
        return self.add_data(var, new_data)

    def __len__(self):
        return 0
    

    def interpolate(self, source_var=None, method=None):
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
        '''
        source = self
        def fun(dest):
            if not hasattr(dest, 'grid'):
                raise TypeError('destination must have a grid defined')

            if method == 'cubic' and dest.grid.ndim > 2:
                raise NotImplementedError('cubic interpolation only supported for 1 or 2 dimensions')
            source_data = source.get_array(source_var, flat=True)

            mask = np.isfinite(source_data)

            # check source has grid variables
            for var in dest.grid.vars:
                assert(var in source.vars), '%s not in %s'%(var, source.vars)

            # prepare arrays
            sample = [source.get_array(var, flat=True)[mask] for var in dest.grid.vars]
            sample = np.vstack(sample)
            xi = dest.mgrid

            output = griddata(points=sample.T, values=source_data[mask], xi=tuple(xi), method=method)

            return output

        return fun


    def histogram(self, source_var=None, method=None, function=None, **kwargs):
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

        def fun(dest):
            if not hasattr(dest, 'grid'):
                raise TypeError('destination must have a grid defined')

            source_data = source.get_array(source_var, flat=True)
            
            # check source has grid variables
            for var in dest.grid.vars:
                assert(var in source.vars), '%s not in %s'%(var, source.vars)

            # prepare arrays
            sample = [source.get_array(var, flat=True) for var in dest.grid.vars]

            if method is not None:
                bins = dest.grid.edges    
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
                indices = dest.grid.compute_indices(sample)
                output_map = np.zeros(dest.grid.shape) * np.nan

                grid_shape = dest.grid.shape

                it = np.nditer(output_map, flags=['multi_index'])

                while not it.finished:
                    out_idx = it.multi_index
                    mask = True
                    for i,idx in enumerate(out_idx):
                        mask = np.logical_and(indices[i] == idx, mask)
                    bin_source_data = source_data[mask]
                    if len(bin_source_data) > 0:
                        result = function(bin_source_data, **kwargs)
                        output_map[out_idx] = result
                    it.iternext()

            return output_map

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


        def fun(dest):
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
            for i in xrange(len(output_array)):
                # check we're inside grid:
                ind = indices[:,i]
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

        def fun(dest):
            if not hasattr(dest, 'grid'):
                raise TypeError('destination must have a grid defined')

            assert dest.grid.vars == source.grid.vars, 'grid variables of source and destination must be identical'

            if method == 'simple':
                # first histogram points of source into destination
                source_data = source.get_array(source_var, flat=True)
                # prepare arrays
                sample = [source.get_array(var, flat=True) for var in dest.grid.vars]

                bins = dest.grid.edges    
                # generate hists
                hist, _ = np.histogramdd(sample=sample, bins=bins, weights=source_data)
                counts, _ = np.histogramdd(sample=sample, bins=bins)
                mask = counts > 0
                hist[mask] /= counts[mask]

                # then lookup destination points in source
                # prepare arrays
                sample = [dest.get_array(var, flat=True) for var in source.grid.vars]

                source_data = source.get_array(source_var)
                indices = source.grid.compute_indices(sample)
                grid_shape = source.grid.shape
                lookup_array = np.ones(np.product(dest.array_shape)) * np.nan
                #TODO: make this better
                for i in xrange(len(lookup_array)):
                    # check we're inside grid:
                    ind = indices[:,i]
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
                return hist
            else:
                raise NotImplementedError('method %s unknown'%method)
        return fun
