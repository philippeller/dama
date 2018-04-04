import numpy as np
import pandas
from scipy.interpolate import griddata

from millefeuille.data import Data
from millefeuille.stat_plot import *

__all__ = ['PointData']

class PointData(Data):
    '''
    Data Layer to hold point-type data structures (Pandas DataFrame, Dict, )
    '''
    def __init__(self, data={}, name=None):
        super(PointData, self).__init__(data=data,
                                         name=name,
                                         )
        self.len = None

    @property
    def vars(self):
        '''
        Available variables in this layer
        '''
        if self.type == 'df':
            return list(self.data.columns)
        elif self.type == 'dict':
            return self.data.keys()
        elif self.type == 'struct_array':
            return list(self.data.dtype.names)
        else:
            return []
   
    @property
    def __len__(self):
        if self.type == 'df':
            return len(self.data)
        elif self.type == 'dict':
            return len(self.data[self.data.keys()[0]])
        elif self.type == 'struct_array':
            return len(self.data.shape[0])
                       
    def set_data(self, data):
        '''
        Set the data
        '''
        # TODO: run some compatibility cheks
        if isinstance(data, pandas.core.frame.DataFrame):
            self.type = 'df'
        elif isinstance(data, dict):
            self.type = 'dict'
        elif isinstance(data, np.ndarray):
            assert data.dtype.names is not None, 'unsctructured arrays not supported'
            self.type = 'struct_array'
        else:
            raise NotImplementedError('data type not supported')
        self.data = data
        
    def get_array(self, var):
        if self.type == 'df':
            return self.data[var].values
        else:
            return self.data[var]
        
    def add_data(self, var, data):
        # TODO do some checks of shape etc
        if self.type == 'struct_array':
            raise TypeError('cannot append rows to structured np array')
        self.data[var] = data
    
    def __getitem__(self, var):
        return self.get_array(var)

    def __setitem__(self, var, data):
        self.add_data(var, data)
    
    def translate(self ,source_var=None, source=None, method=None, dest_var=None):
        '''
        translation from function-like data into point-form
        
        Parameters:
        -----------
        
        var : string
            input variable name
        source : DataLayer
            source data layer
        method : string
            "lookup" = lookup function valiues
        dest_var : string
            name for the destinaty variable name
        '''
        if method == 'lookup':
            args = source.function_args
            points = [self.get_array(arg) for arg in args]
            new_array = source.lookup(source_var, points)
            self.add_data(dest_var, new_array)

    def plot_2d(self, fig, ax, x, y, c=None, s=None, cbar=False, **kwargs):
        plot_points_2d(fig, ax, self, x, y, c=c, s=s, cbar=cbar, **kwargs)

    def interpolate(self, source_var=None, method=None):

        source = self
        def fun(dest):
            if not hasattr(dest, 'grid'):
                raise TypeError('destination layer must have a grid defined')

            if method == 'cubic' and dest.grid.ndim > 2:
                raise NotImplementedError('cubic interpolation only supported for 1 or 2 dimensions')
            source_data = source.get_array(source_var)

            # check source layer has grid variables
            for var in dest.grid.vars:
                assert(var in source.vars), '%s not in %s'%(var, source.vars)

            # prepare arrays
            sample = [source.get_array(var) for var in dest.grid.vars]
            sample = np.vstack(sample)

            xi = dest.meshgrid
            #xi = np.stack(xi)
            #print xi.shape
           
            output = griddata(points=sample.T, values=source_data, xi=tuple(xi), method=method)

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
            'max" = maximum in each bin
            "min" = minimum in each bin
            "count" = histogram
        function : callable
        kwargs : additional keyword arguments
            will be passed to `function`
        '''
        source = self
        def fun(dest):
            if not hasattr(dest, 'grid'):
                raise TypeError('destination layer must have a grid defined')

            source_data = source.get_array(source_var)
            
            # check source has grid variables
            for var in dest.grid.vars:
                assert(var in source.vars), '%s not in %s'%(var, source.vars)

            # prepare arrays
            sample = [source.get_array(var) for var in dest.grid.vars]
            bins = dest.grid.edges    

            if method is not None:
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

                if method in ['min', 'max']:
                    indices = dest.compute_indices(sample)

                    output_map = np.ones(dest.grid.shape)
                    if method == 'min':
                        output_map *= np.max(source_data)
                    elif method == 'max':
                        output_map *= np.min(source_data)
                    
                    grid_shape = dest.grid.shape

                    for i in xrange(len(source_data)):
                        # check we're inside grid:
                        ind = indices[:,i]
                        inside = True
                        for j in range(len(ind)):
                            inside = inside and not ind[j] < 0 and not ind[j] >= grid_shape[j]
                        if inside:
                            idx = tuple(ind)
                            if method == 'min':
                                output_map[idx] =  min(output_map[idx], source_data[i])
                            if method == 'max':
                                output_map[idx] =  max(output_map[idx], source_data[i])

            elif function is not None:

                indices = dest.compute_indices(sample)

                output_map = np.ones(dest.grid.shape)
                grid_shape = dest.grid.shape

                it = np.nditer(output_map, flags=['multi_index'])

                while not it.finished:
                    out_idx = it.multi_index
                    mask = True
                    for i,idx in enumerate(out_idx):
                        mask = np.logical_and(indices[i] == idx, mask)
                    bin_source_data = source_data[mask]
                    result = function(bin_source_data, **kwargs)
                    output_map[out_idx] = result
                    it.iternext()

            return output_map

        return fun
