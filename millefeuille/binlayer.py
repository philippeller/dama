import numpy as np
import pandas

from millefeuille.datalayer import DataLayer

__all__ = ['BinLayer']

class BinLayer(DataLayer):
    '''
    Class to hold binned data (like histograms)
    '''
    def __init__(self, binning, name):
        '''
        Set the binning
        '''
        super(BinLayer, self).__init__(data=None,
                                       name=name,
                                       )
        self.binning = binning
        self.data = {}

    @property
    def function_args(self):
        return self.binning.vars
    
    @property
    def vars(self):
        '''
        Available variables in this layer
        '''
        return self.binning.vars + self.data.keys()
    
    @property
    def shape(self):
        return self.binning.shape
    
    @property
    def meshgrid(self):
        '''
        return meshgrid of binning
        '''
        return self.binning.edge_meshgrid
    
    def add_data(self, var, data):
        # TODO do some checks of shape etc
        self.data[var] = data
        
    def get_array(self, var, flat=False):
        '''
        return array of data
        
        Parameters:
        -----------
        
        var : string
            variable to return
        flat : bool
            if true return flattened (1d) array
        '''
        if var in self.binning.vars:
            array = self.meshgrid[self.binning.vars.index(var)]
        else:
            array = self.data[var]
        if flat:
            return array.ravel()
        else:
            return array

    def __getitem__(self, var):
        return self.get_array(var)
    
    def __setitem__(self, var, data):
        return self.add_data(var, data)
    
    def translate(self, source_var=None, source_layer=None, method=None, function=None, dest_var=None):
        '''
        translation from array data into binned form
        
        Parameters:
        -----------
        
        var : string or array
            input variable
        source_layer : DataLayer
            source data layer
        method : string
            "sum" = weighted historgam
            "mean" = weighted histogram / histogram
            'max" = maximum in each bin
            "min" = minimum in each bin
            "count" = histogram
        function : callable
        dest_var : string
            name for the destinaty variable name
        '''

        if isinstance(source_var, basestring):
            source_var = source_layer.get_array(source_var)
        
        # check source layer has binning variables
        for bin_name in self.binning.vars:
            assert(bin_name in source_layer.vars), '%s not in %s'%(bin_name, source_layer.vars)

        # prepare arrays
        sample = [source_layer.get_array(bin_name) for bin_name in self.binning.vars]
        bins = self.binning.bin_edges    
       

        if method is not None:
            # generate hists
            if method in ['sum', 'mean']:
                weighted_hist, _ = np.histogramdd(sample=sample, bins=bins, weights=source_var)

            if method in ['count', 'mean']:
                hist, _ = np.histogramdd(sample=sample, bins=bins)

            if method in ['min', 'max']:
                indices = self.compute_indices(sample)

                output_map = np.ones(self.binning.shape)
                if method == 'min':
                    output_map *= np.max(source_var)
                if method == 'max':
                    output_map *= np.min(source_var)
                
                binning_shape = self.binning.shape

                for i in xrange(len(source_var)):
                    # check we're inside binning:
                    ind = indices[:,i]
                    inside = True
                    for j in range(len(ind)):
                        inside = inside and not ind[j] < 0 and not ind[j] >= binning_shape[j]
                    if inside:
                        idx = tuple(ind)
                        if method == 'min':
                            output_map[idx] =  min(output_map[idx], source_var[i])
                        if method == 'max':
                            output_map[idx] =  max(output_map[idx], source_var[i])
                self.add_data(dest_var, output_map)

            # make outputs
            if method == 'count':
                self.add_data(dest_var, hist)
            elif method == 'sum':
                self.add_data(dest_var, weighted_hist)
            elif method == 'mean':
                mask = (hist > 0.)
                weighted_hist[mask] /= hist[mask]
                self.add_data(dest_var, weighted_hist)

        elif function is not None:

            indices = self.compute_indices(sample)

            output_map = np.ones(self.binning.shape)
            binning_shape = self.binning.shape

            it = np.nditer(output_map, flags=['multi_index'])

            while not it.finished:
                out_idx = it.multi_index
                mask = True
                for i,idx in enumerate(out_idx):
                    mask = np.logical_and(indices[i] == idx, mask)
                bin_source_var = source_var[mask]
                result = function(bin_source_var)
                output_map[out_idx] = result
                it.iternext()

            self.add_data(dest_var, output_map)
            
    def lookup(self, var, points, ndef_value=0.):
        '''
        lookup the bin content at given points
        
        Parameters:
        -----------
        
        var : string
        ponints : list of k length n arrays or 2-d array with shape (k, n)
            where k is number of bins
        ndef_value : float
            value to assign for points outside the binning
        '''
        indices = self.compute_indices(points)

        binning_shape = self.binning.shape

        output_array = np.empty(len(points[0]))
        # this is stupid
        for i in xrange(len(output_array)):
            # check we're inside binning:
            ind = indices[:,i]
            inside = True
            for j in range(len(ind)):
                inside = inside and not ind[j] < 0 and not ind[j] >= binning_shape[j]
            if inside:
                #print ind
                idx = tuple(ind)
                output_array[i] = self.data[var][idx]
            else:
                output_array[i] = ndef_value
                
        return output_array

    def compute_indices(self, points):
        '''
        calculate the bin indices for a a given sample
        '''

        ndim = self.binning.ndim

        if isinstance(points, np.ndarray):
            assert points.shape[0] == ndim
        elif isinstance(points, list):
            assert len(points) == ndim

        # array to hold indices
        indices = np.empty((ndim, len(points[0])), dtype=np.int)
        #calculate bin indices
        for i in range(ndim):
            indices[i] = np.digitize(points[i], self.binning.bin_edges[i])
        indices -= 1
        #print indices
        return indices
