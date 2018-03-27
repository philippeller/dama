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
        return self.binning.bin_names
    
    @property
    def vars(self):
        '''
        Available variables in this layer
        '''
        return self.binning.bin_names + self.data.keys()
    
    @property
    def shape(self):
        return self.data[self.data.keys()[0]].shape
    
    @property
    def meshgrid(self):
        '''
        return meshgrid of binning
        '''
        return np.meshgrid(*self.binning.bin_edges)
    
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
        if var in self.binning.bin_names:
            array = self.meshgrid[self.binning.bin_names.index(var)]
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
    
    def translate(self, source_var=None, source_layer=None, method=None, dest_var=None):
        '''
        translation from array data into binned form
        
        Parameters:
        -----------
        
        var : string
            input variable name
        source_layer : DataLayer
            source data layer
        method : string
            "sum" = weighted historgam
            "mean" = weighted histogram / histogram
            'max" = maximum in each bin
            "min" = minimum in each bin
            "count" = histogram
        dest_var : string
            name for the destinaty variable name
        '''
        
        # check source layer has binning variables
        for bin_name in self.binning.bin_names:
            assert(bin_name in source_layer.vars), '%s not in %s'%(bin_name, source_layer.vars)

        # prepare arrays
        sample = [source_layer.get_array(bin_name) for bin_name in self.binning.bin_names]
        bins = self.binning.bin_edges    
        
        # generate hists
        if method in ['sum', 'mean']:
            weights = source_layer.get_array(source_var)
            weighted_hist, _ = np.histogramdd(sample=sample, bins=bins, weights=weights)

        if method in ['count', 'mean']:
            hist, _ = np.histogramdd(sample=sample, bins=bins)

        if method in ['min', 'max']:
            indices = self.compute_indices(sample)
            data = source_layer.get_array(source_var)

	    output_map = np.ones(self.binning.shape)
            if method == 'min':
                output_map *= np.max(data)
            if method == 'max':
                output_map *= np.min(data)
            
            binning_shape = self.binning.shape

	    for i in xrange(len(data)):
                # check we're inside binning:
                ind = indices[:,i]
                inside = True
                for j in range(len(ind)):
                    inside = inside and not ind[j] < 0 and not ind[j] >= binning_shape[j]
                if inside:
                    idx = tuple(ind)
                    if method == 'min':
                        output_map[idx] =  min(output_map[idx], data[i])
                    if method == 'max':
                        output_map[idx] =  max(output_map[idx], data[i])
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
