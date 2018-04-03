import numpy as np
import pandas
from scipy.interpolate import griddata

from millefeuille.datalayer import DataLayer

__all__ = ['GridLayer']

class GridLayer(DataLayer):
    '''
    Class to hold grid data
    '''
    def __init__(self, grid, name):
        '''
        Set the grid
        '''
        super(GridLayer, self).__init__(data=None,
                                        name=name,
                                        )
        self.grid = grid
        self.data = {}

    @property
    def function_args(self):
        return self.grid.vars
    
    @property
    def vars(self):
        '''
        Available variables in this layer
        '''
        return self.grid.vars + self.data.keys()
    
    @property
    def shape(self):
        return self.grid.shape
    
    @property
    def meshgrid(self):
        return self.grid.point_meshgrid
    
    @property
    def mgrid(self):
        return self.grid.point_mgrid

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
        if var in self.grid.vars:
            array = self.meshgrid[self.grid.vars.index(var)]
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
            nearest
            linear
            cubic (only for 1d or 2d grids)
        dest_var : string
            name for the destinaty variable name
        '''
        if method == 'cubic' and self.grid.ndim > 2:
            raise NotImplementedError('cubic interpolation only supported for 1 or 2 dimensions')

        if isinstance(source_var, basestring):
            source_var = source_layer.get_array(source_var)
        
        # check source layer has grid variables
        for var in self.grid.vars:
            assert(var in source_layer.vars), '%s not in %s'%(var, source_layer.vars)

        # prepare arrays
        sample = (source_layer.get_array(bin_name) for bin_name in self.grid.vars)
       
        output = griddata(sample, source_var, self.grid.mgrid, method=method)

        self.add_data(dest_var, output)
            
    def lookup(self, var, points, ndef_value=0.):
        pass

