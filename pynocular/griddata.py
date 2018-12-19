import numpy as np
import pandas

from pynocular.data import Data
from pynocular.stat_plot import *

__all__ = ['GridData']

class GridData(Data):
    '''
    Class to hold grid data
    '''
    def __init__(self, grid):
        '''
        Set the grid
        '''
        super(GridData, self).__init__(data=None,
                                        )
        self.grid = grid
        self.data = {}
        self.mask = None

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
    def array_shape(self):
        '''
        shape of a single variable
        '''
        return self.shape
    
    @property
    def meshgrid(self):
        return self.grid.point_meshgrid
    
    @property
    def mgrid(self):
        return self.grid.point_mgrid

    def add_data(self, var, data):
        # TODO do some checks of shape etc
        self.data[var] = data
        
    def get_array(self, var, flat=False, mask=False):
        '''
        return array of data
        
        Parameters:
        -----------
        
        var : string
            variable to return
        flat : bool
            if true return flattened (1d) array
        '''
        if mask:
            if not self.mask is None:
                raise NotImplementedError('masking for griddata not yet implemented')

        if var in self.grid.vars:
            array = self.mgrid[self.grid.vars.index(var)]
        else:
            array = self.data[var]
        if flat:
            return array.ravel()
        else:
            return array

    def plot_map(self, var, cbar=False, fig=None, ax=None, **kwargs):
        '''
        plot a variable as a map

        ax : pyplot axes object
        var : str
        '''
        if self.grid.ndim == 2:
            return plot_map(self, var, cbar=cbar, fig=fig, ax=ax, **kwargs)

    def plot_contour(self, var, fig=None, ax=None, **kwargs):
        return plot_contour(self, var, fig=fig, ax=ax, **kwargs)

    def plot_step(self, var, fig=None, ax=None, **kwargs):
        return plot_step(self, var, fig=fig, ax=ax, **kwargs)

    def plot_band(self, var1, var2, fig=None, ax=None, **kwargs):
        return plot_band(self, var1, var2, fig=fig, ax=ax, **kwargs)

    def plot_errorband(self, var, errors, fig=None, ax=None, **kwargs):
        return plot_errorband(self, var, errors, fig=fig, ax=ax, **kwargs)
