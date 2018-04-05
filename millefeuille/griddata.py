import numpy as np
import pandas

from millefeuille.data import Data
from millefeuille.stat_plot import *

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

    def plot_map(self, fig, ax, var, cbar=False, **kwargs):
        '''
        plot a variable as a map

        ax : pyplot axes object
        var : str
        '''
        if self.grid.ndim == 2:
            return plot_map(fig, ax, self, var, cbar=cbar, **kwargs)

    def plot_contour(self, fig, ax, var, **kwargs):
        return plot_contour(fig, ax, self, var, **kwargs)

    def plot_step(self, fig, ax, var, **kwargs):
        return plot_step(fig, ax, self, var, **kwargs)

    def plot_band(self, fig, ax, var1, var2, **kwargs):
        return plot_band(fig, ax, self, var1, var2, **kwargs)

    def plot_errorband(self, fig, ax, var, errors, **kwargs):
        return plot_errorband(fig, ax, self, var, errors, **kwargs)
