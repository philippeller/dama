import numpy as np
import pandas

from millefeuille.datalayer import DataLayer

class PointLayer(DataLayer):
    '''
    Data Layer to hold point-type data structures (Pandas DataFrame, Dict, )
    '''
    def __init__(self, data, name):
        super(PointLayer, self).__init__(data=data,
                                         name=name,
                                         )
        self.len = None

    @property
    def vars(self):
        '''
        Available variables in this layer
        '''
        if isinstance(self.data, pandas.core.frame.DataFrame):
            return list(self.data.columns)
        elif isinstance(self.data, dict):
            return self.data.keys()
        else:
            raise NotImplementedError('data type not implemented')        
   
    @property
    def __len__(self):
        if isinstance(self.data, pandas.core.frame.DataFrame):
            return len(self.data)
        else:
            return len(self.data[self.data.keys()[0]])
                       
    def set_data(self, data):
        '''
        Set the data
        '''
        # TODO: run some compatibility cheks
        self.data = data
        
    def get_array(self, var):
        if isinstance(self.data, pandas.core.frame.DataFrame):
            return self.data[var].values
        else:
            return self.data[var]
        
    def add_data(self, var, data):
        # TODO do some checks of shape etc
        self.data[var] = data
    
    def __getitem__(self, var):
        return self.get_array(var)
    
    def translate(self ,source_var=None, source_layer=None, method=None, dest_var=None):
        '''
        translation from function-like data into point-form
        
        Parameters:
        -----------
        
        var : string
            input variable name
        source_layer : DataLayer
            source data layer
        method : string
            "lookup" = lookup function valiues
        dest_var : string
            name for the destinaty variable name
        '''
        if method == 'lookup':
            args = source_layer.function_args
            points = [self.get_array(arg) for arg in args]
            new_array = source_layer.lookup(source_var, points)
            self.add_data(dest_var, new_array)
