import numpy as np
import pandas

from millefeuille.datalayer import DataLayer

__all__ = ['PointLayer']

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
