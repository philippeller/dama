class DataStack(object):
    '''
    Structure to hold DataLayers and mediate translations between layers
    '''
    def __init__(self, name):
        self.name = name
        self.layers = {}
        self.vars = []
        self.default_layer = {}
    
    def add_layer(self, layer):
        self.layers[layer.name] = layer
        self.vars = list(set(self.vars + layer.vars))
        
    def __getitem__(self, var):
        return self.layers[var]
        
    def translate(self, var=None, source=None, dest=None, method=None, dest_var=None):
        '''
        translation from array data into binned form
        
        Parameters:
        -----------
        
        var : string
            in/output variable name
        source : string
            source layer name
        dest : string
            destination layer name
        method : string
            translation method
        dest_var : string (optional)
            name for the destinaty variable name, if `None` same as `var`
        '''
        if dest_var is None:
            dest_var = var
        
        self[dest].translate(source_var=var,
                             source_layer=self[source],
                             method=method,
                             dest_var=dest_var)
