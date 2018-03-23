class DataLayer(object):
    '''
    Data layer base class to hold any form of data representation
    '''
    def __init__(self, data, name):
        self.data = None
        self.name = name
        self.set_data(data)
        
    def set_data(self, data):
        pass
