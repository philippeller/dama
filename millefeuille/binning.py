class Binning(object):
    '''
    Class to hold bin edges
    '''
    def __init__(self, bin_names, bin_edges):
        self.bin_names = bin_names
        self.bin_edges = bin_edges
        
    @property
    def n_bins(self):
        return len(self.bin_names)
