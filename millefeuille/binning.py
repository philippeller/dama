from numbers import Number
from collections import Iterable, OrderedDict

import numpy as np

class Binning(object):
    '''
    Class to hold bin edges
    '''
    def __init__(self, bins=None):
        '''
        Paramters:
        ----------
        bins : dict, tuple or Binning object, or list thereof
            tuple must be (name, bin_edges)
            dict must be {name1 : bin_edges1, name2 : bin_edges2, ...}
        '''
        self.bins = OrderedDict()
        if isinstance(bins, dict):
            for b in bins.items():
                self.add_dim(b)
        elif isinstance(bins, (self.__class__, Binning)):
            self.add_dim(bins)
        elif isinstance(bins, tuple):
            self.add_dim(bins)
        elif isinstance(bins, list):
            for b in bins:
                self.add_dim(b)
        else:
            raise TypeError('Cannot add type %s'%type(dim))
        
    def add_dim(self, dim):
        '''
        add aditional binning dim

        Paramters:
        ----------
        dim : tuple or Binning
        '''
        if isinstance(dim, tuple):
            self.bins[dim[0]] = dim[1]
        elif isinstance(dim, (self.__class__, Binning)):
            self.bins.update(dim.bins)
        else:
            raise TypeError('Cannot add type %s'%type(dim))

    @property
    def ndim(self):
        '''
        number of binning dimensions
        '''
        return len(self.bins)

    @property
    def bin_names(self):
        '''
        binning names
        '''
        return self.bins.keys()

    @property
    def bin_edges(self):
        '''
        bin edges
        '''
        return self.bins.values()

    @property
    def bin_centers(self):
        '''
        centers of all bin_edges
        '''
        return [0.5*(b[1:] + b[:-1]) for b in self.bin_edges]

    @property
    def size(self):
        '''
        size = total number of bins
        '''
        return np.product([len(x - 1) for x in self.bin_edges])

    def __len__(self):
        if self.ndim == 1:
            return len(self.values()[0])
        else:
            return self.ndim

    def __str__(self):
        '''
        string representation
        '''
        str = []
        for dim in self.dims:
            str.append('%s : %s'%dim)
        return '\n'.join(str)

    def __iter__(self):
        '''
        if ndim > 1:
            iterate over bins
        else:
            iterate over bin_edges
        '''
        if self.ndim == 1:
            return iter(self.bins.values()[0])
        else:
            return iter([self[n] for n in self.bins.keys()])

    @property
    def dims(self):
        return self.bins.items()

    @property
    def shape(self):
        '''
        binning shape
        '''
        shape = []
        for edges in self.bins.values():
            shape.append(len(edges)-1)
        return tuple(shape)

    def __getitem__(self, item):
        '''
        item : int, str, slice, ierable
        '''
        if self.ndim == 1:
            return self.bins.values()[0][item]
        if isinstance(item, Number):
            return self.__class__((self.bins.items()[int(item)]))
        elif isinstance(item, basestring):
            return self.__class__((item, self.bins[item]))
        elif isinstance(item, Iterable):
            new_binnings = []
            for it in item:
                new_binnings.append(self[it])
            return self.__class__(new_binnings)
        elif isinstance(item, slice):
            new_names = self.bins.keys()[item]
            return self[new_names]
        else:
            raise KeyError('Cannot get key from %s'%type(item))


