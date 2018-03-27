from numbers import Number
from collections import Iterable, OrderedDict

import numpy as np

class Binning(object):
    '''
    Class to hold bin edges
    '''
    def __init__(self, bins=None):
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
        #print 'add ',dim
        if isinstance(dim, tuple):
            self.bins[dim[0]] = dim[1]
        elif isinstance(dim, (self.__class__, Binning)):
            self.bins.update(dim.bins)
        else:
            raise TypeError('Cannot add type %s'%type(dim))

    @property
    def ndim(self):
        return len(self.bins)

    @property
    def bin_names(self):
        return self.bins.keys()

    @property
    def bin_edges(self):
        return self.bins.values()

    @property
    def size(self):
        return np.product([len(x) for x in self.bins.values()])

    def __len__(self):
        if self.ndim == 1:
            return len(self.values()[0])
        else:
            return self.ndim

    def __str__(self):
        str = []
        for dim in self.dims:
            str.append('%s : %s'%dim)
        return '\n'.join(str)

    def __iter__(self):
        if self.ndim == 1:
            return iter(self.bins.values()[0])
        else:
            return iter([self[n] for n in self.bins.keys()])

    @property
    def dims(self):
        return self.bins.items()


    @property
    def shape(self):
        shape = []
        for edges in self.bins.values():
            shape.append(len(edges)-1)
        return tuple(shape)

    def __getitem__(self, item):
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


