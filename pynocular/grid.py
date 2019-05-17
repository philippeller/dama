from numbers import Number
from collections import Iterable, OrderedDict

import numpy as np

class Dimension(object):
    '''
    Class to hold a single dimension of a Grid
    which can have points and/or edges
    '''
    def __init__(self, var=None, mode=None, edges=None, points=None, min=None, max=None, n_points=None, n_edges=None):
        self.var = var
        self._mode = mode
        self._min = min
        self._max = max
        self._n_points = n_points
        self._n_edges = n_edges
        self._edges = edges
        self._points = points

    def __len__(self):
        if self._points is not None:
            return len(self._points)
        elif self._edges is not None:
            return len(self._edges) - 1
        return None

    def __str__(self):
        strs = []
        strs.append('(points) %s'%(self._points))
        strs.append('(edges)  %s'%(self._edges))
        return '\n'.join(strs)

    @property
    def has_data(self):
        '''
        True if either edges or points are not None
        '''
        return (self._edges is not None) or (self._points is not None)

    @property
    def edges(self):
        if self._edges is not None:
            return self._edges
        else:
            return self.edges_from_points()

    @property
    def bin_edges(self):
        '''
        just for convenience
        '''
        return self.edges

    @edges.setter
    def edges(self, edges):
        if self.has_data:
            if not len(edges) == len(self) + 1:
                raise IndexError('incompatible length of edges')
        self._edges = edges

    @property
    def points(self):
        if self._points is not None:
            return self._points
        else:
            return self.points_from_edges()

    @points.setter
    def points(self, points):
        if self.has_data:
            if not len(points) == len(self):
                raise IndexError('incompatible length of points')
        self._points = points

    def edges_from_points(self):
        '''
        create edges around points
        '''
        diff = np.diff(self.points)/2.
        return np.concatenate([[self.points[0]-diff[0]], self.points[:-1] + diff, [self.points[-1] + diff[-1]]])

    def points_from_edges(self):
        '''
        create points from centers between edges
        '''
        return 0.5 * (self.edges[1:] + self.edges[:-1])



class Grid(object):
    '''
    Class to hold grid-like points, such as bin edges
    '''
    def __init__(self, dims=None, **kwargs):
        '''
        Paramters:
        ----------
        dims : Dimension or Grid object, or list thereof

        a dimesnion can also be given by kwargs
        '''
        self.dims = OrderedDict()

        if dims is None:
            if len(kwargs) > 0:
                self.add_dim(kwargs)
        elif isinstance(dims, (list, self.__class__, Grid)):
            for d in dims:
                self.add_dim(d)
        elif isinstance(dims, (dict, Dimension)):
            self.add_dim(dims)
        else:
            raise TypeError('Cannot add type %s'%type(dims))
        
    def add_dim(self, dim):
        '''
        add aditional Dimension

        Paramters:
        ----------
        dim : Dimension or dict or basestring

        in case of a basestring, a new empty dimension gets added
        '''
        if isinstance(dim, Dimension):
            self.dims[dim.var] = dim
        elif isinstance(dim, dict):
            dim = Dimension(**dim)
            self.add_dim(dim)
        elif isinstance(dim, str):
            new_dim = Dimension(var=dim)
            self.add_dim(new_dim)
        else:
            raise TypeError('Cannot add type %s'%type(dim))

    @property
    def ndim(self):
        '''
        number of grid dimensions
        '''
        return len(self.dims)

    @property
    def vars(self):
        '''
        grid dimension variables
        '''
        return list(self.dims.keys())

    @property
    def edges(self):
        '''
        all edges
        '''
        return [dim.edges for dim in self.dims.values()]

    @property
    def points(self):
        '''
        all points
        '''
        return [dim.points for dim in self.dims.values()]

    @property
    def point_meshgrid(self):
        return np.meshgrid(*self.points)

    @property
    def point_mgrid(self):
        return [d.T for d in self.point_meshgrid]

    @property
    def edge_meshgrid(self):
        return np.meshgrid(*self.edges)

    @property
    def edge_mgrid(self):
        return [d.T for d in self.edge_meshgrid]

    @property
    def size(self):
        '''
        size = total number of bins / points
        '''
        return np.product([len(x) for x in self.dims.items()])

    def __len__(self):
        return self.ndim

    def __str__(self):
        '''
        string representation
        '''
        str = []
        for dim in self.dims.items():
            str.append('%s : %s'%dim)
        return '\n'.join(str)

    def __iter__(self):
        '''
        iterate over dimensions
        '''
        return iter([self[n] for n in self.dims.keys()])

    @property
    def shape(self):
        '''
        shape
        '''
        shape = []
        for dim in self:
            shape.append(len(dim))
        return tuple(shape)

    def __getitem__(self, item):
        '''
        item : int, str, slice, ierable
        '''
        if isinstance(item, Number):
            return list(self.dims.values())[int(item)]
        elif isinstance(item, str):
            if not item in self.vars:
                self.add_dim(item)
            return self.dims[item]
        elif isinstance(item, Iterable):
            new_dims = []
            for it in item:
                new_dims.append(self[it])
            return self.__class__(new_dims)
        elif isinstance(item, slice):
            new_names = self.dims.keys()[item]
            return self[new_names]
        else:
            raise KeyError('Cannot get key from %s'%type(item))

    def __setitem__(self, item, val):
        raise AttributeError("to set a grid dimension, specify if it is `points` or `edges`, e.g.:\ngrid['%s'].edges = %s"%(item, val))

    def compute_indices(self, sample):
        '''
        calculate the bin indices for a a given sample
        '''
        if isinstance(sample, np.ndarray):
            assert sample.shape[0] == self.ndim
        elif isinstance(sample, list):
            assert len(sample) == self.ndim

        # array to hold indices
        indices = np.empty((self.ndim, len(sample[0])), dtype=np.int)
        #calculate bin indices
        for i in range(self.ndim):
            indices[i] = np.digitize(sample[i], self.edges[i])
        indices -= 1
        return indices


def test():
    a = Grid(var='a', edges=np.linspace(0,1,2))
    print(a)
    print(a.vars)
    a['x'].edges = np.linspace(0,10,11)
    a['y'].points = np.logspace(-1,1,20)
    print(a['x'].points)
    print(a['x'].edges)
    print(a['x', 'y'])

if __name__ == '__main__':
    test()
