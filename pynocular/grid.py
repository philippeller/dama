from __future__ import absolute_import
from numbers import Number
from collections import OrderedDict
from collections.abc import Iterable

import pynocular as pn

import numpy as np

__license__ = '''Copyright 2019 Philipp Eller

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

class Dimension(object):
    '''
    Class to hold a single dimension of a Grid
    which can have points and/or edges
    '''
    def __init__(self, var=None, edges=None, points=None, nbins=10):

        if isinstance(edges, list): edges = np.array(edges)
        if edges is not None:
            assert len(edges) > 1, 'Edges must be at least length 2'
        if isinstance(points, list): points = np.array(points)
        self.var = var
        self._edges = edges
        self._points = points
        self._nbins = nbins

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
        strs.append('(nbins)  %s'%(self.nbins))
        return '\n'.join(strs)

    def __repr__(self):
        strs = []
        strs.append('Dimension("%s",'%self.var)
        strs.append('points = %s,'%(self._points.__repr__()))
        strs.append('edges = %s)'%(self._edges.__repr__()))
        strs.append('nbins = %s)'%(self.nbins))
        return '\n'.join(strs)

    def __getitem__(self, idx):
        if idx is Ellipsis:
            return self
        if isinstance(idx, Number):
            if not abs(idx) < len(self):
                raise IndexError('Index outside range')
            if idx < 0:
                edge_idx = slice(idx-1, idx+1 if not idx == -1 else None)
            else:
                edge_idx = slice(idx, idx+2)
        elif isinstance(idx, slice):
            if idx.step is not None and abs(idx.step) > 1:
                raise IndexError('Can only slice consecutive indices')
            reverse = idx.step is not None and idx.step < 0
            edge_idx = list(range(*idx.indices(len(self))))
            if reverse:
                edge_idx = [edge_idx[0] + 1] + edge_idx
            else:
                edge_idx = edge_idx + [edge_idx[-1] + 1]
        elif isinstance(idx, list) or isinstance(idx, tuple):
            if not all(i == 1 for i in np.diff(idx)):
                raise IndexError('Can only apply consecutive indices')
            edge_idx = list(idx)
            edge_idx = edge_idx + [edge_idx[-1] + 1]
        else:
            raise NotImplementedError('%s index not supported'%type(idx))

        new_obj = pn.grid.Dimension()
        new_obj.var = self.var
        if self._edges is not None:
            new_obj._edges = self._edges[edge_idx]
        if self._points is not None:
            new_obj._points = self._points[idx]
        new_obj._nbins = self._nbins
        return new_obj

    @property
    def has_data(self):
        '''
        True if either edges or points are not None
        '''
        return (self._edges is not None) or (self._points is not None)

    def __eq__(self, other):
        if not type(self) == type(other):
            return False
        equal = self.var == other.var
        equal = equal and np.all(np.equal(self._edges, other._edges))
        equal = equal and np.all(np.equal(self._points, other._points))
        return equal and self._nbins == other._nbins

    @property
    def regular(self):
        '''True if spacing of egdges and/or points is regular'''
        regular = True
        if self._points is not None:
            regular = regular and np.equal.reduce(np.diff(self._points))
        if self._edges is not None:
            regular = regular and np.equal.reduce(np.diff(self._edges))
        return regular
    
    @property
    def edges(self):
        if self._edges is not None:
            return self._edges
        elif self._points is not None:
            return self.edges_from_points()
        return None

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
        elif self._edges is not None:
            return self.points_from_edges()
        return None

    @points.setter
    def points(self, points):
        if self.has_data:
            if not len(points) == len(self):
                raise IndexError('incompatible length of points')
        self._points = points

    @property
    def nbins(self):
        if self._points is None and self._edges is None:
            return self._nbins
        else:
            return len(self.points)

    @nbins.setter
    def nbins(self, nbins):
        if self._points is None and self._edges is None:
            self._nbins = nbins
        else:
            raise ValueError('Cannot set n since bins are already defined')

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
        points = 0.5 * (self.edges[1:] + self.edges[:-1])
        if isinstance(points, Number):
            return np.array(points)
        return points



class Grid(object):
    '''
    Class to hold grid-like points, such as bin edges
    '''
    def __init__(self, *args, **kwargs):
        '''
        Paramters:
        ----------
        dims : Dimension or Grid object, or list thereof

        a dimesnion can also be given by kwargs
        '''
        self.dims = [] 

        for d in args:
            self.add_dim(d)

        for d,x in kwargs.items():
            if isinstance(x, Number):
                self.add_dim(pn.Dimension(var=d, nbins=x))
            else:
                self.add_dim(pn.Dimension(var=d, edges=x))

    def add_dim(self, dim):
        '''
        add aditional Dimension

        Paramters:
        ----------
        dim : Dimension or dict or basestring

        in case of a basestring, a new empty dimension gets added
        '''
        if isinstance(dim, pn.Dimension):
            assert dim.var not in self.vars
            self.dims.append(dim)
        elif isinstance(dim, dict):
            dim = pn.Dimension(**dim)
            self.add_dim(dim)
        elif isinstance(dim, str):
            new_dim = pn.Dimension(var=dim)
            self.add_dim(new_dim)
        else:
            raise TypeError('Cannot add type %s'%type(dim))

    def __eq__(self, other):
        if not type(self) == type(other):
            return False
        equal = self.vars == other.vars
        return equal and all([self[var] == other[var] for var in self.vars])


    @property
    def T(self):
        '''transpose'''
        return pn.Grid(*list(self)[::-1])

    @property
    def initialized(self):
        '''
        wether the gri is set or not
        '''
        return self.ndim > 0 and all([edge is not None for edge in self.edges])

    @property
    def regular(self):
        '''true is all dimensions are reguallarly spaced'''
        return all([d.regular for d in self])

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
        return [d.var for d in self]

    @property
    def edges(self):
        '''
        all edges
        '''
        return [dim.edges for dim in self]

    @property
    def points(self):
        '''
        all points
        '''
        return [dim.points for dim in self]

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
        return np.product([len(d) for d in self])

    def __len__(self):
        return self.ndim

    def __str__(self):
        '''
        string representation
        '''
        strs = []
        for dim in self:
            strs.append('%s'%dim)
        return '\n'.join(strs)

    def __repr__(self):
        strs = []
        strs.append('Grid(')
        for dim in self:
            strs.append('%s,'%dim.__repr__())
        strs[-1] += ')'
        return '\n'.join(strs)

    def __iter__(self):
        '''
        iterate over dimensions
        '''
        return iter(self.dims)

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
        if isinstance(item, str):
            # by name
            if not item in self.vars:
                # if it does not exist, add empty dim: ToDo: really?
                print('needs to be cjecked, is weird behaviour')
                self.add_dim(item)
            idx = self.vars.index(item)
            return self.dims[idx]

        elif isinstance(item, Number):
            return self[(item,)]
        elif isinstance(item, slice):
            return self[(item,)]
        elif isinstance(item, list):
            if all([isinstance(i, str) for i in item]):
                new_obj = pn.grid.Grid()
                for var in item:
                    new_obj.dims.append(self[var])
                return new_obj
            elif all([isinstance(i, int) for i in item]):
                return self[(item,)]
            else:
                raise IndexError('Cannot process list of indices %s'%item)
        elif isinstance(item, tuple):
            if all([isinstance(i, str) for i in item]):
                return self[list(item)]
            new_obj = pn.grid.Grid()
            for i in range(len(self)): 
                if i < len(item):
                    assert item[i] is not Ellipsis
                    if isinstance(item[i], int):
                        # we can skip this dimesnion, as it is one element
                        continue
                    new_obj.dims.append(self.dims[i][item[i]])
                else:
                    new_obj.dims.append(self.dims[i])
            return new_obj



        elif isinstance(item, Iterable):
            # todo
            new_dims = []
            for it in item:
                new_dims.append(self[it])
            return pn.Grid(*new_dims)
        #elif isinstance(item, slice):
        #    new_names = list(self.dims.keys())[item]
        #    return pn.Grid(*new_names)
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

        # array holding raveld indices
        multi_index = [digitize_inclusive(sample[i], self.edges[i]) for i in range(self.ndim)]
        return np.ravel_multi_index(multi_index, [d+2 for d in self.shape])

def digitize_inclusive(x, bins):
    idx = np.digitize(x, bins)
    idx[x == bins[-1]] -= 1
    return idx


def test():
    a = Grid(var='a', edges=np.linspace(0, 1, 2))
    print(a)
    print(a.vars)
    a['x'].edges = np.linspace(0, 10, 11)
    a['y'].points = np.logspace(-1, 1, 20)
    print(a['x'].points)
    print(a['x'].edges)
    print(a['x', 'y'])

if __name__ == '__main__':
    test()
