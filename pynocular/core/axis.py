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

class Axis(object):
    '''
    Class to hold a single Axis of a Grid
    which can have points and/or edges
    '''
    def __init__(self, var=None, edges=None, points=None, nbins=10):

        self.var = var
        self._edges = pn.Edges(edges)
        self._points = points
        self._nbins = nbins

    def __len__(self):
        if self._points is not None:
            return len(self._points)
        elif self._edges is not None:
            return len(self._edges)
        return None

    def __str__(self):
        strs = []
        strs.append('(points) %s'%(self._points))
        strs.append('(edges)  %s'%(self._edges))
        strs.append('(nbins)  %s'%(self.nbins))
        return '\n'.join(strs)

    def __repr__(self):
        strs = []
        strs.append('Axis("%s",'%self.var)
        strs.append('points = %s,'%(self._points.__repr__()))
        strs.append('edges = %s)'%(self._edges.__repr__()))
        strs.append('nbins = %s)'%(self.nbins))
        return '\n'.join(strs)

    def __getitem__(self, idx):
        if idx is Ellipsis:
            return self

        new_obj = pn.Axis()
        new_obj.var = self.var
        if self._edges._edges is not None:
            new_obj._edges = self._edges[idx]
        if self._points is not None:
            new_obj._points = self._points[idx]
        new_obj._nbins = self._nbins
        return new_obj

    @property
    def initialized(self):
        '''wether axis is initialized'''
        return self._edges._edges is not None or self._points is not None


    @property
    def has_data(self):
        '''
        True if either edges or points are not None
        '''
        return (self._edges.edges is not None) or (self._points is not None)

    def __eq__(self, other):
        if not type(self) == type(other):
            return False
        equal = self.var == other.var
        equal = equal and self._edges == other._edges
        equal = equal and np.all(np.equal(self._points, other._points))
        return equal and self._nbins == other._nbins

    @property
    def regular(self):
        '''True if spacing of egdges and/or points is regular'''
        regular = True
        if self._points is not None:
            regular = regular and np.equal.reduce(np.diff(self._points))
        if self._edges.edges is not None:
            regular = regular and self._edges.regular
        return regular
    
    @property
    def edges(self):
        if self._edges._edges is not None:
            return self._edges
        elif self._points is not None:
            return pn.Edges(points=self._points)
        return None

    @edges.setter
    def edges(self, edges):
        edges = pn.Edges(edges)
        if self.has_data:
            if not len(edges) == len(self):
                raise IndexError('incompatible length of edges')
        self._edges = edges

    @property
    def points(self):
        if self._points is not None:
            return self._points
        elif self._edges is not None:
            return self._edges.points
        return None

    @points.setter
    def points(self, points):
        if self.has_data:
            if not len(points) == len(self):
                raise IndexError('incompatible length of points')
        self._points = points

    @property
    def squeezed_edges(self):
        return self.edges.squeezed_edges

    @property
    def nbins(self):
        if self._points is None and self._edges._edges is None:
            return self._nbins
        else:
            return len(self.points)

    @nbins.setter
    def nbins(self, nbins):
        if self._points is None and self._edges is None:
            self._nbins = nbins
        else:
            raise ValueError('Cannot set n since bins are already defined')

