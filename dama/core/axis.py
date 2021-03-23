from __future__ import absolute_import
from numbers import Number
from collections import OrderedDict
from collections.abc import Iterable

import dama as dm

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
    def __init__(
        self, var=None, edges=None, points=None, nbins=None, log=None, label=None, **kwargs
        ):

        if len(kwargs) == 1:
            assert var is None and edges is None and points is None
            var, val = list(kwargs.items())[0]
            if isinstance(val, (list, np.ndarray)):
                points = np.asanyarray(val)
            elif isinstance(val, dm.Edges):
                edges = val
            elif isinstance(val, Number):
                nbins = val
            else:
                raise ValueError()

        self.var = var
        self.label = label
        self._edges = edges
        self._points = points
        self._nbins = nbins
        self._log = log

    @property
    def has_points(self):
        '''True if points are set'''
        return self._points is not None

    @property
    def log(self):
        if self._log is not None:
            return self._log
        if self.has_edges:
            return self._edges.log
        if self.has_points and np.all(self._points > 0):
            d = np.diff(np.log(self._points))
            return np.allclose(d[0], d)
        return False

    @log.setter
    def log(self, log):
        if self.has_edges:
            self._edges.log = log
        self._log = log

    @property
    def has_edges(self):
        '''True if edges are set'''
        return self._edges is not None

    def __len__(self):
        if self._points is not None:
            return len(self._points)
        elif self._edges is not None:
            return len(self._edges)
        return None

    def __str__(self):
        strs = []
        strs.append('(points) %s' % (self._points))
        strs.append('(edges)  %s' % (self._edges))
        strs.append('(nbins)  %s' % (self.nbins))
        return '\n'.join(strs)

    def __repr__(self):
        strs = []
        strs.append('Axis("%s",' % self.var)
        strs.append('points = %s,' % (self._points.__repr__()))
        strs.append('edges = %s)' % (self._edges.__repr__()))
        strs.append('nbins = %s)' % (self.nbins))
        return '\n'.join(strs)

    def __getitem__(self, idx):

        idx = self.convert_slice(idx)

        if idx is Ellipsis:
            return self

        new_obj = dm.Axis()
        new_obj.var = self.var
        if self._edges._edges is not None:
            new_obj._edges = self._edges[idx]
        if self._points is not None:
            new_obj._points = self._points[idx]
        new_obj._nbins = self._nbins
        return new_obj

    def convert_slice(self, idx):
        '''Convert slice

        idx : int, float, slice, Ellipsis
        '''
        if isinstance(idx, (int, np.integer, type(Ellipsis))):
            return idx

        if isinstance(idx, float):
            return self.convert_index(idx)

        if isinstance(idx, (list, np.ndarray)):
            new_indices = []
            for i in idx:
                new_indices.append(self.convert_index(i))
            return new_indices

        if isinstance(idx, slice):
            start = self.convert_index(idx.start)
            stop = self.convert_index(idx.stop)

            return slice(start, stop, idx.step)

        raise IndexError(idx, type(idx))

    def convert_index(self, idx):
        if idx is None:
            return None

        if isinstance(idx, (int, np.integer)):
            return idx

        idx = self.compute_indices(idx)
        if idx >= 0:
            return idx

        raise IndexError('Index out of range')

    def compute_indices(self, sample):
        '''compute bin indices for a sample, return -1 if outise of bins

        Parameters
        ----------
        sample : array, float

        Returns
        -------

        indices : array, int
        '''
        if not self.edges.consecutive:
            raise NotImplementedError()

        bins = self.edges.squeezed_edges
        if np.isscalar(sample):
            if sample == bins[-1]:
                return len(self)
            elif sample < bins[0] or sample > bins[-1]:
                return -1
            else:
                return np.digitize(sample, bins) - 1

        idx = np.digitize(sample, bins) - 1
        # make inclusive right edge
        idx[sample == bins[-1]] -= 1
        # set overflow bin to idx -1
        idx[idx == len(self)] = -1
        return idx

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
            if self.log:
                d = np.diff(self.log(self._points))
            else:
                d = np.diff(self._points)
            regular = regular and np.allclose(d[0], d)
        if self._edges.edges is not None:
            regular = regular and self._edges.regular
        return regular

    @property
    def edges(self):
        if self.has_edges and self._edges._edges is not None:
            return self._edges
        if self.has_points:
            return dm.Edges(points=self._points, log=self.log)
        return None

    @edges.setter
    def edges(self, edges):
        edges = dm.Edges(edges)
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
        if not self.initialized:
            self._nbins = nbins
        else:
            raise ValueError('Cannot set n since bins are already defined')
