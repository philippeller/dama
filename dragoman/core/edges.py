from __future__ import absolute_import
from numbers import Number
from collections.abc import Iterable

import dragoman as dm

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


class Edges(object):
    '''Holding binning edges'''
    def __init__(self, *args, delta=None, points=None, **kwargs):
        '''
        Paramters
        ---------
        args : either (nbins,) or (min, max, nbins) or (array,)
        delta : float (optional):
            specify the the delta between two subsequent edges
        points : array (optional)
            create edges from points

        '''
        #print(args, kwargs)
        self._edges = None
        if points is not None:
            self._add_edges(self.edges_from_points(points))
        elif len(args) == 1 and isinstance(args[0], dm.Edges):
            self._add_edges(args[0].edges)
        elif len(args) == 1 and isinstance(args[0], np.ndarray):
            self._add_edges(args[0])
        elif len(args) == 1 and isinstance(args[0], list):
            self._add_edges(np.array(args[0]))
        elif len(args) == 0 and delta is None and len(kwargs) == 0:
            pass
        elif len(args) == 1 and args[0] is None:
            pass
        else:
            raise NotImplementedError()

    def __array__(self):
        if len(self) == 1:
            return self._edges[0]
        return self._edges

    def __repr__(self):
        return 'edges: ' + self._edges.__repr__()

    def __str__(self):
        return self._edges.__str__()

    def edges_from_points(self, points):
        '''
        create edges around points
        '''
        if len(points) == 1:
            # in this case we cannot do a delta, just make the binwidth = 1.0 by default
            return np.array([[points[0] - 0.5, points[0] + 0.5]])
        diff = np.diff(points) / 2.
        return np.concatenate(
            [
                [points[0] - diff[0]], points[:-1] + diff,
                [points[-1] + diff[-1]]
                ]
            )

    def min(self):
        return np.min(self.edges)

    def max(self):
        return np.max(self.edges)

    @property
    def points(self):
        '''
        create points from centers between edges
        '''
        points = np.average(self._edges, axis=1)
        if isinstance(points, Number):
            return np.array(points)
        return points

    def _add_edges(self, edges):
        if edges.ndim == 2:
            self._edges = edges
        elif edges.ndim == 1:
            assert len(edges) > 1, 'Edges must be at least length 2'
            self._edges = np.empty((len(edges) - 1, 2))
            self._edges[:, 0] = edges[:-1]
            self._edges[:, 1] = edges[1:]
        else:
            raise ValueError()

    @property
    def consecutive(self):
        '''True if edges consecutive, i.e. no gaps'''
        return np.all(self._edges[1:, 0] == self._edges[:-1, 1])

    @property
    def regular(self):
        '''True if spacing of edges is regular'''
        return np.equal.reduce(self._edges[:, 1] - self._edges[:, 0])

    def __len__(self):
        if self._edges is None:
            return 0
        return self._edges.shape[0]

    @property
    def edges(self):
        '''return edges'''
        return self._edges

    @edges.setter
    def edges(self, val):
        self._add_edges(val)

    @property
    def width(self):
        '''return the bin width'''
        return np.abs(self._edges[:, 1] - self._edges[:, 0])

    @property
    def squeezed_edges(self):
        '''return edges in squeezed form
        that is a 1d array as used in historgam functions and the like
        only available for consecutive edges'''
        if self.consecutive:
            squeezed_edges = np.empty(len(self) + 1)
            squeezed_edges[:-1] = self._edges[:, 0]
            squeezed_edges[-1] = self._edges[-1, 1]
            return squeezed_edges
        else:
            raise ValueError(
                'Can only provide squeezed edges for consecutive edges'
                )

    def __getitem__(self, idx):
        if idx is Ellipsis:
            return self
        new_edges = self._edges[idx]
        if np.isscalar(new_edges):
            return new_edges
        return dm.Edges(new_edges)

    def __eq__(self, other):
        return np.all(np.equal(self._edges, other._edges))


def test_edges():
    edges = Edges(np.arange(10))
    assert np.allclose(edges.points, np.linspace(0.5, 8.5, 9))
