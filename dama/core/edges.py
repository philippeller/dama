from __future__ import absolute_import
from numbers import Number
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
        elif len(args) == 3:
            self._add_edges(np.linspace(*args))
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


def BayesianEdges(sample):
    """Bayesian Blocks Implementation

    By Jake Vanderplas.  License: BSD
    Based on algorithm outlined in http://adsabs.harvard.edu/abs/2012arXiv1207.5578S

    Edit: Small bugfix (P. Eller)

    Parameters
    ----------
    sample : ndarray, length N
        data to be histogrammed
        
    weights : ndarray, length N (optional)
        weights of data

    Returns
    -------
    bins : ndarray
        array containing the (N+1) bin edges

    Notes
    -----
    This is an incomplete implementation: it may fail for some
    datasets.  Alternate fitness functions and prior forms can
    be found in the paper listed above.
    """
    # copy and sort the array
    sample = np.sort(sample)
    N = sample.size

    # create length-(N + 1) array of cell edges
    edges = np.concatenate([sample[:1],
                            0.5 * (sample[1:] + sample[:-1]),
                            sample[-1:]])
    block_length = sample[-1] - edges

    
    #print(np.diff(edges))
    #print(block_length)
    
    # arrays needed for the iteration
    nn_vec = np.ones(N)
    best = np.zeros(N, dtype=float)
    last = np.zeros(N, dtype=int)

    
    #-----------------------------------------------------------------
    # Start with first data cell; add one cell at each iteration
    #-----------------------------------------------------------------
    for K in range(N):
        # Compute the width and count of the final bin for all possible
        # locations of the K^th changepoint
        width = block_length[:K + 1] - block_length[K + 1]
        count_vec = np.cumsum(nn_vec[:K + 1][::-1])[::-1]
        # evaluate fitness function for these possibilities
        
        fit_vec = count_vec * (np.log(count_vec) - np.log(width))
        fit_vec -= 4  # 4 comes from the prior on the number of changepoints
        fit_vec[1:] += best[:K]

        # find the max of the fitness: this is the K^th changepoint
        i_max = np.argmax(fit_vec)
        last[K] = i_max
        best[K] = fit_vec[i_max]
        
    #-----------------------------------------------------------------
    # Recover changepoints by iteratively peeling off the last block
    #-----------------------------------------------------------------
    change_points =  np.zeros(N+1, dtype=int)
    i_cp = N+1
    ind = N
    while True:
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 0:
            break
        ind = last[ind - 1]
        
    change_points = change_points[i_cp:]

    return dm.Edges(edges[change_points])

def QuantileEdges(sample, n=10):
    """
    Return binning edges each containing the same quantile of the sample
    """
    qs = np.linspace(0, 1, n+1)
    edges = np.quantile(sample, qs)
    return dm.Edges(edges)

def test_edges():
    edges = Edges(np.arange(10))
    assert np.allclose(edges.points, np.linspace(0.5, 8.5, 9))
