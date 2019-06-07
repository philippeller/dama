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


class Grid(object):
    '''
    Class to hold a number of axes
    '''
    def __init__(self, *args, **kwargs):
        '''
        Paramters:
        ----------
        args : Axis or Grid object, or list thereof

        an axis can also be given by kwargs
        kwargs : str,Number or str,array

        '''
        self.axes = [] 

        for d in args:
            self.add_axis(d)

        for d, x in kwargs.items():
            if isinstance(x, Number):
                self.add_axis(pn.Axis(var=d, nbins=x))
            elif isinstance(x, (list, np.ndarray)):
                self.add_axis(pn.Axis(var=d, points=np.asanyarray(x)))
            elif isinstance(x, pn.edges):
                self.add_axis(pn.Axis(var=d, edges=x))
            else:
                raise ValueError('Did not understand %s : %s'%(d, x))

    def initialize(self, source):
        '''Method to initialize the grid if grid is not fully set up
        it derive information from source
        
        Parameters
        ----------
        source : pn.GridData, pn.PointData
        
        '''
        # check dest grid is set up, otherwise do so
        for var in self.vars:
            if not self[var].initialized:
                # check if it might be from a grid
                if isinstance(source, pn.GridData):
                    if var in source.grid.vars:
                        if isinstance(self[var].nbins, float):
                            # this measn we want to multiply the old nbins
                            new_nbins = int(source.grid[var].nbins * self[var].nbins)
                        else:
                            new_nbins = self[var].nbins
                        self[var].edges = np.linspace(source.grid[var].edges.min(), source.grid[var].edges.max(), new_nbins+1)
                        continue
                # in this case it's pointdata
                self[var].edges = np.linspace(np.nanmin(source[var]), np.nanmax(source[var]), self[var].nbins+1)

    def add_axis(self, axis):
        '''
        add aditional Axis

        Paramters:
        ----------
        axis : Axis or dict or basestring

        in case of a basestring, a new empty axisension gets added
        '''
        if isinstance(axis, pn.Axis):
            assert axis.var not in self.vars
            self.axes.append(axis)
        elif isinstance(axis, dict):
            axis = pn.Axis(**axis)
            self.add_axis(axis)
        elif isinstance(axis, str):
            new_axis = pn.Axis(var=axis)
            self.add_axis(new_axis)
        else:
            raise TypeError('Cannot add type %s'%type(axis))

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
        wether the grid is set or not
        '''
        return self.nax > 0 and all([d.initialized for d in self])

    @property
    def regular(self):
        '''true is all axisensions are reguallarly spaced'''
        return all([d.regular for d in self])

    def consecutive(self):
        '''true is all edges are consecutive'''
        return all([d.edges.consecutive for d in self])

    @property
    def nax(self):
        '''
        number of grid axisensions
        '''
        return len(self.axes)

    @property
    def vars(self):
        '''
        grid axisension variables
        '''
        return [d.var for d in self]

    @property
    def edges(self):
        '''
        all edges
        '''
        return [axis.edges for axis in self]

    @property
    def squeezed_edges(self):
        '''
        all squeezed edges
        '''
        return [axis.squeezed_edges for axis in self]

    @property
    def points(self):
        '''
        all points
        '''
        return [axis.points for axis in self]

    @property
    def point_meshgrid(self):
        return np.meshgrid(*self.points, indexing='ij')

    @property
    def point_mgrid(self):
        return [d.T for d in self.point_meshgrid]

    @property
    def edge_meshgrid(self):
        return np.meshgrid(*self.squeezed_edges)

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
        return self.nax

    def __str__(self):
        '''
        string representation
        '''
        strs = []
        for axis in self:
            strs.append('%s'%axis)
        return '\n'.join(strs)

    def __repr__(self):
        strs = []
        strs.append('Grid(')
        for axis in self:
            strs.append('%s,'%axis.__repr__())
        strs[-1] += ')'
        return '\n'.join(strs)

    def __iter__(self):
        '''
        iterate over axisensions
        '''
        return iter(self.axes)

    @property
    def shape(self):
        '''
        shape
        '''
        shape = []
        for axis in self:
            shape.append(len(axis))
        return tuple(shape)

    def __getitem__(self, item):
        '''
        item : int, str, slice, ierable
        '''
        if isinstance(item, str):
            # by name
            if not item in self.vars:
                # if it does not exist, add empty axis: ToDo: really?
                print('needs to be checked, is weird behaviour')
                self.add_axis(item)
            idx = self.vars.index(item)
            return self.axes[idx]

        elif isinstance(item, Number):
            return self[(item,)]
        elif isinstance(item, slice):
            return self[(item,)]
        elif isinstance(item, list):
            if all([isinstance(i, str) for i in item]):
                new_obj = pn.grid.Grid()
                for var in item:
                    new_obj.axes.append(self[var])
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
                        # we can skip this axisesnion, as it is one element
                        continue
                    new_obj.axes.append(self.axes[i][item[i]])
                else:
                    new_obj.axes.append(self.axes[i])
            return new_obj



        elif isinstance(item, Iterable):
            # todo
            new_axes = []
            for it in item:
                new_axes.append(self[it])
            return pn.Grid(*new_axes)
        #elif isinstance(item, slice):
        #    new_names = list(self.axes.keys())[item]
        #    return pn.Grid(*new_names)
        else:
            raise KeyError('Cannot get key from %s'%type(item))

    def __setitem__(self, item, val):
        raise AttributeError("to set a grid axisension, specify if it is `points` or `edges`, e.g.:\ngrid['%s'].edges = %s"%(item, val))

    def compute_indices(self, sample):
        '''
        calculate the bin indices for a a given sample
        '''
        if isinstance(sample, np.ndarray):
            assert sample.shape[0] == self.nax
        elif isinstance(sample, list):
            assert len(sample) == self.nax
        
        if not self.consecutive:
            raise NotImplementedError()

        # array holding raveld indices
        multi_index = [digitize_inclusive(sample[i], self.edges[i].squeezed_edges) for i in range(self.nax)]
        return np.ravel_multi_index(multi_index, [d + 2 for d in self.shape])

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
