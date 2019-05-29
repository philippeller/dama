from __future__ import absolute_import
from numbers import Number
import numpy as np
import pynocular as pn
import tabulate

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


N_MAX = 6
PRECISION = 2

def as_str(a):
    '''simple string formatter
    
    Parameters
    ----------
    a : Array, Number

    Retrurns
    --------
    string : str
    '''
    if isinstance(a, np.ma.core.MaskedConstant):
        return str(a)
    if isinstance(a, Number):
        return ('%.'+str(PRECISION)+'g')%a
    return np.array2string(np.asanyarray(a), precision=PRECISION, threshold=2, edgeitems=2)


def table_labels(grid, dim):
    '''generate gird labels

    Parameters
    ----------
    grid : pn.Grid
    dim : str

    Returns
    -------
    labels : list of str
    '''
    if grid[grid.vars[dim]]._edges is not None:
        #return ['<b>%s</b>'%(as_str(grid[grid.vars[dim]].edges[i]), as_str(grid[grid.vars[dim]].edges[i+1])) for i in range(grid.shape[dim])]
        return ['<b>%s</b>'%as_str(grid[grid.vars[dim]].edges[i]) for i in range(grid.shape[dim])]
    else:
        return ['<b>%s</b>'%as_str(grid[grid.vars[dim]].points[i]) for i in range(grid.shape[dim])]
