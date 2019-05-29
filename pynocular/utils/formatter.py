from __future__ import absolute_import
from numbers import Number
import numpy as np
import pynocular as pn
import tabulate
import copy

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


def format_html(data):
    if isinstance(data, pn.PointData):
        table = [['<b>%s</b>'%a.name] + [as_str(d) for d in np.array(a)] for a in data]
        return tabulate.tabulate(table, tablefmt='html')

    if isinstance(data, pn.PointDataDim):
        table = [['<b>%s</b>'%data.name] + [as_str(v) for v in np.array(data)]]
        return tabulate.tabulate(table, tablefmt='html')

    if isinstance(data, pn.GridArray):
        if data.naxes == 2:
            table_x = [0] * (data.shape[0] + 1)
            table = [copy.copy(table_x) for _ in range(data.shape[1] + 1)]
            
            table[0][0] = '<b>%s \\ %s</b>'%(data.grid.vars[1], data.grid.vars[0])
            
            x_labels = table_labels(data.grid, 0)
            y_labels = table_labels(data.grid, 1)
                        
            for i in range(data.shape[0]):
                table[0][i+1] = x_labels[i]
            for i in range(data.shape[1]):
                table[i+1][0] = y_labels[i]
                
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    table[j+1][i+1] = as_str(data.data[i, j])
                    
            return tabulate.tabulate(table, tablefmt='html')
        
        elif data.naxes == 1:
            table_x = [0] * (data.shape[0] + 1)
            table = [copy.copy(table_x) for _ in range(2)]
            table[0][0] = '<b>%s</b>'%data.grid.vars[0]
            table[1][0] = '<b>%s</b>'%data.name
            
            x_labels = table_labels(data.grid, 0)

            
            for i in range(data.shape[0]):
                table[0][i+1] = x_labels[i]
                table[1][i+1] = as_str(data.data[i])

            return tabulate.tabulate(table, tablefmt='html')
        
    if isinstance(data, pn.GridData):
        if data.grid.naxes == 2:
            table_x = [0] * (data.grid.shape[0] + 1)
            table = [copy.copy(table_x) for _ in range(data.grid.shape[1] + 1)]
            
            table[0][0] = '<b>%s \\ %s</b>'%(data.grid.vars[1], data.grid.vars[0])
            
            x_labels = table_labels(data.grid, 0)
            y_labels = table_labels(data.grid, 1)
                        
            for i in range(data.shape[0]):
                table[0][i+1] = x_labels[i]
            for i in range(data.shape[1]):
                table[i+1][0] = y_labels[i]
                
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    all_data = []
                    #for var in data.data_vars:
                    for d in data:
                        all_data.append('%s = %s'%(d.name, as_str(d.data[i, j])))
                    table[j+1][i+1] = '<br>'.join(all_data)
                    
            return tabulate.tabulate(table, tablefmt='html')
        
        elif data.ndim == 1:
            table_x = [0] * (data.grid.shape[0] + 1)
            table = [copy.copy(table_x) for _ in range(len(data.data_vars)+1)]
            table[0][0] = '<b>%s</b>'%data.grid.vars[0]
            for i, var in enumerate(data.data_vars):
                table[i+1][0] = '<b>%s</b>'%var
            
            x_labels = table_labels(data.grid, 0)
            
            for i in range(data.shape[0]):
                table[0][i+1] = x_labels[i]
                for j, d in enumerate(data):
                    table[j+1][i+1] = as_str(d.data[i])

            return tabulate.tabulate(table, tablefmt='html')
        
    return data.__repr__()
