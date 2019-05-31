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


N_MAX = 12
'''max rows to display in html formatting'''
PRECISION = 2
'''signfificant figures to dispay'''

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


def make_table_labels(axis):
    '''generate gird labels

    Parameters
    ----------
    axis : pn.Axis

    Returns
    -------
    labels : list of str
    '''
    if axis._points is None:
        if len(axis) <= N_MAX:
            return ['<b>%s</b>'%as_str(axis.edges[i]) for i in range(len(axis))]
        else:
            labels = ['<b>%s</b>'%as_str(axis.edges[i]) for i in range(N_MAX//2)]
            labels += ['...']
            labels += ['<b>%s</b>'%as_str(axis.edges[i]) for i in range(len(axis)-N_MAX//2, len(axis))]
            return labels
    # ToDo: a bit too much copy-paste going on here...
    elif axis._edges._edges is None:
        if len(axis) <= N_MAX:
            return ['<b>%s</b>'%as_str(axis.points[i]) for i in range(len(axis))]
        else:
            labels = ['<b>%s</b>'%as_str(axis.points[i]) for i in range(N_MAX//2)]
            labels += ['...']
            labels += ['<b>%s</b>'%as_str(axis.points[i]) for i in range(len(axis)-N_MAX//2, len(axis))]
            return labels
    else:
        if len(axis) <= N_MAX:
            return ['<b>[%s | %s | %s]</b>'%(as_str(axis.edges[i,0]), as_str(axis.points[i]), as_str(axis.edges[i,1])) for i in range(len(axis))]
        else:
            labels = ['<b>[%s | %s | %s]</b>'%(as_str(axis.edges[i,0]), as_str(axis.points[i]), as_str(axis.edges[i,1])) for i in range(N_MAX//2)]
            labels += ['...']
            labels += ['<b>[%s | %s | %s]</b>'%(as_str(axis.edges[i,0]), as_str(axis.points[i]), as_str(axis.edges[i,1])) for i in range(len(axis)-N_MAX//2, len(axis))]
            return labels



def make_table_row(data):
    '''forma a simgle table row'''
    row = ['<b>%s</b>'%data.name]
    array = np.array(data)
    if array.shape[0] <= N_MAX:
        row += [as_str(v) for v in array]
    else:
        row += [as_str(v) for v in array[:N_MAX//2]]
        row += ['...']
        row += [as_str(v) for v in array[-N_MAX//2:]]

    return row

def make_2d_table(data):
    x_labels = make_table_labels(data.grid.axes[0])
    y_labels = make_table_labels(data.grid.axes[1])

    table = []
    table.append(['<b>%s \\ %s</b>'%(data.grid.vars[1], data.grid.vars[0])] + x_labels)
    for i in range(len(y_labels)):
        table.append([y_labels[i]] + [0] * (len(x_labels)))

    n_data_cols, n_data_rows = data.shape[:2]

    if n_data_rows > N_MAX:
        for i in range(1, min(n_data_cols, 2*(N_MAX//2) + 1) + 1):
            table[N_MAX//2+1][i] = '...'
        data_row_idices = list(range(N_MAX//2)) + list(range(n_data_rows - N_MAX//2, n_data_rows))
        table_row_idices = list(range(N_MAX//2)) + list(range(N_MAX//2+1, 2*(N_MAX//2) + 1))
    else:
        data_row_idices = list(range(n_data_rows))
        table_row_idices = list(range(n_data_rows))

    if n_data_cols > N_MAX:
        for i in range(1, min(n_data_rows, 2*(N_MAX//2) + 1) + 1):
            table[i][N_MAX//2+1] = '...'
        data_col_idices = list(range(N_MAX//2)) + list(range(n_data_cols - N_MAX//2, n_data_cols))
        table_col_idices = list(range(N_MAX//2)) + list(range(N_MAX//2+1, 2*(N_MAX//2) + 1))
    else:
        data_col_idices = list(range(n_data_cols))
        table_col_idices = list(range(n_data_cols))

    for i, r_idx in zip(table_row_idices, data_row_idices):
        for j, c_idx in zip(table_col_idices, data_col_idices):
            table[i+1][j+1] = get_item(data, (c_idx, r_idx))
            
    return table


def get_item(data, idx):
    '''Get a string formatted item from a GridArray or gridData object at index idx'''
    if isinstance(data, pn.GridArray):
        return as_str(data.data[idx])
    elif isinstance(data, pn.GridData):
        # collect all items
        all_data = []
        for d in data:
            all_data.append('%s = %s'%(d.name, as_str(d.data[idx])))
        return '<br>'.join(all_data)


def format_html(data):
    if isinstance(data, pn.PointData):
        table = [make_table_row(a) for a in data]
        return tabulate.tabulate(table, tablefmt='html')

    if isinstance(data, pn.PointDataDim):
        table = [make_table_row(data)]
        return tabulate.tabulate(table, tablefmt='html')

    if isinstance(data, pn.GridArray):
        if data.naxes == 2:
            table = make_2d_table(data)
        
        elif data.naxes == 1:
            table = []
            table.append(['<b>%s</b>'%data.grid.vars[0]] + make_table_labels(data.grid.axes[0]))
            table.append(make_table_row(data))

        else:
            return None

        return tabulate.tabulate(table, tablefmt='html')


    if isinstance(data, pn.GridData):
        if data.grid.naxes == 2:
            table = make_2d_table(data)
        
        elif data.ndim == 1:
            table = []
            table.append(['<b>%s</b>'%data.grid.vars[0]] + make_table_labels(data.grid.axes[0]))
            for d in data:
                table.append(make_table_row(d))

        return tabulate.tabulate(table, tablefmt='html')

    return data.__repr__()
