""" Module for table formatting """
from __future__ import absolute_import

from numbers import Number

import numpy as np
import tabulate

import dragoman as dm

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
'''max rows to display in html formatting'''
PRECISION = 3
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
        return ('%.' + str(PRECISION) + 'g') % a
    return np.array2string(
        np.ma.asarray(a), precision=PRECISION, threshold=2, edgeitems=2
        )


def make_table_labels(axis, bold, brk):
    '''generate gird labels

    Parameters
    ----------
    axis : dm.Axis

    Returns
    -------
    labels : list of str
    '''
    if axis._points is None:
        if len(axis) <= N_MAX:
            return [bold % as_str(axis.edges[i]) for i in range(len(axis))]
        labels = [bold % as_str(axis.edges[i]) for i in range(N_MAX // 2)]
        labels += ['...']
        labels += [
            bold % as_str(axis.edges[i])
            for i in range(len(axis) - N_MAX // 2, len(axis))
            ]
        return labels

    # ToDo: a bit too much copy-paste going on here...
    elif axis._edges._edges is None:
        if len(axis) <= N_MAX:
            return [bold % as_str(axis.points[i]) for i in range(len(axis))]
        labels = [bold % as_str(axis.points[i]) for i in range(N_MAX // 2)]
        labels += ['...']
        labels += [
            bold % as_str(axis.points[i])
            for i in range(len(axis) - N_MAX // 2, len(axis))
            ]
        return labels

    if len(axis) <= N_MAX:
        return [
            bold % (
                '[%s | %s | %s]' % (
                    as_str(axis.edges[i, 0]
                           ), as_str(axis.points[i]), as_str(axis.edges[i, 1])
                    )
                ) for i in range(len(axis))
            ]
    labels = [
        bold % (
            '[%s | %s | %s]' % (
                as_str(axis.edges[i, 0]), as_str(axis.points[i]
                                                 ), as_str(axis.edges[i, 1])
                )
            ) for i in range(N_MAX // 2)
        ]
    labels += ['...']
    labels += [
        bold % (
            '[%s | %s | %s]' % (
                as_str(axis.edges[i, 0]), as_str(axis.points[i]
                                                 ), as_str(axis.edges[i, 1])
                )
            ) for i in range(len(axis) - N_MAX // 2, len(axis))
        ]
    return labels


def make_table_row(name, array, bold, brk):
    '''forma a simgle table row'''
    if name is not None:
        row = [bold % name]
    else:
        row = []
    array = np.ma.asarray(array)
    #if array.ndim == 0:
    #    return [as_str(array)]
    if array.shape[0] <= N_MAX:
        row += [as_str(v) for v in array]
    else:
        row += [as_str(v) for v in array[:N_MAX // 2]]
        row += ['...']
        row += [as_str(v) for v in array[-N_MAX // 2:]]

    return row


def make_2d_table(data, bold, brk):

    x_labels = make_table_labels(data.grid.axes[0], bold, brk)
    y_labels = make_table_labels(data.grid.axes[1], bold, brk)

    table = []
    table.append(
        [bold % ('%s \\ %s' % (data.grid.vars[1], data.grid.vars[0]))] +
        x_labels
        )
    for i in range(len(y_labels)):
        table.append([y_labels[i]] + [0] * (len(x_labels)))

    n_data_cols, n_data_rows = data.shape[:2]

    if n_data_rows > N_MAX:
        for i in range(1, min(n_data_cols, 2 * (N_MAX // 2) + 1) + 1):
            table[N_MAX // 2 + 1][i] = '...'
        data_row_idices = list(range(N_MAX // 2)) + list(
            range(n_data_rows - N_MAX // 2, n_data_rows)
            )
        table_row_idices = list(range(N_MAX // 2)) + list(
            range(N_MAX // 2 + 1, 2 * (N_MAX // 2) + 1)
            )
    else:
        data_row_idices = list(range(n_data_rows))
        table_row_idices = list(range(n_data_rows))

    if n_data_cols > N_MAX:
        for i in range(1, min(n_data_rows, 2 * (N_MAX // 2) + 1) + 1):
            table[i][N_MAX // 2 + 1] = '...'
        data_col_idices = list(range(N_MAX // 2)) + list(
            range(n_data_cols - N_MAX // 2, n_data_cols)
            )
        table_col_idices = list(range(N_MAX // 2)) + list(
            range(N_MAX // 2 + 1, 2 * (N_MAX // 2) + 1)
            )
    else:
        data_col_idices = list(range(n_data_cols))
        table_col_idices = list(range(n_data_cols))

    for i, r_idx in zip(table_row_idices, data_row_idices):
        for j, c_idx in zip(table_col_idices, data_col_idices):
            table[i + 1][j + 1] = get_item(data, (c_idx, r_idx), bold, brk)

    return table


def get_item(data, idx, bold, brk):
    '''Get a string formatted item from a GridArray or gridData object at index idx'''
    if isinstance(data, dm.GridArray):
        return as_str(data[idx])
    elif isinstance(data, dm.GridData):
        # collect all items
        all_data = []
        for n, d in data.items():
            all_data.append('%s = %s' % (n, as_str(d[idx])))
        return brk.join(all_data)


def format_table(data, tablefmt='html'):

    if tablefmt == 'html':
        bold = '<b>%s</b>'
        brk = '<br>'
    else:
        bold = '%s'
        brk = '\n'

    if isinstance(data, dm.PointData):
        table = [make_table_row(n, d, bold, brk) for n, d in data.items()]
        return tabulate.tabulate(table, tablefmt=tablefmt)

    if isinstance(data, dm.PointArray):
        table = [make_table_row(None, data, bold, brk)]
        return tabulate.tabulate(table, tablefmt=tablefmt)

    if isinstance(data, dm.GridArray):
        if data.nax == 2:
            table = make_2d_table(data, bold, brk)

        elif data.nax == 1:
            table = []
            table.append(
                [bold % data.grid.vars[0]] +
                make_table_labels(data.grid.axes[0], bold, brk)
                )
            table.append(make_table_row('', data, bold, brk))

        else:
            return None

        return tabulate.tabulate(table, tablefmt=tablefmt)

    if isinstance(data, dm.GridData):
        if data.grid.nax == 2:
            table = make_2d_table(data, bold, brk)

        elif data.ndim == 1:
            table = []
            table.append(
                [bold % data.grid.vars[0]] +
                make_table_labels(data.grid.axes[0], bold, brk)
                )
            for d in data.items():
                table.append(make_table_row(*d, bold, brk))

        return tabulate.tabulate(table, tablefmt=tablefmt)

    return data.__repr__()
