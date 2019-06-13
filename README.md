# Pynocular

A consistant and pythonic way to handle different datasaets and translations between them.
A dataset can be a simple pandas datafrane or other colum/row data, or it can be data on a grid.

The key feature of pynocular is seamless translations from data represenation into any other. See `notebooks/example.ipynb`

Convenience `pyplot` plotting functions are also defined, in order to produce standard plots without the hassle of figuring out the correct input to the standard matplotlib functions.

## Installation

* `git clone git@github.com:philippeller/pynocular.git`
* `pip install pynocular`

# Intro

Different data representations are available, one being the `GridArray`:
```python
import pynocular as pn

a = pn.GridArray(np.random.rand(100).reshape(20, 5))
a
>>> +-------+-------+-------+-------+--------+-------+--------+-----+--------+--------+--------+---------+----------+--------+
>>> | y \ x | 0     | 1     | 2     | 3      | 4     | 5      | ... | 14     | 15     | 16     | 17      | 18       | 19     |
>>> +-------+-------+-------+-------+--------+-------+--------+-----+--------+--------+--------+---------+----------+--------+
>>> | 0     | 0.279 | 0.426 | 0.353 | 0.315  | 0.446 | 0.0913 | ... |  0.712 |  0.564 |  0.24  |  0.925  |  0.00417 |  0.463 |
>>> +-------+-------+-------+-------+--------+-------+--------+-----+--------+--------+--------+---------+----------+--------+
>>> | 1     | 0.417 | 0.74  | 0.267 | 0.473  | 0.103 | 0.885  | ... |  0.743 |  0.991 |  0.833 |  0.223  |  0.0862  |  0.3   |
>>> +-------+-------+-------+-------+--------+-------+--------+-----+--------+--------+--------+---------+----------+--------+
>>> | 2     | 0.918 | 0.752 | 0.689 | 0.876  | 0.957 | 0.931  | ... |  0.759 |  0.919 |  0.9   |  0.498  |  0.644   |  0.834 |
>>> +-------+-------+-------+-------+--------+-------+--------+-----+--------+--------+--------+---------+----------+--------+
>>> | 3     | 0.836 | 0.626 | 0.737 | 0.584  | 0.33  | 0.414  | ... |  0.874 |  0.203 |  0.299 |  0.0643 |  0.372   |  0.604 |
>>> +-------+-------+-------+-------+--------+-------+--------+-----+--------+--------+--------+---------+----------+--------+
>>> | 4     | 0.896 | 0.494 | 0.489 | 0.0502 | 0.295 | 0.767  | ... |  0.388 |  0.964 |  0.802 |  0.385  |  0.0112  |  0.522 |
>>> +-------+-------+-------+-------+--------+-------+--------+-----+--------+--------+--------+---------+----------+--------+
```

that supports many numpy features. 

```
np.sum(a, axis='x')
np.ones_like(a)
a[a > 0.5] = 0.5
a.T
a[::-2, [1,3,5]]
```
Its `grid` attribute is another object, here it was just instantiated with default values (i.e. axis "x" and "y" with points `[0, 1, 2, ...]`), but one can explicitly specify the axes with points and/or edges etc to be used.

The various objects include:
* `GridArray` : holds a single gridded array with corresponding axes
* `GridData` : a collection (container class) of `GridArray`s that share a common grid
* `PointArray` : a single point-like array (like any old np.ndarray)
* `PointData` : collection of same length `PointArray`s (similar to a pandas DataFrame)
* `Grid` : a grid holding several `Axis`
* `Axis` : a 1-d axis of a grid
* `Edges` : binning edges

## Translation methods

One of the stregth are the avilable translation methods, e.g.

```python

p = pn.PointData(x = [1, 2, 6, 7.33, ...], a = [55, 1e6, 3, 3.3, ...])
h = p.histogram(x = 10)
```
or
```python
g = p.interp(x = np.linspace(0,10,1000))
```
etc.

currently there are the follwoing translations methods:
* histogram
* binswise
* interp
* kde
* lookup
* resample

## Other things

Objects provide matplotlib plotting methods and jupyterlab HTML output for convenience.

```python
a.plot()
```
