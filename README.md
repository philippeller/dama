# Pynocular

A consistant and pythonic way to handle different datasaets and translations between them.
A dataset can be a simple pandas datafrane or other colum/row data, or it can be data on a grid.

The key feature of pynocular is seamless translations from data represenation into any other. See `notebooks/example.ipynb`

Convenience `pyplot` plotting functions are also defined, in order to produce standard plots without the hassle of figuring out the correct input to the standard matplotlib functions.

## Installation

* `git clone git@github.com:philippeller/pynocular.git`
* `cd pynocular`
* `pip install .`

# Intro

Different data representations are available, one being the `GridArray`:
```python
import pynocular as pn

a = pn.GridArray(np.random.rand(200).reshape(20,10))
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
