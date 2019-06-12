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

Different data representations are available, one being the GridArray:
```python
import pynocular as pn

a = pn.GridArray(np.random.rand(200).reshape(20,10))```
```
that supports many numpy features. it's `grid` attribute is another class, here it was just instantiated with default values, but it can be used to specify the axes.

## Translation methods

One of the stregth are the avilable translation methods, e.g.

```python

p = pn.PointData(x = [1, 2, 6, 7.33, ...], a = [55, 1e6, 3, 3.3, ...]

h = p.histogram(x = 10)
```
