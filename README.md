# dama - Data Manipulator

The dama python library guides you through your data and translates between different representations.
Its aim is to offer a consistant and pythonic way to handle different datasaets and translations between them.
A dataset can for instance be simple colum/row data, or it can be data on a grid.

One of the key features of dama is the seamless translation from one data represenation into any other. 
Convenience `pyplot` plotting functions are also available, in order to produce standard plots without any hassle.

## Installation

* `pip install dama`

## Getting Started


```python
import numpy as np
import dama as dm
```

### Grid Data

`GridData` is a collection of individual `GridArrays`. Both have a defined `grid`, here we initialize the grid in the constructor through simple keyword arguments resulting in a 2d grid with axes `x` and `y`


```python
g = dm.GridData(x = np.linspace(0,3*np.pi, 30),
                y = np.linspace(0,2*np.pi, 20),
               )
```

Filling one array with some sinusoidal functions, called `a` here


```python
g['a'] = np.sin(g['x']) * np.cos(g['y'])
```

As a shorthand, we can also use attributes instead of items:


```python
g.a = np.sin(g.x) * np.cos(g.y)
```

in 1-d and 2-d they render as html in jupyter notebooks

It can be plotted easily in case of 1-d and 2-d grids


```python
g.plot(cbar=True);
```


    
![png](https://raw.githubusercontent.com/philippeller/dama/master/README_files/README_10_0.png)
    


Let's interpolate the values to 200 points along each axis and plot


```python
g.interp(x=200, y=200).plot(cbar=True);
```


    
![png](https://raw.githubusercontent.com/philippeller/dama/master/README_files/README_12_0.png)
    


Executions of (most) translation methods is _lazy_. That means that the computation only happens if a specific variable is used. This can have some side effects, that when you maipulate the original data before the translation is evaluated. just something to be aware of.

Masking, and item assignement also is supported


```python
g.a[g.a > 0.3]
```




<table>
<tbody>
<tr><td><b>y \ x</b></td><td><b>0</b></td><td><b>0.325</b></td><td><b>0.65</b></td><td>...</td><td><b>8.77</b></td><td><b>9.1</b></td><td><b>9.42</b></td></tr>
<tr><td><b>0</b>    </td><td>--      </td><td>0.319       </td><td>0.605      </td><td>...</td><td>0.605      </td><td>0.319     </td><td>--         </td></tr>
<tr><td><b>0.331</b></td><td>--      </td><td>0.302       </td><td>0.572      </td><td>...</td><td>0.572      </td><td>0.302     </td><td>--         </td></tr>
<tr><td><b>0.661</b></td><td>--      </td><td>--          </td><td>0.478      </td><td>...</td><td>0.478      </td><td>--        </td><td>--         </td></tr>
<tr><td>...         </td><td>...     </td><td>...         </td><td>...        </td><td>...</td><td>...        </td><td>...       </td><td>...        </td></tr>
<tr><td><b>5.62</b> </td><td>--      </td><td>--          </td><td>0.478      </td><td>...</td><td>0.478      </td><td>--        </td><td>--         </td></tr>
<tr><td><b>5.95</b> </td><td>--      </td><td>0.302       </td><td>0.572      </td><td>...</td><td>0.572      </td><td>0.302     </td><td>--         </td></tr>
<tr><td><b>6.28</b> </td><td>--      </td><td>0.319       </td><td>0.605      </td><td>...</td><td>0.605      </td><td>0.319     </td><td>--         </td></tr>
</tbody>
</table>



The objects are also numpy compatible and indexable by index (integers) or value (floats). Numpy functions with `axis` keywords accept either the name(s) of the axis, e.g. here `x` and therefore is independent of axis ordering, or the usual integer indices.


```python
g[10::-1, :np.pi:2]
```




<table>
<tbody>
<tr><td><b>y \ x</b></td><td><b>3.25</b></td><td><b>2.92</b></td><td><b>2.6</b></td><td>...</td><td><b>0.65</b></td><td><b>0.325</b></td><td><b>0</b></td></tr>
<tr><td><b>0</b>    </td><td>a = -0.108 </td><td>a = 0.215  </td><td>a = 0.516 </td><td>...</td><td>a = 0.605  </td><td>a = 0.319   </td><td>a = 0   </td></tr>
<tr><td><b>0.661</b></td><td>a = -0.0853</td><td>a = 0.17   </td><td>a = 0.407 </td><td>...</td><td>a = 0.478  </td><td>a = 0.252   </td><td>a = 0   </td></tr>
<tr><td><b>1.32</b> </td><td>a = -0.0265</td><td>a = 0.0528 </td><td>a = 0.127 </td><td>...</td><td>a = 0.149  </td><td>a = 0.0784  </td><td>a = 0   </td></tr>
<tr><td><b>1.98</b> </td><td>a = 0.0434 </td><td>a = -0.0864</td><td>a = -0.207</td><td>...</td><td>a = -0.243 </td><td>a = -0.128  </td><td>a = -0  </td></tr>
<tr><td><b>2.65</b> </td><td>a = 0.0951 </td><td>a = -0.189 </td><td>a = -0.453</td><td>...</td><td>a = -0.532 </td><td>a = -0.281  </td><td>a = -0  </td></tr>
</tbody>
</table>




```python
np.sum(g[10::-1, :np.pi:2].T, axis='x')
```




<table>
<tbody>
<tr><td><b>y</b></td><td><b>0</b></td><td><b>0.661</b></td><td><b>1.32</b></td><td><b>1.98</b></td><td><b>2.65</b></td></tr>
<tr><td><b>a</b></td><td>6.03    </td><td>4.76        </td><td>1.48       </td><td>-2.42      </td><td>-5.3       </td></tr>
</tbody>
</table>



### Comparison
As comparison to point out the convenience, an alternative way without using `dama` to achieve the above would look something like the follwoing for creating and plotting the array:
> ```
> x = np.linspace(0,3*np.pi, 30)
> y = np.linspace(0,2*np.pi, 20) 
>
> xx, yy = np.meshgrid(x, y)
> a = np.sin(xx) * np.cos(yy)
>
> import matplotlib.pyplot as plt
> 
> x_widths = np.diff(x)
> x_pixel_boundaries = np.concatenate([[x[0] - 0.5*x_widths[0]], x[:-1] + 0.5*x_widths, [x[-1] + 0.5*x_widths[-1]]])
> y_widths = np.diff(y)
> y_pixel_boundaries = np.concatenate([[y[0] - 0.5*y_widths[0]], y[:-1] + 0.5*y_widths, [y[-1] + 0.5*y_widths[-1]]])
> 
> pc = plt.pcolormesh(x_pixel_boundaries, y_pixel_boundaries, a)
> plt.gca().set_xlabel('x')
> plt.gca().set_ylabel('y')
> cb = plt.colorbar(pc)
> cb.set_label('a')
> ```

and for doing the interpolation:

> ```
> from scipy.interpolate import griddata
> 
> interp_x = np.linspace(0,3*np.pi, 200)
> interp_y = np.linspace(0,2*np.pi, 200) 
> 
> grid_x, grid_y = np.meshgrid(interp_x, interp_y)
> 
> points = np.vstack([xx.flatten(), yy.flatten()]).T
> values = a.flatten()
> 
> interp_a = griddata(points, values, (grid_x, grid_y), method='cubic')
> ```

### PointData

Another representation of data is `PointData`, which is not any different of a dictionary holding same-length nd-arrays or a pandas `DataFrame` (And can actually be instantiated with those).


```python
p = dm.PointData()
p.x = np.random.randn(100_000)
p.a = np.random.rand(p.size) * p.x**2
```


```python
p
```




<table>
<tbody>
<tr><td><b>x</b></td><td style="text-align: right;">-1.69</td><td style="text-align: right;">-0.744</td><td style="text-align: right;">0.796</td><td>...</td><td style="text-align: right;">1.22</td><td style="text-align: right;">-0.653</td><td style="text-align: right;">1.09 </td></tr>
<tr><td><b>a</b></td><td style="text-align: right;"> 1.52</td><td style="text-align: right;"> 0.409</td><td style="text-align: right;">0.418</td><td>...</td><td style="text-align: right;">1.32</td><td style="text-align: right;"> 0.124</td><td style="text-align: right;">0.784</td></tr>
</tbody>
</table>




```python
p.plot()
```


    
![png](https://raw.githubusercontent.com/philippeller/dama/master/README_files/README_23_0.png)
    


Maybe a correlation plot would be more insightful:


```python
p.plot('x', 'a', '.');
```


    
![png](https://raw.githubusercontent.com/philippeller/dama/master/README_files/README_25_0.png)
    


This can now seamlessly be translated into `Griddata`, for example taking the data binwise in `x` in 20 bins, and in each bin summing up points:


```python
p.binwise(x=20).sum()
```




<table>
<tbody>
<tr><td><b>x</b></td><td><b>[-4.554 -4.12 ]</b></td><td><b>[-4.12  -3.686]</b></td><td><b>[-3.686 -3.253]</b></td><td>...</td><td><b>[2.822 3.255]</b></td><td><b>[3.255 3.689]</b></td><td><b>[3.689 4.123]</b></td></tr>
<tr><td><b>a</b></td><td>44                    </td><td>95.8                  </td><td>205                   </td><td>...</td><td>862                 </td><td>252                 </td><td>73.7                </td></tr>
</tbody>
</table>




```python
p.binwise(x=20).sum().plot();
```


    
![png](https://raw.githubusercontent.com/philippeller/dama/master/README_files/README_28_0.png)
    


This is equivalent of making a weighted histogram, while the latter is faster.


```python
p.histogram(x=20).a
```




<table>
<tbody>
<tr><td><b>x</b></td><td><b>[-4.554 -4.12 ]</b></td><td><b>[-4.12  -3.686]</b></td><td><b>[-3.686 -3.253]</b></td><td>...</td><td><b>[2.822 3.255]</b></td><td><b>[3.255 3.689]</b></td><td><b>[3.689 4.123]</b></td></tr>
<tr><td><b></b> </td><td>44                    </td><td>95.8                  </td><td>205                   </td><td>...</td><td>862                 </td><td>252                 </td><td>73.7                </td></tr>
</tbody>
</table>




```python
np.allclose(p.histogram(x=10).a, p.binwise(x=10).sum().a)
```




    True



There is also KDE in n-dimensions available, for example:


```python
p.kde(x=1000).a.plot();
```


    
![png](https://raw.githubusercontent.com/philippeller/dama/master/README_files/README_33_0.png)
    


`GridArrays` can also hold multi-dimensional values, like RGB images or here 5 values from the percentile function. Let's plot those as bands:


```python
p.binwise(x=20).quantile(q=[0.1, 0.3, 0.5, 0.7, 0.9]).plot_bands()
```


    
![png](https://raw.githubusercontent.com/philippeller/dama/master/README_files/README_35_0.png)
    


When we specify `x` with an array, we e gives a list of points to binwise. So the resulting plot will consist of points, not bins.


```python
p.binwise(x=np.linspace(-3,3,10)).quantile(q=[0.1, 0.3, 0.5, 0.7, 0.9]).plot_bands(lines=True, filled=True, linestyles=[':', '--', '-'], lw=1)
```


    
![png](https://raw.githubusercontent.com/philippeller/dama/master/README_files/README_37_0.png)
    


 This is not the same as using edges as in the example below, hence also the plots look different.


```python
p.binwise(x=dm.Edges(np.linspace(-3,3,10))).quantile(q=[0.1, 0.3, 0.5, 0.7, 0.9]).plot_bands(lines=True, filled=True, linestyles=[':', '--', '-'], lw=1)
```


    
![png](https://raw.githubusercontent.com/philippeller/dama/master/README_files/README_39_0.png)
    


### Saving and loading

Dama supports the pickle protocol, and objects can be stored like:
```python
dm.save("filename.pkl", obj)
```

And read back like:
```python
obj = dm.read("filename.pkl")
```

# Example gallery

This is just to illustrate some different, seemingly random applications, resulting in various plots. All starting from some random data points


```python
from matplotlib import pyplot as plt
```


```python
p = dm.PointData()
p.x = np.random.rand(10_000)
p.y = np.random.randn(p.size) * np.sin(p.x*3*np.pi) * p.x
p.a = p.y/p.x
```


```python
fig, ax = plt.subplots(4,4,figsize=(20,20))
ax = ax.flatten()

# First row
p.y.plot(ax=ax[0])
p.plot('x', 'y', '.', ax=ax[1])
p.plot_scatter('x', 'y', c='a', s=1, cmap=dm.cm.spectrum, ax=ax[2])
p.interp(x=100, y=100, method="nearest").a.plot(ax=ax[3])

# Second row
np.log(1 + p.histogram(x=100, y=100).counts).plot(ax=ax[4])
p.kde(x=100, y=100, bw=(0.02, 0.05)).density.plot(cmap=dm.cm.afterburner_r, ax=ax[5])
p.histogram(x=10, y=10).interp(x=100,y=100).a.plot(cmap="RdBu", ax=ax[6])
p.histogram(x=100, y=100).counts.median_filter(10).plot(ax=ax[7])

# Third row
p.binwise(x=100).quantile(q=[0.1, 0.3, 0.5, 0.7, 0.9]).y.plot_bands(ax=ax[8])
p.binwise(x=100).quantile(q=[0.1, 0.3, 0.5, 0.7, 0.9]).y.gaussian_filter((2.5,0)).interp(x=500).plot_bands(filled=False, lines=True, linestyles=[':', '--', '-'],ax=ax[9])
p.binwise(a=100).mean().y.plot(ax=ax[10])
p.binwise(a=100).std().y.plot(ax=ax[10])

# Fourth row
p.histogram(x=100, y=100).counts.std(axis='x').plot(ax=ax[11])
np.log(p.histogram(x=100, y=100).counts + 1).gaussian_filter(0.5).plot_contour(cmap=dm.cm.passion_r, ax=ax[12])
p.histogram(x=30, y=30).gaussian_filter(1).lookup(p).plot_scatter('x', 'y', 'a', 1, cmap='Spectral', ax=ax[13])
h = p.histogram(y=100, x=np.logspace(-1,0,100)).a.T
h[h>0].plot(ax=ax[14])
h[1/3:2/3].plot(ax=ax[15])
```




    <matplotlib.collections.QuadMesh at 0x7f95bb992130>




    
![png](https://raw.githubusercontent.com/philippeller/dama/master/README_files/README_44_1.png)
    



```python

```
