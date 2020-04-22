# Dragoman
<img align="right" src="https://raw.githubusercontent.com/philippeller/dragoman/master/dragoman.png" alt="Dragoman" width=150>

> A dragoman was an interpreter, translator, and official guide between Turkish, Arabic, and Persian-speaking countries and polities of the Middle East and European embassies, consulates, vice-consulates and trading posts. A dragoman had to have a knowledge of Arabic, Persian, Turkish, and European languages. (Source: wikipedia)

Similarly, the dragoman python library guides you through your data and translates between different representations.
Its aim is to offer a consistant and pythonic way to handle different datasaets and translations between them.
A dataset can for instance be simple colum/row data, or it can be data on a grid.

One of the key features of dragoman is the seamless translation from one data represenation into any other. 
Convenience `pyplot` plotting functions are also available, in order to produce standard plots without any hassle.

## Installation

* `git clone git@github.com:philippeller/dragoman.git`
* `pip install dragoman`

## Simple Examples


```python
import numpy as np
import dragoman as dm
```

### Grid Data

GridData is a collection of individual GridArrays. Both have a defined grid, here we initialize the grid in the constructor through simple keyword arguments resulting in a 2d grid with axes `x` and `y`


```python
g = dm.GridData(x = np.linspace(0,3*np.pi, 30),
                y = np.linspace(0,2*np.pi, 20),
               )
```

Filling one array with some sinusoidal functions, called `a` here


```python
g['a'] = np.sin(g['x']) * np.cos(g['y'])
```

in 1-d and 2-d they render as html in jupyter notebooks


```python
g['a']
```




<table>
<tbody>
<tr><td><b>y \ x</b></td><td><b>0</b></td><td><b>0.325</b></td><td><b>0.65</b></td><td><b>0.975</b></td><td><b>1.3</b></td><td><b>1.62</b></td><td>...</td><td><b>7.8</b></td><td><b>8.12</b></td><td><b>8.45</b></td><td><b>8.77</b></td><td><b>9.1</b></td><td><b>9.42</b></td></tr>
<tr><td><b>0</b>    </td><td>0       </td><td>0.319       </td><td>0.605      </td><td>0.828       </td><td>0.964     </td><td>0.999      </td><td>...</td><td>0.999     </td><td>0.964      </td><td>0.828      </td><td>0.605      </td><td>0.319     </td><td>3.67e-16   </td></tr>
<tr><td><b>0.331</b></td><td>0       </td><td>0.302       </td><td>0.572      </td><td>0.783       </td><td>0.911     </td><td>0.944      </td><td>...</td><td>0.944     </td><td>0.911      </td><td>0.783      </td><td>0.572      </td><td>0.302     </td><td>3.47e-16   </td></tr>
<tr><td><b>0.661</b></td><td>0       </td><td>0.252       </td><td>0.478      </td><td>0.653       </td><td>0.76      </td><td>0.788      </td><td>...</td><td>0.788     </td><td>0.76       </td><td>0.653      </td><td>0.478      </td><td>0.252     </td><td>2.9e-16    </td></tr>
<tr><td><b>0.992</b></td><td>0       </td><td>0.175       </td><td>0.331      </td><td>0.453       </td><td>0.527     </td><td>0.546      </td><td>...</td><td>0.546     </td><td>0.527      </td><td>0.453      </td><td>0.331      </td><td>0.175     </td><td>2.01e-16   </td></tr>
<tr><td><b>1.32</b> </td><td>0       </td><td>0.0784      </td><td>0.149      </td><td>0.203       </td><td>0.237     </td><td>0.245      </td><td>...</td><td>0.245     </td><td>0.237      </td><td>0.203      </td><td>0.149      </td><td>0.0784    </td><td>9.02e-17   </td></tr>
<tr><td><b>1.65</b> </td><td>-0      </td><td>-0.0264     </td><td>-0.05      </td><td>-0.0684     </td><td>-0.0796   </td><td>-0.0825    </td><td>...</td><td>-0.0825   </td><td>-0.0796    </td><td>-0.0684    </td><td>-0.05      </td><td>-0.0264   </td><td>-3.03e-17  </td></tr>
<tr><td>...         </td><td>...     </td><td>...         </td><td>...        </td><td>...         </td><td>...       </td><td>...        </td><td>...</td><td>...       </td><td>...        </td><td>...        </td><td>...        </td><td>...       </td><td>...        </td></tr>
<tr><td><b>4.63</b> </td><td>-0      </td><td>-0.0264     </td><td>-0.05      </td><td>-0.0684     </td><td>-0.0796   </td><td>-0.0825    </td><td>...</td><td>-0.0825   </td><td>-0.0796    </td><td>-0.0684    </td><td>-0.05      </td><td>-0.0264   </td><td>-3.03e-17  </td></tr>
<tr><td><b>4.96</b> </td><td>0       </td><td>0.0784      </td><td>0.149      </td><td>0.203       </td><td>0.237     </td><td>0.245      </td><td>...</td><td>0.245     </td><td>0.237      </td><td>0.203      </td><td>0.149      </td><td>0.0784    </td><td>9.02e-17   </td></tr>
<tr><td><b>5.29</b> </td><td>0       </td><td>0.175       </td><td>0.331      </td><td>0.453       </td><td>0.527     </td><td>0.546      </td><td>...</td><td>0.546     </td><td>0.527      </td><td>0.453      </td><td>0.331      </td><td>0.175     </td><td>2.01e-16   </td></tr>
<tr><td><b>5.62</b> </td><td>0       </td><td>0.252       </td><td>0.478      </td><td>0.653       </td><td>0.76      </td><td>0.788      </td><td>...</td><td>0.788     </td><td>0.76       </td><td>0.653      </td><td>0.478      </td><td>0.252     </td><td>2.9e-16    </td></tr>
<tr><td><b>5.95</b> </td><td>0       </td><td>0.302       </td><td>0.572      </td><td>0.783       </td><td>0.911     </td><td>0.944      </td><td>...</td><td>0.944     </td><td>0.911      </td><td>0.783      </td><td>0.572      </td><td>0.302     </td><td>3.47e-16   </td></tr>
<tr><td><b>6.28</b> </td><td>0       </td><td>0.319       </td><td>0.605      </td><td>0.828       </td><td>0.964     </td><td>0.999      </td><td>...</td><td>0.999     </td><td>0.964      </td><td>0.828      </td><td>0.605      </td><td>0.319     </td><td>3.67e-16   </td></tr>
</tbody>
</table>



It can be plotted easily in case of 1-d and 2-d grids


```python
g.plot(cbar=True);
```


![png](README_files/README_9_0.png)


Let's interpolate the values to 200 points along each axis and plot


```python
g.interp(x=200, y=200).plot(cbar=True);
```


![png](README_files/README_11_0.png)


The objects are also numpy compatible and indexable by index (integers) or values (floats). Numpy functions with axis keywords accept either the name of the axis, e.g. here `x` and therefore is independent of axis ordering, or the usual integer indices.


```python
g[10::-1, :np.pi:2]
```




<table>
<tbody>
<tr><td><b>y \ x</b></td><td><b>3.25</b></td><td><b>2.92</b></td><td><b>2.6</b></td><td><b>2.27</b></td><td><b>1.95</b></td><td><b>1.62</b></td><td><b>1.3</b></td><td><b>0.975</b></td><td><b>0.65</b></td><td><b>0.325</b></td><td><b>0</b></td></tr>
<tr><td><b>0</b>    </td><td>a = -0.108 </td><td>a = 0.215  </td><td>a = 0.516 </td><td>a = 0.762  </td><td>a = 0.929  </td><td>a = 0.999  </td><td>a = 0.964 </td><td>a = 0.828   </td><td>a = 0.605  </td><td>a = 0.319   </td><td>a = 0   </td></tr>
<tr><td><b>0.661</b></td><td>a = -0.0853</td><td>a = 0.17   </td><td>a = 0.407 </td><td>a = 0.601  </td><td>a = 0.733  </td><td>a = 0.788  </td><td>a = 0.76  </td><td>a = 0.653   </td><td>a = 0.478  </td><td>a = 0.252   </td><td>a = 0   </td></tr>
<tr><td><b>1.32</b> </td><td>a = -0.0265</td><td>a = 0.0528 </td><td>a = 0.127 </td><td>a = 0.187  </td><td>a = 0.228  </td><td>a = 0.245  </td><td>a = 0.237 </td><td>a = 0.203   </td><td>a = 0.149  </td><td>a = 0.0784  </td><td>a = 0   </td></tr>
<tr><td><b>1.98</b> </td><td>a = 0.0434 </td><td>a = -0.0864</td><td>a = -0.207</td><td>a = -0.306 </td><td>a = -0.373 </td><td>a = -0.401 </td><td>a = -0.387</td><td>a = -0.332  </td><td>a = -0.243 </td><td>a = -0.128  </td><td>a = -0  </td></tr>
<tr><td><b>2.65</b> </td><td>a = 0.0951 </td><td>a = -0.189 </td><td>a = -0.453</td><td>a = -0.67  </td><td>a = -0.817 </td><td>a = -0.878 </td><td>a = -0.847</td><td>a = -0.728  </td><td>a = -0.532 </td><td>a = -0.281  </td><td>a = -0  </td></tr>
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
As comparison to point out the convenience, an alternative way without using Dragoman to achieve the above would look something like the follwoing:


```python
x = np.linspace(0,3*np.pi, 30)
y = np.linspace(0,2*np.pi, 20) 

xx, yy = np.meshgrid(x, y)

a = np.sin(xx) * np.cos(yy)

import matplotlib.pyplot as plt

x_widths = np.diff(x)
x_pixel_boundaries = np.concatenate([[x[0] - 0.5*x_widths[0]], x[:-1] + 0.5*x_widths, [x[-1] + 0.5*x_widths[-1]]])
y_widths = np.diff(y)
y_pixel_boundaries = np.concatenate([[y[0] - 0.5*y_widths[0]], y[:-1] + 0.5*y_widths, [y[-1] + 0.5*y_widths[-1]]])

pc = plt.pcolormesh(x_pixel_boundaries, y_pixel_boundaries, a)
plt.gca().set_xlabel('x')
plt.gca().set_ylabel('y')
cb = plt.colorbar(pc)
cb.set_label('a')
```


![png](README_files/README_16_0.png)



```python
from scipy.interpolate import griddata

interp_x = np.linspace(0,3*np.pi, 200)
interp_y = np.linspace(0,2*np.pi, 200) 

grid_x, grid_y = np.meshgrid(interp_x, interp_y)

points = np.vstack([xx.flatten(), yy.flatten()]).T
values = a.flatten()

interp_a = griddata(points, values, (grid_x, grid_y), method='cubic')
```

### PointData

Another representation of data is `PointData`, which is not any different of a dictionary holding same-length nd-arrays or a pandas `DataFrame` (And can actually be instantiated with those)


```python
p = dm.PointData()
p['x'] = np.random.randn(10000)
p['a'] = np.random.rand(p.size) * p['x']**2
```


```python
p
```




<table>
<tbody>
<tr><td><b>x</b></td><td style="text-align: right;">0.479</td><td style="text-align: right;">-0.00525</td><td style="text-align: right;">0.379</td><td style="text-align: right;">0.561 </td><td style="text-align: right;">0.94 </td><td style="text-align: right;">-0.201 </td><td>...</td><td style="text-align: right;">0.0417 </td><td style="text-align: right;">0.445</td><td style="text-align: right;">-1.37 </td><td style="text-align: right;">0.992</td><td style="text-align: right;">-0.254 </td><td style="text-align: right;">0.441</td></tr>
<tr><td><b>a</b></td><td style="text-align: right;">0.22 </td><td style="text-align: right;"> 1.1e-05</td><td style="text-align: right;">0.107</td><td style="text-align: right;">0.0771</td><td style="text-align: right;">0.242</td><td style="text-align: right;"> 0.0237</td><td>...</td><td style="text-align: right;">0.00123</td><td style="text-align: right;">0.145</td><td style="text-align: right;"> 0.125</td><td style="text-align: right;">0.259</td><td style="text-align: right;"> 0.0386</td><td style="text-align: right;">0.13 </td></tr>
</tbody>
</table>




```python
p.plot()
plt.legend();
```


![png](README_files/README_21_0.png)


Maybe a correlation plot would be more insightful:


```python
p.plot('x', 'a', '.');
```


![png](README_files/README_23_0.png)


This can now seamlessly be translated into `Griddata`, for example taking the data binwise in `x` in 10 bins, and in each bin summing up points:


```python
p.binwise(x=20).sum()
```




<table>
<tbody>
<tr><td><b>x</b></td><td><b>[-3.736 -3.365]</b></td><td><b>[-3.365 -2.994]</b></td><td><b>[-2.994 -2.623]</b></td><td><b>[-2.623 -2.252]</b></td><td><b>[-2.252 -1.882]</b></td><td><b>[-1.882 -1.511]</b></td><td>...</td><td><b>[1.456 1.827]</b></td><td><b>[1.827 2.198]</b></td><td><b>[2.198 2.569]</b></td><td><b>[2.569 2.939]</b></td><td><b>[2.939 3.31 ]</b></td><td><b>[3.31  3.681]</b></td></tr>
<tr><td><b>a</b></td><td>4.87                  </td><td>29.3                  </td><td>99.7                  </td><td>279                   </td><td>315                   </td><td>556                   </td><td>...</td><td>513                 </td><td>402                 </td><td>214                 </td><td>162                 </td><td>50.1                </td><td>21.4                </td></tr>
</tbody>
</table>




```python
p.binwise(x=20).sum().plot();
```


![png](README_files/README_26_0.png)


This is equivalent of making a weighted histogram, while the latter is faster.


```python
p.histogram(x=20)
```




<table>
<tbody>
<tr><td><b>x</b>     </td><td><b>[-3.736 -3.365]</b></td><td><b>[-3.365 -2.994]</b></td><td><b>[-2.994 -2.623]</b></td><td><b>[-2.623 -2.252]</b></td><td><b>[-2.252 -1.882]</b></td><td><b>[-1.882 -1.511]</b></td><td>...</td><td><b>[1.456 1.827]</b></td><td><b>[1.827 2.198]</b></td><td><b>[2.198 2.569]</b></td><td><b>[2.569 2.939]</b></td><td><b>[2.939 3.31 ]</b></td><td><b>[3.31  3.681]</b></td></tr>
<tr><td><b>a</b>     </td><td>4.87                  </td><td>29.3                  </td><td>99.7                  </td><td>279                   </td><td>315                   </td><td>556                   </td><td>...</td><td>513                 </td><td>402                 </td><td>214                 </td><td>162                 </td><td>50.1                </td><td>21.4                </td></tr>
<tr><td><b>counts</b></td><td>3                     </td><td>8                     </td><td>25                    </td><td>85                    </td><td>156                   </td><td>389                   </td><td>...</td><td>385                 </td><td>199                 </td><td>85                  </td><td>42                  </td><td>10                  </td><td>3                   </td></tr>
</tbody>
</table>




```python
np.allclose(p.histogram(x=10)['a'], p.binwise(x=10).sum()['a'])
```




    True



There is also KDE in n-dimensions available, for example:


```python
p.kde(x=1000)['a'].plot();
```


![png](README_files/README_31_0.png)


GridArrays can also hold multi-dimensional values, like RGB images or here 5 values from th percentile function. We can plot those as bands for example:


```python
p.binwise(x=20).quantile(q=[0.1, 0.3, 0.5, 0.7, 0.9]).plot_bands()
```


![png](README_files/README_33_0.png)



```python

```
