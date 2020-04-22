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
x = np.linspace(0,3*np.pi, 20)
y = np.linspace(0,2*np.pi, 10) 

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
interp_y = np.linspace(0,2*np.pi, 100) 

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
p['a'] = np.random.rand(p.size)
```


```python
p
```




<table>
<tbody>
<tr><td><b>x</b></td><td style="text-align: right;">-1.01 </td><td style="text-align: right;">-0.896</td><td style="text-align: right;">-0.154</td><td style="text-align: right;">0.291</td><td style="text-align: right;">0.775</td><td style="text-align: right;">0.178</td><td>...</td><td style="text-align: right;">0.112</td><td style="text-align: right;">1.52 </td><td style="text-align: right;">1.09 </td><td style="text-align: right;">-0.0441</td><td style="text-align: right;">-0.248</td><td style="text-align: right;">0.267 </td></tr>
<tr><td><b>a</b></td><td style="text-align: right;"> 0.194</td><td style="text-align: right;"> 0.455</td><td style="text-align: right;"> 0.482</td><td style="text-align: right;">0.656</td><td style="text-align: right;">0.329</td><td style="text-align: right;">0.74 </td><td>...</td><td style="text-align: right;">0.405</td><td style="text-align: right;">0.861</td><td style="text-align: right;">0.535</td><td style="text-align: right;"> 0.966 </td><td style="text-align: right;"> 0.169</td><td style="text-align: right;">0.0294</td></tr>
</tbody>
</table>




```python
p.plot()
plt.legend();
```


![png](README_files/README_21_0.png)



```python
p.plot('x', 'a', '.');
```


![png](README_files/README_22_0.png)


This can now seamlessly be translated into `Griddata`, for example taking the data binwise in `x` in 10 bins, and in each bin summing up points:


```python
p.binwise(x=10).sum()
```




<table>
<tbody>
<tr><td><b>x</b></td><td><b>[-3.68  -2.961]</b></td><td><b>[-2.961 -2.242]</b></td><td><b>[-2.242 -1.524]</b></td><td><b>[-1.524 -0.805]</b></td><td><b>[-0.805 -0.086]</b></td><td><b>[-0.086  0.633]</b></td><td><b>[0.633 1.351]</b></td><td><b>[1.351 2.07 ]</b></td><td><b>[2.07  2.789]</b></td><td><b>[2.789 3.507]</b></td></tr>
<tr><td><b>a</b></td><td>8.86                  </td><td>62.9                  </td><td>256                   </td><td>721                   </td><td>1.22e+03              </td><td>1.39e+03              </td><td>856                 </td><td>345                 </td><td>82.5                </td><td>19                  </td></tr>
</tbody>
</table>




```python
p.binwise(x=10).sum().plot();
```


![png](README_files/README_25_0.png)


This is equivalent of making a weighted histogram, while the latter is faster.


```python
p.histogram(x=10)
```




<table>
<tbody>
<tr><td><b>x</b>     </td><td><b>[-3.68  -2.961]</b></td><td><b>[-2.961 -2.242]</b></td><td><b>[-2.242 -1.524]</b></td><td><b>[-1.524 -0.805]</b></td><td><b>[-0.805 -0.086]</b></td><td><b>[-0.086  0.633]</b></td><td><b>[0.633 1.351]</b></td><td><b>[1.351 2.07 ]</b></td><td><b>[2.07  2.789]</b></td><td><b>[2.789 3.507]</b></td></tr>
<tr><td><b>a</b>     </td><td>8.86                  </td><td>62.9                  </td><td>256                   </td><td>721                   </td><td>1.22e+03              </td><td>1.39e+03              </td><td>856                 </td><td>345                 </td><td>82.5                </td><td>19                  </td></tr>
<tr><td><b>counts</b></td><td>17                    </td><td>117                   </td><td>508                   </td><td>1.46e+03              </td><td>2.5e+03               </td><td>2.8e+03               </td><td>1.72e+03            </td><td>691                 </td><td>157                 </td><td>35                  </td></tr>
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


![png](README_files/README_30_0.png)

