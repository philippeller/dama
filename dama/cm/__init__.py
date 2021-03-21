import numpy as np
import matplotlib
import glob
import os

maps = glob.glob('*.npy')

cms = {}

for m in maps:
    name = os.path.splitext(os.path.basename(m))[0]
    array = np.load(m)
    cms[name] = matplotlib.colors.ListedColormap(array, name=name)
    cms[name+'_r'] = matplotlib.colors.ListedColormap(array[::-1], name=name+'_r')

locals().update(cms)
