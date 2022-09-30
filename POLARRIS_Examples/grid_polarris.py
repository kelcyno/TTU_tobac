#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from matplotlib import pyplot as plt
import os
import shutil
from copy import deepcopy
import cartopy.crs as ccrs
import shapely
import pyart

import pandas as pd
import glob


# In[ ]:


def get_grid(radar):
    """ Returns grid object from radar object. """
    grid = pyart.map.grid_from_radars(
        radar, grid_shape=(31, 1001, 1001),
        grid_limits=((0, 15000), (-250000,250000), (-250000, 250000)),
        fields=['CZ','DR','KD','RH','VR','W'],
        gridding_algo='map_gates_to_grid',
        h_factor=0., nb=0.6, bsp=1., min_radius=200.)
    return grid


# In[ ]:


arq = sorted(glob.glob('/Volumes/Seagate Backup Plus Drive/KNB_TTU/20210629/POLARRIS/polarris0629/VCP12_CfRadial_2021_*.nc'))
radar = pyart.io.read_cfradial(arq[0])
#fname = arq[0]


# In[ ]:


filenames = []
for num, key in enumerate(arq):
    print(key)
    print('saving grid', num)
    radar = pyart.io.read_cfradial(key)
    grid = get_grid(radar)
    fname = os.path.split(str(key))[1][:-3]
    name = os.path.join('grid_' + fname + '.nc')
    pyart.io.write_grid(name, grid)
    del radar, grid

