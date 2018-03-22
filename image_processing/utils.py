# -*- coding: utf-8 -*-

import numpy as np


def convert_grid_to_coords(grid):
    coords = np.vstack([g.flatten()[None, ...] for g in grid]) 
    return coords
