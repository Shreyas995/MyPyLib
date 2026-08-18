#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_flow_plane.py -- load ONE z-plane of a tlab field file and plot it.

Run in Spyder: set `fname` and `pl_id` below (pl_id is 1-indexed, as in
functions.readplane, so pl_id = 1 is the first plane).

Reads with the existing helpers in functions.py:
    read_header(path)                    -> offset, nx, ny, nz, nt, params
    readplane(path, nx, ny, pl_id, hdr)  -> plane[ny, nx]
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from functions import read_header, readplane, read_grid

# ---------------------------------------------------------------------------
# user input
# ---------------------------------------------------------------------------
cwd   = os.getcwd() + '/'
fname = 'flow.324000.1'        # component 1 = u (2 = v wall-normal, 3 = w)
pl_id = 1                      # z-plane to plot, 1-indexed (1 = first plane)

# ---------------------------------------------------------------------------
# grid + header
# ---------------------------------------------------------------------------
x, y, z = read_grid(cwd)
nx = np.size(x)
ny = np.size(y)
nz = np.size(z)

path = cwd + fname
offset, nxf, nyf, nzf, nt, params = read_header(path)
print('header: offset=%d  nx=%d  ny=%d  nz=%d  nt=%d' % (offset, nxf, nyf, nzf, nt))

# ---------------------------------------------------------------------------
# read the plane
# ---------------------------------------------------------------------------
plane = readplane(path, nx, ny, pl_id, offset)     # plane[ny, nx]

print('plane %d: min %.6g  mean %.6g  max %.6g'
      % (pl_id, plane.min(), plane.mean(), plane.max()))

# ---------------------------------------------------------------------------
# plot (x streamwise horizontal, y wall-normal vertical)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.4))
m = ax.pcolormesh(x, y, plane, cmap='RdBu_r', shading='auto')
fig.colorbar(m, ax=ax, label=fname)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('%s   plane k = %d of %d   (it = %d)' % (fname, pl_id, nz, nt))
fig.tight_layout()
plt.show()
