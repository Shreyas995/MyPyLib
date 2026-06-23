#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:13:16 2024

@author: shreyas deshpande

PhAvgAllPlanes.py
-----------------
Robust variant of PhAvgCurta.py.

Each tlab `avg_*` file holds `Restart` per-iteration, z-direction-averaged
planes plus a final plane (`Restart+1`) that is the precomputed average of those
planes.  PhAvgCurta.py read only that final plane, but a tlab bug corrupts it
with NaN/Inf.  This script instead reads ALL per-iteration planes (every plane
except the final precomputed-average plane), skips any plane corrupted by
NaN/Inf in the FLUID, and averages the good planes -- each component file with
its OWN accumulator and its OWN good-plane counter.

Standalone: depends only on the standard library and numpy.
"""

import os
import re
import sys
import csv
import struct
import numpy as np

###############################################################################
############################## Function defintion #############################

def read_fortran_record(f_h, dtype):
    dum1 = np.fromfile(f_h, dtype, count=1)[0]

    return dum1

def read_header(FilePath):
    # Define sizes based on Fortran implementation
    int_dtype = np.dtype('<i4')  # 4-byte integer, little-endian
    float_dtype = np.dtype('<f8')  # 8-byte float, little-endian
    sizeofint = 4
    sizeofreal = 8

    try:
        with open(FilePath, 'rb') as f:
            # Read the offset first
            offset = read_fortran_record(f, int_dtype)

            if offset <= sizeofint:
                raise ValueError("Offset value is too small, it nust be greater than the size of an integer.")

            # Read the grid dimensions and nt
            nx = read_fortran_record(f, np.dtype('<i4'))
            ny = read_fortran_record(f, np.dtype('<i4'))
            nz = read_fortran_record(f, np.dtype('<i4'))
            nt = read_fortran_record(f, np.dtype('<i4'))
            # Calculate the size of params
            remaining_header_size = offset - 5 * sizeofint
            params_size = int(remaining_header_size/sizeofreal)

            # Read params if there are any
            params = []
            if params_size > 0:
                for i in range (params_size):
                    params_record = read_fortran_record(f, np.dtype('<f8'))  # 'f8' for double precision float
                    params.append(params_record)

            return offset, nx, ny, nz, nt, params

    except Exception as e:
        # Print the error message and return a default value
        # print(f'Error reading header: {e}')
        return None, None, None, None, None, None

def read_grid(path):
    #---------------------------------------------------------------------------#
    # Read grid
    #---------------------------------------------------------------------------#

    # open grid file
    seek = 0
    f = open(path+'grid','rb')
    f.seek(seek,0)

    # header - number of nodes
    print("--------------------------------------------------")
    h = np.fromfile(f, '<i4', 1)
    print('iheader length = ', h)
    nmax = np.fromfile(f, '<i4', 3)
    h = np.fromfile(f, '<i4', 1)
    print('check iheader  = ', h)

    # header - grid scales
    print("--------------------------------------------------")
    h = np.fromfile(f, '<i4', 1)
    print('fheader length  = ', h)
    scales = np.fromfile(f, '<f8', 3)
    print('scales         = ', scales)
    h = np.fromfile(f, '<i4', 1)
    print('check fheader  = ', h)

    # x - nodes
    print("--------------------------------------------------")
    h = np.fromfile(f, '<i4', 1)
    print('fheader length  = ', h)
    x = np.fromfile(f, '<f8', nmax[0])
    print('x-nodes       =  ', x[:5])
    h = np.fromfile(f, '<i4', 1)
    print('check fheader  = ', h)

    # y - nodes
    print("--------------------------------------------------")
    h = np.fromfile(f, '<i4', 1)
    print('fheader length  = ', h)
    y = np.fromfile(f, '<f8', nmax[1])
    print('y-nodes       =  ', y[:5])
    h = np.fromfile(f, '<i4', 1)
    print('check fheader  = ', h)

    # z - nodes
    print("--------------------------------------------------")
    h = np.fromfile(f, '<i4', 1)
    print('fheader length  = ', h)
    z = np.fromfile(f, '<f8', nmax[2])
    print('z-nodes       =  ', z[:5])
    h = np.fromfile(f, '<i4', 1)
    print('check fheader  = ', h)
    print("--------------------------------------------------")

    # close grid file
    f.close()
    return x, y, z

def epsfield():
    #-----------------------------------------------------------------------------#
    # data specification of eps field
    #-----------------------------------------------------------------------------#
    # path to data
    current_path = os.getcwd() + '/'
    path         = current_path
    fname        ='eps0.1'

    # data types (little endian)
    type_i1 = np.dtype('<i1'); type_i4 = np.dtype('<i4'); type_f8 = np.dtype('<f8')
    sizeofdata_int1 = 1; sizeofdata_int4 = 4; sizeofdata_float = 8

    # header
    head_params = 5
    head_size   = head_params * sizeofdata_int4

    #-----------------------------------------------------------------------------#
    # read
    #-----------------------------------------------------------------------------#
    # header
    f = open(path + fname,'rb')
    f.seek(0,0)
    header = np.fromfile(f, type_i4, head_params)
    f.close()
    print('Header size           :', header[0])
    print('Grid   size (nx*ny*nz):', header[1]*8,'x',header[2],'x',header[3])

    # data size (attention: h[1] = grid.nx*8!)
    bsize = np.prod(header[1:3])
    rsize = bsize * 8

    # read eps field as int1
    f = open(path + fname,'rb')
    f.seek(header[0],0)
    data = np.fromfile(f, np.dtype('<i1'), bsize)
    f.close()

    #-----------------------------------------------------------------------------#
    # convert to bitwise
    #-----------------------------------------------------------------------------#

    eps = np.zeros(rsize)
    eps = int2bit_2(eps,data) # eps = int2bit_2(eps,data) # faster
    eps = eps.reshape((header[1]*8,header[2]),order='F') # (attention: h[1] = grid.nx*8!)
    return eps.T #eps[:,:,1].T

def int2bit_2(out,data): # option 2 (bit faster then option 1)
    bsize = data.size
    for i in range(bsize):
        ip = i * 8
        by   = struct.pack('b',data[i])
        by2b = ''.join(format(ord(by), '08b') for byte in by)
        j = 0
        for k in range(-1,-9,-1):
            out[j+ip] = int(str(by2b)[k])
            j += 1
    return out

def epsVolume(eps,ny,nx, hill_height):
    eps_vol = np.zeros((ny,nx))

    for j in range (hill_height):
        for i in range (nx):
            if i == 1023:
                print (i)

            # Top
            if j == 0:
                # Top left cornor
                if i == 0:
                    if (eps[j,i] + eps[j+1,i+1] + eps[j+1,i] + eps[j,i+1] == 4):
                        eps_vol[j,i] = 1
                    else:
                        print ('i:', i , 'j:', j, 'Case undefined')

                # Top right cornor
                elif i == nx-1:
                    if (eps[j,i] + eps[j,i-1] + eps[j+1,i-1] + eps[j+1,i] == 4):
                        eps_vol[j,i] = 1
                    else:
                        print ('i:', i , 'j:', j, 'Case undefined')

                # Top edge
                if i !=0 and i != nx-1:
                    if (eps[j,i] + eps[j,i-1] + eps[j+1,i-1] + eps[j+1,i] + eps[j+1,i+1] + eps[j,i+1] == 6):
                            eps_vol[j,i] = 1

                    elif (eps[j,i] + eps[j,i-1] + eps[j+1,i-1] + eps[j+1,i] + eps[j+1,i+1] + eps[j,i+1] == 5):
                            eps_vol[j,i] = 0.75

                    elif (eps[j,i] + eps[j,i-1] + eps[j+1,i-1] + eps[j+1,i] + eps[j+1,i+1] + eps[j,i+1] == 4):
                            eps_vol[j,i] = 0.5

                    elif (eps[j,i] + eps[j,i-1] + eps[j+1,i-1] + eps[j+1,i] + eps[j+1,i+1] + eps[j,i+1] == 2):
                            eps_vol[j,i] = 0.25

                    elif (eps[j,i] + eps[j,i-1] + eps[j+1,i-1] + eps[j+1,i] + eps[j+1,i+1] + eps[j,i+1] == 3):
                        if (eps[j+1,i] == 0) and ((eps[j,i+1] == 0) or (eps[j,i-1] == 0)):
                            eps_vol[j,i] = 0.25
                        else:
                            eps_vol[j,i] = 0.5
                    else:
                        print ('i:', i , 'j:', j, 'Case undefined')

            # Generalized area
            elif i != 0 and j != 0 and i != nx-1:
                if (eps[j,i] + eps[j-1,i] + eps[j-1,i+1] + eps[j,i+1] + eps[j+1,i+1] + eps[j+1,i] + eps[j+1,i-1] + eps[j,i-1] + eps[j-1,i-1] == 9):
                    eps_vol[j,i] = 1
                elif (eps[j,i] + eps[j-1,i] + eps[j-1,i+1] + eps[j,i+1] + eps[j+1,i+1] + eps[j+1,i] + eps[j+1,i-1] + eps[j,i-1] + eps[j-1,i-1] == 8):
                    eps_vol[j,i] = 0.75
                elif (eps[j,i] + eps[j-1,i] + eps[j-1,i+1] + eps[j,i+1] + eps[j+1,i+1] + eps[j+1,i] + eps[j+1,i-1] + eps[j,i-1] + eps[j-1,i-1] == 7):
                    eps_vol[j,i] = 0.5
                elif (eps[j,i] + eps[j-1,i] + eps[j-1,i+1] + eps[j,i+1] + eps[j+1,i+1] + eps[j+1,i] + eps[j+1,i-1] + eps[j,i-1] + eps[j-1,i-1] == 6):
                    eps_vol[j,i] = 0.5
                elif (eps[j,i] + eps[j-1,i] + eps[j-1,i+1] + eps[j,i+1] + eps[j+1,i+1] + eps[j+1,i] + eps[j+1,i-1] + eps[j,i-1] + eps[j-1,i-1] == 5):
                    eps_vol[j,i] = 0.25
                elif (eps[j,i] + eps[j-1,i] + eps[j-1,i+1] + eps[j,i+1] + eps[j+1,i+1] + eps[j+1,i] + eps[j+1,i-1] + eps[j,i-1] + eps[j-1,i-1] == 4):
                    eps_vol[j,i] = 0.25
                else:
                    print ('i:', i , 'j:', j, 'Case undefined')

            # Left edge
            elif i == 0 and j != 0:
                if (eps[j,i] + eps[j-1,i] + eps[j-1,i+1] + eps[j,i+1] + eps[j+1,i+1] + eps[j+1,i] == 6):
                    eps_vol[j,i] = 1

                elif (eps[j,i] + eps[j-1,i] + eps[j-1,i+1] + eps[j,i+1] + eps[j+1,i+1] + eps[j+1,i] == 5):
                    eps_vol[j,i] = 0.5

                elif (eps[j,i] + eps[j-1,i] + eps[j-1,i+1] + eps[j,i+1] + eps[j+1,i+1] + eps[j+1,i] == 4):
                    eps_vol[j,i] = 0.5

                else:
                    print ('i:', i , 'j:', j, 'Case undefined')

            # Right edge
            elif i == nx-1 and j != 0:
                if (eps[j,i] + eps[j-1,i] + eps[j-1,i-1] + eps[j,i-1] + eps[j+1,i-1] + eps[j+1,i] == 6):
                    eps_vol[j,i] = 1

                elif (eps[j,i] + eps[j-1,i] + eps[j-1,i-1] + eps[j,i-1] + eps[j+1,i-1] + eps[j+1,i] == 5):
                    eps_vol[j,i] = 0.5

                elif (eps[j,i] + eps[j-1,i] + eps[j-1,i-1] + eps[j,i-1] + eps[j+1,i-1] + eps[j+1,i] == 4):
                    eps_vol[j,i] = 0.5
                else:
                    print ('i:', i , 'j:', j, 'Case undefined')

            else:
                print ('i:', i , 'j:', j, 'Case undefined')
    return eps_vol

def writefield(path, Nx, Ny, Nz, field):
    output_FilePath = path
    data_block = np.zeros((Ny,Nx))
    ofile=open(output_FilePath,'ab')
    ofile.seek(52)
    for iz in range(Nz):
        data_block[:,:]=field[:,:,iz]
        ofile.write(data_block)

def readfield(path, Nx, Ny, Nz, hdr):
    field = np.zeros((Ny,Nx,Nz))
    input_FilePath = path
    data_block = np.zeros((Ny,Nx))
    ifile=open(input_FilePath,'rb')
    ifile.seek(hdr)
    for iz in range(Nz):
        data_block[:,:]=np.fromfile(ifile, dtype=np.float64, count=Nx*Ny).reshape([Ny,Nx])
        field[:,:,iz] = data_block
    ifile.close()
    return field

def readplane(path, Nx, Ny, pl_id, hdr):
    plane = np.zeros((Ny,Nx))
    input_FilePath = path
    ifile=open(input_FilePath,'rb')
    ifile.seek(hdr + Nx*Ny*(pl_id-1)*8)
    plane[:,:]=np.fromfile(ifile, dtype=np.float64, count=Nx*Ny).reshape([Ny,Nx])
    ifile.close()
    return plane

def read_next_plane(fh, Nx, Ny):
    # Read the NEXT z-plane (Nx*Ny float64) sequentially from an open file
    # handle already positioned at the start of a plane. Returns (Ny, Nx).
    # Used to stream every plane of an avg_* file without re-opening/re-seeking.
    return np.fromfile(fh, dtype=np.float64, count=Nx*Ny).reshape([Ny, Nx])

def report_nan_location(field, eps, name):
    # Report whether non-finite values (NaN/Inf) in a 2D field (ny,nx) sit in
    # the solid (eps==1) or the fluid (eps!=1). Solid values are expected/
    # harmless; fluid values mean the field is most likely corrupt. Prints
    # nothing if the field is fully finite.
    bad_mask = ~np.isfinite(field)
    n_tot = int(bad_mask.sum())
    if (n_tot == 0):
        return
    in_solid = int((bad_mask & (eps == 1)).sum())
    in_fluid = int((bad_mask & (eps != 1)).sum())
    print("NaN/Inf report [%s]: %d total -> %d in SOLID (eps==1), %d in FLUID (eps!=1)."
          % (name, n_tot, in_solid, in_fluid))
    if (in_fluid > 0):
        jj, ii = np.where(bad_mask & (eps != 1))
        print("   --> FLUID NaN/Inf present, field most likely corrupt; first at (j=%d, i=%d)."
              % (jj[0], ii[0]))

def print_fluid_nans(field, eps, name, x=None, y=None, max_print=50):
    # Standalone locator: print every location where `field` (2D, ny x nx) is
    # non-finite (NaN/Inf) OUTSIDE the solid (eps != 1). Independent of any eps
    # case logic. If x and y grids are given, also print physical coordinates
    # (x[i], y[j]). Prints nothing if there are no non-finite values in fluid.
    outside = (~np.isfinite(field)) & (eps != 1)
    n = int(outside.sum())
    if (n == 0):
        return
    jj, ii = np.where(outside)
    print("FLUID NaN/Inf in [%s]: %d location(s) outside the solid (eps != 1):" % (name, n))
    for c in range(min(n, max_print)):
        j = int(jj[c])
        i = int(ii[c])
        if (x is not None) and (y is not None):
            print("   j=%d  i=%d   (x=%.6g, y=%.6g)" % (j, i, x[i], y[j]))
        else:
            print("   j=%d  i=%d" % (j, i))
    if (n > max_print):
        print("   ... %d more (raised max_print to see all)." % (n - max_print))

def diffu_dy(field, ny, nx, eps, y):
    coef_f  = np.array([-49/20, 6, -15/2, 20/3, -15/4, 6/5, -1/6])
    coef_f1 = np.array([-1/6, -77/60, 5/2, -5/3, 5/6, -1/4, 1/30])
    coef_f2 = np.array([1/30, -2/5, -7/12, 4/3, -1/2, 2/15, -1/60])
    coef_c =  np.array([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
    coef_b2 = np.array([1/60, -2/15, 1/2, -4/3, 7/12, 2/5, -1/30])
    coef_b1 = np.array([-1/30, 1/4, -5/6, 5/3, -5/2, 77/60, 1/6])
    coef_b = np.array([1/6, -6/5, 15/4, -20/3, 15/2, -6, 49/20])
    du = np.zeros((ny,nx))
    for i in range (nx):
        for j in range (ny):
            if ((eps[j,i] == 1 and eps[j+1,i] == 0) or (eps[j,i] == 0 and j == 0)):
                # Forward
                du[j,i] = np.dot(field[j:j+7,i],coef_f)/np.dot(y[j:j+7],coef_f)

            elif ((eps[j-1,i] == 1 and eps[j,i] == 0) or (eps[j-1,i] == 0 and j == 1)):
                # Forward Bias 1 (1,c,5)
                du[j,i] = np.dot(field[j-1:j+6,i],coef_f1)/np.dot(y[j-1:j+6],coef_f1)

            elif ((eps[j-2,i] == 1 and eps[j,i] == 0) or (eps[j-2,i] == 0 and j == 2)):
                # Forward Bias 2 (2,c,4)
                du[j,i] = np.dot(field[j-2:j+5,i],coef_f2)/np.dot(y[j-2:j+5],coef_f2)

            elif eps[j,i] != 1 and eps[j-2,i] != 1 and j < ny-3 and j > 2:
                # Center difference (3,c,3)
                du[j,i] = np.dot(field[j-3:j+4,i],coef_c)/np.dot(y[j-3:j+4],coef_c)

            elif j == ny-1:
                # Backward
                du[j,i] = np.dot(field[j-6:j+1,i],coef_b)/np.dot(y[j-6:j+1],coef_b)

            elif j == ny-2:
                # Backward Bias 1 (5,c,1)
                du[j,i] = np.dot(field[j-5:j+2,i],coef_b1)/np.dot(y[j-5:j+2],coef_b1)

            elif j == ny-3:
                # Backward Bias 2 (4,c,2)
                du[j,i] = np.dot(field[j-4:j+3,i],coef_b2)/np.dot(y[j-4:j+3],coef_b2)
            else:
                print('Undefined case')
    return du

def diffu_dx(field, ny, nx, eps, x):
    coef_f  = np.array([-49/20, 6, -15/2, 20/3, -15/4, 6/5, -1/6])
    coef_f1 = np.array([-1/6, -77/60, 5/2, -5/3, 5/6, -1/4, 1/30])
    coef_f2 = np.array([1/30, -2/5, -7/12, 4/3, -1/2, 2/15, -1/60])
    coef_c =  np.array([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
    coef_b2 = np.array([1/60, -2/15, 1/2, -4/3, 7/12, 2/5, -1/30])
    coef_b1 = np.array([-1/30, 1/4, -5/6, 5/3, -5/2, 77/60, 1/6])
    coef_b = np.array([1/6, -6/5, 15/4, -20/3, 15/2, -6, 49/20])
    du = np.zeros((ny,nx))
    for j in range (ny):
        for i in range (nx-1):
            if ((eps[j,i] == 1 and eps[j,i+1] == 0 and i < nx-7) or (eps[j,i] == 0 and i == 0)):
                # Forward
                du[j,i] = np.dot(field[j,i:i+7],coef_f)/(x[2]-x[1])

            elif ((eps[j,i-1] == 1 and eps[j,i] == 0 and i < nx-6) or (eps[j,i-1] == 0 and i == 1)):
                # Forward Bias 1 (1,c,5)
                du[j,i] = np.dot(field[j,i-1:i+6],coef_f1)/(x[2]-x[1])

            elif ((eps[j,i-2] == 1 and eps[j,i] == 0 and i < nx-5) or (eps[j,i-2] == 0 and i == 2)):
                # Forward Bias 2 (2,c,4)
                du[j,i] = np.dot(field[j,i-2:i+5],coef_f2)/(x[2]-x[1])

            elif eps[j,i] != 1 and eps[j,i-2] != 1 and i < nx-3 and i > 2:
                # Center difference (3,c,3)
                du[j,i] = np.dot(field[j,i-3:i+4],coef_c)/(x[2]-x[1])

            elif ((eps[j,i] == 0 and eps[j,i+1] == 1 and i < 5) or (eps[j,i] == 0 and i == nx-1)):
                # Backward
                du[j,i] = np.dot(field[j,i-6:i+1],coef_b)/(x[2]-x[1])

            elif ((eps[j,i-1] == 0 and eps[j,i] == 1 and i < 4) or (eps[j,i-1] == 0 and i == nx-2)):
                # Backward Bias 1 (5,c,1)
                du[j,i] = np.dot(field[j,i-5:i+2],coef_b1)/(x[2]-x[1])

            elif ((eps[j,i-2] == 0 and eps[j,i] == 1) or (eps[j,i-2] == 0 and i == nx-3)):
                # Backward Bias 2 (4,c,2)
                du[j,i] = np.dot(field[j,i-4:i+3],coef_b2)/(x[2]-x[1])
            else:
                print('Undefined case')
    return du



###############################################################################
############################# Varaible decleration ############################
hill_height = 94
Re = 500
Re_lambda = 0.5*Re*Re
nu = 0.8e-5
dt = 0.827E-04
index = 1
limity_range = 150
f = 1
alpha = -0.430511
Gx = np.cos(alpha)
Gz = np.sin(alpha)
u_star = 0.0845510893389349
Re_tau = (u_star**2)/nu
l_visc = nu/u_star
l_in = l_visc
l_out = u_star
time_scale = 2*np.pi
restart = 500      # legacy reference only; plane count is read from each file
counter = 0        # legacy reference only; per-file counters are used below
start = 263000     # legacy reference only; file sets are auto-discovered
last = 332500      # legacy reference only

# Controls
cal_Avg = 1
verify_TimeAvg = 1
save_avg = 1
###############################################################################
############################# Main Code #######################################

# Parameter decleration
cwd = str(os.path.dirname(__file__) + '/' )

# Read grid
x, y, z = read_grid(cwd)
eps = epsfield()

nx = np.size(x)
ny = np.size(y)
nz = np.size(z)

x_fill = x
x_fill = np.append(0, x_fill)
x_fill = np.append(x_fill, x[-1])
dx = (2*np.pi/x[-1])
y_fill = np.round((hill_height/(2**1))*(1 + np.cos(dx*(x))))
y_fill = y[y_fill.astype(int)]
y_fill = np.append(0,y_fill)
y_fill = np.append(y_fill, 0)

x_fill_plus = x_fill/l_in
y_fill_plus = y_fill/l_in

x_plus = x/l_in
y_plus = y/l_in

eps_vol = epsVolume(eps,ny,nx,hill_height)
eps_s = np.mean(eps_vol,axis=1)
eps_f = 1 - eps_s

# Forcing values in solid zero. If not it will introduce error when calculating average in x direction.
mask_zero = 1 - eps

# Precompute fluid / solid masks once (eps == 1 is solid, eps != 1 is fluid).
fluid_mask = (eps != 1)
solid_mask = (eps == 1)

# A plane is treated as corrupt if the FLUID holds NaN/Inf OR any |value| above
# this cap. Such values are finite but physically impossible (all components are
# O(1) or smaller), and they blow up the running accumulation, so the plane is
# discarded just like a NaN/Inf plane.
MAX_ABS = 1.0e4

# Calculating phase average by summing ALL good per-iteration planes
if (1 == cal_Avg):
    AvgPh = np.zeros((ny,nx,3))
    AvgStress = np.zeros((ny,nx,6))
    AvgP = np.zeros((ny,nx))
    AvgScal = np.zeros((ny,nx))
    SpaceAvgStr = np.zeros((ny,6))
    VelGbl = np.zeros((ny,3))
    VelGbl2D = np.zeros((ny,nx,3))
    Turb = np.zeros((ny,nx,6))
    DispVel = np.zeros((ny,nx,3))
    PGbl   = np.zeros((ny,))
    DispP  = np.zeros((ny,nx))
    turb1D = np.zeros((ny,6))

    # ── Buoyancy-flux cross-moments (Route C, Research.md goal 4/5) ──────────
    # The solver writes the velocity joint moments ⟨u_iu_j⟩ (avg_stress) but NOT
    # the velocity-scalar joint moments ⟨u_iθ⟩, so the turbulent buoyancy flux
    # cannot be recovered post-hoc.  Here we accumulate, across iterations, the
    # product of the per-iteration z-averaged planes  Σ_t (Ū_i·Θ̄):  its time
    # mean minus ⟨Ū_i⟩⟨Θ̄⟩ is Cov_t(Ū_i,Θ̄) — the buoyancy flux carried by the
    # unsteady spanwise-mean (wave-scale) field.  θ = scalar = buoyancy directly.
    AccUTheta = np.zeros((ny,nx))   # u·θ  (streamwise)
    AccVTheta = np.zeros((ny,nx))   # v·θ  (WALL-NORMAL — the vertical buoyancy flux)
    AccWTheta = np.zeros((ny,nx))   # w·θ  (spanwise)
    AccThTh   = np.zeros((ny,nx))   # θ·θ  (scalar variance)
    good_flux = np.int64(0)         # iterations with u,v,w AND θ all good

    # ------------------------------------------------------------------ #
    # Robust phase average                                                #
    #                                                                     #
    # Read EVERY per-iteration plane (the final plane in each file is the #
    # precomputed - and bug-corrupted - average, so it is excluded).      #
    # A plane is "good" if it has no NaN/Inf in the FLUID (eps != 1);      #
    # solid (eps == 1) non-finite values are expected and ignored.        #
    # Each of the 10/11 component files keeps its OWN accumulator and its  #
    # OWN good-plane counter; counters/accumulators are NOT reset between  #
    # iteration sets.                                                      #
    # ------------------------------------------------------------------ #

    # Per-file good-plane counters (order: flow .1/.2/.3, stress .1..6,
    # pressure .1, scalar .1 [, scalar .2]).
    good_flow = np.zeros(3, dtype=np.int64)
    good_str  = np.zeros(6, dtype=np.int64)
    good_p    = np.int64(0)
    nscal     = None          # detected (1 or 2) from the first complete set
    good_sc   = None
    AvgScal2  = None

    # Discover all avg_flow*_*.1 files and extract the <srt>_<end> tokens.
    flow1_files = [fn for fn in os.listdir(cwd)
                   if re.fullmatch(r'avg_flow\d+_\d+\.1', fn)]
    tokens = []
    for fn in flow1_files:
        m = re.fullmatch(r'avg_flow(\d+_\d+)\.1', fn)
        if m:
            tokens.append(m.group(1))
    # Deterministic order, sorted by the start iteration.
    tokens = sorted(set(tokens), key=lambda t: int(t.split('_')[0]))
    if (len(tokens) == 0):
        print("ERROR: no avg_flow*_*.1 files found in %s" % cwd)
        quit()
    print("Found %d iteration set(s): %s" % (len(tokens), ', '.join(tokens)))

    # Overlap check: two files covering the same iteration number is a logical
    # error in the file naming or the simulation itself.
    parsed_ranges = [(int(t.split('_')[0]), int(t.split('_')[1]), t) for t in tokens]
    overlap_found = False
    for i in range(1, len(parsed_ranges)):
        prev_start, prev_end, prev_tok = parsed_ranges[i - 1]
        curr_start, curr_end, curr_tok = parsed_ranges[i]
        if curr_start <= prev_end:
            print("ERROR: iteration overlap between sets '%s' (ends at %d) and "
                  "'%s' (starts at %d) — two files cover the same iteration. "
                  "This is a file-naming or simulation error; cannot continue."
                  % (prev_tok, prev_end, curr_tok, curr_start))
            overlap_found = True
    if overlap_found:
        quit()

    nplanes_ref = None        # plane count of the first valid set (for warnings)

    for tok in tokens:
        # Build EVERY component path from THIS token only, so components of
        # different iteration ranges can never be mixed up.
        flow_paths = [cwd + 'avg_flow'   + tok + '.' + str(c) for c in (1, 2, 3)]
        str_paths  = [cwd + 'avg_stress' + tok + '.' + str(c) for c in range(1, 7)]
        p_path     =  cwd + 'avg_p'      + tok + '.1'
        scal_paths = [cwd + 'avg_scal'   + tok + '.1']
        if os.path.exists(cwd + 'avg_scal' + tok + '.2'):
            scal_paths.append(cwd + 'avg_scal' + tok + '.2')

        # Number of scalars must be consistent across all sets.
        this_nscal = len(scal_paths)
        if (nscal is None):
            nscal   = this_nscal
            good_sc = np.zeros(nscal, dtype=np.int64)
            # Persistent bad-plane totals per component, summed over ALL sets.
            tot_skip_nan = np.zeros(10 + nscal, dtype=np.int64)   # NaN/Inf in fluid
            tot_skip_big = np.zeros(10 + nscal, dtype=np.int64)   # |val| > MAX_ABS in fluid
            if (nscal == 2):
                AvgScal2 = np.zeros((ny, nx))
            print("Detected %d scalar field(s)." % nscal)
        elif (this_nscal != nscal):
            print("WARNING: set %s has %d scalar(s) but %d expected; skipping set."
                  % (tok, this_nscal, nscal))
            continue

        comp_paths = flow_paths + str_paths + [p_path] + scal_paths

        # Sanity: all component files must exist.
        missing = [p for p in comp_paths if not os.path.exists(p)]
        if missing:
            print("WARNING: set %s missing %d component file(s); skipping. First missing: %s"
                  % (tok, len(missing), os.path.basename(missing[0])))
            continue

        # Read EACH file's own header (offsets differ between files) and verify
        # header dims and a consistent plane count across all components.
        offsets = []
        nplanes_set = None
        ok = True
        for p in comp_paths:
            offset, hnx, hny, hnz, _, _ = read_header(p)
            if (offset is None):
                print("WARNING: cannot read header of %s; skipping set %s."
                      % (os.path.basename(p), tok))
                ok = False
                break
            if (hnx != nx) or (hny != ny):
                print("WARNING: %s header dims (%d x %d) != grid (%d x %d); skipping set %s."
                      % (os.path.basename(p), hnx, hny, nx, ny, tok))
                ok = False
                break
            nbytes = os.path.getsize(p) - offset
            if (nbytes % (nx * ny * 8) != 0):
                print("WARNING: %s data size is not a whole number of planes; skipping set %s."
                      % (os.path.basename(p), tok))
                ok = False
                break
            np_this = nbytes // (nx * ny * 8)
            if (nplanes_set is None):
                nplanes_set = np_this
            elif (np_this != nplanes_set):
                print("WARNING: %s has %d planes but %d expected in set %s; skipping set."
                      % (os.path.basename(p), np_this, nplanes_set, tok))
                ok = False
                break
            offsets.append(offset)
        if (not ok):
            continue

        if (nplanes_set < 2):
            print("WARNING: set %s has only %d plane(s); nothing to average; skipping."
                  % (tok, nplanes_set))
            continue

        # Restart size can legitimately differ between sets when the simulation
        # was restarted with a different averaging interval; log it as information.
        if (nplanes_ref is None):
            nplanes_ref = nplanes_set
        elif (nplanes_set != nplanes_ref):
            print("Note: set %s uses Restart=%d (%d planes) vs Restart=%d (%d planes) "
                  "in the first set — expected when the simulation was restarted with "
                  "a different averaging interval; proceeding."
                  % (tok, nplanes_set - 1, nplanes_set, nplanes_ref - 1, nplanes_ref))

        ndata = nplanes_set - 1   # exclude the final (corrupt) precomputed-average plane
        print("Set %s: %d planes total; averaging the first %d (final avg plane excluded)."
              % (tok, nplanes_set, ndata))

        # Open every component once, seek to its own data start, then stream the
        # planes sequentially. Decisions are made PER FILE.
        handles = [open(p, 'rb') for p in comp_paths]
        for h, off in zip(handles, offsets):
            h.seek(off)
        skipped_nan = np.zeros(len(comp_paths), dtype=np.int64)
        skipped_big = np.zeros(len(comp_paths), dtype=np.int64)
        try:
            for kpl in range(ndata):
                cur = {}   # this iteration's good u/v/w/θ planes (for the flux cross-moments)
                for c in range(len(comp_paths)):
                    plane = read_next_plane(handles[c], nx, ny)
                    nf   = ~np.isfinite(plane)             # NaN or Inf
                    huge = np.abs(plane) > MAX_ABS         # finite but crazy large
                    # Corrupt if the FLUID holds NaN/Inf or any |value| > MAX_ABS.
                    if (nf & fluid_mask).any():
                        skipped_nan[c] += 1
                        continue
                    if (huge & fluid_mask).any():
                        skipped_big[c] += 1
                        continue
                    # mask0 cannot clear these (NaN*0 = NaN); zero any non-finite
                    # OR over-cap SOLID cells so they never pollute the
                    # accumulation or the x-averages (VelGbl).
                    bad_solid = (nf | huge) & solid_mask
                    if bad_solid.any():
                        plane[bad_solid] = 0.0
                    if (c <= 2):
                        AvgPh[:, :, c]       += plane
                        good_flow[c]         += 1
                        cur[c] = plane                      # u (0), v (1), w (2)
                    elif (c <= 8):
                        AvgStress[:, :, c-3] += plane
                        good_str[c-3]        += 1
                    elif (c == 9):
                        AvgP[:, :]           += plane
                        good_p               += 1
                    elif (c == 10):
                        AvgScal[:, :]        += plane
                        good_sc[0]           += 1
                        cur[10] = plane                     # θ (scalar 1 = buoyancy)
                    else:                                   # c == 11 (second scalar)
                        AvgScal2[:, :]       += plane
                        good_sc[1]           += 1
                # Buoyancy-flux cross-moments: only when u,v,w AND θ were all good
                # for THIS iteration (so the product is from one consistent snapshot).
                if all(k in cur for k in (0, 1, 2, 10)):
                    _th = cur[10]
                    AccUTheta += cur[0] * _th
                    AccVTheta += cur[1] * _th
                    AccWTheta += cur[2] * _th
                    AccThTh   += _th    * _th
                    good_flux += 1
        finally:
            for h in handles:
                h.close()

        # Per-set skip summary.
        for c, p in enumerate(comp_paths):
            if (skipped_nan[c] > 0) or (skipped_big[c] > 0):
                print("   %s: skipped %d of %d plane(s) -> %d NaN/Inf, %d |val|>%.0e (in fluid)."
                      % (os.path.basename(p), skipped_nan[c] + skipped_big[c], ndata,
                         skipped_nan[c], skipped_big[c], MAX_ABS))

        # Accumulate this set's bad-plane counts into the run totals.
        tot_skip_nan += skipped_nan
        tot_skip_big += skipped_big

    print('Avg stress calculation')

    # ------------------------------------------------------------------ #
    # Good / bad plane accounting (totals over ALL sets, per component).   #
    # ------------------------------------------------------------------ #
    good_all = np.array(list(good_flow) + list(good_str) + [int(good_p)] + list(good_sc),
                        dtype=np.int64)
    comp_labels = (['flow.1', 'flow.2', 'flow.3',
                    'stress.1', 'stress.2', 'stress.3', 'stress.4', 'stress.5', 'stress.6',
                    'p.1'] + ['scal.%d' % (s + 1) for s in range(nscal)])
    print("---------------------------------------------------------------")
    print("Plane accounting (good vs bad), summed over all iteration sets:")
    tot_good = 0
    tot_bad  = 0
    for c in range(len(good_all)):
        g  = int(good_all[c])
        bn = int(tot_skip_nan[c])
        bb = int(tot_skip_big[c])
        b  = bn + bb
        print("   %-9s : good=%d  bad=%d (NaN/Inf=%d, |val|>%.0e=%d)  total=%d"
              % (comp_labels[c], g, b, bn, MAX_ABS, bb, g + b))
        tot_good += g
        tot_bad  += b
    print("   %-9s : good=%d  bad=%d  total=%d"
          % ('ALL', tot_good, tot_bad, tot_good + tot_bad))
    print("---------------------------------------------------------------")

    # ------------------------------------------------------------------ #
    # Averages: divide each accumulator by its OWN good-plane counter.     #
    # ------------------------------------------------------------------ #
    if any(int(cnt) == 0 for cnt in good_all):
        print("ERROR: a component has 0 good planes (fully corrupt); cannot average.")
        quit()

    for i in range (3):
        AvgPh[:,:,i]   = AvgPh[:,:,i] / good_flow[i]
        VelGbl[:,i]    = np.mean((AvgPh[:,:,i]), axis = 1)
        DispVel[:,:,i] = (AvgPh[:,:,i] - VelGbl[:,i][:,np.newaxis])*mask_zero
    for i in range (6):
        AvgStress[:, :, i] = AvgStress[:, :, i] / good_str[i]
        SpaceAvgStr[:,i]   = np.mean((AvgStress[:,:,i]), axis = 1)
    AvgP[:,:] = (AvgP[:,:]*mask_zero)/good_p
    PGbl[:,]  = np.mean(AvgP, axis=1)
    DispP[:]  = (AvgP - PGbl[:,np.newaxis]) * mask_zero
    AvgScal[:,:] = (AvgScal[:,:]*mask_zero)/good_sc[0]
    if (nscal == 2):
        AvgScal2[:,:] = (AvgScal2[:,:]*mask_zero)/good_sc[1]

    # Time means of the buoyancy cross-moments ⟨Ū_i·Θ̄⟩_t  (Route C).  Downstream
    # (PhAvg_rotated.py) the turbulent buoyancy flux is recovered by subtracting
    # the mean product:  Cov_t(Ū_i,Θ̄) = MeanU_iTheta − AvgPh_i·AvgScal.
    if (good_flux > 0):
        MeanUTheta = (AccUTheta * mask_zero) / good_flux
        MeanVTheta = (AccVTheta * mask_zero) / good_flux
        MeanWTheta = (AccWTheta * mask_zero) / good_flux
        MeanThTh   = (AccThTh   * mask_zero) / good_flux
        print("Buoyancy cross-moments accumulated over %d good iteration(s)." % int(good_flux))
    else:
        MeanUTheta = np.zeros((ny, nx)); MeanVTheta = np.zeros((ny, nx))
        MeanWTheta = np.zeros((ny, nx)); MeanThTh   = np.zeros((ny, nx))
        print("WARNING: no iteration had u,v,w AND θ all good; buoyancy flux = 0.")

    # Report placement (solid vs fluid) of any non-finite values left in the
    # averaged fields. After skipping corrupt planes these should all be clean.
    vel_names = ['AvgPh_u', 'AvgPh_v', 'AvgPh_w']
    str_names = ['AvgStress_0', 'AvgStress_1', 'AvgStress_2',
                 'AvgStress_3', 'AvgStress_4', 'AvgStress_5']
    for i in range (3):
        report_nan_location(AvgPh[:, :, i], eps, vel_names[i])
        print_fluid_nans(AvgPh[:, :, i], eps, vel_names[i], x, y)
    for i in range (6):
        report_nan_location(AvgStress[:, :, i], eps, str_names[i])
        print_fluid_nans(AvgStress[:, :, i], eps, str_names[i], x, y)
    report_nan_location(AvgP, eps, 'AvgP')
    print_fluid_nans(AvgP, eps, 'AvgP', x, y)
    report_nan_location(AvgScal, eps, 'AvgScal')
    print_fluid_nans(AvgScal, eps, 'AvgScal', x, y)
    if (nscal == 2):
        report_nan_location(AvgScal2, eps, 'AvgScal2')
        print_fluid_nans(AvgScal2, eps, 'AvgScal2', x, y)

    for i in range (3):
        VelGbl2D[:,:,i] = (np.tile(VelGbl[:,i].reshape(ny,1), nx).reshape(ny,nx))*mask_zero

    for i in range(6):
        turb1D[:,i] = np.mean(Turb[:,:,i], axis=1)

    uu_t = DispVel[:,:,0]*DispVel[:,:,0]
    uv_t = DispVel[:,:,0]*DispVel[:,:,1]
    uw_t = DispVel[:,:,0]*DispVel[:,:,2]
    vv_t = DispVel[:,:,1]*DispVel[:,:,1]
    vw_t = DispVel[:,:,1]*DispVel[:,:,2]
    ww_t = DispVel[:,:,2]*DispVel[:,:,2]

    uu_g = VelGbl2D[:,:,0]*VelGbl2D[:,:,0]
    uv_g = VelGbl2D[:,:,0]*VelGbl2D[:,:,1]
    uw_g = VelGbl2D[:,:,0]*VelGbl2D[:,:,2]
    vv_g = VelGbl2D[:,:,1]*VelGbl2D[:,:,1]
    vw_g = VelGbl2D[:,:,1]*VelGbl2D[:,:,2]
    ww_g = VelGbl2D[:,:,2]*VelGbl2D[:,:,2]

    udug = DispVel[:,:,0]*VelGbl2D[:,:,0]
    ugud = udug
    udvg = DispVel[:,:,0]*VelGbl2D[:,:,1]
    ugvd = VelGbl2D[:,:,0]*DispVel[:,:,1]
    udwg = DispVel[:,:,0]*VelGbl2D[:,:,2]
    ugwd = VelGbl2D[:,:,0]*DispVel[:,:,2]

    vgvd = DispVel[:,:,1]*VelGbl2D[:,:,1]
    vdvg = vgvd
    vdwg = DispVel[:,:,1]*VelGbl2D[:,:,2]
    vgwd = VelGbl2D[:,:,1]*DispVel[:,:,2]

    wdwg = DispVel[:,:,2]*VelGbl2D[:,:,2]
    wgwd = wdwg


    uu_d = AvgStress[:,:,0] - uu_g - uu_t - udug - ugud
    uv_d = AvgStress[:,:,1] - uv_g - uv_t - udvg - ugvd
    uw_d = AvgStress[:,:,2] - uw_g - uw_t - udwg - ugwd
    vv_d = AvgStress[:,:,3] - vv_g - vv_t - vdvg - vgvd
    vw_d = AvgStress[:,:,4] - vw_g - vw_t - vdwg - vgwd
    ww_d = AvgStress[:,:,5] - ww_g - ww_t - wdwg - wgwd

# Write the required varaible in a file

if (1 == save_avg):
    print('Writing averages')
    np.save('uu_d.npy', uu_d)
    np.save('uv_d.npy', uv_d)
    np.save('uw_d.npy', uw_d)
    np.save('vv_d.npy', vv_d)
    np.save('vw_d.npy', vw_d)
    np.save('ww_d.npy', ww_d)

    np.save('AvgStrUU.npy', AvgStress[:,:,0])
    np.save('AvgStrUV.npy', AvgStress[:,:,1])
    np.save('AvgStrUW.npy', AvgStress[:,:,2])
    np.save('AvgStrVV.npy', AvgStress[:,:,3])
    np.save('AvgStrVW.npy', AvgStress[:,:,4])
    np.save('AvgStrWW.npy', AvgStress[:,:,5])

    np.save('uu_g.npy', uu_g)
    np.save('uv_g.npy', uv_g)
    np.save('uw_g.npy', uw_g)
    np.save('vv_g.npy', vv_g)
    np.save('vw_g.npy', vw_g)
    np.save('ww_g.npy', ww_g)

    np.save('uu_t.npy', uu_t)
    np.save('uv_t.npy', uv_t)
    np.save('uw_t.npy', uw_t)
    np.save('vv_t.npy', vv_t)
    np.save('vw_t.npy', vw_t)
    np.save('ww_t.npy', ww_t)

    np.save('AvgPhU.npy', AvgPh[:,:,0])
    np.save('AvgPhV.npy', AvgPh[:,:,1])
    np.save('AvgPhW.npy', AvgPh[:,:,2])
    np.save('AvgP.npy', AvgP[:,:])
    np.save('PGbl.npy', PGbl)
    np.save('DispP.npy', DispP)
    np.save('AvgScal.npy', AvgScal[:,:])
    if (nscal == 2):
        np.save('AvgScal2.npy', AvgScal2[:,:])

    # Buoyancy-flux cross-moments (Route C) consumed by PhAvg_rotated.py.
    np.save('MeanUTheta.npy', MeanUTheta)
    np.save('MeanVTheta.npy', MeanVTheta)
    np.save('MeanWTheta.npy', MeanWTheta)
    np.save('MeanThTh.npy',   MeanThTh)

    np.save('VelGblU.npy', VelGbl[:,0])
    np.save('VelGblV.npy', VelGbl[:,1])
    np.save('VelGblW.npy', VelGbl[:,2])

    np.save('DispVelU', DispVel[:,:,0])
    np.save('DispVelV', DispVel[:,:,1])
    np.save('DispVelW', DispVel[:,:,2])

    np.save('udug.npy', udug)
    np.save('udvg.npy', udvg)
    np.save('udwg.npy', udwg)
    np.save('vdvg.npy', vdvg)
    np.save('vdwg.npy', vdwg)
    np.save('wdwg.npy', wdwg)

    np.save('ugud.npy', ugud)
    np.save('ugvd.npy', ugvd)
    np.save('ugwd.npy', ugwd)
    np.save('vgvd.npy', vgvd)
    np.save('vgwd.npy', vgwd)
    np.save('wgwd.npy', wgwd)
