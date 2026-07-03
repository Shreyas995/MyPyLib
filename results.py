#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:13:16 2024

@author: shreyas deshpande
"""

import os
import re
import sys
import glob as _glob
import csv
import struct
import math
import pickle
import netCDF4 as nc
import numpy as np
from PlotField import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import simpson
from scipy.integrate import trapezoid
from scipy.stats import linregress
import matplotlib.animation as animation
from matplotlib import cm
from PIL import Image
from functions import *

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
    f = open(cwd1 + fname,'rb')
    f.seek(0,0)
    header = np.fromfile(f, type_i4, head_params)
    f.close()
    print('Header size           :', header[0])
    print('Grid   size (nx*ny*nz):', header[1]*8,'x',header[2],'x',header[3])

    # data size (attention: h[1] = grid.nx*8!)
    bsize = np.prod(header[1:3])
    rsize = bsize * 8

    # read eps field as int1
    f = open(cwd1 + fname,'rb')
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

def epsVolume(eps,ny,nx, hill_hgt):
    eps_vol = np.zeros((ny,nx))
    
    for j in range (hill_hgt):
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
    ifile=open(input_FilePath,'r')
    ifile.seek(hdr)
    for iz in range(Nz):
        data_block[:,:]=np.fromfile(ifile, dtype=np.float64, count=Nx*Ny).reshape([Ny,Nx])
        field[:,:,iz] = data_block
    ifile.close
    return field

def readplane(path, Nx, Ny, pl_id, hdr):
    plane = np.zeros((Ny,Nx))
    input_FilePath = path 
    ifile=open(input_FilePath,'r')
    ifile.seek(hdr + Nx*Ny*(pl_id-1)*8)
    plane[:,:]=np.fromfile(ifile, dtype=np.float64, count=Nx*Ny).reshape([Ny,Nx])
    ifile.close
    return plane

def diffu_dy(field, ny, nx, eps, y):    # ny is number of points in vertical
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
                du[j,i] = np.dot(field[j:j+7,i], coef_f)/np.dot(y[j:j+7], coef_f)
                
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
    return du

def diffu_dx(field, ny, nx, eps, x):    # ny is number of points in vertical
    coef_f  = np.array([-49/20, 6, -15/2, 20/3, -15/4, 6/5, -1/6])
    coef_f1 = np.array([-1/6, -77/60, 5/2, -5/3, 5/6, -1/4, 1/30])
    coef_f2 = np.array([1/30, -2/5, -7/12, 4/3, -1/2, 2/15, -1/60])
    coef_c =  np.array([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
    coef_b2 = np.array([1/60, -2/15, 1/2, -4/3, 7/12, 2/5, -1/30])
    coef_b1 = np.array([-1/30, 1/4, -5/6, 5/3, -5/2, 77/60, 1/6])
    coef_b = np.array([1/6, -6/5, 15/4, -20/3, 15/2, -6, 49/20])
    du = np.zeros((ny,nx))
    for j in range (ny):
        for i in range (nx):
            if ((eps[j,i] == 1 and eps[j,(i+1)%nx] == 0 and i < nx-7) or (eps[j,i] == 0 and i == 0)):
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
                
            elif ((eps[j,i] == 0 and eps[j,(i+1)%nx] == 1 and i < 5) or (eps[j,i] == 0 and i == nx-1)):
                # Backward 
                du[j,i] = np.dot(field[j,i-6:i+1],coef_b)/(x[2]-x[1])
                
            elif ((eps[j,i-1] == 0 and eps[j,i] == 1 and i < 4) or (eps[j,i-1] == 0 and i == nx-2)):
                # Backward Bias 1 (5,c,1)
                du[j,i] = np.dot(field[j,i-5:i+2],coef_b1)/(x[2]-x[1])
                
            elif ((eps[j,i-2] == 0 and eps[j,i] == 1) or (eps[j,i-2] == 0 and i == nx-3)):
                # Backward Bias 2 (4,c,2)
                du[j,i] = np.dot(field[j,i-4:i+3],coef_b2)/(x[2]-x[1])
                
            elif (j == 0):
                # Center difference 2 (4,c,2)
                du[j,i] = 0
                
            elif (i>2 and (i<(nx-3)) and (eps[j,i] == 0)):
                du[j,i] = np.dot(field[j,i-3:i+4],coef_c)/(x[2]-x[1])
                
            elif (i<3 and (i>(nx-3)) and (eps[j,i] == 0)):
                du[j,i] = np.dot(np.concatenate((field[j,i-3:],field[j,i+4])),coef_c)/(x[2]-x[1])
    return du

def vIntegral(varaible, ny, y): # ny is number of points in vertical
    I = np.zeros((ny))
    for j in range (1,ny):
        if j == 1:
            I[j] = trapezoid(varaible[:j],x=y[:j])
        elif j > 1:
            I[j] = simpson(varaible[:j],x=y[:j])
    return I

def vIntegral2(varaible, ny, y):
    I = np.zeros((ny))
    for j in range (1,ny):
        I[j] = I[j-1] + 0.5*(np.abs(varaible[j]) + np.abs(varaible[j-1]))/(y[j]-y[j-1])
    return I

def createIntegrate(surf_hor, n, i_id, variable, x, side):
    if side == 'LHS':
        indj = np.where(surf_hor[0,:int(n)] == i_id)[0]
    else:
        indj = np.where(surf_hor[0,int(n):] == i_id)[0]
    min_ind = indj.min()
    max_ind = indj.max()+1
    sigma = variable[0,min_ind:max_ind]
    coords = x[min_ind:max_ind]
    I = simpson(sigma, x=coords)
    return I

def compute_r_squared(d, z, u):
    # Apply the same filtering to both z and u
    valid_indices = z > d
    z_d = z[valid_indices] - d
    ln_z_d = np.log(z_d)
    u_valid = u[valid_indices]
    slope, intercept, r_value, _, _ = linregress(ln_z_d, u_valid)
    return r_value**2, slope, intercept

def update_frame(frame):
    # Read the corresponding plane file
    filename = f'Plane{frame}.txt'  # Assuming your filenames are Plane0.txt, Plane1.txt, ...
    field = read_field(filename)    # Read the field for this frame
    im.set_data(field)              # Update the data for the image
    return [im]

def plot_frame(ax, x, y, field_2D):
    Y, X = np.meshgrid(y, x)
    if np.max(field_2D) > abs(np.min(field_2D)):
        ll = -np.max(field_2D)
        ul = np.max(field_2D)
    else:
        ll = np.min(field_2D)
        ul = -np.min(field_2D)
    
    contourf = ax.contourf(X, Y, field_2D.T, cmap='seismic', levels=100, vmin=ll, vmax=ul)
    return contourf

# def update_frame(frame):
#     # path =
#     pl_id = frame_ids[frame]  # Get the specific plane ID from the list
#     field = read_plane(path, Nx, Ny, pl_id)  # Read the field for this plane
#     im.set_data(field)  # Update the data for the image
#     return [im]
###############################################################################
############################# Varaible decleration ############################

hill_hgt = 94 # This value is dummy. True value is calculated after reading the eps field
step = 1 # Vertical step intervals of the valley in terms of grid points
Re = 500
Re_lambda = 0.5*Re*Re
nu = 1/Re_lambda
dt = 0.827E-04
index = 1
limity_range = 150
limity = 380
f = 1
alpha = -0.430511
Gx = np.cos(alpha)
Gz = -np.sin(alpha)
# SINGLE-REFERENCE friction velocity for ALL cross-case nondimensionalisation.
# Settled choice: every case (smooth + neutral + all finite-Fr orographic) is
# scaled by the Re=500 Fr=inf smooth-neutral reference u* = 0.0618 (the smooth
# .nc stored FrictionVelocity, == ustr_s1), NOT by each case's own u_star2.  This
# fixes inner units (z+, u+), outer units (y/u*), stress/u*^2 and the BL-thickness
# markers onto one common yardstick so the comparison is on equal footing.
# (The per-case PHYSICAL u* — Method-2 plateau of u_star2 — is still pickled per
#  case and used for each case's own physical scales: delta, Re_tau, C_D, Psi.)
u_star = 0.0618
kappa = 0.42
Re_tau = (u_star**2)/nu
l_visc = nu/u_star
l_in = l_visc
l_out = u_star
time_scale = 2*np.pi
restart = 500
counter = 0
wall_units = nu/u_star
scal = 1
dim = 3 

# Controls
cal_Avg = 0
verify_TimeAvg = 0
save_avg = 0
load_ncfiles = 0
load_arrays = 1
postprocess = 1
plotRes = 1
animate = 0

###############################################################################
############################# Main Code #######################################

# Parameter decleration
# cwd = the directory in which THIS results.py lives.  It is meant to sit in the
# central "examples" tree that holds every simulation directory, so all case
# paths are taken relative to it (os.path.dirname(__file__) follows the symlink's
# own location, not its MyPyLib target, so the script is relocatable: move the
# whole examples tree and the paths still resolve).
cwd = str(os.path.dirname(__file__) + '/' )
# Six cases ordered by decreasing Froude number (increasing stratification), Re = 500 for all.
# Smooth (NetCDF): flat wall, Fr = ∞  — loaded separately below from Re500 NetCDF.
# Re=500 Froude ladder (neutral + Fr = 1, 0.1, 0.01).  The neutral run lives in
# Ekman18/; the stratified runs live on a finer grid (1056x672x1056) in a
# separate subtree.  All are sub-directories of cwd (the examples root), so the
# data is found wherever that root is placed.  Re=750 not yet available.
_base = cwd                                      # examples root = where results.py sits
cwd1 = _base + 'Ekman18/'                        # Neutral      Fr = ∞    (valley present)
cwd2 = _base + '1056x672x1056/EkRe500Fr1/'       # Strat        Fr = 1    (valley present)
cwd3 = _base + '1056x672x1056/EkRe500Fr0.1/'     # Strat        Fr = 0.1  (valley present)
cwd4 = _base + '1056x672x1056/EkRe500Fr0.01/'    # Strat        Fr = 0.01 (valley present)
# Reference grid + IBM geometry come from the NEUTRAL case (cwd1): the central
# examples/ root holds no grid of its own, and the valley geometry (eps, nx, ny,
# hill_hgt) used for the 2-D orographic plots is the neutral-case grid.
x, y, z = read_grid(cwd1)

nx = np.size(x)
ny = np.size(y)
nz = np.size(z)
    
try:
    eps = np.load(cwd1 + 'eps_save.npy')   # IBM indicator from the neutral case grid
    print('eps loaded')
except:
    print('Needed to read eps field')
    eps = epsfield()

eps_top = int(0)         # horizontal grid position at valley top
eps_lf = int(nx/4)       # horizontal grid position at valley left flank
eps_bottom = int(nx/2)   # horizontal grid position at valley bottom
eps_rf = int(nx*0.75)    # horizontal grid position at valley right flank
        
eps_hgt = np.sum(eps, axis=0).astype(int)
hill_hgt = np.max(eps_hgt) - 1 # Directly take hill height from the eps field. THe real height is value -1.
# If no geomtery is created, there is 1 row where velocity is zero so we have + 1 no of eps 
eps_vol = epsVolume(eps,ny,nx,hill_hgt)
eps_s = np.mean(eps_vol,axis=1)
eps_f = 1 - eps_s

flk_hgt = eps_hgt[int(eps_lf)]
flk_wdt = np.where(eps_hgt == flk_hgt)[0]
lf_ind = flk_wdt[:int((len(flk_wdt))/2)]
rf_ind = flk_wdt[int((len(flk_wdt))/2):]
x_oro = x 
x_oro = np.append(0, x_oro)
x_oro = np.append(x_oro, x[-1])
dx = (2*np.pi/x[-1])
y_oro = np.round((hill_hgt/(2**1))*(1 + np.cos(dx*(x))))
y_oro = y[y_oro.astype(int)]
y_oro = np.append(0,y_oro)
y_oro = np.append(y_oro, 0)

x_oro_in = x_oro/l_in
y_oro_in = y_oro/l_in

x_in = x/l_in
y_in = y/l_in

# Forcing values in solid zero. If not it will introduce error when calculating average in x direction.
mask0 = 1 - eps

###############################################################################
# MASTER PLOT CONTROL
# Set PLOT_SMOOTH and ACTIVE_CASES to choose which simulations appear in
# every plot.  Comment out any name to exclude it; all styling is automatic.
###############################################################################
PLOT_SMOOTH = True   # smooth-wall NetCDF reference (flat wall, Fr = inf)
ACTIVE_CASES = {
    'nu_oro',       # Neutral,     Fr = inf
    'fr_1_oro',     # Strat,       Fr = 1
    'fr_0p1_oro',   # Strat,       Fr = 0.1
    'fr_0p01_oro',  # Strat,       Fr = 0.01
}
###############################################################################

###############################################################################
# CASES — all six simulations (smooth first, then rough-wall by decreasing Fr)
# Re = 500 for all cases.
# Coordinate convention: simulation uses engineering coords
#   u/x = streamwise, v/y = wall-normal, w/z = spanwise.
#   Plots label the wall-normal axis as z+ (meteorological convention).
###############################################################################
CASES = [
    {'name': 'Sm_Neu',      'label': r'Smooth ($Fr=\infty$, flat)', 'color': '#636363', 'ls': '-',       'marker': 'o'},
    {'name': 'nu_oro',      'label': r'Neutral ($Fr=\infty$)',      'color': '#1565C0', 'ls': '--',      'marker': 's'},
    {'name': 'fr_1_oro',    'label': r'$Fr=1$',                     'color': '#00838F', 'ls': '-.',      'marker': '^'},
    {'name': 'fr_0p1_oro',  'label': r'$Fr=0.1$',                   'color': '#2E7D32', 'ls': ':',       'marker': 'D'},
    {'name': 'fr_0p01_oro', 'label': r'$Fr=0.01$',                  'color': '#E65100', 'ls': (0,(5,2)), 'marker': 'v'},
]

SIM_DIRS = {
    'nu_oro':      cwd1,
    'fr_1_oro':    cwd2,
    'fr_0p1_oro':  cwd3,
    'fr_0p01_oro': cwd4,
}

###############################################################################
# Cross-case statistics log.  Tee ALL subsequent stdout (data-loading messages,
# the existing research-matrix / coupling tables, the new Ch.6 diagnostics, and
# the end-of-run summary) to sim_stats.log in the examples root.  This is the
# CROSS-CASE log; it is distinct from the per-case sim_stats.log that
# PhAvg_rotated.py writes inside each case directory (different directories, no
# clobber).  Rewritten fresh ('w') on each run by start_stats_log.
###############################################################################
import datetime as _dt
start_stats_log(cwd + 'sim_stats.log')
print('=' * 78)
print('results.py — CROSS-CASE post-processing log   %s'
      % _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print('Examples root : %s' % cwd)
print('Case dirs     : %s' % {k: v for k, v in SIM_DIRS.items()})
print('=' * 78)

###############################################################################
# Load pickles — one per rough-wall simulation
###############################################################################
sims = {}
# Per-case provenance for the end-of-run summary (filled during loading below).
_prov = {_n: {'pickle': False, 'per_case_grid': False,
              'inst': {}, 'inst_skip': {}} for _n in SIM_DIRS}
for _name, _d in SIM_DIRS.items():
    _pkl = _d + 'sim1_results.pkl'
    if os.path.exists(_pkl):
        with open(_pkl, 'rb') as _fh:
            sims[_name] = pickle.load(_fh)
        print(f'Loaded: {_pkl}')
        _prov[_name]['pickle'] = True
        _prov[_name]['per_case_grid'] = ('y' in sims[_name])
        if 'y' not in sims[_name]:
            print(f'  Note: {_name} pickle carries no per-case grid (legacy/stale '
                  f'— regenerate with the current PhAvg_rotated.py + saveresults.py). '
                  f'It will be placed on the neutral axis only if its profile length '
                  f'matches; otherwise it is skipped on the shared z+ plots.')
    else:
        print(f'Warning: {_pkl} not found — skipping {_name}')

###############################################################################
# Load smooth-wall reference (Re=500, NetCDF) — used as benchmark in plots
###############################################################################
_nc_smooth = cwd1 + 'Re500/ri00.00_re0500_2048x0192x2048_20110615_avg_all.nc'
_smooth_loaded = False
if os.path.exists(_nc_smooth):
    # Smooth case computed by the SINGLE shared loader (functions.load_smooth_case),
    # identical to PhAvg.py — the two scripts can no longer diverge.  Uses PhAvg's
    # validated formulas: TKE_s = 0.5*(Rxx+Ryy+Rzz); Coriolis cor_yx_s = -(W_s-G_z)
    # with scalar geostrophic G_z = max(W_s); Fornberg du_dy_s.
    _sm = load_smooth_case(_nc_smooth, x, nu, Re_lambda)
    sy = _sm['sy']; nys = _sm['nys']
    U_s = _sm['U_s']; V_s = _sm['V_s']; W_s = _sm['W_s']
    su = _sm['su']; sw = _sm['sw']; alpha_s = _sm['alpha_s']
    ustr_s1 = _sm['ustr_s1']; alpha_str_s = _sm['alpha_str_s']
    y_s = _sm['y_s']; y_in_s = _sm['y_s_p']          # results.py uses the name y_in_s
    rU_s = _sm['rU_s']; rV_s = _sm['rV_s']; rW_s = _sm['rW_s']
    G_x_s = _sm['G_x_s']; G_z_s = _sm['G_z_s']; G_s = _sm['G_s']
    U_s_p = _sm['U_s_p']; W_s_p = _sm['W_s_p']
    GblU_s = _sm['GblU_s']; GblW_s = _sm['GblW_s']
    Rxx_s = _sm['Rxx_s']; Rxy_s = _sm['Rxy_s']; Ryy_s = _sm['Ryy_s']
    Ryz_s = _sm['Ryz_s']; Rzz_s = _sm['Rzz_s']
    TKE_s = _sm['TKE_s']
    cor_yx_s = _sm['cor_yx_s']; I_corr_yx_s = _sm['I_corr_yx_s']
    du_dy_s = _sm['du_dy_s']; visc_yx_s = _sm['visc_yx_s']; tau_yx_s = _sm['tau_yx_s']
    cor_yz_s = _sm['cor_yz_s']; I_corr_yz_s = _sm['I_corr_yz_s']
    dw_dy_s = _sm['dw_dy_s']; visc_yz_s = _sm['visc_yz_s']; tau_yz_s = _sm['tau_yz_s']
    AVG_TKE_V_s = _sm['AVG_TKE_V_s']; AVG_TKE_V_s_i = _sm['AVG_TKE_V_s_i']

    # results.py-specific derived quantities (TKE advection field), from loader outputs
    sx       = np.linspace(0, 1.08, rU_s.shape[1])
    TKE_s_dx = np.gradient(TKE_s, sx,  axis=1)
    TKE_s_dy = np.gradient(TKE_s, y_s, axis=0)
    Adv_s    = rU_s * TKE_s_dx + rV_s * TKE_s_dy
    _smooth_loaded = True
else:
    print(f'Warning: Smooth NetCDF not found at {_nc_smooth}')

if not PLOT_SMOOTH:
    _smooth_loaded = False
_ustar_ref = ustr_s1 if os.path.exists(_nc_smooth) else u_star

###############################################################################
# Derived quantities stored back into each sim dict
###############################################################################
for _name, _sd in sims.items():
    if 'rey_uv' in _sd and 'du_dy' in _sd:
        _sd['P'] = -_sd['rey_uv'] * _sd['du_dy']
    if 'TKE' in _sd and 'AvgPhU' in _sd and 'AvgPhV' in _sd:
        # Each pickle is self-contained: use THIS case's own grid + eps (saved by
        # saveresults.py) so cases on a different grid than the neutral reference
        # (the stratified runs are 1056x672x1056) are differentiated correctly.
        # Fall back to the neutral globals only if a (legacy) pickle lacks them.
        _eps = _sd.get('eps', eps)
        _nx  = _sd.get('nx',  nx)
        _ny  = _sd.get('ny',  ny)
        _xg  = _sd.get('x',   x)
        if _sd['TKE'].shape == _eps.shape:
            _sd['dTKE_dx'] = diffu_dx(_sd['TKE'], _ny, _nx, _eps, _xg)
            _sd['dTKE_dy'] = diffu_dy(_sd['TKE'], _ny, _nx, _eps, _xg)
            _sd['Adv']     = _sd['AvgPhU'] * _sd['dTKE_dx'] + _sd['AvgPhV'] * _sd['dTKE_dy']
        else:
            print(f'Note: {_name} TKE shape {_sd["TKE"].shape} != its eps grid '
                  f'{_eps.shape}; skipping TKE-advection (grid/eps inconsistent).')

###############################################################################
# Load instantaneous plane data (flow.*.1-3, scal.*.1) for each rough-wall case.
# Each downloaded file contains a complete binary header (offset bytes read via
# read_header) followed by the first x-y plane as Nx*Ny float64 values.
# Turbulent fluctuation = plane minus x-averaged profile (function of y only),
# then solid region zeroed with mask0.
###############################################################################
# Use each file's OWN header dimensions (read_header returns nx, ny) so a case on
# a different grid than the neutral reference is read at the right shape, and mask
# the solid region with THAT case's own pickled mask0 (falling back to the neutral
# mask0 only for a legacy pickle that lacks it / matches the neutral grid).
def _inst_fluct(_path, _mask):
    _ihdr, _inx, _iny, *_ = read_header(_path)
    if _ihdr is None or _inx is None or _iny is None:
        return None
    # The downloaded field files are deliberately truncated (~30 MB) so only the
    # header + first x-y plane are guaranteed present.  readplane() reshapes to
    # (ny, nx) and would crash on an incomplete plane, so confirm the file really
    # holds the whole first plane (offset + nx*ny*8 bytes) before reading it.
    _need = int(_ihdr) + int(_inx) * int(_iny) * 8
    try:
        _have = os.path.getsize(_path)
    except OSError:
        return None
    if _have < _need:
        print(f'  Skip {os.path.basename(_path)}: {_have} B present < {_need} B '
              f'needed for the first {_inx}x{_iny} plane (truncated download).')
        return None
    try:
        _ipl = readplane(_path, _inx, _iny, 1, _ihdr)
    except Exception as _e:
        print(f'  Skip {os.path.basename(_path)}: could not read first plane ({_e}).')
        return None
    _fluc = _ipl - _ipl.mean(axis=1, keepdims=True)
    if _mask is not None and _fluc.shape == _mask.shape:
        _fluc = _fluc * _mask
    return _fluc

for _iname, _idir in SIM_DIRS.items():
    if _iname not in sims:
        sims[_iname] = {}
    _mask = sims[_iname].get('mask0', mask0)
    # Each component glob is iteration-number agnostic (flow.<iter>.{1,2,3} /
    # scal.<iter>.1); the last match is used.  Record which file was read or
    # skipped-as-truncated per case for the end-of-run summary.
    for _comp, _ikey in [('1', 'inst_u'), ('2', 'inst_v'), ('3', 'inst_w'),
                         ('scal', 'inst_scal')]:
        _pat = (_idir + 'scal.*.1') if _comp == 'scal' else (_idir + 'flow.*.' + _comp)
        _hits = sorted(_glob.glob(_pat))
        if not _hits:
            continue
        _src = _hits[-1]
        _f = _inst_fluct(_src, _mask)
        if _f is not None:
            sims[_iname][_ikey] = _f
            _prov[_iname]['inst'][_ikey] = os.path.basename(_src)
        else:
            _prov[_iname]['inst_skip'][_ikey] = os.path.basename(_src)

# Plot results
if (1 == plotRes):
    import matplotlib.lines as mlines
    from matplotlib.lines import Line2D
    import os as _os

    ###########################################################################
    # Per-simulation style objects — CASES is the single source of truth.
    # Direct lookup: CASE_MAP['nu_oro']['color'], CASE_MAP['Sm_Neu']['ls'], …
    # Derived lists (SIM_NAMES, SIM_COLORS, …) kept for loop compatibility.
    ###########################################################################
    CASE_MAP  = {c['name']: c for c in CASES}
    _smooth   = CASE_MAP['Sm_Neu']
    _sims     = [c for c in CASES if c['name'] != 'Sm_Neu' and c['name'] in ACTIVE_CASES]

    SMOOTH_COLOR  = _smooth['color']
    SMOOTH_LS     = _smooth['ls']
    SMOOTH_LABEL  = _smooth['label']
    SMOOTH_MARKER = _smooth['marker']

    SIM_NAMES      = [c['name']   for c in _sims]
    SIM_LABELS     = [c['label']  for c in _sims]
    SIM_COLORS     = [c['color']  for c in _sims]
    SIM_LINESTYLES = [c['ls']     for c in _sims]
    SIM_MARKERS    = [c['marker'] for c in _sims]
    _figdir = cwd + 'fig/'
    _os.makedirs(_figdir, exist_ok=True)

    # Global line-quality settings
    plt.rcParams.update({'lines.linewidth': 1.5, 'font.size': 10})

    def gv(name, case='nu_oro'):
        """Return sims[case][name]; None if absent."""
        return sims.get(case, {}).get(name)

    def gy_in(case='nu_oro'):
        """Per-case wall-normal grid in SINGLE-REFERENCE inner units.

        Returns the case's own physical grid y (which may have a different ny
        than the neutral grid) divided by l_in = nu/0.0618, i.e. scaled by the
        common reference u* — NOT by the case's own u_star2.  Use this instead of
        the pickled per-case 'y_inner' (which is scaled by that case's own
        Method-2 u*) wherever a profile is plotted on the shared z+ axis.

        Legacy/stale pickles (predating per-case grid bundling) carry no 'y'.
        For those we fall back to the neutral reference axis y_in ONLY when the
        case's profile length matches it (i.e. the case really is on the neutral
        grid); a legacy pickle on a different grid (e.g. an old 1056x672x1056
        stratified run) can't be placed on the shared z+ axis, so we return None
        and the caller skips it rather than crashing on a length mismatch.
        """
        _sd = sims.get(case, {})
        _yg = _sd.get('y')
        if _yg is not None:
            return _yg / l_in
        _probe = _sd.get('u_plus_rot')
        if _probe is not None and len(_probe) == len(y_in):
            return y_in
        return None

    def geps(case='nu_oro'):
        """Per-case IBM indicator eps (1 in solid). Falls back to the reference
        eps for a case whose pickle predates the per-case grid bundling."""
        return sims.get(case, {}).get('eps', eps)

    def gmask0(case='nu_oro'):
        """Per-case fluid mask (1-eps); falls back to the reference mask0."""
        return sims.get(case, {}).get('mask0', mask0)

    def geps_f(case='nu_oro'):
        """Per-case fluid-fraction column weight for intrinsic x-averages:
        mean_x(mask0) with zeros replaced by NaN (avoids divide-by-zero)."""
        _m = np.mean(gmask0(case), axis=1)
        return np.where(_m > 0, _m, np.nan)

    def gx_in(case='nu_oro'):
        """Per-case streamwise grid in SINGLE-REFERENCE inner units (x / l_in).
        A case on a different grid than the neutral reference has a different nx,
        so x-distribution profiles (e.g. AVG_TKE_V) must be plotted against THIS
        case's own x.  Falls back to the neutral x_in for a legacy pickle that
        carries no per-case 'x'."""
        _xg = sims.get(case, {}).get('x')
        return x_in if _xg is None else _xg / l_in

    def ghill(case='nu_oro'):
        """Per-case valley-crest ROW index from THIS case's eps, using the same
        definition as the global hill_hgt (max solid-cell column count - 1).
        Cases on different wall-normal grids place the crest at a different index,
        so any per-case profile sampled 'at the crest' (u*, hodograph markers)
        must use this, not the neutral hill_hgt (= 94).  Falls back to the neutral
        hill_hgt when the case has no per-case eps (geps returns the reference)."""
        _e = geps(case)
        if _e is eps:
            return hill_hgt
        return int(np.max(np.sum(_e, axis=0).astype(int)) - 1)

    def _xprof(case, field):
        """Return a 1-D wall-normal profile from a budget/stress term that may be
        stored EITHER 2-D (x,z) or already 1-D (intrinsically x-averaged).  This
        is needed because PhAvg_rotated.py pickles some terms 1-D (visc_yx,
        visc_yz = (1/Re_λ)·avg_c(d·_dy)) and others 2-D (rey_uv, rey_vw); older
        pickles may differ again.  A 2-D field is intrinsically x-averaged with
        THIS case's eps (matching avg_c); a 1-D field is returned as-is.  None
        passes through as None so the caller's `is not None` guards still work."""
        if field is None:
            return None
        _f = np.asarray(field)
        return avg_c(geps(case), _f, axis=1) if _f.ndim == 2 else _f

    def all_handles():
        """Legend handles for active cases (smooth if loaded + active rough-wall)."""
        _h = []
        if _smooth_loaded:
            _h.append(Line2D([0],[0], color=_smooth['color'], linestyle=_smooth['ls'],
                             label=_smooth['label']))
        _h += [Line2D([0],[0], color=c['color'], linestyle=c['ls'], label=c['label'])
               for c in _sims]
        return _h

    def sim_handles():
        """Legend handles for the active rough-wall cases only — lines only."""
        return [Line2D([0],[0], color=c['color'], linestyle=c['ls'],
                       label=c['label'])
                for c in _sims]

    def _mark_h(orient='v', label=True, lblpos=0.04, ha='right'):
        """Mark the valley-crest height with a dashed black line labelled 'h'.

        The height is z+ = y_in[hill_hgt], where hill_hgt = np.max(eps_hgt) - 1
        (= 94 for this geometry, i.e. the line sits at y_in[94] in inner units).
        orient='v' draws a vertical line (use when z+ is on the x-axis);
        orient='h' draws a horizontal line (use when z+ is on the y-axis).
        The 'h' label is placed in axes-fraction coordinates along the marked
        line so it renders correctly on linear or log axes and after re-scaling.
        """
        _ax  = plt.gca()
        _pos = y_in[hill_hgt]
        if orient == 'v':
            _ax.axvline(x=_pos, color='black', linestyle='--', linewidth=0.8)
            if label:
                _ax.text(_pos, lblpos, r'$h$', rotation=90, va='bottom', ha=ha,
                         fontsize=9, transform=_ax.get_xaxis_transform())
        else:
            _ax.axhline(y=_pos, color='black', linestyle='--', linewidth=0.8)
            if label:
                _ax.text(lblpos, _pos, r'$h$', va='bottom', ha='left',
                         fontsize=9, transform=_ax.get_yaxis_transform())

    # Derived 2D fields (neutral-oro case)
    _n0 = sims.get('nu_oro', {})
    res_dispz    = np.sqrt(_n0.get('DispVelV', np.zeros((ny, nx)))**2
                           + _n0.get('DispVelU', np.zeros((ny, nx)))**2)
    res_phavg_uv = np.sqrt(_n0.get('AvgPhU', np.zeros((ny, nx)))**2
                           + _n0.get('AvgPhV', np.zeros((ny, nx)))**2)

    ###########################################################################
    # Helper: side-by-side 2D pcolormesh panels for all available Fr.
    # A single shared colorbar is placed in an explicit dedicated axes at the
    # far right.  Colour limits are the global max/min across ALL panels.
    ###########################################################################
    _DIVERGING_CMAPS = {'RdBu_r', 'coolwarm', 'seismic', 'PiYG', 'bwr', 'RdYlBu'}

    # --- per-case 2D axes helpers -------------------------------------------
    # Each case carries its OWN grid + orography in its pickle (saveresults.py),
    # so a case on a different grid than the neutral reference (the stratified
    # runs are 1056x672x1056) is plotted against its own coordinates.  Inner
    # units use the SINGLE-REFERENCE l_in (common z+ yardstick) per the cross-
    # case scaling convention, NOT each case's own u*.
    def _case_grid(cn, use_inner=True):
        _sd = sims.get(cn, {})
        _xc = _sd.get('x', x);         _yc = _sd.get('y', y)
        _xo = _sd.get('x_oro', x_oro); _yo = _sd.get('y_oro', y_oro)
        if use_inner:
            return _xc / l_in, _yc / l_in, _xo / l_in, _yo / l_in
        return _xc, _yc, _xo, _yo

    def _row_to_height(ylim, use_inner=True):
        """Physical/inner z-height of neutral reference row index `ylim` — used
        as a common z-extent so all panels show the same height across grids."""
        _ref = y_in if use_inner else y
        return _ref[ylim] if ylim < len(_ref) else _ref[-1]

    def _clip_rows(_yp, _zmax):
        _j = int(np.searchsorted(_yp, _zmax)) + 1
        return min(max(_j, 1), len(_yp))

    def plot2D_allFr(field_key, suptitle, cmap_name, savename,
                     ylim=None, use_inner=True, cbar_label=None):
        if ylim is None:
            ylim = limity
        _avail = [(cn, lb) for cn, lb in zip(SIM_NAMES, SIM_LABELS)
                  if gv(field_key, cn) is not None]
        if not _avail:
            print(f'plot2D_allFr: no data for {field_key}')
            return
        nsims = len(_avail)

        # sharey=False: each panel has its own grid; a common z-extent (_zmax)
        # keeps them visually comparable across different grids.
        fig, axes = plt.subplots(1, nsims, figsize=(4 * nsims + 1.0, 5),
                                 sharey=False)
        if nsims == 1:
            axes = [axes]

        fig.subplots_adjust(left=0.07, bottom=0.12, top=0.87, wspace=0.04)

        _zmax = _row_to_height(ylim, use_inner)

        # Per-case axes + row-clip to the common physical height _zmax
        _cax = {}
        for cn, _ in _avail:
            _xp, _yp, _xo, _yo = _case_grid(cn, use_inner)
            _cax[cn] = (_xp, _yp, _xo, _yo, _clip_rows(_yp, _zmax))

        # Global colour limits: single scale shared by every panel, from the
        # actual min/max across all available Fr cases (each clipped to _zmax)
        _epsilon = 1e-4
        _gmin = min(np.nanmin(gv(field_key, cn)[:_cax[cn][4], :]) for cn, _ in _avail)
        _gmax = max(np.nanmax(gv(field_key, cn)[:_cax[cn][4], :]) for cn, _ in _avail)
        _has_neg = _gmin < -_epsilon
        _diverging = (cmap_name in _DIVERGING_CMAPS) and _has_neg
        if _diverging:
            _vmax = max(abs(_gmin), _gmax)
            _vmin = -_vmax
        else:
            _vmin = 0.0 if _gmin >= -_epsilon else _gmin
            _vmax = _gmax

        _pcm = None
        for _i, (ax, (case, lbl)) in enumerate(zip(axes, _avail)):
            _xp, _yp, _xo, _yo, _jl = _cax[case]
            _fld = gv(field_key, case)[:_jl, :]
            _pcm = ax.pcolormesh(_xp, _yp[:_jl], _fld, cmap=cmap_name,
                                 vmin=_vmin, vmax=_vmax, shading='auto')
            ax.fill(_xo, _yo, color='dimgray')
            ax.set_ylim(0, _zmax)
            ax.set_title(lbl, fontsize=9)
            ax.set_xlabel(r'$x^+$' if use_inner else r'$x$')
            if _i > 0:
                ax.tick_params(labelleft=False)
        axes[0].set_ylabel(r'$z^+$' if use_inner else r'$z$')

        # Single shared colorbar — steals space from the axes array so it
        # always renders regardless of bbox_inches='tight' clipping
        _cb = fig.colorbar(_pcm, ax=axes, orientation='vertical',
                           shrink=0.85, pad=0.02)
        if cbar_label is not None:
            _cb.set_label(cbar_label, fontsize=9)
        _cb.ax.tick_params(labelsize=8)

        fig.suptitle(suptitle, fontsize=11)
        _out = _figdir + savename
        fig.savefig(_out, dpi=300, bbox_inches='tight')
        print(f'Saved: {_out}')

    def plot2D_div_allcases(panels, field_label, suptitle, savename, cmap='seismic'):
        """Plot multiple 2D diverging fields side-by-side with a shared colourbar.

        panels : list of (label, x_arr, y_arr, field_arr[ny,nx], xfill, yfill)
        """
        n = len(panels)
        if n == 0:
            return
        fig, axes = plt.subplots(1, n, figsize=(4 * n + 1.0, 5))
        if n == 1:
            axes = [axes]
        _gmin = min(np.nanmin(fld) for _, _, _, fld, _, _ in panels)
        _gmax = max(np.nanmax(fld) for _, _, _, fld, _, _ in panels)
        _vmax = max(abs(_gmin), _gmax)
        _vmin = -_vmax
        _pcm = None
        for _i, (ax, (lbl, _x, _y, fld, xfill, yfill)) in enumerate(zip(axes, panels)):
            _pcm = ax.pcolormesh(_x, _y, fld, cmap=cmap,
                                 vmin=_vmin, vmax=_vmax, shading='auto')
            if len(xfill) > 0:
                ax.fill(xfill, yfill, color='dimgray')
            ax.set_title(lbl, fontsize=9)
            ax.set_xlabel(r'$x$')
            if _i > 0:
                ax.tick_params(labelleft=False)
        axes[0].set_ylabel(r'$z$')
        _cb = fig.colorbar(_pcm, ax=axes, orientation='vertical',
                           shrink=0.85, pad=0.02)
        _cb.set_label(field_label, fontsize=9)
        _cb.ax.tick_params(labelsize=8)
        fig.suptitle(suptitle, fontsize=11)
        fig.subplots_adjust(left=0.07, bottom=0.12, top=0.87, wspace=0.12)
        _out = _figdir + savename
        fig.savefig(_out, dpi=300, bbox_inches='tight')
        plt.show()
        print(f'Saved: {_out}')

    ###########################################################################
    # SECTION 1 — 2D SIDE-BY-SIDE COLORMAPS (rough-wall cases, all available Fr)
    # x-axis = x+ (streamwise); y-axis = z+ (wall-normal, meteorological label).
    # Re = 500 for all cases.
    ###########################################################################

    plot2D_allFr('AvgPhU',   r'Ph-avg $\langle\bar{u}\rangle$ — Re=500',              'Reds',  'PhAvgU_allFr.png')
    plot2D_allFr('AvgPhV',   r'Ph-avg $\langle\bar{v}\rangle$ — Re=500',              'RdBu_r',  'PhAvgV_allFr.png')
    plot2D_allFr('AvgPhW',   r'Ph-avg $\langle\bar{w}\rangle$ — Re=500',              'RdBu_r',  'PhAvgW_allFr.png')
    plot2D_allFr('AvgP',     r'Ph-avg pressure $\langle\bar{p}\rangle$ — Re=500',     'PiYG',    'AvgP_allFr.png')
    plot2D_allFr('AvgScal',  r'Ph-avg potential temperature $\langle\bar{\theta}\rangle$ — Re=500', 'inferno', 'PotTemp_allFr.png',
                 cbar_label=r'$\langle\overline{\theta}\rangle$ (buoyancy $b$)')
    plot2D_allFr('DispVelU', r'Dispersive streamwise $\tilde{u}$ — Re=500',           'RdBu_r',  'DispU_allFr.png')
    plot2D_allFr('DispVelV', r'Dispersive wall-normal $\tilde{v}$ — Re=500',          'RdBu_r',  'DispV_allFr.png')
    plot2D_allFr('DispVelW', r'Dispersive spanwise $\tilde{w}$ — Re=500',             'RdBu_r',  'DispW_allFr.png')
    # Raw turbulent kinetic energy k = ½⟨u_i'u_i'⟩ (NOT wall-normalised — the z+/x+
    # axes use the single 0.0618 reference l_in, but the field is raw, shared scale).
    plot2D_allFr('TKE',      r'Turbulent kinetic energy — Re=500',                   'hot_r',   'TKE_allFr.png',
                 cbar_label=r"$k=\frac{1}{2}\,\overline{u_i'u_i'}$ (raw)")
    plot2D_allFr('disp_vortz', r'Dispersive vorticity $\tilde{\omega}_z$ — Re=500',   'coolwarm','disp_vortz_allFr.png', ylim=200)
    plot2D_allFr('vort_z',   r'Ph-avg vorticity $\langle\bar{\omega}_z\rangle$ — Re=500', 'coolwarm','vort_z_allFr.png', ylim=200)
    plot2D_allFr('rey_uv',   r"Reynolds stress $\overline{u'v'}$ — Re=500",           'RdBu_r',  'rey_uv_allFr.png')
    plot2D_allFr('rey_uu',   r"Reynolds normal $\overline{u'u'}$ — Re=500",           'hot_r',   'rey_uu_allFr.png')
    plot2D_allFr('rey_vv',   r"Reynolds normal $\overline{v'v'}$ — Re=500",           'hot_r',   'rey_vv_allFr.png')
    plot2D_allFr('rey_ww',   r"Reynolds normal $\overline{w'w'}$ — Re=500",           'hot_r',   'rey_ww_allFr.png')
    plot2D_allFr('UU_disp',  r'Dispersive stress $\tilde{u}\tilde{u}$ — Re=500',      'hot_r',   'UU_disp_allFr.png')
    plot2D_allFr('VV_disp',  r'Dispersive stress $\tilde{v}\tilde{v}$ — Re=500',      'hot_r',   'VV_disp_allFr.png')
    plot2D_allFr('WW_disp',  r'Dispersive stress $\tilde{w}\tilde{w}$ — Re=500',      'hot_r',   'WW_disp_allFr.png')

    ###########################################################################
    # SECTION 1b — 2D INSTANTANEOUS PLANE COLORMAPS (all available Fr)
    # First x-y plane of flow.* / scal.* binary files; turbulent fluctuation
    # (subtract x-averaged y-profile) zeroed over solid region.
    ###########################################################################
    plot2D_allFr('inst_u',    r"Inst. $u' = u - \langle u\rangle_x$ — Re=500",               'RdBu_r', 'inst_u_allFr.png', 530, False)
    plot2D_allFr('inst_v',    r"Inst. $v' = v - \langle v\rangle_x$ — Re=500",               'RdBu_r', 'inst_v_allFr.png', 530, False)
    plot2D_allFr('inst_w',    r"Inst. $w' = w - \langle w\rangle_x$ — Re=500",               'RdBu_r', 'inst_w_allFr.png', 530, False)
    plot2D_allFr('inst_scal', r"Inst. $\theta' = \theta - \langle\theta\rangle_x$ — Re=500", 'RdBu_r', 'inst_scal_allFr.png', 700, False)

    # Neutral only: streamlines overlaid on dispersive vorticity / magnitude
    _du_n = _n0.get('DispVelU')
    _dv_n = _n0.get('DispVelV')
    _dz_n = _n0.get('disp_vortz')
    _au_n = _n0.get('AvgPhU')
    _av_n = _n0.get('AvgPhV')
    _vz_n = _n0.get('vort_z')
    if _du_n is not None:
        plot2D_streamlines_vorticity(
            x_in, y_in[:250], _du_n[:250,:], _dv_n[:250,:],
            res_dispz[:250,:], eps[:250,:], '', '',
            r'$x^+$', r'$z^+$',
            _figdir+'Streamlines_disp_mag.png', x_oro_in, y_oro_in, 1000)
        if _dz_n is not None:
            plot2D_streamlines_vorticityZ(
                x_in, y_in[:200], _du_n[:200,:], _dv_n[:200,:], _dz_n[:200,:],
                r'Dispersive streamlines + $\tilde{\omega}_z$ (Neutral)',
                r'$x$', r'$y$',
                _figdir+'Streamlines_disp_vortZ.png', x_oro_in, y_oro_in, 1000)
    if _au_n is not None and _vz_n is not None:
        plot2D_streamlines_vorticityZ(
            x_in, y_in[:200], _au_n[:200,:], _av_n[:200,:], _vz_n[:200,:],
            r'Ph.avg streamlines + $\langle\bar{\omega}_z\rangle$ (Neutral)',
            r'$x$', r'$y$',
            _figdir+'Streamlines_PhAvg_vortZ.png', x_oro_in, y_oro_in, 1000)

    # TKE shear production — all active cases in one subplot figure
    _dvdx_n = _n0.get('dv_dx')
    _prod_panels = []
    _zmax_lim = _row_to_height(limity, use_inner=False)
    for _cname, _clbl in zip(SIM_NAMES, SIM_LABELS):
        _P_c = sims.get(_cname, {}).get('P')
        if _P_c is not None:
            _xc, _yc, _xo, _yo = _case_grid(_cname, use_inner=False)
            _jl = _clip_rows(_yc, _zmax_lim)
            _prod_panels.append((_clbl, _xc, _yc[:_jl], _P_c[:_jl, :], _xo, _yo))
    if _prod_panels:
        plot2D_div_allcases(
            _prod_panels,
            r'$-\overline{u^\prime v^\prime}\,\partial\langle\bar{u}\rangle/\partial z$',
            r'TKE production — all cases', 'TKEprod_allFr.png')

    # TKE advection — smooth (if loaded) + all active rough cases in one subplot figure
    _adv_panels = []
    if _smooth_loaded:
        _adv_panels.append((_smooth['label'],
                            sx, y_s[:limity_range], Adv_s[:limity_range, :],
                            np.array([]), np.array([])))
    for _cname, _clbl in zip(SIM_NAMES, SIM_LABELS):
        _Adv_c = sims.get(_cname, {}).get('Adv')
        if _Adv_c is not None:
            _xc, _yc, _xo, _yo = _case_grid(_cname, use_inner=False)
            _jl = _clip_rows(_yc, _zmax_lim)
            _adv_panels.append((_clbl, _xc, _yc[:_jl], _Adv_c[:_jl, :], _xo, _yo))
    if _adv_panels:
        plot2D_div_allcases(
            _adv_panels,
            r'$u\,\partial k/\partial x + v\,\partial k/\partial z$',
            r'TKE advection — all cases', 'TKEadv_allFr.png')

    # dv/dx derivative 2D and phase-averaged velocity magnitude
    if _dvdx_n is not None:
        plot2D_div(x, y[:limity], _dvdx_n[:limity,:], '',
                   r'$\partial\langle\bar{v}\rangle/\partial x$', r'$x^+$', r'$z^+$',
                   _figdir+'dvdx.png', x_oro, y_oro, 1000)
    plot2D_div(x, y[:limity], res_phavg_uv[:limity,:], '',
               r'$|\langle\bar{u},\bar{v}\rangle|$ — Neutral', r'$x^+$', r'$z^+$',
               _figdir+'ResPhAvg_UV.png', x_oro, y_oro, 1000)

    ###########################################################################
    # SECTION 2 -- 1D VERTICAL PROFILES (all 5 Fr)
    ###########################################################################

    ###########################################################################
    # Layer boundary precomputation — used by all layer-zoomed profile plots.
    # Viscous/canopy sublayer : z+ < 5
    # Buffer layer            : 5 ≤ z+ < 37   (extended to cover transition)
    # Log-law region          : 30 ≤ z+ < 130 (overlaps buffer 30–37 by design)
    # Outer region            : z+ ≥ 130
    ###########################################################################
    _LYR_NAMES  = ['viscous_canopy', 'buffer', 'log', 'outer']
    _LYR_TITLES = [r'Viscous/canopy sublayer ($z^+<5$)',
                   r'Buffer layer ($5\leq z^+<37$)',
                   r'Log-law region ($30\leq z^+<130$)',
                   r'Outer region ($z^+\geq130$)']
    _LYR_XLIMS  = [(0, 5), (5, 37), (30, 130), (130, None)]

    def _lyr_idx(y):
        """Return (i_lo, i_hi) index pairs for each layer in array y.
        Buffer ends at 37, log starts at 30 (overlap by design)."""
        i5   = int(np.searchsorted(y,   5, side='left'))
        i30  = int(np.searchsorted(y,  30, side='left'))
        i37  = int(np.searchsorted(y,  37, side='left'))
        i130 = int(np.searchsorted(y, 130, side='left'))
        return [(0, i5), (i5, i37), (i30, i130), (i130, len(y))]

    _LYR_IDX   = _lyr_idx(y_in)
    _LYR_IDX_S = _lyr_idx(y_in_s) if _smooth_loaded else [(0, 0)] * 4

    def _autoscale_y(_ax, _xlo, _xhi):
        """Set y-limits to the data range of all lines visible within [_xlo, _xhi]."""
        _logy = _ax.get_yscale() == 'log'
        _ylo, _yhi = np.inf, -np.inf
        for _ln in _ax.get_lines():
            _xd = np.asarray(_ln.get_xdata(), dtype=float)
            _yd = np.asarray(_ln.get_ydata(), dtype=float)
            if _xd.size < 2:
                continue
            # Skip axvline (ydata ≈ [0,1] axes-coords, constant x)
            if _xd[0] == _xd[-1] and np.allclose(_yd[[0,-1]], [0, 1], atol=1e-9):
                continue
            # Skip axhline (xdata ≈ [0,1] axes-coords, constant y)
            if _yd[0] == _yd[-1] and np.allclose(_xd[[0,-1]], [0, 1], atol=1e-9):
                continue
            _mask = (_xd >= _xlo) & (_xd <= _xhi)
            if _logy:
                _mask &= (_yd > 0)
            _yv = _yd[_mask]
            if _yv.size:
                _ylo = min(_ylo, float(np.nanmin(_yv)))
                _yhi = max(_yhi, float(np.nanmax(_yv)))
        if np.isfinite(_ylo) and np.isfinite(_yhi) and _ylo < _yhi:
            if _logy:
                _ax.set_ylim(_ylo * 0.5, _yhi * 2.0)
            else:
                _mg = 0.05 * (_yhi - _ylo)
                _ax.set_ylim(_ylo - _mg, _yhi + _mg)

    def _autoscale_x(_ax, _ylo, _yhi):
        """Set x-limits to the data range of all lines visible within [_ylo, _yhi]."""
        _logx = _ax.get_xscale() == 'log'
        _xlo, _xhi = np.inf, -np.inf
        for _ln in _ax.get_lines():
            _xd = np.asarray(_ln.get_xdata(), dtype=float)
            _yd = np.asarray(_ln.get_ydata(), dtype=float)
            if _yd.size < 2:
                continue
            if _xd[0] == _xd[-1] and np.allclose(_yd[[0,-1]], [0, 1], atol=1e-9):
                continue
            if _yd[0] == _yd[-1] and np.allclose(_xd[[0,-1]], [0, 1], atol=1e-9):
                continue
            _mask = (_yd >= _ylo) & (_yd <= _yhi)
            if _logx:
                _mask &= (_xd > 0)
            _xv = _xd[_mask]
            if _xv.size:
                _xlo = min(_xlo, float(np.nanmin(_xv)))
                _xhi = max(_xhi, float(np.nanmax(_xv)))
        if np.isfinite(_xlo) and np.isfinite(_xhi) and _xlo < _xhi:
            if _logx:
                _ax.set_xlim(_xlo * 0.5, _xhi * 2.0)
            else:
                _mg = 0.05 * (_xhi - _xlo)
                _ax.set_xlim(_xlo - _mg, _xhi + _mg)

    def _save_layers_x(base_path, base_title, is_log=False):
        """Save one zoomed PNG per BL layer; z+ on x-axis.
        Both axes are rescaled to the data within each layer window."""
        _ax    = plt.gca()
        _xlim0 = _ax.get_xlim()
        _ylim0 = _ax.get_ylim()
        for _ln, _lt, (_x0, _x1) in zip(_LYR_NAMES, _LYR_TITLES, _LYR_XLIMS):
            _xlo = max(_x0, float(y_in[1])) if is_log else _x0
            _xhi = min(_x1, _xlim0[1]) if _x1 is not None else _xlim0[1]
            if _xlo >= _xhi:
                continue
            _ax.set_xlim(_xlo, _xhi)
            _autoscale_y(_ax, _xlo, _xhi)
            plt.title(f'{base_title}\n{_lt}', fontsize=9)
            plt.savefig(f'{base_path}_{_ln}.png', dpi=300)
        _ax.set_xlim(*_xlim0)
        _ax.set_ylim(*_ylim0)
        plt.title(base_title)

    def _save_layers_y(base_path, base_title):
        """Save one zoomed PNG per BL layer; z+ on y-axis.
        Both axes are rescaled to the data within each layer window."""
        _ax    = plt.gca()
        _ylim0 = _ax.get_ylim()
        _xlim0 = _ax.get_xlim()
        for _ln, _lt, (_y0, _y1) in zip(_LYR_NAMES, _LYR_TITLES, _LYR_XLIMS):
            _yhi = min(_y1, _ylim0[1]) if _y1 is not None else _ylim0[1]
            if _y0 >= _yhi:
                continue
            _ax.set_ylim(_y0, _yhi)
            _autoscale_x(_ax, _y0, _yhi)
            plt.title(f'{base_title}\n{_lt}', fontsize=9)
            plt.savefig(f'{base_path}_{_ln}.png', dpi=300)
        _ax.set_ylim(*_ylim0)
        _ax.set_xlim(*_xlim0)
        plt.title(base_title)

    # Log-law references
    if _smooth_loaded:
        u_most    = (1/0.43)*np.log(y_in_s) + 4.9
        u_most[0] = 0
    u_most_v    = (1/0.43)*np.log(y_in) + 4.7
    u_most_v[0] = 0

    # 2a. Log-law velocity profile (u+ and w+ vs z+)
    # Solid lines = streamwise (u+), faded (alpha=0.4) = spanwise (w+).
    # Vertical dashed lines mark the per-case BL thickness δ⁺ = u_*(h) × u_star/ν.
    plt.figure(figsize=(8, 6), dpi=300)
    if _smooth_loaded:
        plt.plot(y_in_s, np.mean(U_s_p, axis=1),  color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
        plt.plot(y_in_s, -np.mean(W_s_p, axis=1), color=SMOOTH_COLOR, linestyle=SMOOTH_LS, alpha=0.4)
        _delta_smooth = ustr_s1**2 / nu
        plt.axvline(x=_delta_smooth, color=SMOOTH_COLOR, linestyle='--', linewidth=1.0, alpha=0.8)
    for case, clr, ls, mrkr in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_MARKERS):
        _upr  = gv('u_plus_rot', case)
        _wpr  = gv('w_plus_rot', case)
        _us2  = gv('u_star2',    case)
        _yi   = gy_in(case)
        if _upr is None or _yi is None:
            continue
        plt.plot(_yi, _upr/ustr_s1, color=clr, linestyle=ls)
        plt.plot(_yi, _wpr/ustr_s1, color=clr, linestyle=ls, alpha=0.4)
        # BL thickness on the SINGLE-REFERENCE axis: δ⁺ = u*_ref²/(f·ν) = Re_tau
        # (f = 1).  Same yardstick for every case (settled single-reference choice),
        # so this line coincides with the smooth δ marker above.
        if _us2 is not None:
            _delta_case = Re_tau
            plt.axvline(x=_delta_case, color=clr, linestyle='--', linewidth=1.0, alpha=0.8)
    if _smooth_loaded:
        plt.plot(y_in_s, u_most, color='black', linestyle='--', linewidth=1.0, alpha=0.6)
    _mark_h('v')
    plt.xscale('log')
    plt.xlabel(r'$z^+$')
    plt.ylabel(r'$\langle\bar{u}_i\rangle^+$')
    _lgh_2a = (all_handles()
               + [Line2D([0],[0], color='k', ls='-',   lw=1.5, label=r'$u^+$ (solid)'),
                  Line2D([0],[0], color='k', ls='-',   lw=1.5, alpha=0.4, label=r'$w^+$ (faded)'),
                  Line2D([0],[0], color='k', ls='--',  lw=1.0, alpha=0.6, label='Log-law'),
                  Line2D([0],[0], color='k', ls='--',  lw=1.0, alpha=0.8, label=r'$\delta_o$ per case')])
    plt.legend(handles=_lgh_2a, fontsize=7, ncol=2)
    plt.grid(True, which='both', linestyle='--', linewidth=0.4)
    plt.title(r'Log-law velocity profile — all Fr, Re=500')
    plt.savefig(cwd+'fig'+'/'+'Velocity_LogLaw_allFr.png', dpi=300)
    _save_layers_x(cwd+'fig'+'/'+'Velocity_LogLaw_allFr', r'Log-law velocity profile — all Fr, Re=500', is_log=True)
    plt.show()

    # 2b. Roughness sublayer velocity profile (log-log, rough-wall cases only)
    u_roughnesslayer = 0.1018*np.exp(1.3165*y_in)
    plt.figure(figsize=(8, 6), dpi=300)
    for case, clr, ls, lbl in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_LABELS):
        _upr = gv('u_plus_rot', case)
        _yi  = gy_in(case)
        if _upr is None or _yi is None:
            continue
        plt.plot(_yi[:157], (_upr/ustr_s1)[:157], color=clr, linestyle=ls, label=lbl)
    plt.plot(y_in[:10], u_roughnesslayer[:10], color='black', linestyle='--', alpha=0.5, label='RSL fit')
    _mark_h('v')
    plt.xscale('log')
    plt.xlabel(r'$z^+$')
    plt.ylabel(r'$\langle\bar{u}\rangle^+$')
    plt.legend(fontsize=8)
    plt.grid(True, which='both', linestyle='--', linewidth=0.4)
    plt.title(r'Roughness sublayer velocity — rough-wall cases, Re=500')
    plt.savefig(cwd+'fig'+'/'+'Velocity_RSL_allFr.png', dpi=300)
    plt.show()

    # 2c. Hodograph — all 6 cases
    # Normalisation by the geostrophic wind components at the domain top:
    #   G_u = <u>|_top  (last row of x-averaged streamwise velocity)
    #   G_w = <w>|_top  (last row of x-averaged spanwise velocity)
    #   G   = sqrt(G_u² + G_w²)  — geostrophic wind magnitude (for reference)
    #   u_norm = <u> / G_u ,  w_norm = <w> / G_w
    # Markers flag key BL heights; height encoded by SIZE, case by SHAPE + colour.
    #   small  (6 pt) = h       (valley crest, rough cases only)
    #   medium (9 pt) = 3h      (approx. RSL top, rough cases only)
    #   large (12 pt) = δ_o     (per-case outer BL scale)
    _Re_tau_s = ustr_s1**2 / nu   # BL thickness in smooth inner units
    _fig_hodo, _ax_hodo = plt.subplots(figsize=(7, 6), dpi=300)
    _mkw = dict(zorder=5, markeredgewidth=0.8)
    _hodo_data = []   # stores (un, wn) 1-D arrays for each available rough case

    if _smooth_loaded:
        _Us_ref = GblU_s[-1]                           # geostrophic u-component
        _Ws_ref = GblW_s[-1]                           # geostrophic w-component
        _G_ref  = np.sqrt(_Us_ref**2 + _Ws_ref**2)    # geostrophic wind magnitude
        _un_s   = GblU_s / _G_ref
        _wn_s   = -GblW_s / _G_ref
        _ax_hodo.plot(_un_s, _wn_s, color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
        _id_s = np.argmin(np.abs(y_in_s - _Re_tau_s))
        _ax_hodo.plot(_un_s[_id_s], _wn_s[_id_s],
                      marker=SMOOTH_MARKER, color=SMOOTH_COLOR, ms=5,
                      markeredgecolor='k', **_mkw)

    for case, clr, ls, mrkr in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_MARKERS):
        # Use rotation-corrected profiles saved by PhAvg.py:
        #   u_plus_rot =  avg_c(eps,U,x)*cos(α) - avg_c(eps,W,x)*sin(α)
        #   w_plus_rot = -(avg_c(eps,U,x)*sin(α) + avg_c(eps,W,x)*cos(α))
        _u_rot = gv('u_plus_rot', case)
        _w_rot = gv('w_plus_rot', case)
        _us2   = gv('u_star2', case)
        _yi    = gy_in(case)
        if _u_rot is None or _w_rot is None or _yi is None:
            continue
        _G_ref_case = np.sqrt(_u_rot[-1]**2 + _w_rot[-1]**2)
        _un    = _u_rot / _G_ref_case
        _wn    = _w_rot / _G_ref_case
        _ax_hodo.plot(_un, _wn, color=clr, linestyle=ls)
        _hodo_data.append((_un.copy(), _wn.copy()))
        # h — small marker (crest index on THIS case's own grid)
        _hc = ghill(case)
        _ax_hodo.plot(_un[_hc], _wn[_hc],
                      marker=mrkr, color=clr, ms=3, markeredgecolor='k', **_mkw)
        # 3h — medium marker (height index on this case's own grid)
        _i_3h = np.argmin(np.abs(_yi - 3*_yi[_hc]))
        _ax_hodo.plot(_un[_i_3h], _wn[_i_3h],
                      marker=mrkr, color=clr, ms=5, markeredgecolor='k', **_mkw)
        # δ_o — large marker; BL depth on the SINGLE-REFERENCE axis:
        # δ⁺ = u*_ref²/(f·ν) = Re_tau (f = 1), the same yardstick for every case.
        _delta_plus = Re_tau
        _id = np.argmin(np.abs(_yi - _delta_plus))
        _ax_hodo.plot(_un[_id], _wn[_id],
                      marker=mrkr, color=clr, ms=6, markeredgecolor='k', **_mkw)

    _ax_hodo.set_xlabel(r'$u_{\mathrm{rot}}\,/\,G$')
    _ax_hodo.set_ylabel(r'$w_{\mathrm{rot}}\,/\,G$')
    # Legend: case handles + height-size key
    _lgh_hodo = (all_handles()
                 + [Line2D([0],[0], color='k', ls='none', marker='o', ms=3,
                            markeredgecolor='k', markeredgewidth=0.8, label=r'$h$'),
                    Line2D([0],[0], color='k', ls='none', marker='o', ms=5,
                            markeredgecolor='k', markeredgewidth=0.8, label=r'$3h$'),
                    Line2D([0],[0], color='k', ls='none', marker='o', ms=6,
                            markeredgecolor='k', markeredgewidth=0.8, label=r'$\delta_o$')])
    _ax_hodo.legend(handles=_lgh_hodo, fontsize=7, ncol=2)
    _ax_hodo.grid(True)
    _ax_hodo.set_title(r'Hodograph — all Fr, Re=500')
    _fig_hodo.savefig(cwd+'fig'+'/'+'Hodograph_allFr.png', dpi=300, bbox_inches='tight')
    # Layer-zoomed hodographs: restrict visible window to each z+ range
    for _ln, _lt, (_i0, _i1), (_i0_s, _i1_s) in zip(
            _LYR_NAMES, _LYR_TITLES, _LYR_IDX, _LYR_IDX_S):
        _uall, _wall = [], []
        if _smooth_loaded and _i1_s > _i0_s:
            _uall.extend(_un_s[_i0_s:_i1_s].tolist())
            _wall.extend(_wn_s[_i0_s:_i1_s].tolist())
        for _ud, _wd in _hodo_data:
            _ihi = min(_i1, len(_ud))
            if _ihi > _i0:
                _uall.extend(_ud[_i0:_ihi].tolist())
                _wall.extend(_wd[_i0:_ihi].tolist())
        if not _uall:
            continue
        _umg = 0.05 * max(abs(max(_uall) - min(_uall)), 1e-6)
        _wmg = 0.05 * max(abs(max(_wall) - min(_wall)), 1e-6)
        _ax_hodo.set_xlim(min(_uall) - _umg, max(_uall) + _umg)
        _ax_hodo.set_ylim(min(_wall) - _wmg, max(_wall) + _wmg)
        _ax_hodo.set_title(f'Hodograph — all Fr, Re=500\n{_lt}', fontsize=9)
        _fig_hodo.savefig(cwd+'fig'+'/'+f'Hodograph_allFr_{_ln}.png', dpi=300, bbox_inches='tight')
    _ax_hodo.autoscale()
    _ax_hodo.set_title(r'Hodograph — all Fr, Re=500')
    plt.show()

    # 2c (outer). Hodograph — outer units (y^- = y / u_star2(h) per case)
    # Curves normalised by G are unchanged; markers locate h, 3h, δ_o
    # in outer-unit coordinates.
    _fig_hodo_out, _ax_hodo_out = plt.subplots(figsize=(7, 6), dpi=300)
    if _smooth_loaded:
        _ax_hodo_out.plot(_un_s, _wn_s, color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
        _id_s_out = np.argmin(np.abs(y_s / ustr_s1 - 1.0))
        _ax_hodo_out.plot(_un_s[_id_s_out], _wn_s[_id_s_out],
                          marker=SMOOTH_MARKER, color=SMOOTH_COLOR, ms=5,
                          markeredgecolor='k', **_mkw)
    for case, clr, ls, mrkr in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_MARKERS):
        _u_rot = gv('u_plus_rot', case)
        _w_rot = gv('w_plus_rot', case)
        _us2   = gv('u_star2', case)
        if _u_rot is None or _w_rot is None or _us2 is None:
            continue
        _hc      = ghill(case)
        _yc      = gv('y', case)
        if _yc is None:
            _yc = y
        _us2_hgt = _us2[_hc]
        _G_ref_c = np.sqrt(_u_rot[-1]**2 + _w_rot[-1]**2)
        _un_o    = _u_rot / _G_ref_c
        _wn_o    = _w_rot / _G_ref_c
        _ax_hodo_out.plot(_un_o, _wn_o, color=clr, linestyle=ls)
        # h marker
        _ax_hodo_out.plot(_un_o[_hc], _wn_o[_hc],
                          marker=mrkr, color=clr, ms=3, markeredgecolor='k', **_mkw)
        # 3h marker (per-case grid)
        _i_3h_o = np.argmin(np.abs(_yc - 3*_yc[_hc]))
        _ax_hodo_out.plot(_un_o[_i_3h_o], _wn_o[_i_3h_o],
                          marker=mrkr, color=clr, ms=5, markeredgecolor='k', **_mkw)
        # δ_o marker: y^- = y / u_star2(h) = 1 → y = u_star2(h)  (per-case grid)
        _id_do = np.argmin(np.abs(_yc / _us2_hgt - 1.0))
        _ax_hodo_out.plot(_un_o[_id_do], _wn_o[_id_do],
                          marker=mrkr, color=clr, ms=6, markeredgecolor='k', **_mkw)
    _ax_hodo_out.set_xlabel(r'$u_{\mathrm{rot}}\,/\,G$')
    _ax_hodo_out.set_ylabel(r'$w_{\mathrm{rot}}\,/\,G$')
    _ax_hodo_out.legend(handles=_lgh_hodo, fontsize=7, ncol=2)
    _ax_hodo_out.grid(True)
    _ax_hodo_out.set_title(r'Hodograph (outer units, $u_{\star 2}(h)$) — all Fr, Re=500')
    _fig_hodo_out.savefig(cwd+'fig'+'/'+'Hodograph_allFr_outer.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2d. Wind turning angle vs z+ (rough-wall cases)
    plt.figure(figsize=(8, 6), dpi=300)
    for case, clr, ls, lbl in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_LABELS):
        _ia = gv('inst_alpha', case)
        _yi = gy_in(case)
        if _ia is None or _yi is None:
            continue
        plt.plot(_yi[1:], _ia[1:]*(180/np.pi), color=clr, linestyle=ls, label=lbl)
    _mark_h('v')
    plt.xscale('log')
    plt.xlabel(r'$z^+$')
    plt.ylabel(r'$\alpha\;(\mathrm{deg})$')
    plt.legend(handles=sim_handles(), fontsize=7, ncol=2)
    plt.grid(True, which='both', linestyle='--', linewidth=0.4)
    plt.title(r'Wind turning angle — rough-wall cases, Re=500')
    plt.savefig(cwd+'fig'+'/'+'TurningAngle_allFr.png', dpi=300)
    _save_layers_x(cwd+'fig'+'/'+'TurningAngle_allFr', r'Wind turning angle — rough-wall cases, Re=500', is_log=True)
    plt.show()

    # 2e. TKE vertical profile — all 6 cases
    plt.figure(figsize=(8, 6), dpi=300)
    if _smooth_loaded:
        plt.plot(y_in_s[:130], np.mean(TKE_s, axis=1)[:130]/ustr_s1**2,
                 color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
    for case, clr, ls in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES):
        _tke = gv('TKE', case)
        _yi  = gy_in(case)
        if _tke is None or _yi is None:
            continue
        plt.plot(_yi[:460], np.mean(_tke, axis=1)[:460]/_ustar_ref**2,
                 color=clr, linestyle=ls)
    _mark_h('v')
    if _smooth_loaded:
        plt.axvline(x=ustr_s1**2/nu, color='black', linestyle='--', linewidth=0.8)
        plt.text(ustr_s1**2/nu, 0.5, r'$\delta_s$', rotation=90, va='center', ha='right', fontsize=9)
    plt.xlabel(r'$z^+$')
    plt.ylabel(r'$k/u_*^2$')
    plt.legend(handles=all_handles(), fontsize=7, ncol=2)
    plt.grid(True, which='both', linestyle='--', linewidth=0.4)
    plt.title(r'TKE vertical profile — all Fr, Re=500')
    plt.savefig(cwd+'fig'+'/'+'TKE_vertical_allFr.png', dpi=300)
    _save_layers_x(cwd+'fig'+'/'+'TKE_vertical_allFr', r'TKE vertical profile — all Fr, Re=500')
    plt.show()

    # 2f. TKE horizontal (column-integrated) distribution
    plt.figure(figsize=(8, 6), dpi=300)
    # Smooth reference now comes from the shared loader (was wrongly read as zeros
    # from the rough-wall pickle via _n0.get('AVG_TKE_V_s_i')).
    if _smooth_loaded:
        plt.plot(x_in, AVG_TKE_V_s_i/u_star**2,
                 color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
    for case, clr, ls, lbl in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_LABELS):
        _atv = gv('AVG_TKE_V', case)
        if _atv is None:
            continue
        plt.plot(gx_in(case), _atv/u_star**2, color=clr, linestyle=ls, label=lbl)
    _hill_line = (y[hill_hgt]/u_star)*(1 + np.cos(2*x_in*np.pi/x_in[-1]))
    plt.fill_between(x_in, _hill_line, color='black', alpha=1.0, label='IBM solid')
    plt.xlabel(r'$x^+$')
    plt.ylabel(r'$\langle k\rangle_z / u_*^2$')
    plt.legend(handles=sim_handles(), fontsize=7, ncol=2)
    plt.grid(True)
    plt.title(r'Horizontal TKE distribution — rough-wall cases, Re=500')
    plt.savefig(cwd+'fig'+'/'+'TKE_horizontal_allFr.png', dpi=300)
    plt.show()

    # 2g. Friction velocity profile u*(z+) — rough-wall cases
    plt.figure(figsize=(8, 6), dpi=300)
    for case, clr, ls, lbl in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_LABELS):
        _us2 = gv('u_star2', case)
        _yi  = gy_in(case)
        if _us2 is None or _yi is None:
            continue
        plt.plot(_us2[:430], _yi[:430], color=clr, linestyle=ls, label=lbl)
    _mark_h('h')
    plt.xlabel(r'$u_*(z)$')
    plt.ylabel(r'$z^+$')
    plt.legend(handles=sim_handles(), fontsize=7, ncol=2)
    plt.grid(True)
    plt.title(r'Friction velocity profile — rough-wall cases, Re=500')
    plt.savefig(cwd+'fig'+'/'+'FrictionVelocity_allFr.png', dpi=300)
    _save_layers_y(cwd+'fig'+'/'+'FrictionVelocity_allFr', r'Friction velocity profile — rough-wall cases, Re=500')
    plt.show()

    # 2h. Reynolds and dispersive normal stress profiles (uu, vv, ww) — all 6 cases
    # Solid lines = Reynolds stress; faded (alpha=0.4) = dispersive stress.
    _eps_f = np.where(np.mean(mask0, axis=1) > 0, np.mean(mask0, axis=1), np.nan)
    for _key_r, _key_d, _key_sm, _lbl_r, _lbl_d in [
        ('rey_uu', 'UU_disp', 'Rxx_s', r"$\overline{u'u'}$", r'$\tilde{u}\tilde{u}$'),
        ('rey_vv', 'VV_disp', 'Ryy_s', r"$\overline{v'v'}$", r'$\tilde{v}\tilde{v}$'),
        ('rey_ww', 'WW_disp', 'Rzz_s', r"$\overline{w'w'}$", r'$\tilde{w}\tilde{w}$'),
    ]:
        plt.figure(figsize=(8, 6), dpi=300)
        _sm = globals().get(_key_sm)
        if _sm is not None and _smooth_loaded:
            plt.semilogy(y_in_s, np.mean(_sm, axis=1)/ustr_s1**2,
                         color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
        for case, clr, ls in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES):
            _r = gv(_key_r, case)
            _d = gv(_key_d, case)
            _yi = gy_in(case)
            if _r is None or _yi is None:
                continue
            plt.semilogy(_yi, np.mean(_r*gmask0(case), axis=1)/geps_f(case)/_ustar_ref**2,
                         color=clr, linestyle=ls)
            if _d is not None:
                plt.semilogy(_yi, np.mean(_d, axis=1)/_ustar_ref**2,
                             color=clr, linestyle=ls, alpha=0.4)
        _mark_h('v', ha='left')
        plt.xlim(y_in[hill_hgt], None)
        plt.xlabel(r'$z^+$')
        plt.ylabel(_lbl_r + ', ' + _lbl_d + r' / $u_*^2$')
        _leg_n = ([Line2D([0],[0], color='k', ls='-', lw=2,   label=_lbl_r + r' (solid)'),
                   Line2D([0],[0], color='k', ls='-', lw=0.8, alpha=0.4, label=_lbl_d + r' (faded)')]
                  + all_handles())
        plt.legend(handles=_leg_n, fontsize=7, ncol=2)
        plt.grid(True, which='both', linestyle='--', linewidth=0.4)
        plt.title('Normal stress: ' + _lbl_r + ' and ' + _lbl_d + r' — all Fr, Re=500')
        plt.savefig(cwd+'fig'+'/'+'Stress_'+_key_r+'_allFr.png', dpi=300)
        _save_layers_x(cwd+'fig'+'/'+'Stress_'+_key_r+'_allFr',
                       'Normal stress: ' + _lbl_r + ' and ' + _lbl_d + r' — all Fr, Re=500')
        plt.show()

    # 2i. Reynolds + dispersive shear stress uv — all 6 cases
    # Solid lines = Reynolds stress; faded (alpha=0.4) = dispersive stress.
    plt.figure(figsize=(8, 6), dpi=300)
    if _smooth_loaded:
        plt.plot(y_in_s, np.mean(Rxy_s, axis=1)/ustr_s1**2,
                 color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
    for case, clr, ls in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES):
        _r = gv('rey_uv', case)
        _d = gv('UV_disp', case)
        _yi = gy_in(case)
        if _r is None or _yi is None:
            continue
        plt.plot(_yi, np.mean(_r*gmask0(case), axis=1)/geps_f(case)/_ustar_ref**2,
                 color=clr, linestyle=ls)
        if _d is not None:
            plt.plot(_yi, np.mean(_d, axis=1)/_ustar_ref**2,
                     color=clr, linestyle=ls, alpha=0.4)
    _mark_h('v')
    plt.xlim(0, y_in[429])
    plt.xlabel(r'$z^+$')
    plt.ylabel(r"$\overline{u'v'},\;\tilde{u}\tilde{v}\;/ u_*^2$")
    _leg_uv = ([Line2D([0],[0], color='k', ls='-', lw=2,   label=r"$\overline{u'v'}$ (solid)"),
                Line2D([0],[0], color='k', ls='-', lw=0.8, alpha=0.4, label=r'$\tilde{u}\tilde{v}$ (faded)')]
               + all_handles())
    plt.legend(handles=_leg_uv, fontsize=7, ncol=2)
    plt.grid(True)
    plt.title(r"Shear stress $\overline{u'v'}$ and $\tilde{u}\tilde{v}$ — all Fr, Re=500")
    plt.savefig(cwd+'fig'+'/'+'Stress_uv_allFr.png', dpi=300)
    plt.show()

    # 2i (outer). Reynolds + dispersive shear stress uv — outer units
    # y^- = y / u_star2(h) per case; stress normalised by u_star2(h)^2.
    plt.figure(figsize=(8, 6), dpi=300)
    if _smooth_loaded:
        _y_out_s = y_s / ustr_s1
        plt.plot(_y_out_s, np.mean(Rxy_s, axis=1)/ustr_s1**2,
                 color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
    for case, clr, ls in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES):
        _r   = gv('rey_uv',  case)
        _d   = gv('UV_disp', case)
        _us2 = gv('u_star2', case)
        _yc  = gv('y', case)
        if _r is None or _us2 is None or _yc is None:
            continue
        _us2_hgt = _us2[ghill(case)]
        _y_out   = _yc / _us2_hgt
        plt.plot(_y_out, np.mean(_r*gmask0(case), axis=1)/geps_f(case)/_us2_hgt**2,
                 color=clr, linestyle=ls)
        if _d is not None:
            plt.plot(_y_out, np.mean(_d, axis=1)/_us2_hgt**2,
                     color=clr, linestyle=ls, alpha=0.4)
    plt.xlabel(r'$z^-$')
    plt.ylabel(r"$\overline{u'v'},\;\tilde{u}\tilde{v}\;/\;u_{\star 2}^2(h)$")
    _leg_uv_out = ([Line2D([0],[0], color='k', ls='-', lw=2,   label=r"$\overline{u'v'}$ (solid)"),
                    Line2D([0],[0], color='k', ls='-', lw=0.8, alpha=0.4, label=r'$\tilde{u}\tilde{v}$ (faded)')]
                   + all_handles())
    plt.legend(handles=_leg_uv_out, fontsize=7, ncol=2)
    plt.grid(True)
    plt.title(r"Shear stress $\overline{u'v'}$ and $\tilde{u}\tilde{v}$ — outer units, Re=500")
    plt.savefig(cwd+'fig'+'/'+'Stress_uv_allFr_outer.png', dpi=300)
    plt.show()

    # 2j. p'v' pressure transport — only plotted if pdvd2D is in the pickle
    if gv('pdvd2D', 'nu_oro') is not None:
        plt.figure(figsize=(6, 8), dpi=300)
        for case, clr, ls, lbl in zip(SIM_NAMES[:2], SIM_COLORS[:2],
                                      SIM_LINESTYLES[:2], SIM_LABELS[:2]):
            _pv = gv('pdvd2D', case)
            _yi = gy_in(case)
            if _pv is None or _yi is None:
                continue
            plt.semilogy(np.mean(_pv, axis=1)/u_star**3, _yi,
                         color=clr, linestyle=ls, label=lbl)
        _mark_h('h')
        plt.xlabel(r"$\langle p'v'\rangle / u_*^3$")
        plt.ylabel(r'$z^+$')
        plt.legend(handles=sim_handles()[:2], fontsize=7)
        plt.grid(True, which='both', ls='--', alpha=0.5)
        plt.title(r"Pressure transport $\langle p'v'\rangle$ — Re=500")
        plt.savefig(cwd+'fig'+'/'+'PressureTransport.png', dpi=300)
        plt.show()

    ###########################################################################
    # SECTION 3 -- MOMENTUM BALANCE (all 5 Fr, zoomed to y+ <= 200)
    ###########################################################################

    # Shared term colour handles (defined once, used in both tau_yx and tau_yz).
    # Double encoding: term by colour, case by linestyle (see all_handles() for case key).
    _term_handles = [
        Line2D([0],[0], color='steelblue',   ls='-', lw=1.5, label='Coriolis'),
        Line2D([0],[0], color='firebrick',   ls='-', lw=1.5, label='Viscous'),
        Line2D([0],[0], color='darkorange',  ls='-', lw=1.5, label='Reynolds'),
        Line2D([0],[0], color='saddlebrown', ls='-', lw=1.5, label='Temporal'),
    ]

    # 3a. tau_yx — streamwise/wall-normal shear stress, all 6 cases
    # Colour = stress term; linestyle = case (see legend).
    plt.figure(figsize=(10, 6), dpi=300)
    if _smooth_loaded:
        plt.plot(y_in_s[:160], -I_corr_yx_s[:160]/ustr_s1**2,
                 color='steelblue',   linestyle=SMOOTH_LS, linewidth=1.5)
        plt.plot(y_in_s[:160], np.mean(visc_yx_s, axis=1)[:160]/ustr_s1**2,
                 color='firebrick',   linestyle=SMOOTH_LS, linewidth=1.5)
        plt.plot(y_in_s[:160], -np.mean(Rxy_s, axis=1)[:160]/ustr_s1**2,
                 color='darkorange',  linestyle=SMOOTH_LS, linewidth=1.5)
    for case, ls in zip(SIM_NAMES, SIM_LINESTYLES):
        _Ic = gv('I_corr_yx', case)
        _vx = gv('visc_yx',   case)
        _rv = gv('rey_uv',    case)
        _dt = gv('dudt',      case)
        _yi = gy_in(case)
        if _Ic is None or _yi is None:
            continue
        _yn = _yi[:limity]
        plt.plot(_yn, -_Ic[:limity]/_ustar_ref**2,    color='steelblue',   linestyle=ls)
        if _vx is not None:
            plt.plot(_yn,  _vx[:limity]/_ustar_ref**2, color='firebrick',   linestyle=ls)
        if _rv is not None:
            plt.plot(_yn, -_xprof(case, _rv)[:limity]/_ustar_ref**2,
                     color='darkorange', linestyle=ls)
        if _dt is not None:
            plt.plot(_yn,  _dt[:limity]/_ustar_ref**2, color='saddlebrown', linestyle=ls)
    plt.legend(handles=_term_handles + all_handles(), fontsize=7, ncol=2, loc='upper right')
    _mark_h('v')
    plt.xlim(0, 200)
    plt.ylim(-0.1, 1.1)
    plt.xlabel(r'$z^+$')
    plt.ylabel(r'$\langle\bar{\tau}_{yx}\rangle^+$')
    plt.title(r'Momentum balance $\tau_{yx}$ — all Fr, Re=500 (zoomed $z^+\leq200$)')
    plt.grid(True)
    plt.savefig(cwd+'fig'+'/'+'MomBal_tauyx_allFr.png', dpi=300)
    _save_layers_x(cwd+'fig'+'/'+'MomBal_tauyx_allFr',
                   r'Momentum balance $\tau_{yx}$ — all Fr, Re=500 (zoomed $z^+\leq200$)')
    plt.show()

    # 3b. tau_yz — spanwise/wall-normal shear stress, all 6 cases
    # Colour = stress term; linestyle = case (see legend).
    plt.figure(figsize=(10, 6), dpi=300)
    if _smooth_loaded:
        plt.plot(y_in_s[:160],  I_corr_yz_s[:160]/ustr_s1**2,
                 color='steelblue',   linestyle=SMOOTH_LS, linewidth=1.5)
        plt.plot(y_in_s[:160], np.mean(visc_yz_s, axis=1)[:160]/ustr_s1**2,
                 color='firebrick',   linestyle=SMOOTH_LS, linewidth=1.5)
        plt.plot(y_in_s[:160], np.mean(Ryz_s, axis=1)[:160]/ustr_s1**2,
                 color='darkorange',  linestyle=SMOOTH_LS, linewidth=1.5)
    for case, ls in zip(SIM_NAMES, SIM_LINESTYLES):
        _Iz = -gv('I_corr_yz', case)      # negated: view as positive contribution
        _vz = gv('visc_yz',   case)
        _rw = gv('rey_vw',    case)
        _dw = gv('dwdt',      case)
        _yi = gy_in(case)
        if _Iz is None or _yi is None:
            continue
        _yn = _yi[:limity]
        plt.plot(_yn,  _Iz[:limity]/_ustar_ref**2,   color='steelblue',   linestyle=ls)
        if _vz is not None:
            plt.plot(_yn, _xprof(case, _vz)[:limity]/_ustar_ref**2,
                     color='firebrick', linestyle=ls)
        if _rw is not None:
            plt.plot(_yn, _xprof(case, _rw)[:limity]/_ustar_ref**2,
                     color='darkorange', linestyle=ls)
        if _dw is not None:
            plt.plot(_yn,  _dw[:limity]/_ustar_ref**2, color='saddlebrown', linestyle=ls)
    plt.legend(handles=_term_handles + all_handles(), fontsize=7, ncol=2, loc='upper right')
    _mark_h('v')
    plt.xlim(0, 200)
    plt.ylim(-0.5, 1.0)
    plt.xlabel(r'$z^+$')
    plt.ylabel(r'$\langle\bar{\tau}_{yz}\rangle^+$')
    plt.title(r'Momentum balance $\tau_{yz}$ — all Fr, Re=500 (zoomed $z^+\leq200$)')
    plt.grid(True)
    plt.savefig(cwd+'fig'+'/'+'MomBal_tauyz_allFr.png', dpi=300)
    _save_layers_x(cwd+'fig'+'/'+'MomBal_tauyz_allFr',
                   r'Momentum balance $\tau_{yz}$ — all Fr, Re=500 (zoomed $z^+\leq200$)')
    plt.show()

    ###########################################################################
    # SECTION 3 (outer units) — MOMENTUM BALANCE
    # y^- = y / u_star2(h) per case; stresses normalised by u_star2(h)^2.
    # Smooth case: y^-_s = y_s / ustr_s1, normalised by ustr_s1^2.
    ###########################################################################

    # 3a (outer). tau_yx — streamwise/wall-normal shear stress, outer units
    plt.figure(figsize=(10, 6), dpi=300)
    if _smooth_loaded:
        _y_out_s = y_s / ustr_s1
        plt.plot(_y_out_s, -I_corr_yx_s/ustr_s1**2,
                 color='steelblue',   linestyle=SMOOTH_LS, linewidth=1.5)
        plt.plot(_y_out_s, np.mean(visc_yx_s, axis=1)/ustr_s1**2,
                 color='firebrick',   linestyle=SMOOTH_LS, linewidth=1.5)
        plt.plot(_y_out_s, -np.mean(Rxy_s, axis=1)/ustr_s1**2,
                 color='darkorange',  linestyle=SMOOTH_LS, linewidth=1.5)
    for case, ls in zip(SIM_NAMES, SIM_LINESTYLES):
        _Ic  = gv('I_corr_yx',   case)
        _vx  = gv('visc_yx',     case)
        _rv  = gv('rey_uv',      case)
        _dt  = gv('dudt',        case)
        _us2 = gv('u_star2',     case)
        _yc  = gv('y',           case)
        if _Ic is None or _us2 is None or _yc is None:
            continue
        _us2_hgt = _us2[ghill(case)]
        _y_out   = _yc / _us2_hgt
        plt.plot(_y_out, -_Ic/_us2_hgt**2,  color='steelblue',  linestyle=ls)
        if _vx is not None:
            plt.plot(_y_out,  _vx/_us2_hgt**2, color='firebrick',  linestyle=ls)
        if _rv is not None:
            plt.plot(_y_out, -_xprof(case, _rv)/_us2_hgt**2,
                     color='darkorange', linestyle=ls)
        if _dt is not None:
            plt.plot(_y_out,  _dt/_us2_hgt**2, color='saddlebrown', linestyle=ls)
    plt.legend(handles=_term_handles + all_handles(), fontsize=7, ncol=2, loc='upper right')
    plt.xlabel(r'$z^-$')
    plt.ylabel(r'$\langle\bar{\tau}_{yx}\rangle^-$')
    plt.title(r'Momentum balance $\tau_{yx}$ — outer units, Re=500')
    plt.grid(True)
    plt.savefig(cwd+'fig'+'/'+'MomBal_tauyx_allFr_outer.png', dpi=300)
    plt.show()

    # 3b (outer). tau_yz — spanwise/wall-normal shear stress, outer units
    plt.figure(figsize=(10, 6), dpi=300)
    if _smooth_loaded:
        _y_out_s = y_s / ustr_s1
        plt.plot(_y_out_s,  I_corr_yz_s/ustr_s1**2,
                 color='steelblue',   linestyle=SMOOTH_LS, linewidth=1.5)
        plt.plot(_y_out_s, np.mean(visc_yz_s, axis=1)/ustr_s1**2,
                 color='firebrick',   linestyle=SMOOTH_LS, linewidth=1.5)
        plt.plot(_y_out_s, np.mean(Ryz_s, axis=1)/ustr_s1**2,
                 color='darkorange',  linestyle=SMOOTH_LS, linewidth=1.5)
    for case, ls in zip(SIM_NAMES, SIM_LINESTYLES):
        _Iz  = -gv('I_corr_yz',   case)
        _vz  =  gv('visc_yz',     case)
        _rw  =  gv('rey_vw',      case)
        _dw  =  gv('dwdt',        case)
        _us2 =  gv('u_star2',     case)
        _yc  =  gv('y',           case)
        if _Iz is None or _us2 is None or _yc is None:
            continue
        _us2_hgt = _us2[ghill(case)]
        _y_out   = _yc / _us2_hgt
        plt.plot(_y_out,  _Iz/_us2_hgt**2, color='steelblue',  linestyle=ls)
        if _vz is not None:
            plt.plot(_y_out, _xprof(case, _vz)/_us2_hgt**2,
                     color='firebrick',  linestyle=ls)
        if _rw is not None:
            plt.plot(_y_out, _xprof(case, _rw)/_us2_hgt**2,
                     color='darkorange', linestyle=ls)
        if _dw is not None:
            plt.plot(_y_out,  _dw/_us2_hgt**2, color='saddlebrown', linestyle=ls)
    plt.legend(handles=_term_handles + all_handles(), fontsize=7, ncol=2, loc='upper right')
    plt.xlabel(r'$z^-$')
    plt.ylabel(r'$\langle\bar{\tau}_{yz}\rangle^-$')
    plt.title(r'Momentum balance $\tau_{yz}$ — outer units, Re=500')
    plt.grid(True)
    plt.savefig(cwd+'fig'+'/'+'MomBal_tauyz_allFr_outer.png', dpi=300)
    plt.show()

    ###########################################################################
    # SECTION 4 -- NEW SCIENTIFICALLY INTERESTING PLOTS
    ###########################################################################

    # 4a. Form drag vs skin friction partition (bar chart)
    _dlbls, _form_d, _skin_d = [], [], []
    for case, lbl in zip(SIM_NAMES, SIM_LABELS):
        _pd = gv('P_drag', case)
        _fx = gv('Fyx',    case)
        if _pd is None or _fx is None:
            continue
        _dlbls.append(lbl)
        _form_d.append(float(_pd)/u_star**2)
        _skin_d.append(float(_fx)/u_star**2)
    if _dlbls:
        _xp = np.arange(len(_dlbls))
        _bw = 0.35
        plt.figure(figsize=(8, 5), dpi=300)
        plt.bar(_xp - _bw/2, _form_d, _bw, color='steelblue', label='Form drag')
        plt.bar(_xp + _bw/2, _skin_d, _bw, color='firebrick',  label='Skin friction')
        plt.xticks(_xp, _dlbls, fontsize=8)
        plt.ylabel(r'Drag $/ u_*^2$')
        plt.legend()
        plt.grid(True, axis='y')
        plt.title('Form drag vs skin friction partition -- all Fr')
        plt.savefig(cwd+'fig'+'/'+'DragPartition_allFr.png', dpi=300)
        plt.show()

    # 4b. Dispersive kinetic energy (DKE) profile — rough-wall cases
    plt.figure(figsize=(8, 6), dpi=300)
    for case, clr, ls, lbl in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_LABELS):
        _du = gv('DispVelU', case)
        _dv = gv('DispVelV', case)
        _dw = gv('DispVelW', case)
        _yi = gy_in(case)
        if _du is None or _yi is None:
            continue
        _dke = (0.5*(np.mean(_du**2, axis=1) + np.mean(_dv**2, axis=1)
                     + np.mean(_dw**2, axis=1)) / u_star**2)
        plt.semilogy(_yi, _dke, color=clr, linestyle=ls, label=lbl)
    _mark_h('v')
    plt.xlabel(r'$z^+$')
    plt.ylabel(r'$\frac{1}{2}\langle\tilde{u}_i\tilde{u}_i\rangle / u_*^2$')
    plt.legend(handles=sim_handles(), fontsize=7, ncol=2)
    plt.grid(True, which='both', linestyle='--', linewidth=0.4)
    plt.title(r'Dispersive kinetic energy (DKE) — rough-wall cases, Re=500')
    plt.savefig(cwd+'fig'+'/'+'DKE_allFr.png', dpi=300)
    _save_layers_x(cwd+'fig'+'/'+'DKE_allFr', r'Dispersive kinetic energy (DKE) — rough-wall cases, Re=500')
    plt.show()

    # 4c. TKE shear production profile P(z+)/u*3 — rough-wall cases
    plt.figure(figsize=(8, 6), dpi=300)
    for case, clr, ls, lbl in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_LABELS):
        _rv = gv('rey_uv', case)
        _dd = gv('du_dy',  case)
        _yi = gy_in(case)
        if _rv is None or _dd is None or _yi is None:
            continue
        _P1d = -(_xprof(case, _rv)) * _xprof(case, _dd) / u_star**3
        plt.plot(_yi[:430], _P1d[:430], color=clr, linestyle=ls, label=lbl)
    _mark_h('v')
    plt.xlim(0, y_in[429])
    plt.xlabel(r'$z^+$')
    plt.ylabel(r"$-\langle\overline{u'v'}\rangle\,\partial\langle\bar{u}\rangle/\partial z\;/ u_*^3$")
    plt.legend(handles=sim_handles(), fontsize=7, ncol=2)
    plt.grid(True)
    plt.title(r'TKE shear production — rough-wall cases, Re=500')
    plt.savefig(cwd+'fig'+'/'+'TKEproduction_allFr.png', dpi=300)
    _save_layers_x(cwd+'fig'+'/'+'TKEproduction_allFr', r'TKE shear production — rough-wall cases, Re=500')
    plt.show()

    # 4d. Streamwise advection at orographic landmarks -- all Fr
    _loc_colors = {'top': 'magenta', 'lf': 'red', 'bottom': 'black', 'rf': 'blue'}
    _loc_labels = {'top': 'Valley top', 'lf': 'Left flank',
                   'bottom': 'Valley bottom', 'rf': 'Right flank'}
    plt.figure(figsize=(6, 7), dpi=300)
    for case, ls in zip(SIM_NAMES, SIM_LINESTYLES):
        _yi = gy_in(case)
        if _yi is None:
            continue
        for loc, clr in _loc_colors.items():
            _cv = gv('conv_'+loc, case)
            if _cv is None:
                continue
            plt.plot(_cv[:450]/u_star**3, _yi[:450], color=clr, linestyle=ls)
    _lh = [Line2D([0],[0], color=c, ls='-', label=_loc_labels[loc])
           for loc, c in _loc_colors.items()]
    plt.legend(handles=_lh + sim_handles(), fontsize=7, ncol=2)
    _mark_h('h')
    plt.axhline(y=Re_tau, color='black', linestyle='--', linewidth=0.8)
    plt.text(0, Re_tau, r'$\delta_o$', va='bottom', ha='left', fontsize=9)
    plt.xlabel(r'$u_j\,\partial u_i/\partial x_j\;/ u_*^3$')
    plt.ylabel(r'$z^+$')
    plt.grid(True, linestyle=':')
    plt.title(r'Streamwise advection at orographic landmarks — rough-wall cases, Re=500')
    plt.savefig(cwd+'fig'+'/'+'Advection_landmarks_allFr.png', dpi=300)
    _save_layers_y(cwd+'fig'+'/'+'Advection_landmarks_allFr',
                   r'Streamwise advection at orographic landmarks — rough-wall cases, Re=500')
    plt.show()

    # ═══════════════════════════════════════════════════════════════════════
    # ░░  SECTION 5 — CHAPTER-6 IMMEDIATELY-ACHIEVABLE DIAGNOSTICS (all Fr)  ░░
    # Research.md "Immediately achievable from existing data" (lines 568-610) +
    # the medium-priority items computable from the existing phase-averaged 2-D
    # fields and the first-plane snapshots.  EVERY plot here is a cross-case
    # visualization (all simulations overlaid / side-by-side panels) — never an
    # individual per-case figure (those stay in PhAvg_rotated.py's fig_rotated/).
    # Each block is gated: a case missing a required field is skipped, not crashed.
    # Reduced numbers are collected in _ch6 for the end-of-run summary.  Scaling
    # is single-reference (u_star / l_in / Re_tau); grids/eps/geometry per-case.
    # ═══════════════════════════════════════════════════════════════════════
    print('\n' + '=' * 78)
    print('SECTION 5 — Chapter-6 immediately-achievable diagnostics (all Fr)')
    print('=' * 78)
    _ch6 = {}
    def _ch6set(case, key, val):
        _ch6.setdefault(case, {})[key] = val

    def _stations(case):
        """Per-case station COLUMNS as fractions of this case's own nx:
        top/crest (i=0), windward (nx/4, descending), floor (nx/2, valley
        bottom), lee (3nx/4, ascending).  Matches eps_top/lf/bottom/rf."""
        _f = gv('AvgPhU', case)
        _nxc = _f.shape[1] if _f is not None else nx
        return {'top': 0, 'wind': _nxc // 4, 'floor': _nxc // 2, 'lee': (3 * _nxc) // 4}

    def _surf_rows(case):
        """First fluid ROW index per column (= number of solid cells in the
        column), from this case's own eps; clipped into range."""
        _e = geps(case)
        _h = np.sum(_e, axis=0).astype(int)
        return np.clip(_h, 0, _e.shape[0] - 1)

    def _col_at_xplus(case, xp):
        """Column index nearest streamwise position xp (inner units) on this
        case's own x⁺ axis."""
        _xc = gx_in(case)
        return int(np.argmin(np.abs(_xc - xp)))

    # ── D1. Surface wind-veer angle α_s(x⁺) (immediate #1) ──────────────────
    # Veer = ∠ of the near-wall (first-fluid-cell) wind to the geostrophic
    # (rotated x), per x-column.  All cases overlaid.
    plt.figure(figsize=(9, 5), dpi=300)
    for case, clr, ls, lbl in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_LABELS):
        _U = gv('AvgPhU', case); _W = gv('AvgPhW', case)
        if _U is None or _W is None:
            continue
        _sr = _surf_rows(case); _ii = np.arange(_U.shape[1])
        _veer = np.degrees(np.arctan2(_W[_sr, _ii], _U[_sr, _ii]))
        plt.plot(gx_in(case), _veer, color=clr, linestyle=ls, label=lbl)
        _st = _stations(case)
        _ch6set(case, 'veer_surf_range', (float(np.nanmin(_veer)), float(np.nanmax(_veer))))
        _ch6set(case, 'veer_wind_lee', (float(_veer[_st['wind']]), float(_veer[_st['lee']])))
    plt.xlabel(r'$x^+$'); plt.ylabel(r'$\alpha_s\;(\mathrm{deg})$')
    plt.legend(handles=sim_handles(), fontsize=7, ncol=2)
    plt.grid(True, linestyle='--', linewidth=0.4)
    plt.title(r'Surface wind-veer angle $\alpha_s(x^+)$ — all Fr, Re=500')
    plt.savefig(cwd + 'fig/' + 'Ch6_veer_surface_allFr.png', dpi=300)
    plt.show()

    # ── D4. Depth-integrated Ekman transport M_y(x⁺)=∫₀^δ⟨U⟩dz (immediate #4) ─
    # P1 falsification test for the τ_zy depth-integral mechanism: M_y should NOT
    # be suppressed over the valley by a factor comparable to the τ_zx reduction
    # (~0.59, Research.md §6.14.5).  All cases overlaid; per-case windward
    # suppression logged + a verdict.
    _TAU_ZX_REDUCTION = 0.59          # reference valley τ_zx reduction (Ch. 6)
    plt.figure(figsize=(9, 5), dpi=300)
    for case, clr, ls, lbl in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_LABELS):
        _U = gv('AvgPhU', case); _yc = gv('y', case)
        if _U is None or _yc is None:
            continue
        _m = gmask0(case)
        _Um = _U * _m if np.shape(_m) == np.shape(_U) else _U
        _jtop = int(np.argmin(np.abs(gy_in(case) - Re_tau)))     # δ row (z⁺≈Re_tau)
        _My = vIntegral_2d(_Um, _U.shape[0], _yc)[_jtop, :]
        plt.plot(gx_in(case), _My, color=clr, linestyle=ls, label=lbl)
        _Mmean = float(np.nanmean(_My))
        _supp = (1.0 - _My[_stations(case)['wind']] / _Mmean) if _Mmean != 0 else float('nan')
        _ch6set(case, 'My_windward_suppression', float(_supp))
        print(f"  [D4] {case:<12} M_y windward suppression = {_supp:+.3f} "
              f"(τ_zx ref {_TAU_ZX_REDUCTION:.2f}) → "
              f"{'mechanism HOLDS' if abs(_supp) < 0.5*_TAU_ZX_REDUCTION else 'check: comparable to τ_zx'}")
    plt.xlabel(r'$x^+$'); plt.ylabel(r'$M_y(x^+)=\int_0^\delta\langle\bar{u}\rangle\,dz$')
    plt.legend(handles=sim_handles(), fontsize=7, ncol=2)
    plt.grid(True, linestyle='--', linewidth=0.4)
    plt.title(r'Depth-integrated Ekman transport $M_y(x^+)$ — all Fr, Re=500')
    plt.savefig(cwd + 'fig/' + 'Ch6_My_x_allFr.png', dpi=300)
    plt.show()

    # ── D5. Form-drag windward/lee split (immediate #5) ─────────────────────
    # P_Lag = windward face (high pressure), P_Front = lee face (low pressure);
    # net retarding form drag = ΣP_Lag − ΣP_Front.  Bar chart all cases.
    _d5 = []
    for case, lbl in zip(SIM_NAMES, SIM_LABELS):
        _pl = gv('P_Lag', case); _pf = gv('P_Front', case)
        if _pl is None or _pf is None:
            continue
        _w = float(np.sum(_pl)); _le = float(np.sum(_pf))
        _d5.append((lbl, _w, _le))
        _ch6set(case, 'Dform_wind_lee', (_w, _le))
    if _d5:
        print('\n  [D5] FORM DRAG windward/lee split (Σ over faces):')
        print(f"    {'case':<14}{'windward(P_Lag)':>16}{'lee(P_Front)':>14}{'net':>12}")
        for _lb, _w, _le in _d5:
            print(f"    {_lb:<14}{_w:>16.5e}{_le:>14.5e}{_w - _le:>12.5e}")
        _xp = np.arange(len(_d5)); _bw = 0.38
        plt.figure(figsize=(8, 5), dpi=300)
        plt.bar(_xp - _bw / 2, [r[1] for r in _d5], _bw, color='steelblue', label='windward (P_Lag)')
        plt.bar(_xp + _bw / 2, [r[2] for r in _d5], _bw, color='firebrick', label='lee (P_Front)')
        plt.xticks(_xp, [r[0] for r in _d5], fontsize=8)
        plt.ylabel('Form-drag contribution'); plt.legend(); plt.grid(True, axis='y')
        plt.title('Form-drag windward vs lee split — all Fr')
        plt.savefig(cwd + 'fig/' + 'Ch6_Dform_windlee_allFr.png', dpi=300)
        plt.show()

    # ── D8. Streamwise momentum budget at a station x⁺≈1050 (immediate #8) ───
    # Recompute C/V/R at one x-column from the 2-D fields (the pickled I_corr_yx
    # etc. are x-averaged 1-D): V=ν∂⟨u⟩/∂z, R=−⟨u'v'⟩, C=−∫⟨v⟩dz (g2≈0 rotated).
    # All cases overlaid; smooth reference (x-independent) for context.
    plt.figure(figsize=(8, 6), dpi=300)
    for case, clr, ls in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES):
        _U = gv('AvgPhU', case); _V = gv('AvgPhV', case); _ruv = gv('rey_uv', case)
        _yc = gv('y', case)
        if _U is None or _V is None or _ruv is None or _yc is None:
            continue
        _i = _col_at_xplus(case, 1050.0)
        _V_visc = nu * np.gradient(_U[:, _i], _yc)
        _R = -_ruv[:, _i]
        _C = -vIntegral(_V[:, _i], _U.shape[0], _yc)
        _zc = gy_in(case)
        _u2 = _ustar_ref ** 2
        plt.plot(_zc, _C / _u2, color='steelblue',  linestyle=ls)
        plt.plot(_zc, _V_visc / _u2, color='firebrick', linestyle=ls)
        plt.plot(_zc, _R / _u2, color='darkorange', linestyle=ls)
        plt.plot(_zc, (_C + _V_visc + _R) / _u2, color='black', linestyle=ls)
    _mark_h('v')
    plt.xlim(0, 200)
    plt.xlabel(r'$z^+$'); plt.ylabel(r'$\tau_{zx}$ budget $/u_*^2$ at $x^+\!\approx\!1050$')
    _d8h = [Line2D([0], [0], color=c, label=l) for c, l in
            [('steelblue', 'Coriolis C'), ('firebrick', 'Viscous V'),
             ('darkorange', 'Reynolds R'), ('black', 'Total T')]]
    plt.legend(handles=_d8h + sim_handles(), fontsize=7, ncol=2)
    plt.grid(True, linestyle='--', linewidth=0.4)
    plt.title(r'Streamwise momentum budget at $x^+\approx1050$ — all Fr')
    plt.savefig(cwd + 'fig/' + 'Ch6_MomBudget_x1050_allFr.png', dpi=300)
    plt.show()

    # ── D9. Log-log |⟨W⟩|(z⁺) at the windward peak (immediate #9) ───────────
    plt.figure(figsize=(8, 6), dpi=300)
    for case, clr, ls, lbl in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_LABELS):
        _W = gv('AvgPhW', case)
        if _W is None:
            continue
        _i = _stations(case)['wind']
        _Wp = np.abs(_W[:, _i]); _zc = gy_in(case)
        plt.loglog(_zc, _Wp, color=clr, linestyle=ls, label=lbl)
        _jpk = int(np.argmax(_Wp))
        _ch6set(case, 'Wwind_peak', (float(_Wp[_jpk]), float(_zc[_jpk])))
    plt.xlabel(r'$z^+$'); plt.ylabel(r'$|\langle\bar{w}\rangle|$ (windward)')
    plt.legend(handles=sim_handles(), fontsize=7, ncol=2)
    plt.grid(True, which='both', linestyle='--', linewidth=0.4)
    plt.title(r'Log-log $|\langle\bar{w}\rangle|(z^+)$ at windward peak — all Fr')
    plt.savefig(cwd + 'fig/' + 'Ch6_Wwind_loglog_allFr.png', dpi=300)
    plt.show()

    # ── D10. Lee–windward W symmetry test (immediate #10) ───────────────────
    plt.figure(figsize=(8, 6), dpi=300)
    for case, clr, ls, lbl in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_LABELS):
        _W = gv('AvgPhW', case)
        if _W is None:
            continue
        _st = _stations(case); _zc = gy_in(case)
        _Ww = _W[:, _st['wind']]; _Wl = _W[:, _st['lee']]
        plt.plot(_zc, _Ww, color=clr, linestyle=ls)
        plt.plot(_zc, _Wl, color=clr, linestyle=ls, alpha=0.4)
        _aw = float(np.max(np.abs(_Ww))); _al = float(np.max(np.abs(_Wl)))
        _ratio = _al / _aw if _aw != 0 else float('nan')
        _ch6set(case, 'W_lee_wind_ratio', _ratio)
        print(f"  [D10] {case:<12} lee/windward |W| peak ratio = {_ratio:.3f}")
    _mark_h('v')
    plt.xlabel(r'$z^+$'); plt.ylabel(r'$\langle\bar{w}\rangle$ (solid=windward, faded=lee)')
    plt.legend(handles=sim_handles(), fontsize=7, ncol=2)
    plt.grid(True, linestyle='--', linewidth=0.4)
    plt.title(r'Lee vs windward $\langle\bar{w}\rangle(z^+)$ — all Fr')
    plt.savefig(cwd + 'fig/' + 'Ch6_W_leewind_allFr.png', dpi=300)
    plt.show()

    # ── D12. TKE production at valley centre vs smooth (immediate #15) ───────
    # P(z⁺) = −⟨u'v'⟩ ∂⟨u⟩/∂z at the valley-centre column; smooth reference is
    # x-mean.  Logs the production-peak height (Ch. 6: ≈17 smooth → ≈34 valley).
    plt.figure(figsize=(8, 6), dpi=300)
    if _smooth_loaded:
        _Ps = -np.mean(Rxy_s, axis=1) * np.mean(du_dy_s, axis=1)
        plt.semilogx(y_in_s, _Ps / ustr_s1 ** 3, color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
        _ch6set('smooth', 'TKEprod_peak_z', float(y_in_s[int(np.argmax(_Ps))]))
    for case, clr, ls, lbl in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_LABELS):
        _ruv = gv('rey_uv', case); _dd = gv('du_dy', case)
        if _ruv is None or _dd is None:
            continue
        _i = _stations(case)['floor']
        _P = -_ruv[:, _i] * _dd[:, _i]; _zc = gy_in(case)
        plt.semilogx(_zc, _P / _ustar_ref ** 3, color=clr, linestyle=ls, label=lbl)
        _jpk = int(np.argmax(_P))
        _ch6set(case, 'TKEprod_peak_z', float(_zc[_jpk]))
    _mark_h('v')
    plt.xlabel(r'$z^+$'); plt.ylabel(r'$\mathcal{P}/u_*^3$ (valley centre)')
    plt.legend(handles=all_handles(), fontsize=7, ncol=2)
    plt.grid(True, which='both', linestyle='--', linewidth=0.4)
    plt.title(r'TKE production at valley centre vs smooth — all Fr')
    plt.savefig(cwd + 'fig/' + 'Ch6_TKEprod_centre_allFr.png', dpi=300)
    plt.show()

    # ── D14. Outer-layer mean-velocity surplus ΔU⁺(z⁺) (Fig 6.19 #3) ────────
    # ΔU⁺ = ⟨u⟩⁺_valley − ⟨u⟩⁺_smooth (both on the single-reference axis), per
    # case interpolated onto the smooth z⁺ grid.  Logs max surplus above z⁺≈100.
    if _smooth_loaded:
        _Usm = np.mean(U_s_p, axis=1)
        plt.figure(figsize=(8, 6), dpi=300)
        for case, clr, ls, lbl in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_LABELS):
            _upr = gv('u_plus_rot', case); _yi = gy_in(case)
            if _upr is None or _yi is None:
                continue
            _uval = np.interp(y_in_s, _yi, _upr / ustr_s1)
            _dU = _uval - _Usm
            plt.plot(y_in_s, _dU, color=clr, linestyle=ls, label=lbl)
            _outer = y_in_s > 100.0
            if np.any(_outer):
                _jm = np.where(_outer)[0][int(np.argmax(_dU[_outer]))]
                _ch6set(case, 'dUplus_max', (float(_dU[_jm]), float(y_in_s[_jm])))
        plt.axhline(0, color='k', lw=0.6)
        _mark_h('v'); plt.xscale('log')
        plt.xlabel(r'$z^+$'); plt.ylabel(r'$\Delta\langle\bar{u}\rangle^+$ (valley − smooth)')
        plt.legend(handles=sim_handles(), fontsize=7, ncol=2)
        plt.grid(True, which='both', linestyle='--', linewidth=0.4)
        plt.title(r'Outer-layer velocity surplus $\Delta U^+$ — all Fr')
        plt.savefig(cwd + 'fig/' + 'Ch6_dUplus_allFr.png', dpi=300)
        plt.show()

    # ── D15. TKE anisotropy: normal-stress components (TKE #5) ──────────────
    # ⟨u'u'⟩, ⟨v'v'⟩, ⟨w'w'⟩ (x-averaged, intrinsic) vs z⁺, all cases overlaid;
    # smooth reference.  Logs which component the valley most enhances at peak.
    plt.figure(figsize=(8, 6), dpi=300)
    if _smooth_loaded:
        plt.plot(y_in_s, np.mean(Rxx_s, axis=1) / ustr_s1 ** 2, color=SMOOTH_COLOR, linestyle='-')
        plt.plot(y_in_s, np.mean(Ryy_s, axis=1) / ustr_s1 ** 2, color=SMOOTH_COLOR, linestyle='--')
        plt.plot(y_in_s, np.mean(Rzz_s, axis=1) / ustr_s1 ** 2, color=SMOOTH_COLOR, linestyle=':')
    for case, clr, ls, lbl in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_LABELS):
        _uu = gv('rey_uu', case); _vv = gv('rey_vv', case); _ww = gv('rey_ww', case)
        _yi = gy_in(case)
        if _uu is None or _vv is None or _ww is None or _yi is None:
            continue
        _pu = _xprof(case, _uu) / _ustar_ref ** 2
        _pv = _xprof(case, _vv) / _ustar_ref ** 2
        _pw = _xprof(case, _ww) / _ustar_ref ** 2
        plt.plot(_yi, _pu, color=clr, linestyle='-')
        plt.plot(_yi, _pv, color=clr, linestyle='--')
        plt.plot(_yi, _pw, color=clr, linestyle=':')
        _comp = ['uu', 'vv', 'ww'][int(np.argmax([np.max(_pu), np.max(_pv), np.max(_pw)]))]
        _ch6set(case, 'TKE_dominant_component', _comp)
    _mark_h('v'); plt.xscale('log')
    plt.xlabel(r'$z^+$'); plt.ylabel(r"$\langle u_i'^2\rangle/u_*^2$")
    _d15h = [Line2D([0], [0], color='k', ls=s, label=l) for s, l in
             [('-', r"$u'u'$"), ('--', r"$v'v'$"), (':', r"$w'w'$")]]
    plt.legend(handles=_d15h + sim_handles(), fontsize=7, ncol=2)
    plt.grid(True, which='both', linestyle='--', linewidth=0.4)
    plt.title(r'TKE anisotropy (normal stresses) — all Fr')
    plt.savefig(cwd + 'fig/' + 'Ch6_TKEanisotropy_allFr.png', dpi=300)
    plt.show()

    # ── D2/D3. Mean & dispersive streamfunction ψ(x⁺,z⁺) (immediate #2,#19) ──
    # ψ = ∫₀^z ⟨u⟩ dz' (wall-anchored).  Computed per case, stored into the sim
    # dict, then drawn as side-by-side panels by the existing plot2D_allFr.
    for case in SIM_NAMES:
        _U = gv('AvgPhU', case); _V = gv('AvgPhV', case); _yc = gv('y', case)
        if _U is not None and _yc is not None:
            sims[case]['psi_mean'] = streamfunction_2d(_U, _V, gv('x', case), _yc,
                                                       mask=gmask0(case))
            _pm = sims[case]['psi_mean']
            _jm, _im = np.unravel_index(int(np.nanargmin(_pm)), _pm.shape)
            _ch6set(case, 'psi_min', (float(_pm[_jm, _im]),
                                      float(gx_in(case)[_im]), float(gy_in(case)[_jm])))
        _dU = gv('DispVelU', case); _dV = gv('DispVelV', case)
        if _dU is not None and _yc is not None:
            sims[case]['psi_disp'] = streamfunction_2d(_dU, _dV, gv('x', case), _yc,
                                                       mask=gmask0(case))
            _pd = sims[case]['psi_disp']
            if 'psi_min' in _ch6.get(case, {}):
                _pmin = _ch6[case]['psi_min'][0]
                _ratio = (float(np.nanmin(_pd)) / _pmin) if _pmin != 0 else float('nan')
                _ch6set(case, 'psi_disp_ratio', _ratio)
    plot2D_allFr('psi_mean', r'Mean streamfunction $\psi(x^+,z^+)$ — all Fr',
                 'RdBu_r', 'Ch6_streamfunction_allFr.png', ylim=250)
    plot2D_allFr('psi_disp', r"Dispersive streamfunction $\psi''(x^+,z^+)$ — all Fr",
                 'RdBu_r', 'Ch6_streamfunction_disp_allFr.png', ylim=250)
    print('  [D2/D3] ψ note: 2-D spanwise-mean projection; the spanwise drift '
          '⟨w̄⟩ (AvgPhW) carries fluid through the apparent recirculation — a '
          'true 3-D closed-orbit test needs spanwise-resolved fields (gated).')

    # ── D17. Streamwise-resolved Coriolis integrand C(x⁺,z⁺) (immediate #3) ──
    # C(x,z)=∫₀^z(g2−⟨v⟩)dz' (g2≈0 rotated).  R(x,z)=−⟨u'v'⟩ is already the
    # rey_uv panel (Section 1), so only the new C map is added here.
    for case in SIM_NAMES:
        _V = gv('AvgPhV', case); _yc = gv('y', case)
        if _V is None or _yc is None:
            continue
        _m = gmask0(case); _Vm = _V * _m if np.shape(_m) == np.shape(_V) else _V
        sims[case]['C2D'] = -vIntegral_2d(_Vm, _V.shape[0], _yc)
    plot2D_allFr('C2D', r'Streamwise-resolved Coriolis integrand $\mathcal{C}(x^+,z^+)$ — all Fr',
                 'RdBu_r', 'Ch6_Coriolis2D_allFr.png', ylim=200)

    # ── D18. Pressure-Poisson source decomposition (medium 3) ───────────────
    # ∇²P = −∂²(u_iu_j)/∂x_i∂x_j.  Split the RHS into mean-strain / Reynolds /
    # dispersive sources; store the total for the panel, log which dominates at
    # the Cp extrema (windward / lee floor columns).
    def _d2(f, c, ax):
        return np.gradient(np.gradient(f, c, axis=ax), c, axis=ax)
    def _dxz(f, xc, yc):
        return np.gradient(np.gradient(f, xc, axis=1), yc, axis=0)
    for case in SIM_NAMES:
        _U = gv('AvgPhU', case); _V = gv('AvgPhV', case)
        _xc = gv('x', case); _yc = gv('y', case)
        if _U is None or _V is None or _xc is None or _yc is None:
            continue
        _Smean = -(_d2(_U * _U, _xc, 1) + 2 * _dxz(_U * _V, _xc, _yc) + _d2(_V * _V, _yc, 0))
        _uu = gv('rey_uu', case); _uv = gv('rey_uv', case); _vv = gv('rey_vv', case)
        _Srey = (-(_d2(_uu, _xc, 1) + 2 * _dxz(_uv, _xc, _yc) + _d2(_vv, _yc, 0))
                 if (_uu is not None and _uv is not None and _vv is not None)
                 else np.zeros_like(_Smean))
        _Du = gv('UU_disp', case); _Duv = gv('UV_disp', case); _Dvv = gv('VV_disp', case)
        _Sdisp = (-(_d2(_Du, _xc, 1) + 2 * _dxz(_Duv, _xc, _yc) + _d2(_Dvv, _yc, 0))
                  if (_Du is not None and _Duv is not None and _Dvv is not None)
                  else np.zeros_like(_Smean))
        sims[case]['Psource_total'] = _Smean + _Srey + _Sdisp
        # Dominant source over the near-wall valley band (z⁺<50) at the floor.
        _band = gy_in(case) < 50.0
        _i0, _i1 = _stations(case)['wind'], _stations(case)['lee']
        _fm = float(np.nanmean(np.abs(_Smean[_band][:, _i0:_i1 + 1])))
        _fr = float(np.nanmean(np.abs(_Srey[_band][:, _i0:_i1 + 1])))
        _fd = float(np.nanmean(np.abs(_Sdisp[_band][:, _i0:_i1 + 1])))
        _tot = _fm + _fr + _fd
        if _tot > 0:
            _ch6set(case, 'poisson_fracs', (_fm / _tot, _fr / _tot, _fd / _tot))
            _dom = ['mean-strain', 'Reynolds', 'dispersive'][int(np.argmax([_fm, _fr, _fd]))]
            print(f"  [D18] {case:<12} Poisson source over valley: mean={_fm/_tot:.2f} "
                  f"rey={_fr/_tot:.2f} disp={_fd/_tot:.2f} → dominant: {_dom}")
    plot2D_allFr('Psource_total', r'Pressure-Poisson source $-\partial^2(u_iu_j)/\partial x_i\partial x_j$ — all Fr',
                 'RdBu_r', 'Ch6_Poisson_source_allFr.png', ylim=200)

    # ── D11/D16. Terrain-following maps (immediate #14, #20) ────────────────
    # Re-sample to ζ⁺ = z⁺ − local-surface⁺ so a constant-ζ row sits a constant
    # distance above the local surface, removing the leading-order kinematic
    # crest/valley artefact.  Side-by-side panels, all cases.
    def _panels_zeta(compute_fn, title, cmap, savename, zmax_plus=200):
        _av = []
        for cn, lb in zip(SIM_NAMES, SIM_LABELS):
            _r = compute_fn(cn)
            if _r is not None:
                _av.append((cn, lb, _r[0], _r[1]))
        if not _av:
            print(f'  [Ch6] no data for {savename}')
            return
        _vmax = max(float(np.nanmax(np.abs(_f))) for _, _, _f, _z in _av) or 1.0
        fig, axes = plt.subplots(1, len(_av), figsize=(4 * len(_av) + 1.0, 5), sharey=True)
        if len(_av) == 1:
            axes = [axes]
        _mesh = None
        for ax, (cn, lb, _f, _zp) in zip(axes, _av):
            _mesh = ax.pcolormesh(gx_in(cn), _zp, _f, cmap=cmap,
                                  vmin=-_vmax, vmax=_vmax, shading='auto')
            ax.set_ylim(0, zmax_plus); ax.set_title(lb, fontsize=9); ax.set_xlabel(r'$x^+$')
        axes[0].set_ylabel(r'$\zeta^+$ (terrain-following)')
        if _mesh is not None:
            fig.colorbar(_mesh, ax=axes, shrink=0.8)
        fig.suptitle(title)
        fig.savefig(cwd + 'fig/' + savename, dpi=300, bbox_inches='tight'); plt.show()

    def _tf_disp(cn):
        _du = gv('DispVelU', cn); _dv = gv('DispVelV', cn); _dw = gv('DispVelW', cn)
        _yc = gv('y', cn)
        if _du is None or _yc is None:
            return None
        _mag = np.sqrt(_du ** 2
                       + (_dv ** 2 if _dv is not None else 0.0)
                       + (_dw ** 2 if _dw is not None else 0.0))
        _f, _z = terrain_follow_remap(_mag, _yc, _yc[_surf_rows(cn)])
        return _f, _z / l_in

    def _tf_tauzx(cn):
        _U = gv('AvgPhU', cn); _ruv = gv('rey_uv', cn); _yc = gv('y', cn)
        if _U is None or _ruv is None or _yc is None:
            return None
        _tau = nu * np.gradient(_U, _yc, axis=0) - _ruv
        _f, _z = terrain_follow_remap(_tau, _yc, _yc[_surf_rows(cn)])
        return _f, _z / l_in

    _panels_zeta(_tf_disp,
                 r'Terrain-following dispersive velocity magnitude $|\tilde{u}_i|(x^+,\zeta^+)$ — all Fr',
                 'hot_r', 'Ch6_dispTF_allFr.png', zmax_plus=250)
    _panels_zeta(_tf_tauzx,
                 r'Terrain-following streamwise stress $\nu\partial\bar{u}/\partial z-\overline{u^\prime v^\prime}$ — all Fr',
                 'RdBu_r', 'Ch6_tauzxTF_allFr.png', zmax_plus=120)

    # ── D6. Wall-normal pressure equilibrium ∂P/∂z at IBM surface (#6) ──────
    print('\n  [D6] Wall-normal pressure equilibrium  |∂P/∂z|  (IBM band vs outer):')
    print(f"    {'case':<14}{'max|dP/dz| IBM':>18}{'mean|dP/dz| outer':>20}")
    for case, lbl in zip(SIM_NAMES, SIM_LABELS):
        _P = gv('AvgP', case); _yc = gv('y', case)
        if _P is None or _yc is None:
            continue
        _dPdz = np.gradient(_P, _yc, axis=0)
        _zc = gy_in(case)
        _band = _zc < 1.5 * y_in[hill_hgt]
        _outer = _zc > 100.0
        _ibm = float(np.nanmax(np.abs(_dPdz[_band]))) if np.any(_band) else float('nan')
        _out = float(np.nanmean(np.abs(_dPdz[_outer]))) if np.any(_outer) else float('nan')
        _ch6set(case, 'pressure_equil', (_ibm, _out))
        print(f"    {lbl:<14}{_ibm:>18.4e}{_out:>20.4e}")

    # ── D7. Surface-pressure streamwise spectrum / streak spacing (#7) ──────
    print('\n  [D7] Surface-pressure streamwise spectrum (dominant wavelength):')
    print(f"    {'case':<14}{'λ_dom+':>12}{'2·dx+ (Nyquist)':>18}{'flag':>10}")
    for case, lbl in zip(SIM_NAMES, SIM_LABELS):
        _P = gv('AvgP', case)
        if _P is None:
            continue
        _sr = _surf_rows(case); _ii = np.arange(_P.shape[1])
        _ps = _P[_sr, _ii]; _ps = _ps - np.mean(_ps)
        _xc = gx_in(case); _Lxp = float(_xc[-1] - _xc[0])
        _F = np.abs(np.fft.rfft(_ps)) ** 2
        _kdom = int(np.argmax(_F[1:])) + 1 if _F.size > 1 else 0
        _lam = (_Lxp / _kdom) if _kdom > 0 else float('inf')
        _dxp = 2.0 * float(gv('dx', case) or np.nan) / l_in
        _flag = 'grid?' if np.isfinite(_dxp) and _lam < 2.0 * _dxp else 'ok'
        _ch6set(case, 'cp_lambda', float(_lam))
        print(f"    {lbl:<14}{_lam:>12.1f}{_dxp:>18.2f}{_flag:>10}")

    # ── D13. Log-law parameters: matched-range κ (smooth) + z₀/d (#16,#17) ──
    print('\n  [D13] Log-law parameters (κ, z₀ₘ⁺, d⁺):')
    if _smooth_loaded:
        _Us = np.mean(U_s_p, axis=1)
        _msk = (y_in_s >= 60) & (y_in_s <= 200) & (_Us > 0)
        if np.count_nonzero(_msk) >= 2:
            _sl, _ic, _rv, _pv, _se = linregress(np.log(y_in_s[_msk]), _Us[_msk])
            _ksm = (1.0 / _sl) if _sl != 0 else float('nan')
            _ch6set('smooth', 'kappa_matched', float(_ksm))
            print(f"    smooth (matched z⁺∈[60,200]): κ={_ksm:.4f}  R²={_rv**2:.4f}")
    print(f"    {'case':<14}{'κ':>9}{'z0m+':>11}{'d+':>9}")
    for case, lbl in zip(SIM_NAMES, SIM_LABELS):
        _k = gv('kappa_loglaw', case)
        if _k is None:
            continue
        _z0 = gv('z0m_loglaw', case); _d = gv('d_m_loglaw', case)
        _z0f = float(_z0) if _z0 is not None else float('nan')
        _df = float(_d) if _d is not None else float('nan')
        _ch6set(case, 'loglaw', (float(_k), _z0f, _df))
        print(f"    {lbl:<14}{float(_k):>9.4f}{_z0f:>11.5f}{_df:>9.2f}")

    # ── D19. Linear potential-flow Cp vs measured (min-location shift) (#med4) ─
    # Linear inviscid theory over the sinusoid → Cp ∝ +surface elevation (flow
    # accelerates over the crest → low pressure there), so the linear Cp minimum
    # sits at the crest (i=0).  The measured displacement of the minimum is the
    # elliptic/nonlinear (or rotation) effect flagged in Ch. 6.
    print('\n  [D19] Cp minimum: measured vs linear-potential (displacement):')
    print(f"    {'case':<14}{'x+(Cp_min) meas':>18}{'x+(Cp_min) lin':>16}{'Cp_min':>10}{'Cp_max':>10}")
    for case, lbl in zip(SIM_NAMES, SIM_LABELS):
        _P = gv('AvgP', case); _yc = gv('y', case)
        if _P is None or _yc is None:
            continue
        _sr = _surf_rows(case); _ii = np.arange(_P.shape[1])
        _cp = _P[_sr, _ii]; _xc = gx_in(case)
        _imeas = int(np.argmin(_cp))
        _elev = _yc[_sr] - float(np.mean(_yc[_sr]))
        _cplin = -_elev                              # Cp_lin ∝ −elevation → min at crest
        _ilin = int(np.argmin(_cplin))
        _ch6set(case, 'cp_extrema', (float(_cp[_imeas]), float(np.max(_cp))))
        _ch6set(case, 'cp_min_shift', (float(_xc[_imeas]), float(_xc[_ilin])))
        print(f"    {lbl:<14}{float(_xc[_imeas]):>18.1f}{float(_xc[_ilin]):>16.1f}"
              f"{float(_cp[_imeas]):>10.4f}{float(np.max(_cp)):>10.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # ░░  CROSS-CASE RESEARCH AGGREGATION  ░░   (Research.md:536-550)
    # The matrix views a single-run PhAvg_rotated.py cannot build: each Fr placed
    # on the stability axis, dispersive share / scales / similarity / intermittency
    # tracked vs Ri_B, and Re=500-vs-750 (gated until the 750 data exists).
    # Reads the per-run research keys pickled by PhAvg_rotated.py; skips silently
    # for any case whose pickle lacks them.
    # ═══════════════════════════════════════════════════════════════════════
    _FR = {'nu_oro': np.inf, 'fr_1_oro': 1.0, 'fr_0p1_oro': 0.1,
           'fr_0p01_oro': 0.01}
    _xc = [c for c in CASES
           if c['name'] in sims and gv('Ri_B', c['name']) is not None]
    if len(_xc) == 0:
        print('[results] cross-case research aggregation: no pickles carry the '
              'research diagnostics yet — run PhAvg_rotated.py per case first.')
    else:
        _figdir_x = cwd + 'fig' + '/'
        _os.makedirs(_figdir_x, exist_ok=True)
        _nm  = [c['name'] for c in _xc]
        _lab = [c['label'] for c in _xc]
        _col = [c['color'] for c in _xc]
        _mk  = [c['marker'] for c in _xc]
        _RiB = np.array([float(gv('Ri_B', n)) for n in _nm])
        _Lp  = np.array([float(gv('L_col_plus', n)) for n in _nm])

        def _scl(n, key, meth='M2'):
            s = gv('scales', n)
            return float(s[meth][key]) if (s is not None and meth in s) else np.nan
        def _share_BL(n, key):
            v = gv(key, n)
            if v is None:
                return np.nan
            jt = gv('bl_top_j', n)
            seg = v[:max(int(jt), 1)] if jt is not None else v
            return float(np.nanmean(seg)) if np.size(seg) else np.nan
        def _depmean(n, key):
            d = gv(key, n)
            if d is None:
                return np.nan
            vals = [v for v in d.values() if np.isfinite(v)]
            return float(np.mean(vals)) if vals else np.nan

        # ── [X1] Stability axis: each Fr at its measured Ri_B + Ansorge bins ──
        fig, ax = plt.subplots(figsize=(9, 3.6), dpi=300)
        _hi = max(0.2, float(np.nanmax(np.abs(_RiB))) * 1.3)
        ax.axvspan(0,    0.05, color='green',  alpha=0.10)
        ax.axvspan(0.05, 0.15, color='orange', alpha=0.10)
        ax.axvspan(0.15, _hi,  color='red',    alpha=0.10)
        for i, n in enumerate(_nm):
            ax.scatter(_RiB[i], 0.0, color=_col[i], marker=_mk[i], s=90, zorder=5, label=_lab[i])
        ax.set_yticks([]); ax.set_xlim(0, _hi)
        ax.set_xlabel(r'$Ri_B = B_0\,\delta_{neu}/G^2$')
        ax.set_title('Stability axis: weak | intermediate | strong')
        ax.legend(fontsize=7, ncol=3, loc='upper center')
        fig.savefig(_figdir_x + 'Xcase_stability_axis.png', dpi=300, bbox_inches='tight'); plt.show()

        # ── [X2] Dispersive share vs Ri_B (momentum & buoyancy; Goal 4) ───────
        _sm = np.array([_share_BL(n, 'disp_share_mom')  for n in _nm])
        _sb = np.array([_share_BL(n, 'disp_share_buoy') for n in _nm])
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        ax.plot(_RiB, _sm, 'bo-',  label='momentum')
        ax.plot(_RiB, _sb, 'rs--', label='buoyancy')
        ax.set_xlabel(r'$Ri_B$'); ax.set_ylabel('BL-mean dispersive share')
        ax.set_title('Dispersive share vs $Ri_B$')
        ax.legend(); ax.grid(True, ls='--', lw=0.5)
        fig.savefig(_figdir_x + 'Xcase_dispshare_vs_RiB.png', dpi=300, bbox_inches='tight'); plt.show()

        # ── [X3] Scales & Obukhov vs Ri_B (Goals 3 & 1) ───────────────────────
        _panels = [('u_star', 'u*'), ('delta', r'$\delta$'), ('Psi', r'$\Psi=L_x/2\delta$'),
                   ('H_delta', r'$H/\delta$'), ('H_plus', r'$H^+$')]
        fig, axs = plt.subplots(2, 3, figsize=(14, 8), dpi=300)
        for axi, (key, ttl) in zip(axs.flat, _panels):
            axi.plot(_RiB, [_scl(n, key) for n in _nm], 'ko-')
            axi.set_xlabel(r'$Ri_B$'); axi.set_title(ttl); axi.grid(True, ls='--', lw=0.5)
        axs.flat[5].plot(_RiB, _Lp, 'ko-'); axs.flat[5].set_title(r'$L_{col}^+$')
        axs.flat[5].axhline(100, color='r', ls=':', label='collapse ~100')
        axs.flat[5].set_xlabel(r'$Ri_B$'); axs.flat[5].legend(fontsize=7); axs.flat[5].grid(True, ls='--', lw=0.5)
        fig.suptitle('Scales & Obukhov length vs $Ri_B$')
        fig.tight_layout()
        fig.savefig(_figdir_x + 'Xcase_scales_vs_RiB.png', dpi=300, bbox_inches='tight'); plt.show()

        # ── [X4] Similarity departure vs Ri_B (Goal 5) ────────────────────────
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        ax.plot(_RiB, [_depmean(n, 'phi_m_dep') for n in _nm], 'bo-',  label=r'$\phi_m$')
        ax.plot(_RiB, [_depmean(n, 'phi_h_dep') for n in _nm], 'rs--', label=r'$\phi_h$')
        ax.set_xlabel(r'$Ri_B$'); ax.set_ylabel('RMS departure from MOST (station mean)')
        ax.set_title('Similarity departure vs $Ri_B$')
        ax.legend(); ax.grid(True, ls='--', lw=0.5)
        fig.savefig(_figdir_x + 'Xcase_phidep_vs_RiB.png', dpi=300, bbox_inches='tight'); plt.show()

        # ── [X5] Intermittency collapse vs Ri_B (Goal 6; only if γ computed) ──
        _gc = []
        for n in _nm:
            g = gv('gamma_z', n); jt = gv('bl_top_j', n)
            _gc.append(float(np.nanmean(g[:max(int(jt), 1)] if jt is not None else g))
                       if g is not None else np.nan)
        _gc = np.array(_gc)
        if np.any(np.isfinite(_gc)):
            fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
            ax.plot(_RiB, _gc, 'ko-')
            ax.set_xlabel(r'$Ri_B$'); ax.set_ylabel(r'BL-mean intermittency $\gamma$')
            ax.set_title('Intermittency collapse vs $Ri_B$')
            ax.grid(True, ls='--', lw=0.5)
            fig.savefig(_figdir_x + 'Xcase_gamma_vs_RiB.png', dpi=300, bbox_inches='tight'); plt.show()

        # ── [X6] Coriolis–topography COUPLING observables vs Ψ ────────────────
        # First look at how the coupling observables organise on Ψ = Lx/(2δ)
        # (Research.md candidate finding #3, line 558).  HONEST BOUND: across this
        # Froude ladder Ψ is varied only through δ, so it covaries with Ri_B, H/δ
        # and the stability state — read as organisation along a covaried path,
        # NOT a γ_veer(Ψ) scaling law (which needs fixed-Re_τ / varying-Lx runs).
        _Psi  = np.array([_scl(n, 'Psi')                 for n in _nm])
        # γ_veer = α_oro / α_smooth, formed here: per-case orographic global veer
        # (pickled as veer_oro) over the smooth-wall surface veer (smooth ref).
        _vo    = np.array([(gv('veer_oro', n) if gv('veer_oro', n) is not None
                            else np.nan) for n in _nm], float)
        _smv   = (abs(np.degrees(float(alpha_str_s))) if _smooth_loaded else np.nan)
        _gveer = (_vo / _smv if (np.isfinite(_smv) and _smv != 0.0)
                  else np.full(len(_nm), np.nan))
        _dmom = np.array([_share_BL(n, 'disp_share_mom') for n in _nm])
        _Hdel = np.array([_scl(n, 'H_delta')             for n in _nm])
        _o    = np.argsort(_Psi)                          # order by increasing Ψ
        _cpl_panels = [(_gveer, r'$\gamma_{veer}=\alpha_{oro}/\alpha_{smooth}$'),
                       (_dmom,  'BL-mean dispersive momentum share'),
                       (_Hdel,  r'$H/\delta$'),
                       (_Lp,    r'$L_{col}^+$')]
        fig, axs = plt.subplots(2, 2, figsize=(11, 8), dpi=300)
        for axi, (yv, ttl) in zip(axs.flat, _cpl_panels):
            if np.any(np.isfinite(yv)):
                axi.plot(_Psi[_o], yv[_o], '-', color='0.6', lw=1.0, zorder=1)
            for i, n in enumerate(_nm):
                axi.scatter(_Psi[i], yv[i], color=_col[i], marker=_mk[i],
                            s=80, zorder=3, label=_lab[i])
            axi.set_xlabel(r'$\Psi = L_x/(2\delta)$'); axi.set_title(ttl)
            axi.grid(True, ls='--', lw=0.5)
        axs.flat[3].axhline(100, color='r', ls=':', lw=1, label='collapse ~100')
        axs.flat[0].legend(fontsize=7, loc='best')
        fig.suptitle('Coriolis–topography coupling vs $\\Psi$ (first look; '
                     'covaried path, not a scaling law)')
        fig.tight_layout()
        fig.savefig(_figdir_x + 'Xcase_coupling_vs_Psi.png', dpi=300, bbox_inches='tight'); plt.show()

        # ── Console matrix summary + Goal 8 (Re 500 vs 750) status ────────────
        print('\n=== CROSS-CASE RESEARCH MATRIX (Re_D=500) ===')
        print(f"  {'case':<12}{'Ri_B':>11}{'L_col+':>10}{'class':>22}{'Psi':>8}{'H/delta':>9}")
        for n in _nm:
            _s = gv('scales', n)
            _ps = _s['M2']['Psi']     if _s else float('nan')
            _hd = _s['M2']['H_delta'] if _s else float('nan')
            print(f"  {n:<12}{float(gv('Ri_B', n)):>11.3e}{float(gv('L_col_plus', n)):>10.1f}"
                  f"{str(gv('stab_class', n)):>22}{_ps:>8.3f}{_hd:>9.4f}")
        _re_set = {int(gv('Re', n)) for n in _nm if gv('Re', n) is not None}
        print('  Reynolds numbers present: %s' % sorted(_re_set))
        if 750 not in _re_set:
            print('  Goal 8 (Re_D=750): data absent — inner-vs-outer collapse test pending 750 pickles.')

        # ── COUPLING vs Ψ table (Coriolis–topography; ordered by increasing Ψ) ─
        print('\n=== CORIOLIS–TOPOGRAPHY COUPLING vs Psi (Re_D=500; covaried path) ===')
        print(f"  {'case':<13}{'Psi':>8}{'H/delta':>9}{'Ri_B':>11}"
              f"{'gamma_veer':>12}{'disp_mom':>10}{'class':>22}")
        for i in _o:
            n   = _nm[i]
            _s  = gv('scales', n)
            _ps = _s['M2']['Psi']     if _s else float('nan')
            _hd = _s['M2']['H_delta'] if _s else float('nan')
            print(f"  {n:<13}{_ps:>8.3f}{_hd:>9.4f}{float(gv('Ri_B', n)):>11.3e}"
                  f"{_gveer[i]:>12.4f}{_dmom[i]:>10.4f}{str(gv('stab_class', n)):>22}")
        print('  NOTE: Psi covaries with Ri_B / H/delta / stability along this '
              'ladder — a first look, not a gamma_veer(Psi) scaling law.')

    # ═══════════════════════════════════════════════════════════════════════
    # ░░  END-OF-RUN SUMMARY  ░░  (detailed; teed to sim_stats.log)
    # Per-case inputs found/skipped, scales & stability, the Chapter-6
    # observations gathered in _ch6, and the honestly-gated/blocked items.
    # ═══════════════════════════════════════════════════════════════════════
    def _fmt(v, f='{:.4g}'):
        try:
            return f.format(float(v))
        except (TypeError, ValueError):
            return str(v)

    def _print_run_summary():
        print('\n' + '#' * 78)
        print('# END-OF-RUN SUMMARY — results.py cross-case post-processing')
        print('#' * 78)
        _plotted, _skipped = [], []
        for case in SIM_DIRS:
            _lbl = next((c['label'] for c in CASES if c['name'] == case), case)
            _pv = _prov.get(case, {})
            _ob = _ch6.get(case, {})
            _has = _pv.get('pickle', False)
            (_plotted if _has else _skipped).append(case)
            print('\n--- %s  (%s) ---' % (case, _lbl))
            # Inputs / provenance
            print('  inputs : pickle=%s  per_case_grid=%s'
                  % (_has, _pv.get('per_case_grid', False)))
            if _pv.get('inst'):
                print('           inst planes read : %s'
                      % ', '.join('%s=%s' % (k, v) for k, v in _pv['inst'].items()))
            if _pv.get('inst_skip'):
                print('           inst planes SKIPPED (truncated/corrupt): %s'
                      % ', '.join('%s=%s' % (k, v) for k, v in _pv['inst_skip'].items()))
            if not _has:
                print('  (no pickle — case skipped in all diagnostics)')
                continue
            # Scales / stability
            _sc = gv('scales', case); _m2 = _sc.get('M2') if isinstance(_sc, dict) else None
            print('  scales : u*=%s  delta=%s  Psi=%s  H/delta=%s  H+=%s'
                  % (_fmt(_m2.get('u_star')) if _m2 else 'NA',
                     _fmt(_m2.get('delta')) if _m2 else 'NA',
                     _fmt(gv('Psi', case)), _fmt(gv('H_delta', case)),
                     _fmt(gv('H_plus_r', case))))
            print('  stab   : Ri_B=%s  class=%s  L_col+=%s'
                  % (_fmt(gv('Ri_B', case), '{:.3e}'), gv('stab_class', case),
                     _fmt(gv('L_col_plus', case), '{:.1f}')))
            if 'loglaw' in _ob:
                _k, _z0, _d = _ob['loglaw']
                print('  loglaw : kappa=%s  z0m+=%s  d+=%s'
                      % (_fmt(_k), _fmt(_z0, '{:.5f}'), _fmt(_d, '{:.2f}')))
            # Chapter-6 observations
            if 'My_windward_suppression' in _ob:
                _s = _ob['My_windward_suppression']
                print('  D4  M_y windward suppression=%s  (tau_zx ref 0.59 -> %s)'
                      % (_fmt(_s, '{:+.3f}'),
                         'mechanism holds' if abs(_s) < 0.295 else 'comparable to tau_zx'))
            if 'psi_min' in _ob:
                _pm, _px, _pz = _ob['psi_min']
                print('  D2  psi_min=%s at (x+=%s, z+=%s)'
                      % (_fmt(_pm, '{:.3e}'), _fmt(_px, '{:.0f}'), _fmt(_pz, '{:.0f}')))
            if 'psi_disp_ratio' in _ob:
                print('  D3  psi_disp/psi_mean amplitude ratio=%s' % _fmt(_ob['psi_disp_ratio']))
            if 'veer_surf_range' in _ob:
                _v0, _v1 = _ob['veer_surf_range']
                print('  D1  surface veer range=[%s, %s] deg' % (_fmt(_v0, '{:.1f}'), _fmt(_v1, '{:.1f}')))
            if 'W_lee_wind_ratio' in _ob:
                print('  D10 lee/windward |W| peak ratio=%s' % _fmt(_ob['W_lee_wind_ratio']))
            if 'dUplus_max' in _ob:
                _du, _dz = _ob['dUplus_max']
                print('  D14 max dU+ (z+>100)=%s at z+=%s' % (_fmt(_du, '{:+.3f}'), _fmt(_dz, '{:.0f}')))
            if 'TKEprod_peak_z' in _ob:
                print('  D12 TKE-production peak height z+=%s' % _fmt(_ob['TKEprod_peak_z'], '{:.1f}'))
            if 'TKE_dominant_component' in _ob:
                print('  D15 dominant normal stress=%s' % _ob['TKE_dominant_component'])
            if 'Dform_wind_lee' in _ob:
                _w, _le = _ob['Dform_wind_lee']
                print('  D5  form drag windward=%s lee=%s net=%s'
                      % (_fmt(_w, '{:.3e}'), _fmt(_le, '{:.3e}'), _fmt(_w - _le, '{:.3e}')))
            if 'cp_extrema' in _ob:
                _cmin, _cmax = _ob['cp_extrema']
                print('  D19 Cp_min=%s Cp_max=%s' % (_fmt(_cmin, '{:.4f}'), _fmt(_cmax, '{:.4f}')))
            if 'pressure_equil' in _ob:
                _ib, _ot = _ob['pressure_equil']
                print('  D6  |dP/dz| IBM=%s outer=%s' % (_fmt(_ib, '{:.3e}'), _fmt(_ot, '{:.3e}')))
            if 'poisson_fracs' in _ob:
                _fm, _fr, _fd = _ob['poisson_fracs']
                print('  D18 Poisson source share mean/rey/disp=%s/%s/%s'
                      % (_fmt(_fm, '{:.2f}'), _fmt(_fr, '{:.2f}'), _fmt(_fd, '{:.2f}')))
        # Honestly gated / data-blocked items
        print('\n--- gated / data-blocked (need data not on hand) ---')
        for _ln in (
                'plan-view vorticity at z+=5,15,30   : needs wall-parallel planes (only first x-y plane downloaded)',
                '3-D streamline topology             : phase-avg is spanwise-mean (D2 gives the 2-D projection)',
                'sampling convergence / inertial osc.: needs time series (only final-time averages + 1 snapshot)',
                'grid-sensitivity, Ch.3/4 tables     : need extra runs / not flow post-processing',
                'stratified flat-wall reference (G2) : all current .nc are neutral (ri00.00)',
                'Re_D=750 robustness (G8)            : 750 statistics not yet available'):
            print('  N/A  ' + _ln)
        print('\n--- tally ---')
        print('  cases with pickle (diagnosed): %s' % (_plotted or 'none'))
        print('  cases skipped (no pickle)    : %s' % (_skipped or 'none'))
        print('  smooth reference loaded      : %s' % _smooth_loaded)
        print('#' * 78)

    _print_run_summary()
