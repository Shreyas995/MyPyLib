#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:13:16 2024

@author: shreyas deshpande
"""

import os
import re
import sys
import csv
import struct
import math
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
            # else:
            #     print('Undefined case')
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
                
            else:
                print('Undefined case', 'i:',i,'j:',j)
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

def createIntegrate(eps_horizontal, n, i_id, variable, x, side):
    if side == 'LHS':
        indj = np.where(eps_horizontal[0,:int(n)] == i_id)[0]
    else:
        indj = np.where(eps_horizontal[0,int(n):] == i_id)[0]
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

limity = 700
hill_height = 94
step = 2
Re = 500
Re_lambda = 0.5*Re*Re
nu = 1/Re_lambda
dt = 0.827E-04
index = 1
limity_range = 150
limity = 453
f = 1
alpha = -0.430511
Gx = np.cos(alpha)
Gz = -np.sin(alpha)
u_star = 0.0659
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
plotRes = 0
animate = 0

###############################################################################
############################# Main Code #######################################

# Parameter decleration
cwd = str(os.path.dirname(__file__) + '/' )

# Read grid
x, y, z = read_grid(cwd)
    
try:
    eps = np.load(cwd + 'eps_save.npy')
    print('eps loaded')
except:
    print('Needed to read eps field')
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

eps_hill = np.sum(eps, axis=0).astype(int)
eps_vol = epsVolume(eps,ny,nx,hill_height)
eps_s = np.mean(eps_vol,axis=1)
eps_f = 1 - eps_s

# Forcing values in solid zero. If not it will introduce error when calculating average in x direction.
mask_zero = 1 - eps

# Calculating phase average by summing last planes
if (1 == cal_Avg):
    AvgPh = np.zeros((ny,nx,dim))
    AvgStress = np.zeros((ny,nx,6))
    SpaceAvgStr = np.zeros((ny,6))
    VelGbl = np.zeros((ny,dim))
    VelGbl2D = np.zeros((ny,nx,dim))
    Turb = np.zeros((ny,nx,6))
    DispVel = np.zeros((ny,nx,dim))
    turb1D = np.zeros((ny,6))
    ugud = np.zeros((ny,nx))
    udug = np.zeros((ny,nx))
    udvg = np.zeros((ny,nx))
    ugvd = np.zeros((ny,nx))
    udwg = np.zeros((ny,nx))
    ugwd = np.zeros((ny,nx))
    vgvd = np.zeros((ny,nx))
    vdwg = np.zeros((ny,nx))
    vgwd = np.zeros((ny,nx))
    wdwg = np.zeros((ny,nx))
    
    for i in range(30):
        files = 0
        FilePath = []
        base = 234500
        srt = base + 1 + restart * i 
        end = base + restart * (i + 1)
        for i in range (9):
            if (i <= 2):
                path = cwd + 'avg_flow' + str(srt) + '_' + str(end) + '.' + str(i+1)
            else:
                path = cwd + 'avg_stress' + str(srt) + '_' + str(end) + '.' + str(i-2)
            if (os.path.exists(path)):
                FilePath.append([path])
                files += 1
        if (files == 9):
            counter += 1
            for i in range (9):
                hdr, _, _, _, _, _ = read_header(FilePath[i][0])
                if (i <= 2):
                    AvgPh[:, :, i] += readplane(FilePath[i][0], nx, ny, restart + 1, hdr)
                else:
                    AvgStress[:, :, i-3] += readplane((FilePath[i][0]), nx, ny, restart + 1, hdr)
    
    for i in range (9):
        if (i <= 2):
            AvgPh[:,:,i] = AvgPh[:,:,i]/counter
            VelGbl[:,i] = np.mean((AvgPh[:,:,i]), axis = 1)
            DispVel[:,:,i] = (AvgPh[:,:,i] - VelGbl[:,i][:,np.newaxis])*mask_zero
        else:
            AvgStress[:, :, i-dim] = AvgStress[:, :, i-dim]/counter
            SpaceAvgStr[:,i-dim] = np.mean((AvgStress[:,:,i-dim]), axis = 1)
         
            # plot2D_div(x, y[:limity], DispVel[:limity,:,0], 'DispVel_u', 'DispVel', cwd + '/fig/' + 'DispVel_u' + '.png', x_fill, y_fill)
    
    for i in range (dim):
        VelGbl2D[:,:,i] = (np.tile(VelGbl[:,i].reshape(ny,1), nx).reshape(ny,nx))*mask_zero
    
    
    for i in range(6):
        turb1D[:,i] = np.mean(Turb[:,:,i], axis=1)

    uu_t = DispVel[:,:,0]*DispVel[:,:,0]
    uv_t = DispVel[:,:,0]*DispVel[:,:,1]
    uw_t = DispVel[:,:,0]*DispVel[:,:,2]
    vv_t = DispVel[:,:,1]*DispVel[:,:,1]
    vw_t = DispVel[:,:,1]*DispVel[:,:,2]
    ww_t = DispVel[:,:,2]*DispVel[:,:,2]
    
    space_uu_t = (np.mean(uu_t, axis =1))/eps_f
    space_uv_t = (np.mean(uv_t, axis =1))/eps_f
    space_uw_t = (np.mean(uw_t, axis =1))/eps_f
    space_vv_t = (np.mean(vv_t, axis =1))/eps_f
    space_vw_t = (np.mean(vw_t, axis =1))/eps_f
    space_ww_t = (np.mean(ww_t, axis =1))/eps_f
    
    uu_g = VelGbl2D[:,:,0]*VelGbl2D[:,:,0]
    uv_g = VelGbl2D[:,:,0]*VelGbl2D[:,:,1]
    uw_g = VelGbl2D[:,:,0]*VelGbl2D[:,:,2]
    vv_g = VelGbl2D[:,:,1]*VelGbl2D[:,:,1]
    vw_g = VelGbl2D[:,:,1]*VelGbl2D[:,:,2]
    ww_g = VelGbl2D[:,:,2]*VelGbl2D[:,:,2]

# uu_d = SpaceAvgStr[:,0] - uu_g - space_uu_t
# uv_d = SpaceAvgStr[:,1] - uv_g - space_uv_t
# uw_d = SpaceAvgStr[:,2] - uw_g - space_uw_t
# vv_d = SpaceAvgStr[:,3] - vv_g - space_vv_t
# vw_d = SpaceAvgStr[:,4] - vw_g - space_vw_t
# ww_d = SpaceAvgStr[:,5] - ww_g - space_ww_t

if(1 == verify_TimeAvg):
    for i in range(30):
        files = 0
        FilePath = []
        base = 234500
        srt = base + 1 + restart * i 
        end = base + restart * (i + 1)
        pathi = cwd + 'avg_flow' + str(srt) + '_' + str(end) + '.' + str(1)
        pathj = cwd + 'avg_flow' + str(srt) + '_' + str(end) + '.' + str(2)
        pathk = cwd + 'avg_flow' + str(srt) + '_' + str(end) + '.' + str(3)
    
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

if (100 == save_avg):
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

if (1 == load_arrays):
    # declares arrays to load 
    du_dt = np.zeros((ny,nx,dim))
    ds_dt = np.zeros((ny,nx,scal))
    
    Rey_UU = np.load('uu_d.npy')
    Rey_UV = np.load('uv_d.npy')
    Rey_UW = np.load('uw_d.npy')
    Rey_VV = np.load('vv_d.npy')
    Rey_VW = np.load('vw_d.npy')
    Rey_WW = np.load('ww_d.npy')
    
    AvgStrUU = np.load('AvgStrUU.npy')
    AvgStrUV = np.load('AvgStrUV.npy')
    AvgStrUW = np.load('AvgStrUW.npy')
    AvgStrVV = np.load('AvgStrVV.npy')
    AvgStrVW = np.load('AvgStrVW.npy')
    AvgStrWW = np.load('AvgStrWW.npy')
    
    UU_G = np.load('uu_g.npy')
    UV_G = np.load('uv_g.npy')
    UW_G = np.load('uw_g.npy')
    VV_G = np.load('vv_g.npy')
    VW_G = np.load('vw_g.npy')
    WW_G = np.load('ww_g.npy')
    
    UU_disp = np.load('uu_t.npy')
    UU_disp = np.load('uv_t.npy')
    UU_disp = np.load('uw_t.npy')
    UU_disp = np.load('vv_t.npy')
    VW_disp = np.load('vw_t.npy')
    WW_disp = np.load('ww_t.npy')
        
    AvgPhU = np.load('AvgPhU.npy')
    AvgPhV = np.load('AvgPhV.npy')
    AvgPhW = np.load('AvgPhW.npy')
    
    VelGblU = np.load('VelGblU.npy')
    VelGblV = np.load('VelGblV.npy')
    VelGblW = np.load('VelGblW.npy')
    
    DispVelU = np.load('DispVelU.npy')
    DispVelV = np.load('DispVelV.npy')
    DispVelW = np.load('DispVelW.npy')
    
    udug = np.load('udug.npy')
    udvg = np.load('udvg.npy')
    udwg = np.load('udwg.npy')
    vdvg = np.load('vdvg.npy')
    vdwg = np.load('vdwg.npy')
    wdwg = np.load('wdwg.npy')
    
    ugud = np.load('ugud.npy')
    ugvd = np.load('ugvd.npy')
    ugwd = np.load('ugwd.npy')
    vgvd = np.load('vgvd.npy')
    vgwd = np.load('vgwd.npy')
    wgwd = np.load('wgwd.npy')
    
    # dq_dt 
    du_dt[:,:,0] = np.load('du_dt1.npy')
    du_dt[:,:,1] = np.load('du_dt2.npy')
    du_dt[:,:,2] = np.load('du_dt3.npy')
    ds_dt[:,:,0] = np.load('ds_dt.npy')
    
    s1 = nc.Dataset(cwd + 'Re500/'+'ri00.00_re0500_2048x0192x2048_20110615_avg_all.nc', 'r')
    r1 = nc.Dataset(cwd + 'Re1000/'+'ri00.00_re1000_3072x0512x6144_20130520_avg_all.nc', 'r')
    r2 = nc.Dataset(cwd + 'RoughRe1000/'+'ri00.00_re1000_3072x0656x3072_20230119_s_avg_all.nc', 'r')
    sy = (s1.variables['y'][:])
    nys = np.size(sy)
    
    su = np.reshape(np.mean((s1.variables['rU'][:]).T, axis=1), (nys,1))
    sv = np.reshape(np.mean((s1.variables['rV'][:]).T, axis=1), (nys,1))
    sw = np.reshape(np.mean((s1.variables['rW'][:]).T, axis=1), (nys,1))
    alpha_s = (np.flip(np.mean(sw,axis=1)))/(np.mean(su,axis=1))
    ustr_s1 = 0.0618
    
    y_s=s1.variables['y'][:]

    y_s_p = (y_s*ustr_s1)/nu
    U_s = (s1.variables['fU'][:]).T
    V_s = (s1.variables['fV'][:]).T
    W_s = (s1.variables['fW'][:]).T
    rU_s = (s1.variables['rU'][:]).T
    rV_s = (s1.variables['rV'][:]).T
    rW_s = (s1.variables['rW'][:]).T
    G_x_s = np.max (U_s)
    G_z_s = np.max (W_s)
    G_s = np.sqrt(G_x_s**2 + G_z_s**2)
    U_s_p=(U_s/ustr_s1)/G_s
    V_s_p=(V_s/ustr_s1)/G_s
    W_s_p=(W_s/ustr_s1)/G_s
    
    GblU_s = np.mean(rU_s, axis=1)
    GblV_s = np.mean(rV_s, axis=1)
    GblW_s = np.mean(rW_s, axis=1)
    
    Rxx_s = (s1.variables['Rxx'][:]).T
    Rxy_s = (s1.variables['Rxy'][:]).T
    Ryy_s = (s1.variables['Ryy'][:]).T
    Ryz_s = (s1.variables['Ryz'][:]).T
    Rzz_s = (s1.variables['Rzz'][:]).T
    TKE_s = 0.5 *np.sqrt(Rxx_s**2 + Ryy_s**2 + Rzz_s**2)
    
    cor_yx_s = -(W_s - G_z_s)
    I_corr_yx_s = vIntegral(np.mean(cor_yx_s, axis=1), y_s.size, y_s)
    du_dy_s = diffu_dy((np.reshape(GblU_s,(y_s.size,1))), y_s.size, 1, np.zeros((y_s.size,1)), y_s)
    visc_yx_s = (1/Re_lambda) * du_dy_s
    tau_yx_s = I_corr_yx_s + np.mean(visc_yx_s, axis=1) - np.mean(Rxy_s, axis=1)
    
    cor_yz_s = (U_s - G_x_s)
    I_corr_yz_s = vIntegral(np.mean(cor_yz_s, axis=1), y_s.size, y_s)
    dw_dy_s = diffu_dy((np.reshape(GblW_s,(y_s.size,1))), y_s.size, 1, np.zeros((y_s.size,1)), y_s)
    visc_yz_s = (1/Re_lambda) * dw_dy_s
    tau_yz_s = -I_corr_yz_s + np.mean(visc_yz_s, axis=1) - np.mean(Ryz_s, axis=1)
    
# Postprocess
if (1 == postprocess):
    # Momentum Balance to find u*
    # Time derivative is zero
    # $f \int_0^y \epsilon_{1 2 3}\left(\langle\bar{v}\rangle_k-g_v\right) \mathrm{d} y + \frac{1}{\operatorname{Re} e_{\Lambda}} \frac{\partial\langle\bar{u}\rangle}{\partial y}-\left\langle\overline{u^{\prime} w^{\prime}}\right\rangle $
    # Turining angle is 23.29 degrees
    corr_yx = -(AvgPhW - Gz)*mask_zero
    I_corr_yx = vIntegral(np.mean(corr_yx, axis=1), ny, y)
    delu_dely = diffu_dy((np.reshape(VelGblU,(ny,1))), ny, 1, np.zeros((ny,1)), y) # in the grid file ny is vertical number of points and y is the vertical grid
    visc_yx = (1/Re_lambda) * delu_dely
    stress_yx = Rey_UV
    tau_yx = I_corr_yx + np.mean(visc_yx, axis=1) - np.mean(stress_yx, axis=1)
    
    # $f \int_0^z \epsilon_{2 1 3}\left(\langle\bar{u}\rangle_k-g_u\right) \mathrm{d} z + \frac{1}{\operatorname{Re} e_{\Lambda}} \frac{\partial\langle\bar{v}\rangle}{\partial z}-\left\langle\overline{v^{\prime} w^{\prime}}\right\rangle $
    corr_yz = (AvgPhU - Gx)*mask_zero
    I_corr_yz = vIntegral(np.mean(corr_yz, axis=1), ny, y)
    delw_dely = diffu_dy((np.reshape(VelGblW,(ny,1))), ny, 1, np.zeros((ny,1)), y) # in the grid file 'ny' is vertical number of points and y is the vertical grid
    visc_yz = (1/Re_lambda) * delw_dely
    stress_yz = Rey_VW
    tau_yz = -I_corr_yz + np.mean(visc_yz, axis=1) - np.mean(stress_yz, axis=1)
    
    delv_delx = diffu_dx((np.reshape(AvgPhV,(ny,nx))), ny, nx, eps, x)
    tau_corrctn = ((1/Re_lambda) * np.mean(delv_delx, axis=1))
    
    u_star2 = ((tau_yx**2 + tau_yz**2 + tau_corrctn**2)**0.5)**0.5
    
    y_inner =  y*(u_star/nu)
    y_outer = y/u_star
    
    # Re 500 Smooth casae
    s1corr_yx = -(sw - sw[-1])
    s1I_corr_yx = vIntegral(s1corr_yx[:,0], nys, sy)
    du_dy_s = (1/Re_lambda)*(diffu_dy(su, nys, 1, np.zeros((nys,1)), sy))
    
    
    
    # Turbulent Kinetic Energy
    TKE = 0.5*(Rey_UU + Rey_VV + Rey_WW)
    
    
    # Turning angle
    inst_alpha = (np.mean(AvgPhW,axis=1)/np.mean(AvgPhU,axis=1))
    
    dudt = np.mean(du_dt[:,:,0], axis=1)
    dwdt = np.mean(du_dt[:,:,2], axis=1)
    
    
    # Horizontal wind
    u_plus = AvgPhU/u_star
    w_plus = AvgPhW/u_star
    v_plus = AvgPhV/u_star
    
    alphacos = np.cos(alpha)
    alphasin = np.sin(alpha)
    u_plus_rot = np.mean(AvgPhU,axis=1)*alphacos - np.mean(AvgPhW,axis=1)*alphasin
    w_plus_rot = -(np.mean(AvgPhU,axis=1)*alphasin + np.mean(AvgPhW,axis=1)*alphacos)
    
    uh_plus = np.sqrt(u_plus**2 + w_plus**2)
    uh_pl1D = np.mean(uh_plus, axis=1)
        
    # Compute Friction veocity method 1
    ### calcualte horizontal surfaces
    eps_horizontal = np.zeros((2,nx))
    for i in range (nx):
        for j in range (ny):
            if eps[j,i] == 1 and eps[j+1,i] == 0:
                eps_horizontal[0,i] = j
                eps_horizontal[1,i] = i
            elif eps[j,i] == 0 and j == 0:
                eps_horizontal[0,i] = j
                eps_horizontal[1,i] = i
            else:
                print('Not on the solid fluid interface')
                
    eps_vertical1 = np.zeros((2,hill_height))
    eps_vertical2 = np.zeros((2,hill_height))
    for j in range (hill_height):
        for i in range(nx-1):
            if eps[j,i] == 1 and eps[j,i+1] == 0:
                eps_vertical1[0,j] = j
                eps_vertical1[1,j] = i
            elif eps[j,i] == 0 and eps[j,i+1] == 1:
                eps_vertical2[0,j] = j
                eps_vertical2[1,j] = i
            else:
                print('Not on the solid fluid interface')
    du_dy = diffu_dy(AvgPhU, ny, nx, eps, y) 
    du_dx = diffu_dx(AvgPhU, ny, nx, eps, x)

    print('Computed dv/dy')
    dv_dy = diffu_dy(AvgPhV, ny, nx, eps, y) 
    dv_dx = diffu_dx(AvgPhV, ny, nx, eps, x)

    print('Computed dw/dy')
    dw_dy = diffu_dy(AvgPhW, ny, nx, eps, y)
    dw_dx = diffu_dx(AvgPhW, ny, nx, eps, x)
    
    tau_ux = np.zeros((2,hill_height))
    tau_uz = np.zeros((2,hill_height))
    cordsy = np.zeros((2,hill_height))

    tau_vx = np.zeros((1,nx))
    tau_vz = np.zeros((1,nx))
    

    for i in range (nx):
        tau_vx[0,i] = du_dy[int(eps_horizontal[0,i]), int(eps_horizontal[1,i])]
        tau_vz[0,i] = dw_dy[int(eps_horizontal[0,i]), int(eps_horizontal[1,i])]
        
    for j in range(hill_height):
        tau_ux[0,j] = du_dx[int(eps_vertical1[0,j]), int(eps_vertical1[1,j])]
        tau_uz[0,j] = dw_dx[int(eps_vertical1[0,j]), int(eps_vertical1[1,j])]
        tau_ux[1,j] = du_dx[int(eps_vertical2[0,j]), int(eps_vertical2[1,j])]
        tau_uz[1,j] = dw_dx[int(eps_vertical2[0,j]), int(eps_vertical2[1,j])]
        cordsy[0,j] = y[int(eps_vertical1[0,j])]
        cordsy[1,j] = y[int(eps_vertical2[0,j])]

    I_tau_vx = 0 #np.zeros((hill_height))
    I_tau_vz = 0 #np.zeros((hill_height))
    I_tau_ux = 0
    I_tau_uz = 0

    for j in range (hill_height):
        if j == 0:
            Ix = createIntegrate(eps_horizontal, nx, j, tau_vx, x, 'LHS')
            I_tau_vx = I_tau_vx + Ix
            Iz = createIntegrate(eps_horizontal, nx, j, tau_vz, x, 'LHS')
            I_tau_vz = I_tau_vz + Iz
        elif (j%step != 0):
            Ix = createIntegrate(eps_horizontal, nx/2, j, tau_vx, x, 'LHS')
            I_tau_vx = I_tau_vx + Ix
            Ix = createIntegrate(eps_horizontal, nx/2, j, tau_vx, x, 'RHS')
            I_tau_vx = I_tau_vx + Ix
            Iz = createIntegrate(eps_horizontal, nx/2, j, tau_vz, x, 'LHS')
            I_tau_vz = I_tau_vz + Iz
            Iz = createIntegrate(eps_horizontal, nx/2, j, tau_vz, x, 'RHS')
            I_tau_vz = I_tau_vz + Iz
            
    for j in range (hill_height-1):
        I_tau_ux = I_tau_ux + trapezoid(tau_ux[0,j:j+step]) + trapezoid(tau_ux[1,j:j+step])
        I_tau_uz = I_tau_uz + trapezoid(tau_uz[0,j:j+step]) + trapezoid(tau_uz[1,j:j+step])

    tau_horizontal = np.sqrt((nu*I_tau_vx)**2 + (nu*I_tau_vz)**2 )
    tau_vertical = np.sqrt((nu*0)**2 + (nu*I_tau_uz)**2 )
    tau_total = np.sqrt(tau_horizontal**2 + tau_vertical**2 )
    u_star1 = np.sqrt(tau_total)
    
    # Vorticity 
    dv_dz = diffu_dx((np.reshape(AvgPhV,(ny,nx))), ny, nx, eps, z)
    du_dz = diffu_dx((np.reshape(AvgPhU,(ny,nx))), ny, nx, eps, z)
    omega_x = dw_dy - dv_dz
    omega_y = du_dz - dw_dx
    omega_z = dv_dx - du_dy
    
    # Dispersive component Vorticity to isolate the rotatiaon and gravityy waves
    dvd_dx = diffu_dx(DispVelV, ny, nx, eps, x)
    dud_dy = diffu_dy(DispVelU, ny, nx, eps, y)
    
    tmp = np.gradient(DispVelU, y, axis=0)
    disp_vortz = (dvd_dx - dud_dy)*(1-eps)
    res_dispz = np.sqrt(DispVelV**2+DispVelU**2)
    
    dwd_dy = diffu_dx(DispVelW, ny, nx, eps, y)
    dvd_dz = dvd_dx
    disp_vortx = (dwd_dy - dvd_dz)*(1-eps)
    res_dispx = np.sqrt(DispVelW**2+DispVelV**2)
    
    dud_dz = diffu_dx(DispVelU, ny, nx, eps, z)
    dwd_dx = diffu_dx(DispVelW, ny, nx, eps, x)
    disp_vorty = (dud_dz - dwd_dx)*(1-eps)
    res_dispy = np.sqrt(DispVelU**2+DispVelW**2)
    res_phavg_uv = np.sqrt(AvgPhU**2 + AvgPhV**2)
    
    # Monin Obukhov log layer
    u_most = (1/kappa)*np.log(y_s_p) + 4.5 # Smooth case
    u_most[0]=0
    
    d = 0.01*u_star; y0 =5*u_star
    u_most_v = (1/kappa)*np.log(((y-d)/(y0))/l_in) + 1
    u_most_v[0]=0
    
    # Tr_u = np.mean(AvgPhU, axis=1)*np.cos(-30*180/np.pi)+np.mean(AvgPhW, axis=1)*np.sin(-30*180/np.pi)
    # Tr_w = np.mean(AvgPhU, axis=1)*np.sin(-30*180/np.pi)-np.mean(AvgPhW, axis=1)*np.cos(-30*180/np.pi)

    # TKE Vertical Profile
    TKE_V = np.sum((TKE), axis=0)
    eps_vert = np.sum((1-eps_vol), axis=0)
    AVG_TKE_V = TKE_V/eps_vert
    
    AVG_TKE_V_s = np.mean(TKE_s,axis=0)
    x_s = np.linspace(0, 1, 250)
    AVG_TKE_V_s_i = np.interp(x, x_s, AVG_TKE_V_s)
    
if (1 == plotRes):
    # Plot derivatives
    # plot2D_div(x, y[:150], delv_delx[:150,:],'', 'dv_dx',r'$x^{+}$',r'$z^{+}$' , cwd + '/fig/' + 'dv_dx' + '.png', x_fill, y_fill ,1000)
    plot2D_div(x, y[:150], delv_delx[:150,:],'', 'dv_dx',r'$x^{+}$',r'$z^{+}$' , cwd + '/fig/' + 'dv_dx' + '.png', x_fill, y_fill ,1000)
    
    # Phase Average
    plot2D_div(x, y[:limity], AvgPhU[:limity,:],'','Phase Avg U',r'$x$',r'$z$', cwd + '/fig/' + 'PhAvgU' + '.png', x_fill, y_fill, 1000) #, contour=True)
    plot2D_div(x, y[:limity], AvgPhV[:limity,:],'','Phase Avg W',r'$x$',r'$z$', cwd + '/fig/' + 'PhAvgW' + '.png', x_fill, y_fill, 1000)
    plot2D_div(x, y[:limity], AvgPhW[:limity,:],'','Phase Avg V',r'$x$',r'$z$', cwd + '/fig/' + 'PhAvgV' + '.png', x_fill, y_fill, 1000)
    
    # Streamlines
    # plot2D_cont_log(x, y[:limity], (disp_vortz[:limity,:]),'','VorticityY',r'$x$',r'$z$', cwd + '/fig/' + 'VorticityY' + '.png', x_fill, y_fill, 1000)
    plot2D_streamlines_vorticity(x_plus, y_plus[:250], DispVelU[:250,:], DispVelV[:250,:],res_dispz[:250,:],eps[:250,:],'','',r'$x$',r'$z$', cwd + '/fig/' + 'Streamlinexy' + '.png', x_fill, y_fill,1000)
    plot2D_streamlines_vorticity(x_plus, y_plus[:250], DispVelU[:250,:], DispVelV[:250,:],disp_vortz[:250,:],eps[:250,:],'','',r'$x$',r'$z$', cwd + '/fig/' + 'Streamlinexy' + '.png', x_fill_plus, y_fill_plus,1000)
    
    plot2D_streamlines_vorticityX(x, y[:limity], DispVelV[:limity,:], DispVelW[:limity,:],disp_vortx[:limity,:],'','',r'$x$',r'$z$', cwd + '/fig/' + 'Streamlineyz' + '.png', x_fill, y_fill,1000)
    plot2D_streamlines_vorticityX(x, y[:limity], DispVelU[:limity,:], DispVelW[:limity,:],disp_vorty[:limity,:],'','',r'$x$',r'$z$', cwd + '/fig/' + 'Streamlinezx' + '.png', x_fill, y_fill,1000)
    
    # Streamlines of the phase average
    plot2D_streamlines_vorticity(x_plus, y_plus[:limity], AvgPhU[:limity,:], AvgPhV[:limity,:], res_phavg_uv[:limity,:], eps[:limity,:],'','',r'$x$',r'$z$', cwd + '/fig/' + 'Streamlinexy' + '.png', x_fill_plus, y_fill_plus,1000)
    plot2D_div(x, y[:limity], res_phavg_uv[:limity,:],'', 'ResPhXY',r'$x^{+}$',r'$z^{+}$' , cwd + '/fig/' + 'ResPhXY' + '.png', x_fill, y_fill ,1000)
    
    # orographic wave drag
    plot2D_div(x, y, AvgPhU,'','Phase Avg U',r'$x$',r'$z$', cwd + '/fig/' + 'PhAvgU_f' + '.png', x_fill, y_fill, 20)
    plot2D_div(x, y, AvgPhV,'','Phase Avg W',r'$x$',r'$z$', cwd + '/fig/' + 'PhAvgW_f' + '.png', x_fill, y_fill, 20)
    plot2D_div(x, y, AvgPhW,'','Phase Avg V',r'$x$',r'$z$', cwd + '/fig/' + 'PhAvgV_f' + '.png', x_fill, y_fill, 20)
    
    # Dispersive Velocity Component
    plot2D_div(x_plus, y_plus[:limity], DispVelU[:limity,:],'','Streamwise Dispersive Velocity', r'$x^+$',r'$z^+$', cwd + '/fig/' + 'DispU' + '.png', x_fill_plus, y_fill_plus, 1000)
    plot2D_div(x_plus, y_plus[:limity], DispVelV[:limity,:],'','Normal Dispersive Velocity', r'$x^+$',r'$z^+$', cwd + '/fig/' + 'DispW' + '.png', x_fill_plus, y_fill_plus, 1000)
    plot2D_div(x_plus, y_plus[:limity], DispVelW[:limity,:],'','Spanwise Dispersive Velocity', r'$x^+$',r'$z^+$', cwd + '/fig/' + 'DispV' + '.png', x_fill_plus, y_fill_plus, 1000)
    
    # TKE
    plot2D_div(x_plus, y_plus[:limity], TKE[:limity,:], '', 'TKE', r'$x^+$',r'$z^+$', cwd + '/fig/' + 'TKE' + '.png', x_fill_plus, y_fill_plus, 1000)
    
    plot2D_div(x, y[:limity], Rey_UU[:limity,:], '', 'Reynolds stress (Ruu)', r'$x$',r'$z$', cwd + '/fig/' + 'Ruu' + '.png', x_fill, y_fill, 1000)
    plot2D_div(x, y[:limity], Rey_UV[:limity,:], '', 'Reynolds stress (Ruw)', r'$x$',r'$z$', cwd + '/fig/' + 'Ruw' + '.png', x_fill, y_fill, 1000)
    plot2D_div(x, y[:limity], Rey_UW[:limity,:], '', 'Reynolds stress (Ruv)', r'$x$',r'$z$', cwd + '/fig/' + 'Ruv' + '.png', x_fill, y_fill, 1000)
    plot2D_div(x, y[:limity], Rey_VV[:limity,:], '', 'Reynolds stress (Rww)', r'$x$',r'$z$', cwd + '/fig/' + 'Rww' + '.png', x_fill, y_fill, 1000)
    plot2D_div(x, y[:limity], Rey_VW[:limity,:], '', 'Reynolds stress (Rwv)', r'$x$',r'$z$', cwd + '/fig/' + 'Rwv' + '.png', x_fill, y_fill, 1000)
    plot2D_div(x, y[:limity], Rey_WW[:limity,:], '', 'Reynolds stress (Rvv)', r'$x$',r'$z$', cwd + '/fig/' + 'Rvv' + '.png', x_fill, y_fill, 1000)
    
    # Vorticity
    plot2D_div(x, y[:limity], omega_x[:limity,:], '', 'Vorticity X', r'$x$',r'$z$', cwd + '/fig/' + 'VorticityX' + '.png', x_fill, y_fill, 50)
    plot2D_div(x, y[:300], omega_y[:300,:], '', 'Vorticity Z', r'$x$',r'$z$', cwd + '/fig/' + 'VorticityZ' + '.png', x_fill, y_fill, 50)
    plot2D_div(x, y[:200], omega_z[:200,:], '', 'Vorticity Y', r'$x$',r'$z$', cwd + '/fig/' + 'VorticityY' + '.png', x_fill, y_fill, 50)
    plot2D_streamlines_vorticityX(x, y[:limity], AvgPhU[:limity,:], AvgPhV[:limity,:],omega_y[:limity,:],'','',r'$x$',r'$z$', cwd + '/fig/' + 'Streamlinezx' + '.png', x_fill, y_fill,1000)
    
    # Hodograph
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(u_plus_rot, w_plus_rot, label='valley', color='blue', linestyle='-')
    plt.plot(np.mean(rU_s,axis=1)/G_s, -np.mean(rW_s,axis=1)/G_s, label='smooth', color='red', linestyle='-')
    plt.title('Hodograph')
    plt.ylabel(r'$\langle \bar{v} \rangle^{-} $')
    plt.xlabel(r'$\langle \bar{u} \rangle^{-} $')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Hodograph Comparion
    plt.plot(y_s_p[1:], -alpha_s[1:]*(180/np.pi), label='smooth case', color='blue', linestyle='-')

    # Turning angle
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(y_inner[1:], (inst_alpha[1:]*(180/np.pi)), label=r'$\alpha (rad)$', color='blue', linestyle='-')
    plt.title('Rotation angle')
    plt.ylabel(r'$\alpha (\degree)$')
    plt.xlabel(r'$z^{+}$')
    plt.xscale("log")
    plt.grid(True)
    plt.show()
    
    # Momentum balance XY
    plt.figure(figsize=(10, 6))
    plt.plot(y_inner[:], I_corr_yx[:], label='coriolis', color='blue', linestyle='-')
    plt.plot(y_inner[:], (np.mean(visc_yx, axis=1))[:], label='viscous', color='red', linestyle='-')
    plt.plot(y_inner[:], -(np.mean(stress_yx, axis=1))[:], label='Rey Stress', color='orange', linestyle='-')
    plt.plot(y_inner[:], dudt, label='Temporal', color='saddlebrown', linestyle='-')
    plt.plot(y_inner[:], tau_yx[:], label='Total', color='black', linestyle='-')
    plt.title(r'Shear stress $\tau_{zx}$')
    plt.xlabel(r'$z^{+}$')
    plt.ylabel(r'${{\langle \bar{\tau} \rangle}^+}_{zx}$')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Zoomed plot
    plt.figure(figsize=(8, 6), dpi=300)
    
    # Plot for Valley case (solid lines)
    plt.plot(y_inner[:limity], I_corr_yx[:limity]/u_star**2, color='blue', linestyle='-', label='Coriolis')
    plt.plot(y_inner[:limity], (np.mean(visc_yx, axis=1))[:limity]/u_star**2, color='red', linestyle='-', label='Viscous')
    plt.plot(y_inner[:limity], -(np.mean(stress_yx, axis=1))[:limity]/u_star**2, color='orange', linestyle='-', label='Rey Stress')
    plt.plot(y_inner[:limity], dudt[:limity]/u_star**2, color='saddlebrown', linestyle='-', label='Temporal')
    
    # Plot for Smooth case (dashed lines)
    plt.plot(y_s_p[:160], I_corr_yx_s[:160]/0.0618**2, color='blue', linestyle='--')
    plt.plot(y_s_p[:160], (np.mean(visc_yx_s, axis=1))[:160]/0.0618**2, color='red', linestyle='--')
    plt.plot(y_s_p[:160], -(np.mean(Rxy_s, axis=1))[:160]/0.0618**2, color='orange', linestyle='--')
    plt.plot(y_s_p[:160], np.zeros((160)), color='saddlebrown', linestyle='--')
    
    # Custom Legend Handles
    import matplotlib.lines as mlines
    valley_legend = mlines.Line2D([], [], color='black', linestyle='-', label='Valley')
    smooth_legend = mlines.Line2D([], [], color='black', linestyle='--', label='Smooth')
    
    plt.title(r'Shear stress $\tau_{zx}$')
    plt.xlabel(r'$z^{+}$')
    plt.ylabel(r'${{\langle \bar{\tau} \rangle}^+}_{zx}$')
    plt.legend(handles=[  
        mlines.Line2D([], [], color='blue', linestyle='-', label='Coriolis'),  
        mlines.Line2D([], [], color='red', linestyle='-', label='Viscous'),  
        mlines.Line2D([], [], color='orange', linestyle='-', label='Rey Stress'),  
        mlines.Line2D([], [], color='saddlebrown', linestyle='-', label='Temporal'),  
        valley_legend, smooth_legend  
    ])
    plt.grid(True)
    plt.xlim(0, 200)
    plt.ylim(-0.1,1.0)
    plt.show()
    
    # Momentum balance ZY
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(y_inner[:], I_corr_yz[:], label='coriolis', color='blue', linestyle='-')
    plt.plot(y_inner[:], (np.mean(visc_yz, axis=1))[:], label='viscous', color='red', linestyle='-')
    plt.plot(y_inner[:], (np.mean(stress_yz, axis=1))[:], label='Rey Stress', color='orange', linestyle='-')
    plt.plot(y_inner[:], dwdt, label='Temporal', color='saddlebrown', linestyle='-')
    plt.plot(y_inner[:], tau_yz[:], label='Total', color='black', linestyle='-')
    
    
    plt.title(r'Shear stress $\tau_{zx}$')
    plt.xlabel(r'$z^{+}$')
    plt.ylabel(r'${{\langle \bar{\tau} \rangle}^+}_{zy}$')
    plt.legend()
    plt.grid(True)
    plt.show()
        
    # Zoomed plot
    plt.figure(figsize=(8, 6), dpi=300)

    # Valley case (solid lines)
    plt.plot(y_inner[:limity], -I_corr_yz[:limity]/u_star**2, color='blue', linestyle='-', label='Coriolis')
    plt.plot(y_inner[:limity], (np.mean(visc_yz, axis=1))[:limity]/u_star**2, color='red', linestyle='-', label='Viscous')
    plt.plot(y_inner[:limity], (np.mean(stress_yz, axis=1))[:limity]/u_star**2, color='orange', linestyle='-', label='Rey Stress')
    plt.plot(y_inner[:limity], dwdt[:limity]/u_star**2, color='saddlebrown', linestyle='-', label='Temporal')
    
    # Smooth case (dashed lines)
    plt.plot(y_s_p[:160], -I_corr_yz_s[:160]/0.0618**2, color='blue', linestyle='--')
    plt.plot(y_s_p[:160], (np.mean(visc_yz_s, axis=1))[:160]/0.0618**2, color='red', linestyle='--')
    plt.plot(y_s_p[:160], (np.mean(Ryz_s, axis=1))[:160]/0.0618**2, color='orange', linestyle='--')
    plt.plot(y_s_p[:160], np.zeros((160)), color='saddlebrown', linestyle='--')
    
    # Custom Legend Handles
    valley_legend = mlines.Line2D([], [], color='black', linestyle='-', label='Valley')
    smooth_legend = mlines.Line2D([], [], color='black', linestyle='--', label='Smooth')
    
    plt.title(r'Shear stress $\tau_{zy}$')
    plt.xlabel(r'$z^{+}$')
    plt.ylabel(r'${{\langle \bar{\tau} \rangle}^+}_{zy}$')
    
    plt.legend(handles=[
        mlines.Line2D([], [], color='blue', linestyle='-', label='Coriolis'),
        mlines.Line2D([], [], color='red', linestyle='-', label='Viscous'),
        mlines.Line2D([], [], color='orange', linestyle='-', label='Rey Stress'),
        mlines.Line2D([], [], color='saddlebrown', linestyle='-', label='Temporal'),
        valley_legend, smooth_legend
    ])
    
    plt.grid(True)
    plt.xlim(0,200)
    plt.ylim(-0.5,1)
    plt.show()
    
    # Wind profile
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(np.mean(corr_yx, axis=1), y, label='coriolis', color='blue', linestyle='-')
    plt.title('Wind profile')
    plt.ylabel(r'$z^{+}$')
    plt.xlabel(r'$wind$')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Friction Velocity Profile
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(u_star2, y, label='u_{star}', color='blue', linestyle='-')
    plt.title('Friction Velocity')
    plt.ylabel(r'$y$')
    plt.xlabel(r'$u_{*}$')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Velocity profile
    plt.figure(figsize=(8,6))
    plt.plot(y_plus[(eps_hill[0]-1):]-y_plus[(eps_hill[0]-1)] ,u_plus[(eps_hill[0]-1):,0], label='top', color='blue', linestyle='-')
    plt.plot(y_plus[(eps_hill[128]-1):]-y_plus[eps_hill[128]] ,u_plus[(eps_hill[128]-1):,128], label='Flank left', color='saddlebrown', linestyle='-')
    plt.plot(y_plus[eps_hill[512]:]-y_plus[eps_hill[512]]     ,u_plus[(eps_hill[512]):,512], label='Bottom', color='red', linestyle='-')
    plt.plot(y_plus[(eps_hill[896]-1):]-y_plus[(eps_hill[896]-1)] ,u_plus[(eps_hill[896]-1):,896], label='Flank right', color='magenta', linestyle='-')
    
    plt.plot(y_plus[(eps_hill[0]-1):]-y_plus[(eps_hill[0]-1)],  w_plus[(eps_hill[0]-1):,0], label='top', color='blue', linestyle='--')
    plt.plot(y_plus[(eps_hill[128]-1):]-y_plus[(eps_hill[128]-1)],w_plus[(eps_hill[128]-1):,128], label='Flank left', color='saddlebrown', linestyle='--')
    plt.plot(y_plus[eps_hill[512]:]-y_plus[eps_hill[512]],w_plus[eps_hill[512]:,512], label='Bottom', color='red', linestyle='--')
    plt.plot(y_plus[(eps_hill[896]-1):]-y_plus[(eps_hill[896]-1)],w_plus[(eps_hill[896]-1):,896], label='Flank right', color='magenta', linestyle='--')
    
    custom_labels = ['Hill top', 'Left Flank', 'Valley Bottom', 'Right Flank', r'$\langle \bar{u} \rangle$', r'$\langle \bar{v} \rangle$']
    color_handles = [
    Line2D([0], [0], color='blue', lw=4, label='Blue'),
    Line2D([0], [0], color='saddlebrown', lw=4, label='SaddleBrown'),
    Line2D([0], [0], color='red', lw=4, label='Red'),
    Line2D([0], [0], color='magenta', lw=4, label='Magenta')]
    style_handles = [
    Line2D([0], [0], color='black', linestyle='-', lw=2, label='(-)'),
    Line2D([0], [0], color='black', linestyle='--', lw=2, label='(--)')]
    custom_handles = color_handles + style_handles
    plt.title('Velocity Profile ')
    plt.ylabel(r'$\langle \bar{u}_i \rangle ^+$')
    plt.xlabel(r'$z^{+}$')
    plt.xscale("log")
    plt.legend(custom_handles, custom_labels, loc='upper left')
    plt.grid(True)
    plt.show()
    
    # zoomed
    plt.figure(figsize=(8,6))
    plt.plot(y_plus[(eps_hill[0]-1):limity]-y_plus[(eps_hill[0]-1)] ,u_plus[(eps_hill[0]-1):limity,0], label='top', color='blue', linestyle='-')
    plt.plot(y_plus[(eps_hill[128]-1):limity]-y_plus[eps_hill[128]] ,u_plus[(eps_hill[128]-1):limity,128], label='Flank left', color='saddlebrown', linestyle='-')
    plt.plot(y_plus[eps_hill[512]:limity]-y_plus[eps_hill[512]]     ,u_plus[(eps_hill[512]):limity,512], label='Bottom', color='red', linestyle='-')
    plt.plot(y_plus[(eps_hill[896]-1):limity]-y_plus[(eps_hill[896]-1)] ,u_plus[(eps_hill[896]-1):limity,896], label='Flank right', color='magenta', linestyle='-')
    
    plt.plot(y_plus[(eps_hill[0]-1):limity]-y_plus[(eps_hill[0]-1)],  w_plus[(eps_hill[0]-1):limity,0], label='top', color='blue', linestyle='--')
    plt.plot(y_plus[(eps_hill[128]-1):limity]-y_plus[(eps_hill[128]-1)],w_plus[(eps_hill[128]-1):limity,128], label='Flank left', color='saddlebrown', linestyle='--')
    plt.plot(y_plus[eps_hill[512]:limity]-y_plus[eps_hill[512]],w_plus[eps_hill[512]:limity,512], label='Bottom', color='red', linestyle='--')
    plt.plot(y_plus[(eps_hill[896]-1):limity]-y_plus[(eps_hill[896]-1)],w_plus[(eps_hill[896]-1):limity,896], label='Flank right', color='magenta', linestyle='--')
    
    plt.axvline(x=(y[hill_height]*u_star/nu), color='black', linestyle='--', linewidth=1)
    plt.text((y[hill_height]*u_star/nu), 0.5, '$h$', rotation=90, verticalalignment='center', horizontalalignment='right')
    plt.axvline(x=(u_star**2/nu), color='black', linestyle='--', linewidth=1)
    plt.text((u_star**2/nu), 0.5, '$\delta$', rotation=90, verticalalignment='center', horizontalalignment='right')
    
    custom_labels = ['Hill top', 'Left Flank', 'Valley Bottom', 'Right Flank', r'$\langle \bar{u} \rangle$', r'$\langle \bar{v} \rangle$']
    color_handles = [
    Line2D([0], [0], color='blue', lw=4, label='Blue'),
    Line2D([0], [0], color='saddlebrown', lw=4, label='SaddleBrown'),
    Line2D([0], [0], color='red', lw=4, label='Red'),
    Line2D([0], [0], color='magenta', lw=4, label='Magenta')]
    style_handles = [
    Line2D([0], [0], color='black', linestyle='-', lw=2, label='(-)'),
    Line2D([0], [0], color='black', linestyle='--', lw=2, label='(--)')]
    custom_handles = color_handles + style_handles
    plt.title('Velocity Profile ')
    plt.ylabel(r'$\langle \bar{u}_i \rangle ^+$')
    plt.xlabel(r'$z^{+}$')
    plt.xscale("log")
    plt.legend(custom_handles, custom_labels, loc='upper left')
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(8,6))
    plt.plot(y_s_p, np.mean(U_s_p, axis=1),color='blue', linestyle='-' )
    plt.plot(y_s_p, np.mean(V_s_p, axis=1),color='red', linestyle='-')
    plt.plot(y_s_p, -np.mean(W_s_p, axis=1),color='black', linestyle='-')
    
    plt.figure(figsize=(8, 6), dpi=300)
    # Without orography (solid lines)
    plt.plot(y_s_p, np.mean(U_s_p, axis=1), color='red', linestyle='--', label='Streamwise')
    plt.plot(y_s_p, -np.mean(W_s_p, axis=1), color='blue', linestyle='--', label='Spanwise')
    # With orography (dashed lines)
    plt.plot(y_plus, u_plus_rot/ustr_s1, color='red', linestyle='-')
    plt.plot(y_plus, w_plus_rot/ustr_s1, color='blue', linestyle='-')
    plt.plot(y_s_p, u_most, linestyle='dotted', label='MOST_log law')
    # plt.plot(y_plus, u_most_v, linestyle='dotted', label='MOST_log law2')
    plt.axvline(x=(u_star**2/nu), color='black', linestyle='-', linewidth=1)
    plt.text((u_star**2/nu), 0.5, '$\delta_{hill}$', rotation=90, verticalalignment='center', horizontalalignment='right')
    plt.axvline(x=(ustr_s1**2/nu), color='black', linestyle='--', linewidth=1)
    plt.text((ustr_s1**2/nu), 0.5, '$\delta_{s}$', rotation=90, verticalalignment='center', horizontalalignment='right')
    plt.axvline(x=(y_plus[hill_height]), color='black', linestyle='--', linewidth=1)
    plt.text((y_plus[hill_height]), 0.5, '$h$', rotation=90, verticalalignment='center', horizontalalignment='right')
    # Formatting
    plt.xscale('log')  # Logarithmic x-axis
    plt.xlabel(r'$z^+$')  # x-axis label
    plt.ylabel(r'$\langle \bar{u_i} \rangle^+$')  # y-axis label
    # Creating a custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='-', label='Streamwise'),
        Line2D([0], [0], color='blue', linestyle='-', label='Spanwise'),
        Line2D([0], [0], color='black', linestyle='-', label='— With orography'),
        Line2D([0], [0], color='black', linestyle='--', label='— Smooth case')
    ]
    plt.legend(handles=legend_elements)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title("Velocity Profile with and without Orography")
    plt.show()
    
    plt.figure(figsize=(8,6))
    plt.plot(y_plus[:460], (np.mean(TKE, axis=1)/u_star**2)[:460] , label='valley', color='blue', linestyle='-')
    plt.plot(y_s_p[:130], (np.mean(TKE_s, axis=1)/ustr_s1**2)[:130] , label = 'smooth', color='red', linestyle='-')
    
    plt.axvline(x=(ustr_s1**2/nu), color='black', linestyle='--', linewidth=1)
    plt.text((ustr_s1**2/nu), 0.5, '$\delta_{s}$', rotation=90, verticalalignment='center', horizontalalignment='right')
    
    plt.axvline(x=(u_star**2/nu), color='black', linestyle='-', linewidth=1)
    plt.text((u_star**2/nu), 0.5, '$\delta_{v}$', rotation=90, verticalalignment='center', horizontalalignment='right')
    
    plt.axvline(x=(y[hill_height]/l_in), color='black', linestyle='-', linewidth=1)
    plt.text((y[hill_height]/l_in), 0.5, '$h$', rotation=90, verticalalignment='center', horizontalalignment='right')
    
    plt.title('TKE profile')
    plt.xlabel(r'$z^{+}$')
    plt.ylabel(r'$TKE$')
    # plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(8, 6), dpi=300)
    # Plotting the TKE distributions
    plt.plot(x_plus, AVG_TKE_V / (u_star**2), label="valley", color="blue", linestyle="-")
    plt.plot(x_plus, AVG_TKE_V_s_i / (u_star**2), label="smooth", color="red", linestyle="-")
    # Define the black line
    black_line = (y[hill_height] / u_star) * (1 + np.cos(2 * x_plus * np.pi / x_plus[-1]))
    # Fill the area below the black line in black
    plt.fill_between(x_plus, black_line, color="black", alpha=1.0, label="IBM solid")
    # Plot the black line again so it's visible on top of the fill
    plt.plot(x_plus, black_line, color="black", linestyle="-")
    # Labels and formatting
    plt.title("TKE distribution")
    plt.xlabel(r"$z^{+}$")
    plt.ylabel(r"$TKE$")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
if animate == 1:
    print('animating')
    for r in range(262509,263501):
        path = cwd + 'planesK.' + str(r)
        hdr, _, _, _, _, _ = read_header(path)
        if (os.path.exists(path)):
            a1 = readfield(path, nx, ny, 1, 0)
            plotanimatelog(x_plus, y_plus[:260], a1[:260,:,0], '', '', r'$x^+$',r'$z^+$', cwd + '/fig/' + str(r) + '.png', x_fill_plus, y_fill_plus, 1000)
    
    image_folder = os.path.join(cwd, "fig")  # Path to images
    start_num, end_num = 262510, 264500  #263500  # Image number range
    
    # Generate list of valid image files
    image_files = []
    for i in range(start_num, end_num + 1, 10):  # Increment by 10
        file_path = os.path.join(image_folder, f"{i}.png")
        if os.path.exists(file_path):  # Check if file exists
            image_files.append(file_path)
    
    # Check if images exist
    if not image_files:
        print("No valid images found in the specified range.")
        exit()
    
    # Load first image to get figure size
    first_image = Image.open(image_files[0])
    dpi = 300  # Match with plotanimate
    fig, ax = plt.subplots(figsize=(first_image.width / dpi, first_image.height / dpi), dpi=dpi)
    ax.axis("off")  # Remove axes
    
    # Function to update frames in the animation
    def update(frame):
        img = Image.open(image_files[frame])
        ax.imshow(img)
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(image_files), interval=50)
    
    # Save animation as a GIF or MP4
    output_gif = os.path.join(cwd, "animation.gif")
    output_mp4 = os.path.join(cwd, "animation.mp4")
    
    # Save as GIF
    ani.save(output_gif, writer="pillow", fps=20)
    print(f"Animation saved as {output_gif}")
    
    # Save as MP4
    ani.save(output_mp4, writer="ffmpeg", fps=20)
    print(f"Animation saved as {output_mp4}")
    
    plt.show()