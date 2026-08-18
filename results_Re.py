#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
results_Re.py — Stage c (local) for the REYNOLDS ladder.

Derived from results.py (the Froude ladder) and identical to it in role: it is a
PURE PLOTTING / COMBINING stage.  It loads each case's pickled per-case
quantities (sim1_results.pkl from PhAvg_rotated.py) plus the smooth NetCDF
reference, and lays them side by side.  It reads NO raw record (flow.*, scal.*,
avg_*, planesK/J.*) and builds no derivative field from scratch — see the
pipeline load-balancing rule in CLAUDE.md.

WHAT DIFFERS FROM results.py
----------------------------
results.py varies the Froude number at FIXED Re = 500.  This file does the
opposite: stratification is held at NEUTRAL (Fr = inf) and the Reynolds number
varies, so every difference between the cases is a Reynolds effect.

  cwd1  EkRe500FrInf/    Re_D = 500
  cwd2  EkRe750FrInf/    Re_D = 750
  cwd3  EkRe1000FrInf/   Re_D = 1000   (commented out by default)

  nc1   Re500/ri00.00_re0500_..._avg_all.nc    SMOOTH (flat) reference, Re_D=500
  nc2   Re1000/ri00.00_re1000_..._avg_all.nc   SMOOTH (flat) reference, Re_D=1000

  The Re500/ and Re1000/ folders in the examples ROOT hold the SMOOTH (flat
  wall) NetCDF references; the EkRe*FrInf folders are the VALLEY simulations.

0. DATA SOURCES ARE A SINGLE SWITCH — for the valley cases (cwd1/cwd2/cwd3)
   AND for the smooth references (nc1/nc2) alike.  COMMENTING OUT A SOURCE LINE
   REMOVES IT COMPLETELY — a commented source is an ABSENT source.  EVERY
   declared smooth reference is DRAWN (one curve per reference, each in its own
   wall units), so the figures carry as many flat-wall benchmarks as there are
   uncommented nc* lines.  Every derived
   table (SIM_DIRS / CASES / ACTIVE_CASES / USTAR_REF / SIM_FR / SIM_RE /
   SIM_NU / SIM_USTAR_FALLBACK) is built from the CASE REGISTRY, which only
   picks up the cwdN names that exist, so nothing downstream can reference a
   case that was switched off.  The reference grid / IBM geometry is taken from
   the first declared case directory that carries a `grid` file (NOT hardwired
   to cwd1, which used to break the moment cwd1 was commented out).

1. PER-CASE WALL UNITS.  results.py could use the single global nu / u* / l_in
   from config.py because Re was fixed.  Here nu = 1/(0.5*Re_D^2) and
   l_in = nu/u* are PER-CASE — including for the smooth reference, whose nu
   follows the smooth file actually selected.  u* is each case's own measured
   Method-2 value (pickled u_star2) when available, else the CLAUDE.md table
   (the Re=750 / Re=1000 entries are ESTIMATES until those flows are measured).

2. NORMALIZATION IS A USER INPUT.  The single knob is the USTAR_REF dict near
   the top of the file — one reference friction velocity per case:

       z+ = z / l_in(case),  l_in(case) = nu(case) / USTAR_REF[case]
       u+ = u / USTAR_REF[case],   tau / USTAR_REF[case]**2

   Defaults are each case's smooth-wall reference (Re=500 → 0.0618,
   Re=750 → 0.0561, Re=1000 → 0.0531; they live in the case registry as
   `ustar_ref`), so every case is drawn in its OWN wall units — the physically
   correct inner scaling for comparing Reynolds numbers, in which the log
   layers should collapse and the higher Re shows a longer log region.

   SET THE USTAR_REF ENTRIES EQUAL to normalize every case by one common
   velocity scale and see the raw, uncollapsed Reynolds effect.  (That equates
   the velocity scale exactly; l_in still differs through nu, which is fixed by
   the Reynolds number and IS the effect under study.)

   The smooth NetCDF reference follows the same dict via its 'Sm_Neu' entry, so
   it never silently ends up on a different axis than the valley cases.

   ONE set of figures is produced, written directly into fig_Re/.

2b. THE VALLEY IS IDENTICAL; THE AXIS IS NOT.  eps0.1 is generated once and
   carried to every grid by MyPyLib/transfer_eps.py, so the topography is the
   SAME OBJECT IN PHYSICAL SPACE in every case — MyPyLib/compare_eps_geometry.py
   verifies crest / floor / depth / width / solid area agree to better than one
   wall-normal cell across 1024x832x1024, 1152x816x1152 and 1728x784x1728.  A
   fixed physical crest height h nevertheless draws at

       h+ = h*u*/nu   ~ 25 (Re_D=500) | 57 (750) | 96 (1000)     inner units
       h/delta        ~ 0.045 | 0.045 | 0.047                    outer units
       h/h            = 1 always                                 topography

   so on a per-case z+ axis the valley legitimately looks ~2.3x taller at
   Re_D=750 than at 500: nu fell, the valley did not grow.  Z_NORM (below the
   imports) switches the 2-D side-by-side panels between 'inner' (z+, default,
   unchanged) and 'topo' (z/h, in which the topography coincides exactly in
   every panel).  The crest marker _mark_h is per-case in BOTH modes.

3. WALL LAW.  Every case is neutral, so every fit takes the classical
   ln(z+ - d+) branch.  The Obukhov (1971) stability-corrected branch is
   retained but unreachable on this ladder.

4. FROUDE-LADDER BLOCKS PRESERVED, NOT DELETED.  The cross-case vs-Ri_B
   aggregation (X1-X10 / P72-P78) is kept verbatim and gated off by
   PLOT_FR_LADDER, since Ri_B = 0 for every neutral case would collapse every
   such view to a single point.  Intermittency (P79-P84) is deliberately left
   ENABLED — Ansorge gamma is Reynolds-dependent and meaningful at neutral.

FILE LAYOUT (four sections, in this order — see the banners in the body):

  1. FUNCTION DEFINITIONS   every def, nothing else.  First, so anything below
                            can call anything, and so no function can depend on
                            a user parameter at DEFINITION time.
  2. USER PARAMETERS AND VARIABLE DECLARATION
                            the only section you normally edit: case dirs, the
                            NetCDF references, physics constants, Z_NORM, the
                            plot windows, the on/off switches.  An index of
                            every knob heads the section.
  3. CALCULATIONS AND CALLS TO FUNCTIONS
                            grid + eps geometry, pickle and NetCDF loading, the
                            per-case friction velocity, derived case tables,
                            the run summary.  Order-dependent.
  4. PLOTS                  all figure generation, gated by `plotRes`.  Reads
                            only what 2 and 3 produced.

Outputs (examples root): fig_Re/{percase,shared}/*.png + sim_stats_Re.log,
all named so they cannot collide with results.py's fig/ and sim_stats.log.

Created on Mon Nov  4 11:13:16 2024

@author: shreyas deshpande
"""

import os
import re
import sys
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
from scipy.optimize import curve_fit
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from PIL import Image
from functions import *


###############################################################################
###############################################################################
####  1.  FUNCTION DEFINITIONS
#
#  Every def in this file, and nothing else.  Definitions only: no call, no
#  assignment and no print runs in this section.  It is first so that anything
#  below can call anything, and so that a function can never depend on a user
#  parameter at DEFINITION time (defaults resolve inside the body instead).
#
#  Edit here to change BEHAVIOUR.  To change a SETTING, go to section 2.
###############################################################################
###############################################################################


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
    f = open(_REF_DIR + fname,'rb')
    f.seek(0,0)
    header = np.fromfile(f, type_i4, head_params)
    f.close()
    print('Header size           :', header[0])
    print('Grid   size (nx*ny*nz):', header[1]*8,'x',header[2],'x',header[3])

    # data size (attention: h[1] = grid.nx*8!)
    bsize = np.prod(header[1:3])
    rsize = bsize * 8

    # read eps field as int1
    f = open(_REF_DIR + fname,'rb')
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
            # Top
            if j == 0:
                # Top left cornor
                if i == 0:
                    if (eps[j,i] + eps[j+1,i+1] + eps[j+1,i] + eps[j,i+1] == 4):
                        eps_vol[j,i] = 1
                    # else:
                    #     print ('i:', i , 'j:', j, 'Case undefined')
                        
                # Top right cornor
                elif i == nx-1:
                    if (eps[j,i] + eps[j,i-1] + eps[j+1,i-1] + eps[j+1,i] == 4):
                        eps_vol[j,i] = 1
                    # else:
                    #     print ('i:', i , 'j:', j, 'Case undefined')
                        
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
                    # else:
                    #     print ('i:', i , 'j:', j, 'Case undefined')
                
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
                # else:
                #     print ('i:', i , 'j:', j, 'Case undefined')
                    
            # Left edge
            elif i == 0 and j != 0:
                if (eps[j,i] + eps[j-1,i] + eps[j-1,i+1] + eps[j,i+1] + eps[j+1,i+1] + eps[j+1,i] == 6):
                    eps_vol[j,i] = 1
                
                elif (eps[j,i] + eps[j-1,i] + eps[j-1,i+1] + eps[j,i+1] + eps[j+1,i+1] + eps[j+1,i] == 5):
                    eps_vol[j,i] = 0.5
                    
                elif (eps[j,i] + eps[j-1,i] + eps[j-1,i+1] + eps[j,i+1] + eps[j+1,i+1] + eps[j+1,i] == 4):
                    eps_vol[j,i] = 0.5
                    
                # else:
                #     print ('i:', i , 'j:', j, 'Case undefined')
                    
            # Right edge
            elif i == nx-1 and j != 0:
                if (eps[j,i] + eps[j-1,i] + eps[j-1,i-1] + eps[j,i-1] + eps[j+1,i-1] + eps[j+1,i] == 6):
                    eps_vol[j,i] = 1
                
                elif (eps[j,i] + eps[j-1,i] + eps[j-1,i-1] + eps[j,i-1] + eps[j+1,i-1] + eps[j+1,i] == 5):
                    eps_vol[j,i] = 0.5
                    
                elif (eps[j,i] + eps[j-1,i] + eps[j-1,i-1] + eps[j,i-1] + eps[j+1,i-1] + eps[j+1,i] == 4):
                    eps_vol[j,i] = 0.5
                # else:
                #     print ('i:', i , 'j:', j, 'Case undefined')
                    
            # else:
            #     print ('i:', i , 'j:', j, 'Case undefined')
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
############### PLOTTING & ANALYSIS HELPERS (grouped definitions) ##############
# These were previously defined inline inside the `if (1 == plotRes):` block,
# interleaved with the plotting/calculation flow.  They are collected here so
# that ALL function definitions live at the top of the file, separate from the
# calculation/plotting code below (easier to find/debug).  They resolve the
# module-level names they use (sims, the grids, SIM_NAMES, _figdir, _ll_fits, …)
# at CALL time — those names are populated by the main body and the plotRes block
# before any of these functions run, so the relocation does not change behaviour.
###############################################################################
def _dflt_case(case=None):
    """Resolve an omitted `case` argument to the FIRST DECLARED case.

    The per-case helpers below used to default to the literal 're500_oro',
    which silently returned nothing the moment that data source was commented
    out.  SIM_DIRS is read at CALL time, so the default always follows whichever
    sources are actually declared."""
    return case if case is not None else next(iter(SIM_DIRS), None)

def gv(name, case=None):
    """Return sims[case][name]; None if absent."""
    case = _dflt_case(case)
    return sims.get(case, {}).get(name)


def _note_missing(figure, field, case, reason, once=False):
    """Record — and print — one 'case absent from this figure' event.

    once=True prints only the FIRST occurrence of a given (case, reason): the
    case-level reasons (source commented out, no pickle, disabled) are identical
    for every figure, so repeating them dozens of times buries the field-level
    ones that actually differ.  Every event is still logged for the summary."""
    _MISSING_LOG.append((figure, field, case, reason))
    if once:
        if (case, reason) in _MISSING_SEEN:
            return
        _MISSING_SEEN.add((case, reason))
    print('  MISSING DATA: case %-12s field %-16s [%s] — %s'
          % (case, field, figure, reason))


def _case_drop_reason(case):
    """Why `case` carries no plottable data AT ALL (None when it does).

    Case-level only — a loaded case that merely lacks one field returns None
    here and is classified per field by _case_availability()."""
    if case in _CASE_DROP_REASON:
        return _CASE_DROP_REASON[case]
    if case not in sims:
        return ('no sim1_results.pkl loaded from %s (run stage b, '
                'PhAvg_rotated.py, in that directory)'
                % SIM_DIRS.get(case, '<undeclared>'))
    return None


def _case_availability(field_key, need_2d=True):
    """Classify EVERY case in the registry for one plotted field.

    Returns (available, absent):
      available : [(name, label), …] — exactly the old `_avail` list, in
                  registry order (increasing Re), ready to draw.
      absent    : [(name, reason, case_level), …] — every registry case NOT
                  drawn, with the reason.  case_level=True marks a reason that
                  is the same on every figure (commented-out source, missing
                  pickle, disabled) so the caller can print it once.

    Iterating the REGISTRY rather than SIM_NAMES is deliberate: a case dropped
    before SIM_NAMES was built (no pickle, commented out) is invisible to
    SIM_NAMES and would otherwise never be reported by the figure that wanted
    it."""
    _av, _ab = [], []
    for _e in _CASE_REGISTRY:
        _cn = _e['name']
        _why = _case_drop_reason(_cn)
        if _why is not None:
            _ab.append((_cn, _why, True))
            continue
        if _cn not in SIM_NAMES:
            _ab.append((_cn, 'not active (hidden by DISABLE_CASES)', True))
            continue
        _f = gv(field_key, _cn)
        if _f is None:
            # Covers both a directly-pickled field and one this file derives from
            # pickled fields (P, veer_2d, reyn_*, tot_*, Ph*_mean, …): either way
            # the cure is the same — regenerate that case's pickle.
            _ab.append((_cn, "'%s' unavailable — not in %ssim1_results.pkl (nor "
                             "are the pickled fields it is built from).  Re-run "
                             "stage b (PhAvg_rotated.py) there; if it is a new "
                             "quantity, add it to saveresults.py var_names first"
                        % (field_key, SIM_DIRS.get(_cn, '?')), False))
            continue
        if need_2d and np.ndim(_f) != 2:
            _ab.append((_cn, "'%s' is %d-D in the pickle, a 2-D (z,x) map is "
                             "required for this panel" % (field_key, np.ndim(_f)),
                        False))
            continue
        _av.append((_cn, _e['label']))
    return _av, _ab


def _report_missing_summary():
    """Roll-up of every MISSING DATA event, grouped by case (end-of-run)."""
    print('\n--- MISSING DATA (cases that could not be drawn) ---')
    if not _MISSING_LOG:
        print('  none — every declared case supplied every plotted field.')
        return
    _by_case = {}
    for _fig, _fld, _cn, _why in _MISSING_LOG:
        _by_case.setdefault(_cn, []).append((_fig, _fld, _why))
    for _cn in [_e['name'] for _e in _CASE_REGISTRY if _e['name'] in _by_case]:
        _rows    = _by_case[_cn]
        _fields  = sorted({_r[1] for _r in _rows})
        _reasons = sorted({_r[2] for _r in _rows})
        print('  %s (%s): absent from %d figure(s), %d field(s)'
              % (_cn, _case_label(_cn), len({_r[0] for _r in _rows}), len(_fields)))
        print('      fields  : %s' % ', '.join(_fields))
        for _rs in _reasons:
            print('      reason  : %s' % _rs)


def _case_label(case):
    """Registry label of a case (falls back to its name)."""
    return next((_e['label'] for _e in _CASE_REGISTRY if _e['name'] == case),
                case)

def _case_own_grid(case):
    """This case's OWN grid + IBM geometry, read from its data directory.

    Used only as a FALLBACK for a legacy/stale pickle that predates per-case
    grid bundling (no 'x'/'y'/'eps' inside sim1_results.pkl).  Without it such a
    case can be placed on NO wall-normal axis and is skipped by every shared
    z+/z- figure — which is exactly how a case silently disappears from the
    *_allRe plots while the case that happens to own the reference grid still
    draws.  Reads only `grid` and `eps_save.npy` (the same two files stage c
    already reads for the reference case) — never a raw field record.

    Returns {'x','y','z','eps','mask0'} (eps/mask0 may be None) or None.
    Cached per case."""
    if case in _OWN_GRID_CACHE:
        return _OWN_GRID_CACHE[case]
    _d = SIM_DIRS.get(case)
    _out = None
    if _d and os.path.exists(_d + 'grid'):
        try:
            _xc, _yc, _zc = read_grid(_d)
            _ec = None
            if os.path.exists(_d + 'eps_save.npy'):
                _ec = np.load(_d + 'eps_save.npy')
                if _ec.shape != (len(_yc), len(_xc)):
                    print('  %s: eps_save.npy %s does not match its grid (%d, %d)'
                          ' — ignored' % (case, _ec.shape, len(_yc), len(_xc)))
                    _ec = None
            _out = {'x': _xc, 'y': _yc, 'z': _zc, 'eps': _ec,
                    'mask0': (1 - _ec) if _ec is not None else None}
            print('  %s: per-case grid read from its own `grid` file '
                  '(pickle carries none): ny=%d, nx=%d'
                  % (case, len(_yc), len(_xc)))
        except Exception as _e_g:
            print('  %s: could not read its own grid (%s)' % (case, _e_g))
    _OWN_GRID_CACHE[case] = _out
    return _out


def _case_axis_y(case):
    """(y_physical, source) — the wall-normal grid THIS case's profiles are
    plotted against, tried in order:
      1. the per-case grid bundled in its pickle          ('pickle')
      2. this case's own `grid` file in its data dir      ('own grid file')
      3. the shared reference grid                        ('reference grid')
    A candidate is accepted only if its length matches the case's own profile
    length (so a case on a different grid is never silently drawn on the wrong
    axis).  Returns (None, reason) when nothing matches — the caller then skips
    that case, and the PLOT READINESS table prints the reason."""
    _sd = sims.get(case, {})
    if not _sd:
        return None, 'no pickle loaded'
    _probe = _sd.get('u_plus_rot')
    _n = len(_probe) if _probe is not None else None
    _OWN = object()                            # sentinel: resolve lazily
    for _cand, _src in ((_sd.get('y'), 'pickle'), (_OWN, 'own grid file'),
                        (y, 'reference grid')):
        if _cand is _OWN:                      # reads that case's `grid` file
            _own = _case_own_grid(case)
            _cand = _own['y'] if _own else None
        if _cand is None:
            continue
        if _n is None or len(_cand) == _n:
            return _cand, _src
    return None, ('no grid matches this case\'s profile length (ny=%s)' % _n)


def gy_in(case=None):
    """Per-case wall-normal grid in inner units (Reynolds-ladder specific).

    Returns the case's own physical grid y (which may have a different ny than
    the reference grid) divided by norm_l_in(case) = nu(case)/norm_ustar(case),
    i.e. by that case's own wall unit.  With distinct u* per case each case is
    in its own true wall units (log layers should collapse across Reynolds
    number); equate the USTAR_REF entries to put every case on one common
    velocity scale.  results.py divided by the single global l_in
    unconditionally, which is only valid when every case shares one Re.

    The grid itself comes from _case_axis_y: pickle → the case's own `grid`
    file → the reference grid.  None only when none of the three matches the
    case's profile length (the caller then skips it instead of crashing)."""
    _yg, _ = _case_axis_y(_dflt_case(case))
    return None if _yg is None else _yg / norm_l_in(_dflt_case(case))

def geps(case=None):
    """Per-case IBM indicator eps (1 in solid): pickled per-case eps → the eps
    saved beside that case's own grid → the reference eps (last resort, correct
    only for a case actually on the reference grid)."""
    case = _dflt_case(case)
    _e = sims.get(case, {}).get('eps')
    if _e is None:
        _own = _case_own_grid(case)
        _e = _own['eps'] if _own else None
    return eps if _e is None else _e

def gmask0(case=None):
    """Per-case fluid mask (1-eps), same fallback chain as geps."""
    case = _dflt_case(case)
    _m = sims.get(case, {}).get('mask0')
    if _m is None:
        _own = _case_own_grid(case)
        _m = _own['mask0'] if _own else None
    return mask0 if _m is None else _m

def geps_f(case=None):
    """Per-case fluid-fraction column weight for intrinsic x-averages:
    mean_x(mask0) with zeros replaced by NaN (avoids divide-by-zero)."""
    _m = np.mean(gmask0(_dflt_case(case)), axis=1)
    return np.where(_m > 0, _m, np.nan)

def gx_in(case=None):
    """Per-case streamwise grid in inner units under the ACTIVE normalization
    mode (x / norm_l_in(case) — see gy_in).  A case on a different grid than
    the reference has a different nx, so x-distribution profiles (e.g.
    AVG_TKE_V) must be plotted against THIS case's own x.  Falls back to the
    reference x for a legacy pickle that carries no per-case 'x'."""
    case = _dflt_case(case)
    _xg = sims.get(case, {}).get('x')
    if _xg is None:
        _own = _case_own_grid(case)
        _xg = _own['x'] if _own else None
    return (x if _xg is None else _xg) / norm_l_in(case)

def ghill(case=None):
    """Per-case valley-crest ROW index from THIS case's eps, using the same
    definition as the global hill_hgt (max solid-cell column count - 1).
    Cases on different wall-normal grids place the crest at a different index,
    so any per-case profile sampled 'at the crest' (u*, hodograph markers)
    must use this, not the neutral hill_hgt (= 94).  Falls back to the neutral
    hill_hgt when the case has no per-case eps (geps returns the reference)."""
    _e = geps(_dflt_case(case))
    if _e is eps:
        return hill_hgt
    return int(np.max(np.sum(_e, axis=0).astype(int)) - 1)

# ── Valley height h: the ONE length that is identical across the ladder ──────
# The IBM indicator is generated once and carried to every grid with
# transfer_eps.py, so the valley is the SAME OBJECT IN PHYSICAL SPACE on every
# case — verified with MyPyLib/compare_eps_geometry.py (crest, floor, depth,
# width and solid area agree to better than one wall-normal cell between
# 1024x832x1024, 1152x816x1152 and 1728x784x1728).
#
# It therefore does NOT follow that the valley draws at the same height on
# every figure.  A fixed physical h maps to
#     h+     = h*u*/nu      inner  -> Re_D 500/750/1000 give h+ ~ 25 / 57 / 96
#     h/delta= h*f/u*       outer  -> ~0.045 / 0.045 / 0.047   (weakly Re-dep.)
#     h/h    = 1            topography units (Z_NORM='topo')   -> ALWAYS equal
# so a valley that looks taller at higher Re on a z+ axis has not grown: nu
# fell.  Z_NORM (top of file) selects which of these the 2-D panels use.
def case_hill_h(case=None):
    """PHYSICAL crest height h of THIS case's valley = y_case[ghill(case)].

    Read from the case's OWN grid + eps (pickle -> its own `grid`/eps_save.npy
    -> the reference), so a case on a different wall-normal grid still reports
    its true height rather than the reference grid's."""
    case = _dflt_case(case)
    _yc = sims.get(case, {}).get('y')
    if _yc is None:
        _own = _case_own_grid(case)
        _yc = _own['y'] if _own else y
    _j = ghill(case)
    return float(_yc[min(_j, len(_yc) - 1)])

def case_hill_zplus(case=None):
    """Crest height of THIS case on ITS OWN inner axis: h+ = h/norm_l_in(case).
    This is where `h` actually sits on every z+ figure — the global
    y_in[hill_hgt] is the REFERENCE grid over the REFERENCE l_in and matches no
    case once Re varies."""
    case = _dflt_case(case)
    return case_hill_h(case) / norm_l_in(case)

def _chk_mode(mode):
    if mode not in _Z_MODES:
        raise ValueError('unit mode must be one of %s, got %r' % (_Z_MODES, mode))
    return mode

def delta_nu(case=None):
    """INNER length scale of this case: delta_nu = nu(case)/u_tau(case).
    Alias of norm_l_in kept so the two scales read symmetrically."""
    return norm_l_in(_dflt_case(case))

def delta_out(case=None):
    """OUTER length scale of this case: the Ekman depth delta = u_tau/f, built
    from the SAME per-case u_tau as delta_nu (f is the global Coriolis
    parameter, = 1 in these runs, so delta is numerically u_tau)."""
    return norm_ustar(_dflt_case(case)) / float(f)

def norm_z_len(case=None, mode='inner'):
    """The length THIS case's wall-normal coordinate is divided by, per mode:
    delta_nu ('inner'), delta ('outer'), the valley height h ('topo')."""
    case = _dflt_case(case)
    _chk_mode(mode)
    if mode == 'outer':
        return delta_out(case)
    if mode == 'topo':
        return case_hill_h(case)
    return delta_nu(case)

def norm_vel(case=None, power=1, mode='inner'):
    """Divisor for a quantity of VELOCITY DIMENSION `power` in this mode.

    inner -> u_tau(case)**power        outer/topo -> 1.0  (geostrophic units)

    Use it for every velocity (power 1), every second moment / stress / TKE /
    pressure (power 2) and every flux-like third moment (power 3), so a figure
    can never end up with an inner ordinate on an outer abscissa."""
    _chk_mode(mode)
    if mode != 'inner':
        return 1.0
    return float(norm_ustar(_dflt_case(case))) ** power

def gy_out(case=None):
    """Per-case OUTER wall-normal grid z- = y/delta(case)."""
    case = _dflt_case(case)
    _yg, _ = _case_axis_y(case)
    return None if _yg is None else _yg / delta_out(case)

def gy_mode(case=None, mode='inner'):
    """Per-case wall-normal grid in the requested mode (None when the case has
    no usable grid — the caller then skips it, as the z+ helpers already do)."""
    case = _dflt_case(case)
    _yg, _ = _case_axis_y(case)
    return None if _yg is None else _yg / norm_z_len(case, mode)

def case_hill_zminus(case=None):
    """Crest height of THIS case in OUTER units: h/delta = h*f/u_tau."""
    case = _dflt_case(case)
    return case_hill_h(case) / delta_out(case)

def case_hill_pos(case=None, mode='inner'):
    """Crest position of THIS case on the wall-normal axis of `mode`:
    h+ ('inner'), h/delta ('outer'), exactly 1.0 ('topo')."""
    case = _dflt_case(case)
    _chk_mode(mode)
    if mode == 'topo':
        return 1.0
    return case_hill_h(case) / norm_z_len(case, mode)

def z_label(mode='inner'):
    return _Z_LABELS[_chk_mode(mode)]

def x_label(mode='inner'):
    return _X_LABELS[_chk_mode(mode)]

def norm_vel_smooth(power=1, mode='inner'):
    """Same as norm_vel but for the flat smooth NetCDF reference: its own u*
    (ustr_s1) in 'inner', the geostrophic wind (1.0) in 'outer'/'topo'.  Kept
    beside norm_vel so a figure cannot scale the reference and the valley cases
    on two different conventions."""
    _chk_mode(mode)
    return (float(ustr_s1) ** power) if mode == 'inner' else 1.0

def vel_label(sym, power=1, mode='inner'):
    """Ordinate label for a quantity `sym` of velocity power `power`.
    inner -> '<sym>/u_*^power' (or the '+' superscript for power 1);
    outer -> '<sym>/G^power' — G is the geostrophic wind, the outer velocity
    scale, so the curve is the raw stored value (G = 1 in these runs)."""
    _chk_mode(mode)
    if mode == 'inner':
        return (sym + r'$^+$') if power == 1 else sym + r'$/u_*^{%d}$' % power
    return sym + (r'$/G$' if power == 1 else r'$/G^{%d}$' % power)

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
    """Legend handles for everything drawn: ONE handle per loaded smooth
    reference (in registry order) followed by the active valley cases."""
    _h = [Line2D([0], [0], color=SMOOTH_MAP[_k]['color'],
                 linestyle=SMOOTH_MAP[_k]['ls'], label=SMOOTH_MAP[_k]['label'])
          for _k in SMOOTH_KEYS]
    _h += [Line2D([0],[0], color=c['color'], linestyle=c['ls'], label=c['label'])
           for c in _sims]
    return _h

def smooth_handles():
    """Legend handles for the loaded smooth references only."""
    return [Line2D([0], [0], color=SMOOTH_MAP[_k]['color'],
                   linestyle=SMOOTH_MAP[_k]['ls'], label=SMOOTH_MAP[_k]['label'])
            for _k in SMOOTH_KEYS]

def sim_handles():
    """Legend handles for the active rough-wall cases only — lines only."""
    return [Line2D([0],[0], color=c['color'], linestyle=c['ls'],
                   label=c['label'])
            for c in _sims]

def _hill_marks(mode='inner'):
    """[(crest position, case colour), ...] for the ACTIVE cases, expressed on
    the wall-normal axis of `mode` — h+ ('inner'), h/delta ('outer'), 1 ('topo').
    Each case on ITS OWN axis.  Cases whose crest lands within 1 % of each other
    are collapsed to ONE black line, so at fixed Reynolds number (or in 'topo',
    where every crest is exactly 1) the figure shows a single line."""
    _m = []
    for _c, _clr in zip(globals().get('SIM_NAMES', []),
                        globals().get('SIM_COLORS', [])):
        try:
            _p = case_hill_pos(_c, mode)
        except Exception:
            continue
        if np.isfinite(_p) and _p > 0:
            _m.append((float(_p), _clr))
    if not _m:                                   # no case resolved -> reference
        return [(float(y_in[hill_hgt] if mode == 'inner'
                       else (1.0 if mode == 'topo'
                             else y[hill_hgt] / (u_star / float(f)))), 'black')]
    _lo = min(_p for _p, _ in _m)
    _hi = max(_p for _p, _ in _m)
    if _hi <= 1.01 * _lo:                        # all cases agree -> one line
        return [(0.5 * (_lo + _hi), 'black')]
    return _m

def _mark_h(orient='v', label=True, lblpos=0.04, ha='right', mode='inner'):
    """Mark the valley-crest height h with a dashed line labelled 'h'.

    PER CASE, on that case's OWN axis.  The crest of a FIXED physical valley
    sits at h+ = h*u*/nu, so on a Reynolds ladder it lands at a DIFFERENT z+ in
    every case (h+ ~ 25 at Re_D=500 but ~57 at Re_D=750 — the valley has not
    changed, nu has).  This used to draw a single line at the global
    y_in[hill_hgt] = the REFERENCE grid over the REFERENCE l_in (nu=8e-6,
    u*=0.0618), which matches NO case once Re varies and made the geometry look
    mis-normalised on every z+ figure.  One line per case now, in the case
    colour, collapsing to a single black line when the crests coincide (fixed
    Re, or Z_NORM='topo' where every crest is exactly 1 by construction).

    `mode` MUST match the axis being drawn on ('inner' for a z+ figure, 'outer'
    for a z- one, 'topo' for a z/h panel) — the crest of the same valley sits at
    h+ ~ 25-96 but at h/delta ~ 0.045, so a marker on the wrong scale lands off
    the plot entirely.

    orient='v' draws vertical lines (use when z is on the x-axis);
    orient='h' draws horizontal ones (use when z is on the y-axis).  The 'h'
    label is placed in axes-fraction coordinates along the marked line so it
    renders correctly on linear or log axes and after re-scaling.
    """
    _ax = plt.gca()
    _marks = _hill_marks(mode)
    for _i, (_pos, _clr) in enumerate(_marks):
        if orient == 'v':
            _ax.axvline(x=_pos, color=_clr, linestyle='--', linewidth=0.8)
            if label:
                _ax.text(_pos, lblpos, r'$h$', rotation=90, va='bottom', ha=ha,
                         fontsize=9, color=_clr,
                         transform=_ax.get_xaxis_transform())
        else:
            _ax.axhline(y=_pos, color=_clr, linestyle='--', linewidth=0.8)
            if label:
                _ax.text(lblpos, _pos, r'$h$', va='bottom', ha='left',
                         fontsize=9, color=_clr,
                         transform=_ax.get_yaxis_transform())

def _hill_pos_min(mode='inner'):
    """Lowest crest position over the active cases, on the axis of `mode` — the
    sensible lower limit for an axis that starts 'at the valley top' (was
    y_in[hill_hgt], i.e. the reference case's crest, which is below every real
    crest on this ladder)."""
    return min(_p for _p, _ in _hill_marks(mode))

def _axis_start_zplus():
    """Smallest positive z+ any active case can show = min over cases of that
    case's first grid node on its OWN inner axis (was the reference y_in[1])."""
    _v = []
    for _c in globals().get('SIM_NAMES', []):
        _yi = gy_in(_c)
        if _yi is not None and len(_yi) > 1 and np.isfinite(_yi[1]) and _yi[1] > 0:
            _v.append(float(_yi[1]))
    return min(_v) if _v else float(y_in[1])

def _mark_heights_v(entries, lblpos=0.94):
    """Draw dashed vertical height-reference lines, but ONLY for heights that
    fall inside the current z+ axis window (0, Z_PLUS_MAX].  A height outside
    the window (e.g. the boundary-layer depth delta ~ u*^2/nu ~ 477 on a
    z+<=200 axis) is SKIPPED rather than drawn far off-plot.
    entries : list of (z+ position, LaTeX label, colour)."""
    _ax = plt.gca()
    for _zp, _lab, _clr in entries:
        if _zp is None:
            continue
        _zp = float(_zp)
        if not (0.0 < _zp <= Z_PLUS_MAX):
            continue
        _ax.axvline(x=_zp, color=_clr, linestyle='--', linewidth=0.8)
        _ax.text(_zp, lblpos, _lab, rotation=90, va='top', ha='right',
                 fontsize=9, color=_clr, transform=_ax.get_xaxis_transform())

def _z_out(case):
    """Per-case OUTER wall-normal coordinate z- = y/delta, delta = u_tau/f.

    Returns (z_minus_array, delta) or (None, None) when the case has no usable
    grid.  The SECOND return value is a LENGTH (the Ekman depth), not a velocity
    scale: in outer units nothing of velocity dimension is divided by u_tau (see
    THE UNIT SYSTEM above), so callers must NOT reuse it as one.  Use
    norm_vel(case, power, 'outer') — which is 1.0 — for the ordinate.

    u_tau is the SAME per-case Method-2 plateau the inner figures use
    (norm_ustar / case_ustar, read from the pickled u_star2 profile).  This used
    to be u_star2[ghill(case)], the CREST value — a different number (0.0640 vs
    0.0663 on the Re=500 case), so the inner and outer views of one case were
    built on two different friction velocities."""
    _yc = gv('y', case)
    if _yc is None:
        _yc, _ = _case_axis_y(case)
    if _yc is None:
        return None, None
    _d = delta_out(case)
    if not (np.isfinite(_d) and _d > 0.0):
        return None, None
    return _yc / _d, float(_d)

# PhAvg-style boundary-layer LAYER markers (functions.mark_layers): the
# symbol encodes the sublayer, the colour encodes the case.  Indices are
# taken on THIS case's own inner (z+) axis so a case on a different grid is
# marked at the right physical height (mirrors PhAvg_rotated.py _LYR_ORO).
#   'o' viscous top z+~5 | 's' canopy (peak x-avg dispersive uv below log
#   start) | '^' log start z+~75 | 'D' log top z+~200 | 'X' valley crest h
def _oro_layer_idx(case):
    _yi = gy_in(case)
    if _yi is None:
        return None
    def _zi(zv):
        return int(np.argmin(np.abs(_yi - zv)))
    _mk = {'o': _zi(5.0), '^': _zi(75.0), 'D': _zi(200.0), 'X': ghill(case)}
    _uv = gv('UV_disp', case)
    if _uv is not None and np.ndim(_uv) == 2:
        _cmax = _zi(75.0)
        _prof = avg_c(geps(case), _uv * gmask0(case), axis=1)
        if _cmax >= 1 and np.size(_prof):
            _mk['s'] = int(np.argmax(_prof[:_cmax + 1]))
    return _mk

def _smo_layer_idx():
    """Smooth-wall layer indices on the smooth inner axis y_in_s
    ('o' z+~5, '^' z+~30, 'D' z+~100)."""
    def _zi(zv):
        return int(np.argmin(np.abs(y_in_s - zv)))
    return {'o': _zi(5.0), '^': _zi(30.0), 'D': _zi(100.0)}

# --- per-case 2D axes helpers -------------------------------------------
# Each case carries its OWN grid + orography in its pickle (saveresults.py),
# so a case on a different grid than the reference is plotted against its own
# coordinates.  The NORMALISING LENGTH is chosen by Z_NORM (top of file):
#   'inner' -> delta_nu(cn) = nu(cn)/u_tau(cn), one wall unit of THIS case;
#   'outer' -> delta_out(cn) = u_tau(cn)/f, THIS case's Ekman depth;
#   'topo'  -> case_hill_h(cn), THIS case's own valley height, so the identical
#              physical topography lands at exactly 1 in every panel.
# x and z are divided by the SAME length, so the panels stay isotropic and the
# ORIGRAPHY (the black eps region and the x_oro/y_oro polygon) is carried onto
# the active axis with the field — it can never be left on a stale scale.
# (results.py used the single global l_in here, which is only valid at fixed
# Reynolds number.)
def _norm_len(cn):
    """The length the 2-D panels of case `cn` are divided by — see Z_NORM."""
    return norm_z_len(cn, Z_NORM)

def _case_grid(cn, use_inner=True):
    _sd = sims.get(cn, {})
    _xc = _sd.get('x', x);         _yc = _sd.get('y', y)
    _xo = _sd.get('x_oro', x_oro); _yo = _sd.get('y_oro', y_oro)
    if use_inner:
        _li = _norm_len(cn)
        return _xc / _li, _yc / _li, _xo / _li, _yo / _li
    return _xc, _yc, _xo, _yo

def _z_axis_label():
    """Wall-normal axis label of the 2-D panels under the active Z_NORM."""
    return z_label(Z_NORM)

def _x_axis_label():
    """Streamwise axis label of the 2-D panels under the active Z_NORM (both
    axes are divided by the SAME length, so the panels stay isotropic)."""
    return x_label(Z_NORM)

def _ref_norm_len():
    """The Z_NORM length of the REFERENCE grid — used only for the shared panel
    extent and for the flat smooth reference, which has no valley of its own."""
    if Z_NORM == 'topo':
        return float(y[hill_hgt])
    if Z_NORM == 'outer':
        return float(u_star) / float(f)
    return l_in

def _row_to_height(ylim, use_inner=True):
    """Common z-extent for the panels, given as the height of reference row
    `ylim`, expressed in the units of the ACTIVE axis (z+ / z- / z/h), or as a
    physical height when use_inner=False."""
    _yr = y[ylim] if ylim < len(y) else y[-1]
    return _yr if not use_inner else _yr / _ref_norm_len()

def _zplus_to_mode(zp):
    """Convert an axis limit written in REFERENCE wall units (z+) into the
    ACTIVE Z_NORM units, by going through the physical height it stands for.
    This is what keeps every mode showing the SAME PART OF THE FLOW: there is
    one window, expressed three ways, instead of three constants that drift."""
    return float(zp) * l_in / _ref_norm_len()

def _contour_zmax(use_inner=True):
    # Common wall-normal crop for every 2-D contour/pcolormesh panel: ONE
    # physical height (Z_PLUS_CONTOUR_MAX reference wall units, set in the
    # plotRes block), returned in the units of the ACTIVE axis.  Flipping Z_NORM
    # therefore re-labels and re-scales the panels without changing what part of
    # the flow they show.  use_inner=False returns that height in physical units.
    _phys = Z_PLUS_CONTOUR_MAX * l_in
    return _phys if not use_inner else _phys / _ref_norm_len()

def _clip_rows(_yp, _zmax):
    _j = int(np.searchsorted(_yp, _zmax)) + 1
    return min(max(_j, 1), len(_yp))

# Smooth flat-wall (neutral) 2D reference for the side-by-side colormaps.
# The smooth NetCDF stores (y × nt) mean/Reynolds fields — homogeneous in x,
# so they are shown against the pseudo-x axis `sx` (0…1.08), the SAME
# convention plot 24 already uses for the smooth TKE-advection panel.
# Fields with no flat-wall analog return None → no smooth panel:
#   * dispersive velocities / stresses / vorticity  → identically 0 (no x-var)
#   * potential temperature / pressure              → not loaded / neutral
#   * instantaneous fluctuation planes              → no such smooth data
def _smooth_field_2d(field_key):
    if not _smooth_loaded:
        return None
    # Flat-wall reference is x-homogeneous, so ONLY mean / single-point statistics
    # have a smooth analog here (dispersive & instantaneous fields are identically
    # zero on a flat wall and are intentionally absent).  rP_s is the mean pressure
    # profile (None if the .nc lacked rP); rV_s (wall-normal) is ~0 by construction
    # (mean vertical velocity vanishes in a statistically-steady flat Ekman layer).
    #   AvgScal  -> rs_s      : mean scalar ⟨s⟩ (raw Boussinesq solution, tlab `rs`;
    #                           NOT the derived buoyancy `rB`).  ≡0 in the neutral
    #                           (ri00.00) reference -> a blank panel until a
    #                           stratified flat-wall .nc is available.
    #   DispVel* -> Disp_*_s  : temporal-deviation proxy (see the smooth-load block);
    #                           the flat wall has no true spatial dispersive field.
    # ── Stress families on the flat wall ────────────────────────────────────
    # The .nc stores ONLY the single-point Reynolds stresses ⟨u'_i u'_j⟩
    # (Rxx/Rxy/Rxz/Ryy/Ryz/Rzz) — it carries NO triple decomposition.  A flat
    # wall is x-homogeneous, so the dispersive stress ũ_iũ_j ≡ 0 and hence the
    # turbulent part ⟨u''_i u''_j⟩ is not an independent quantity there.
    # Therefore:
    #   reyn_*  (Reynolds stress)   -> R** directly           [smooth panel OK]
    #   tot_*   (total momentum)    -> ⟨u_i⟩⟨u_j⟩ + R**       [smooth panel OK]
    #                                  (same form as the orographic case, whose
    #                                   turbulent part equals R** on a flat wall)
    #   rey_*   (turbulent stress)  -> NOT DEFINED here       [no smooth panel]
    #   *_disp  (dispersive stress) -> identically 0          [no smooth panel]
    # Engineering index -> .nc axis: u=x (streamwise), v=y (wall-normal),
    # w=z (spanwise); means rU_s / rV_s / rW_s respectively.
    _R  = {'uu': Rxx_s, 'uv': Rxy_s, 'uw': Rxz_s,
           'vv': Ryy_s, 'vw': Ryz_s, 'ww': Rzz_s}
    _M  = {'u': rU_s, 'v': rV_s, 'w': rW_s}
    if field_key.startswith('reyn_'):
        return _R.get(field_key[5:])
    if field_key.startswith('tot_'):
        _ek = field_key[4:]
        _r, _a, _b = _R.get(_ek), _M.get(_ek[:1]), _M.get(_ek[1:2])
        if _r is None or _a is None or _b is None:
            return None
        return _a * _b + _r                     # ⟨u_i⟩⟨u_j⟩ + ⟨u'_i u'_j⟩

    return {
        'AvgPhU': rU_s, 'AvgPhV': rV_s, 'AvgPhW': rW_s, 'AvgP': rP_s,
        'AvgScal': rs_s,
        'DispVelU': Disp_U_s, 'DispVelV': Disp_V_s, 'DispVelW': Disp_W_s,
        'TKE':    TKE_s,
        # Product-of-phase-average (mean-flow) stresses.  Flat wall is
        # x-homogeneous so ⟨q⟩≡q̄; only the terms without the ~0 wall-normal
        # mean (rV_s) have a real analog → ⟨u⟩² and ⟨v⟩²(spanwise, rW_s²).
        'PhUU_mean': (rU_s * rU_s) if rU_s is not None else None,
        'PhWW_mean': (rW_s * rW_s) if rW_s is not None else None,
    }.get(field_key)

def _panel_grid_shape(n):
    """2-D panel layout for `n` side-by-side case panels.

    One row up to 3 panels; otherwise two rows filled top-heavy so the extra
    panel goes on the top row (6 -> 2x3, 5 -> 3 above + 2 below, 4 -> 2x2,
    7 -> 4 above + 3 below, ...).  Returns (nrows, ncols)."""
    if n <= 3:
        return 1, n
    return 2, int(np.ceil(n / 2.0))

def _overlay_iso_contours(ax, x, z, fld, vmin, vmax,
                          n_contours=8, fmt='%.3g'):
    """Overlay black iso-contour ('solenoid') lines on a filled panel and label
    each loop near its TOP (its max-z vertex) so the value is easy to read —
    matplotlib's default clabel placement often lands at the bottom of the loop.
    SOLID lines mark positive contour values, DOTTED (':') mark negative ones;
    linewidth is 0.65.  Cells set to NaN (e.g. inside the IBM solid) are skipped
    by contour, so no spurious lines hug the staircase boundary."""
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax <= vmin:
        return
    # 'Nice' round interior levels (drop the two endpoints at the colour limits).
    levels = MaxNLocator(nbins=n_contours + 1).tick_values(vmin, vmax)
    levels = levels[(levels > vmin) & (levels < vmax)]
    if levels.size == 0:
        return
    linestyles = [':' if lv < 0 else '-' for lv in levels]   # solid +, dotted −
    CS = ax.contour(x, z, fld, levels=levels, colors='k',
                    linewidths=0.65, linestyles=linestyles, zorder=4)
    # Manual label positions = the top (max-z) vertex of every non-trivial loop,
    # so each line is annotated where it is easiest to read.
    manual = []
    try:
        for segs in CS.allsegs:                  # one list of loops per level
            for seg in segs:
                if seg.shape[0] < 6:             # skip tiny fragments
                    continue
                jtop = int(np.argmax(seg[:, 1]))
                manual.append((seg[jtop, 0], seg[jtop, 1]))
    except AttributeError:
        manual = []                              # mpl without .allsegs → auto
    ax.clabel(CS, fmt=fmt, fontsize=6, inline=True, inline_spacing=2,
              manual=manual if manual else None)


# ── Colour-scale policy, shared by plot2D_allRe and plot2D_div_allcases ──────
# (1) ZERO-CENTRED DIVERGING SCALES.  Whenever a diverging colormap is used on a
#     field that really goes negative, the limits are made SYMMETRIC about zero
#     (vmin = -V, vmax = +V, V = max|field|).  The neutral colour therefore always
#     means exactly 0, and the mapping stays LINEAR — equal colour distance is
#     equal value distance on both sides of zero.  This holds whether the panels
#     share a scale or not.  Because each panel is symmetric, the global min/max
#     over panels is symmetric too, so a shared scale is centred automatically.
#     The price is that the colorbar ends at ±V rather than at the true [min,max];
#     the true data range is printed in the colorbar label instead (_range_note).
# (2) AUTOMATIC SHARED SCALE.  shared_scale='auto' shares one scale when every
#     panel's magnitude V falls in the SAME power-of-ten decade, and falls back to
#     per-panel scales when the cases differ by more than a decade (a shared scale
#     would then flatten the weaker panels into a single colour).
#     NB: the decade test is a hard boundary — panels at V=0.99 and V=1.01 are 2 %
#     apart but sit in different decades and will be split.  Pass shared_scale=True
#     / False to override on any figure where that matters.
def _panel_limits(fld, cmap_name, epsilon=1e-4):
    """(vmin, vmax) for ONE panel, zero-centred when diverging (see policy above)."""
    _fmin = float(np.nanmin(fld))
    _fmax = float(np.nanmax(fld))
    if (cmap_name in _DIVERGING_CMAPS) and _fmin < -epsilon:
        _v = max(abs(_fmin), abs(_fmax))          # symmetric → white == 0, linear
        return -_v, _v
    return (0.0 if _fmin >= -epsilon else _fmin), _fmax

def _scale_mag(vmin, vmax):
    """Characteristic magnitude of a panel's colour scale (used by the decade test)."""
    return max(abs(vmin), abs(vmax))

def _same_decade(mags):
    """True when every non-zero magnitude shares one power-of-ten decade."""
    _m = [abs(float(v)) for v in mags if np.isfinite(v) and abs(v) > 0.0]
    if len(_m) < 2:
        return True
    _d = [int(np.floor(np.log10(v))) for v in _m]
    return max(_d) == min(_d)

def _resolve_shared(shared_scale, mags, what=''):
    """Resolve shared_scale True / False / 'auto' to a bool (see policy above)."""
    if shared_scale != 'auto':
        return bool(shared_scale)
    _sh = _same_decade(mags)
    if not _sh:
        print('  [auto-scale] %s: panel magnitudes span >1 decade '
              '(%s) → per-panel scales' % (what, ', '.join('%.3g' % m for m in mags)))
    return _sh

def _range_note(cbar_label, dmin, dmax):
    """Colorbar label carrying the TRUE data range, since a symmetric zero-centred
    scale ends at ±V rather than at [min, max]."""
    _note = r'data range: [%.3g, %.3g]' % (dmin, dmax)
    return _note if not cbar_label else '%s\n%s' % (cbar_label, _note)


# ── Wall-unit normalization of the 2-D phase-average field maps ──────────────
# Every 2-D comparison map is drawn in the case's OWN wall (inner) units, so the
# colour value is dimensionless in u* rather than in geostrophic (u/G) units.
# The field is multiplied by  pre / u*_case**unorm  (smooth panel by ustr_s1).
# Powers by physical dimension:
#   velocity          u_i           -> /u*        (unorm=1)
#   velocity^2        p, TKE, ⟨u_iu_j⟩, ũ_iũ_j, ⟨u_i⟩⟨u_j⟩  -> /u*^2   (unorm=2)
#   vorticity         ω_y           -> ·ν/u*^2    (pre=ν, unorm=2)
#   scalar/buoyancy, streamfunction, Poisson source, turning ANGLE:
#       no single clean u* power (θ is a 1→0 deficit; ψ/source need ν too) ->
#       left in their stored units (unorm=0).  See the note printed at run time.
def _wall_power(field_key):    # [U*-NORM] field-map wall-unit classifier
    """(pre, unorm): plotted = pre * field / u*^unorm.  unorm=0 → unchanged."""
    k = field_key
    if k in ('vort_z', 'disp_vortz'):
        return (nu, 2)                       # ω·ν/u*^2
    if k in ('AvgPhU', 'AvgPhV', 'AvgPhW',
             'DispVelU', 'DispVelV', 'DispVelW',
             'inst_u', 'inst_v', 'inst_w'):
        return (1.0, 1)                      # velocity → u+
    if k in ('AvgP', 'TKE'):
        return (1.0, 2)                      # p/ρu*^2 ; k/u*^2
    if k.endswith('_mean') and k.startswith('Ph'):
        return (1.0, 2)                      # mean-flow stress ⟨u_i⟩⟨u_j⟩
    if k.startswith(('tot_', 'reyn_', 'rey_')):
        return (1.0, 2)                      # total / Reynolds / turbulent stress
    if k.endswith('_disp') and len(k) == 7:  # UU_disp … WW_disp (dispersive stress)
        return (1.0, 2)
    return (1.0, 0)                          # scalar / advanced / angle: unchanged

def _wall_note(pre, unorm):
    """Colorbar suffix describing the wall-unit scaling applied."""
    if unorm == 0:
        return ''
    if pre is nu or pre == nu:
        return r'$\,\nu/u_*^2$'
    return r'$/u_*$' if unorm == 1 else r'$/u_*^{%d}$' % unorm

def plot2D_allRe(field_key, suptitle, cmap_name, savename,
                 ylim=None, use_inner=True, cbar_label=None,
                 include_smooth=True, shared_scale='auto',
                 overlay_contours=False, n_contours=8, contour_fmt='%.3g'):
    if ylim is None:
        ylim = limity
    # Case selection is now EXPLICIT (see _case_availability): every declared
    # case is classified as drawn or absent-with-a-reason.  The old form here,
    #     _avail = [... for cn, lb in zip(SIM_NAMES, SIM_LABELS)
    #               if gv(field_key, cn) is not None]
    # dropped a case whose pickle lacked this one field WITHOUT SAYING SO — and
    # cases already dropped upstream (no pickle / commented-out source) never
    # reached the zip at all — which is exactly how a case whose data exists
    # still misses every panel of a figure.
    _avail, _absent = _case_availability(field_key)
    for _cn_a, _why_a, _case_lvl in _absent:
        _note_missing(savename, field_key, _cn_a, _why_a, once=_case_lvl)
    if not _avail:
        print('plot2D_allRe: NO case supplied %r — %s NOT written'
              % (field_key, savename))
        return
    print('  %-32s %-16s drawn: %s%s'
          % (savename, field_key,
             ', '.join(_cn for _cn, _ in _avail),
             ('  |  MISSING: ' + ', '.join(_c for _c, _, _ in _absent))
             if _absent else ''))

    # [U*-NORM] Per-case wall-unit scaling (see _wall_power): each panel is divided
    # by that case's own u*^unorm so the comparison is in inner units, not
    # geostrophic.  Applied to EVERY plot2D_allRe field map (P01,P02,… velocity,
    # pressure, TKE, stresses, vorticity) — that is why the individual
    # plot2D_allRe('AvgPhU',…) call sites below are unchanged: the scaling lives
    # here in the helper, not at each call.
    _pre, _un = _wall_power(field_key)
    def _wscale(_cn):
        return (_pre / norm_ustar(_cn) ** _un) if _un else 1.0
    # (the smooth panels scale by the u* of the reference being drawn — see
    # _sm_scale_r inside the smooth_refs() loop below)
    if _un and cbar_label is None:
        cbar_label = _wall_note(_pre, _un)

    # All contour panels share the common z+=800 extent (ylim retained for
    # API compatibility but no longer sets the wall-normal crop).
    _zmax = _contour_zmax(use_inner)

    # Unified panel list: (label, x, z_full, field_full, jclip, xfill, yfill).
    # The smooth flat-wall reference (when a 2D analog exists) leads, then the
    # rough Fr cases each on their own grid (a common z-extent _zmax keeps the
    # panels comparable across grids).
    _panels = []
    # ONE panel per loaded smooth reference (each on its own grid + own u*),
    # then the valley cases.  smooth_refs() rebinds the flat *_s names, so the
    # body below is the single-reference code unchanged.
    if include_smooth:
        for _ in smooth_refs():
            _sm_arr = _smooth_field_2d(field_key)
            if _sm_arr is None:
                continue
            # The flat reference must land on the SAME KIND of axis as the
            # valley panels beside it, per Z_NORM:
            #   'inner' -> its own wall units y_in_s = y_s/(nu_s/ustr_s1)
            #   'outer' -> its own Ekman depth  y_s/(ustr_s1/f)
            #   'topo'  -> the REFERENCE valley height (it has no valley of its
            #              own), so "z/h" means the same length across the row.
            # It previously handled only 'topo' and fell through to y_in_s for
            # BOTH 'inner' and 'outer' — so in outer mode this panel was drawn
            # in INNER units (z+ up to ~2700) next to panels on z- (0…1.7).
            if not use_inner:
                _zs = y_s
            elif Z_NORM == 'topo':
                _zs = y_s / float(y[hill_hgt])
            elif Z_NORM == 'outer':
                _zs = y_s / (float(ustr_s1) / float(f))
            else:
                _zs = y_in_s
            _sm_scale_r = (_pre / ustr_s1 ** _un) if _un else 1.0
            _panels.append((SMOOTH_LABEL, sx, _zs,
                            _sm_arr * _sm_scale_r if _un else _sm_arr,  # [U*-NORM] smooth panel
                            _clip_rows(_zs, _zmax), np.array([]), np.array([]), None))
    for cn, lb in _avail:
        _xp, _yp, _xo, _yo = _case_grid(cn, use_inner)
        _fld_cn = gv(field_key, cn)
        _panels.append((lb, _xp, _yp,
                        _fld_cn * _wscale(cn) if _un else _fld_cn,   # [U*-NORM] per-case panel
                        _clip_rows(_yp, _zmax), _xo, _yo, geps(cn)))

    npan = len(_panels)
    nrows, ncols = _panel_grid_shape(npan)
    # sharey=False: each panel has its own grid; a common z-extent (_zmax)
    # keeps them visually comparable across different grids.  constrained_layout
    # spaces the grid + per-panel colorbars without manual tuning.
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.6 * ncols, 5.0 * nrows),
                             sharey=False, squeeze=False,
                             constrained_layout=True)
    _axflat = axes.ravel()

    # Colour limits per the policy above: every panel is zero-centred when the
    # map is diverging; the panels then either share one scale (+ one figure-wide
    # colorbar) or keep their own (+ a colorbar each).
    _crop = [_p[3][:_p[4], :] for _p in _panels]
    _lims = [_panel_limits(_f, cmap_name) for _f in _crop]
    _shared = _resolve_shared(shared_scale,
                              [_scale_mag(*_l) for _l in _lims], savename)
    _fb = 1 if _shared else 0   # +1 font bump on the shared-scale figures
    if _shared:
        # Panel limits are already symmetric for a diverging map, so the global
        # min/max over them is symmetric too → the shared scale is zero-centred.
        _gmin = min(_l[0] for _l in _lims)
        _gmax = max(_l[1] for _l in _lims)
        # True data range across all panels — reported in the colorbar label,
        # because a symmetric scale's end ticks are ±V, not the real extremes.
        _dmin = min(float(np.nanmin(_f)) for _f in _crop)
        _dmax = max(float(np.nanmax(_f)) for _f in _crop)

    _pcm = None
    for _i, _pan in enumerate(_panels):
        lbl, _xp, _zp, _fld_full, _jl, _xo, _yo = _pan[:7]
        _eps = _pan[7] if len(_pan) > 7 else None
        ax = _axflat[_i]
        _fld = _crop[_i]
        _vmin, _vmax = (_gmin, _gmax) if _shared else _lims[_i]
        _pcm = ax.pcolormesh(_xp, _zp[:_jl], _fld, cmap=cmap_name,
                             vmin=_vmin, vmax=_vmax, shading='auto')
        if _eps is not None:
            _shade_ibm(ax, _xp, _zp[:_jl], _eps[:_jl, :])   # true eps==1 region
        elif len(_xo) > 0:
            ax.fill(_xo, _yo, color=_IBM_COLOR)
        if overlay_contours:
            # NaN-out the IBM solid so contours stay in the fluid, then overlay
            # labelled black iso-contour lines (labels at each loop's top).
            _fld_c = _fld
            if _eps is not None:
                _fld_c = np.where(_eps[:_jl, :] >= 0.5, np.nan, _fld)
            _overlay_iso_contours(ax, _xp, _zp[:_jl], _fld_c, _vmin, _vmax,
                                  n_contours=n_contours, fmt=contour_fmt)
        ax.set_ylim(0, _zmax)
        ax.set_title(lbl, fontsize=9 + _fb)
        ax.set_xlabel(_x_axis_label() if use_inner else r'$x$', fontsize=10 + _fb)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(_z_axis_label() if use_inner else r'$z$', fontsize=10 + _fb)
        if _fb:
            ax.tick_params(labelsize=10 + _fb)   # axis scale numbers
        if not _shared:
            # One colorbar per panel (own zero-centred scale); its own true range
            # is on the label, since ±V hides the weaker of the two extremes.
            _cb = fig.colorbar(_pcm, ax=ax, orientation='vertical',
                               shrink=0.9, pad=0.02)
            _cb.set_label(_range_note(cbar_label, float(np.nanmin(_fld)),
                                      float(np.nanmax(_fld))), fontsize=LEGEND_FONTSIZE)
            _cb.ax.tick_params(labelsize=7)

    # Blank any unused grid cells (e.g. the 6th slot when 5 panels present)
    for _j in range(npan, nrows * ncols):
        _axflat[_j].axis('off')

    if _shared and _pcm is not None:
        # Single figure-wide colorbar shared by every panel.  Ticks span the
        # symmetric scale (so 0 is always a tick); the TRUE global [min, max] is
        # spelled out on the label.
        _cb = fig.colorbar(_pcm, ax=list(_axflat[:npan]),
                           orientation='vertical', shrink=0.9, pad=0.02)
        _cb.set_ticks(np.linspace(_gmin, _gmax, 5))
        _cb.set_label(_range_note(cbar_label, _dmin, _dmax), fontsize=8 + _fb)
        _cb.ax.tick_params(labelsize=7 + _fb)   # legend (colorbar) numbers

    fig.suptitle(suptitle, fontsize=11 + _fb)
    _out = _figdir + savename
    fig.savefig(_out, dpi=FIG_DPI, bbox_inches='tight')
    print(f'Saved: {_out}')
    # Release the figure (and the panel copies matplotlib holds) — this script
    # draws dozens of multi-panel figures; leaving them open pins every field
    # array for the whole run.  The PNG is already on disk.
    plt.close(fig)
    _panels.clear()

# _shade_ibm / plot2D_div_allcases use _IBM_COLOR (set in the plotRes block).
def _shade_ibm(ax, _x, _y, eps_arr):
    """Shade EXACTLY the eps==1 region (per-case eps) — not the analytic
    cosine polygon x_oro/y_oro, which can miss the true staircase boundary."""
    if eps_arr is None:
        return
    try:
        if np.nanmax(eps_arr) >= 0.5:
            ax.contourf(_x, _y, eps_arr, levels=[0.5, 1.5],
                        colors=[_IBM_COLOR], zorder=5)
    except (ValueError, TypeError):
        pass

def plot2D_div_allcases(panels, field_label, suptitle, savename, cmap='seismic',
                        xname=r'$x$', yname=r'$z$', ylim_top=None,
                        zero_contour=False, vmax_pct=None,
                        shared_scale='auto', overlay_contours=False,
                        n_contours=8, contour_fmt='%.3g'):
    """Plot multiple 2D diverging fields side-by-side.

    panels : (label, x_arr, y_arr, field_arr[ny,nx], xfill, yfill[, eps_arr])
             An optional 7th element eps_arr (cropped to the same rows as
             field_arr) shades the true solid region and blanks it to NaN;
             without it the polygon (xfill,yfill) is filled instead.
    xname/yname  : axis labels (default physical x/z; pass x+/z+ for inner).
    ylim_top     : if set, cap every panel's wall-normal axis at this value.
    zero_contour : draw the field = 0 isoline (used for the separation shear).
    vmax_pct     : if set, clip the symmetric scale to this percentile of
                   |field| (keeps small-signed pockets visible when the field
                   has a large near-wall extreme).
    shared_scale : True / False / 'auto' — one symmetric scale spanning ALL
                   panels + a single figure-wide colorbar, instead of a
                   per-panel scale + colorbar.  'auto' shares when every panel's
                   magnitude sits in the same power-of-ten decade.
    overlay_contours : labelled black iso-contour lines on every panel
                   (same convention as plot2D_allRe: solid +, dotted −).

    Every scale here is symmetric about zero (this is a diverging-field plotter),
    so the neutral colour always means exactly 0 and the mapping stays linear.
    """
    n = len(panels)
    if n == 0:
        return
    nrows, ncols = _panel_grid_shape(n)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.6 * ncols, 5.0 * nrows),
                             squeeze=False, constrained_layout=True)
    _axflat = axes.ravel()

    # Blank the solid to NaN up-front so masked-to-0 derivatives neither set the
    # colour scale nor draw a false 0-isoline along the body boundary.
    _flds = []
    for _pan in panels:
        fld = _pan[3]
        _eps = _pan[6] if len(_pan) > 6 else None
        _flds.append(np.where(_eps > 0.5, np.nan, fld) if _eps is not None else fld)

    def _sym_vmax(_f):
        if vmax_pct is not None:
            return float(np.nanpercentile(np.abs(_f), vmax_pct)) or 1.0
        return max(abs(np.nanmin(_f)), abs(np.nanmax(_f))) or 1.0

    # ONE symmetric scale across every panel, or a symmetric scale per panel so
    # small-signed structure is not washed out.  Either way 0 is at the neutral
    # colour.  'auto' → share only while all panels sit in the same decade.
    _vmaxes = [_sym_vmax(_f) for _f in _flds]
    _shared = _resolve_shared(shared_scale, _vmaxes, savename)
    _gvmax = max(_vmaxes) if _shared else None

    _pcm = None
    for _i, _pan in enumerate(panels):
        lbl, _x, _y, _, xfill, yfill = _pan[:6]
        _eps = _pan[6] if len(_pan) > 6 else None
        fld = _flds[_i]
        ax = _axflat[_i]
        _vmax = _gvmax if _shared else _vmaxes[_i]
        _vmin = -_vmax
        _pcm = ax.pcolormesh(_x, _y, fld, cmap=cmap,
                             vmin=_vmin, vmax=_vmax, shading='auto')
        if zero_contour:
            ax.contour(_x, _y, fld, levels=[0.0], colors='k',
                       linewidths=0.3, linestyles=':')
        if overlay_contours:
            _overlay_iso_contours(ax, _x, _y, fld, _vmin, _vmax,
                                  n_contours=n_contours, fmt=contour_fmt)
        if _eps is not None:
            _shade_ibm(ax, _x, _y, _eps)          # true eps==1 region
        elif len(xfill) > 0:
            ax.fill(xfill, yfill, color=_IBM_COLOR)
        if ylim_top is not None:
            ax.set_ylim(0, ylim_top)
        ax.set_title(lbl, fontsize=9)
        ax.set_xlabel(xname)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(yname)
        if not _shared:
            _cb = fig.colorbar(_pcm, ax=ax, orientation='vertical',
                               shrink=0.9, pad=0.02)
            _cb.set_label(_range_note(field_label, float(np.nanmin(fld)),
                                      float(np.nanmax(fld))), fontsize=LEGEND_FONTSIZE)
            _cb.ax.tick_params(labelsize=7)
    for _j in range(n, nrows * ncols):
        _axflat[_j].axis('off')
    if _shared and _pcm is not None:
        _cb = fig.colorbar(_pcm, ax=list(_axflat[:n]),
                           orientation='vertical', shrink=0.9, pad=0.02)
        _cb.set_ticks(np.linspace(-_gvmax, _gvmax, 5))
        _dmin = min(float(np.nanmin(_f)) for _f in _flds)
        _dmax = max(float(np.nanmax(_f)) for _f in _flds)
        _cb.set_label(_range_note(field_label, _dmin, _dmax), fontsize=8)
        _cb.ax.tick_params(labelsize=7)
    fig.suptitle(suptitle, fontsize=11)
    _out = _figdir + savename
    fig.savefig(_out, dpi=FIG_DPI, bbox_inches='tight')
    print(f'Saved: {_out}')
    plt.close(fig)          # free the figure + its copies of the panel arrays
    _flds.clear()

# Inner-scaled with THIS case's own measured u*: nu/norm_ustar(cn)^3.
def _gradP_case(cn, axis):
    """Wall-normalised mean-pressure gradient for case `cn`.
    axis='x' -> dP/dx (adverse pressure gradient); axis='y' -> dP/dz+ (eng dP/dy).
    Uses the pickled compact-scheme field when present, else a centred
    difference of AvgP on the case's own grid; zeroed inside the solid."""
    _key = 'dP_dx' if axis == 'x' else 'dP_dy'
    _g   = gv(_key, cn)
    if _g is None:
        _P = gv('AvgP', cn)
        if _P is None:
            return None
        _xc = sims.get(cn, {}).get('x', x)
        _yc = sims.get(cn, {}).get('y', y)
        _g  = (np.gradient(_P, _xc, axis=1) if axis == 'x'
               else np.gradient(_P, _yc, axis=0)) * gmask0(cn)
    return _g * (nu / norm_ustar(cn)**3)

def _lyr_idx(y):
    """Return (i_lo, i_hi) index pairs for each layer in array y.
    Buffer ends at 37, log starts at 30 (overlap by design)."""
    i5   = int(np.searchsorted(y,   5, side='left'))
    i30  = int(np.searchsorted(y,  30, side='left'))
    i37  = int(np.searchsorted(y,  37, side='left'))
    i130 = int(np.searchsorted(y, 130, side='left'))
    return [(0, i5), (i5, i37), (i30, i130), (i130, len(y))]

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

# _save_layers_x/_y iterate _LYR_NAMES/_LYR_TITLES/_LYR_XLIMS (set in plotRes).
def _save_layers_x(base_path, base_title, is_log=False):
    """Save one zoomed PNG per BL layer; z+ on x-axis.
    Both axes are rescaled to the data within each layer window."""
    _ax    = plt.gca()
    _xlim0 = _ax.get_xlim()
    _ylim0 = _ax.get_ylim()
    for _ln, _lt, (_x0, _x1) in zip(_LYR_NAMES, _LYR_TITLES, _LYR_XLIMS):
        _xlo = max(_x0, float(_axis_start_zplus())) if is_log else _x0
        _xhi = min(_x1, _xlim0[1]) if _x1 is not None else _xlim0[1]
        if _xlo >= _xhi:
            continue
        _ax.set_xlim(_xlo, _xhi)
        _autoscale_y(_ax, _xlo, _xhi)
        plt.title(f'{base_title}\n{_lt}', fontsize=9)
        plt.savefig(f'{base_path}_{_ln}.png', dpi=FIG_DPI)
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
        plt.savefig(f'{base_path}_{_ln}.png', dpi=FIG_DPI)
    _ax.set_ylim(*_ylim0)
    _ax.set_xlim(*_xlim0)
    plt.title(base_title)


def obu_up_of_xi(xi):
    """Auxiliary parameter u'(xi), stable & unstable, via monotone interp."""
    return np.interp(np.asarray(xi, float), _OBU_XI, _OBU_U)


def obu_eta_of_xi(xi):
    """eta(xi) = Ri/Ri_cr = 1 - u'^4  (Table III column 'eta')."""
    return 1.0 - obu_up_of_xi(xi)**4


def obu_psi(xi):
    """Wind function psi(xi) = (k/v*) v = ln|xi| + psi_hat(xi)  (Table III 'psi').

    Additive constant is arbitrary (it is absorbed by the fit's offset / z0);
    as xi -> 0 psi -> ln|xi|, so the modified law reduces to the neutral log law
    in the unstratified limit (L1 -> inf)."""
    xi = np.asarray(xi, float)
    psih = np.interp(xi, _OBU_XI, _OBU_PSIH)
    with np.errstate(divide='ignore'):
        return np.log(np.abs(xi)) + psih


def obu_wind_profile(z, v_star, L1, offset, kappa=None):
    """Modified log-law wind  v(z) = (v*/k) psi(z/L1) + offset.
    L1 > 0 stable, L1 < 0 unstable (sign of xi follows sign of L1).
    kappa=None -> the user parameter OBU_KAPPA, read at CALL time (it is
    declared in the USER PARAMETERS section, which is below this one)."""
    kappa = OBU_KAPPA if kappa is None else kappa
    return (v_star/kappa)*obu_psi(np.asarray(z, float)/L1) + offset


def fit_modified_loglaw(z, u, kappa=None, L1_0=None):
    """Nonlinear least-squares fit (scipy curve_fit) of the modified log law
        u(z) = (v*/k) psi(z/L1) + offset
    for (v_star, L1, offset).  Returns a dict (or None if too few points).
    Sign of L1_0 seeds the stable(+)/unstable(-) branch."""
    kappa = OBU_KAPPA if kappa is None else kappa   # user param, read at call time
    z = np.asarray(z, float); u = np.asarray(u, float)
    good = np.isfinite(z) & np.isfinite(u) & (z > 0)
    z, u = z[good], u[good]
    if z.size < 4:
        return None
    if L1_0 is None:
        L1_0 = 3.0*float(z.max())                        # weak-stratification seed
    _d = obu_psi(z[-1]/L1_0) - obu_psi(z[0]/L1_0)
    v0   = kappa*(u[-1] - u[0]) / (_d if abs(_d) > 1e-6 else 1e-6)
    off0 = u[0] - (v0/kappa)*obu_psi(z[0]/L1_0)

    def _model(zz, vs, L1, off):
        return (vs/kappa)*obu_psi(np.asarray(zz, float)/L1) + off
    try:
        popt, pcov = curve_fit(_model, z, u, p0=[abs(v0), L1_0, off0], maxfev=20000)
    except Exception as _e:
        return {'ok': False, 'err': str(_e)}
    resid  = u - _model(z, *popt)
    ss_res = float(np.sum(resid**2)); ss_tot = float(np.sum((u - u.mean())**2))
    r2 = (1.0 - ss_res/ss_tot) if ss_tot > 0 else float('nan')
    return {'ok': True, 'v_star': float(popt[0]), 'L1': float(popt[1]),
            'offset': float(popt[2]), 'r2': float(r2),
            'perr': np.sqrt(np.diag(pcov)).tolist()}


def validate_obukhov_tableIII(verbose=True, tol_eta=0.02, tol_psi=0.06):
    """Reproduce Obukhov (1971) Table III (eta and psi vs xi) from the solver.
    Unit-independent (all quantities dimensionless in the paper).  psi carries an
    arbitrary additive constant, so it is compared after a single best-fit shift
    C (= Obukhov's integration constant, ~4.6)."""
    xi, eta_t, psi_t = _OBU_TBL3[:, 0], _OBU_TBL3[:, 1], _OBU_TBL3[:, 2]
    eta_c = obu_eta_of_xi(xi)
    psi_c = obu_psi(xi)
    C     = float(np.mean(psi_t - psi_c))
    e_eta = np.abs(eta_c - eta_t)
    e_psi = np.abs(psi_c + C - psi_t)
    ok = (e_eta.max() < tol_eta) and (float(np.sqrt(np.mean(e_psi**2))) < tol_psi)
    if verbose:
        print(f"[Obukhov Table III] eta(xi): max|err|={e_eta.max():.4f} "
              f"RMS={np.sqrt(np.mean(e_eta**2)):.4f}")
        print(f"[Obukhov Table III] psi(xi): shift C={C:.3f} max|err|={e_psi.max():.4f} "
              f"RMS={np.sqrt(np.mean(e_psi**2)):.4f}")
        print(f"[Obukhov Table III] {'PASS' if ok else 'FAIL'}  "
              f"(k={OBU_KAPPA}, stable+unstable parametric solver)")
    return ok


# _loglaw_fit_case uses the fit-window/κ/Ri_cr constants (_LL_*) set in plotRes.
def _ll_cumtrapz0(_fv, _xv):
    """Cumulative trapezoid of _fv over _xv, starting at 0 (matches PhAvg)."""
    return np.concatenate(([0.0],
                           np.cumsum(0.5*(_fv[1:]+_fv[:-1])*np.diff(_xv))))

def _loglaw_fit_case(case):
    """Fit the Froude-dependent wall law to one case's rotated mean profile.
    Returns a dict (κ, d⁺, z₀ₘ⁺, B, R², law, ⟨Ri⟩, Ri_max, u★, and the fitted
    curve on the shared reference axis for overlay) or None if the case lacks
    the profile / grid."""
    _upr = gv('u_plus_rot', case)
    _yg  = gv('y', case)
    if _upr is None or _yg is None:
        return None
    # Own inner units use the SAME per-case scale as gy_in/gx_in (norm_ustar =
    # the case's measured Method-2 plateau), so this fit's z_own/u_own overlay
    # lands exactly on the self-scaled profile axes.
    _uc  = norm_ustar(case)
    _zown = _yg * _uc / case_nu(case)      # z⁺ in THIS case's own inner units
    _uown = _upr / _uc                     # u⁺ in own units
    _msk  = (_zown >= _LL_ZMIN) & (_zown <= _LL_ZMAX)
    _zf, _uf = _zown[_msk], _uown[_msk]
    if _zf.size < 3:
        return None
    _Fr    = SIM_FR.get(case, np.inf)
    _strat = bool(np.isfinite(_Fr))
    # Full-profile φ(z⁺) so the abscissa can be evaluated on ANY window (the
    # narrow fit window AND the wider display window).  Neutral ⇒ φ ≡ 1.
    if _strat:
        _b = gv('AvgScal', case)
        if _b is not None:
            _b1d   = avg_c(geps(case), _b, axis=1) if np.ndim(_b) == 2 else _b
            _dbdzf = np.gradient(_b1d, _yg)
            _dudzf = np.gradient(_upr, _yg)
            with np.errstate(divide='ignore', invalid='ignore'):
                _Rif = _dbdzf / _dudzf**2
            _Rif = np.nan_to_num(_Rif, nan=0.0, posinf=_LL_RICR, neginf=0.0)
            _Rif = np.minimum(_Rif, 0.999*_LL_RICR)
            _phif = (1.0 - _Rif/_LL_RICR) ** (-0.25)
        else:                              # stratified case but no buoyancy pickled
            _strat = False
            _Rif = np.zeros_like(_yg); _phif = np.ones_like(_yg)
    else:
        _Rif = np.zeros_like(_yg); _phif = np.ones_like(_yg)
    _Ri, _phi = _Rif[_msk], _phif[_msk]

    _kap, _dm, _z0m, _B, _r2 = 0.41, 0.0, 0.068, np.nan, -np.inf
    for _d in np.linspace(0.0, 0.9*_LL_ZMIN, 1001):
        _zs = _zf - _d
        if np.any(_zs <= 0):
            break
        _absc = (np.log(_zs[0]) + _ll_cumtrapz0(_phi/_zs, _zf)) if _strat \
                else np.log(_zs)
        _sl, _ic, _rv, *_ = linregress(_absc, _uf)
        if _sl <= 0:
            continue
        _k = 1.0/_sl
        if not (_LL_KBND[0] <= _k <= _LL_KBND[1]):
            continue
        if _rv**2 > _r2:
            _r2, _kap, _dm, _B = _rv**2, _k, _d, _ic
            _z0m = np.exp(-_ic/_sl)         # z₀ₘ⁺ = exp(−B·κ)
    _ok = np.isfinite(_r2) and _r2 > -np.inf and np.isfinite(_B)
    # Fitted curve on a WIDE display window, mapped onto the shared single-
    # reference axis for overlay:  plotted u = u⁺_own·u★_case/ustr_s1  vs
    # z⁺_ref = y/l_in.  The stratified abscissa Ξ is referenced to the SAME
    # anchor used in the fit (z = z_fit[0]), so the display line coincides with
    # the fit over [_LL_ZMIN,_LL_ZMAX] and extrapolates the straight law beyond.
    _zref = _uref = None
    _zown_d = _uown_d = None                 # same fit curve in THIS case's own units
    if _ok:
        _dmask = ((_zown >= _LL_DISP_LO) & (_zown <= _LL_DISP_HI)
                  & ((_zown - _dm) > 1e-9))
        _zd = _zown[_dmask]
        if _zd.size >= 2:
            _zsd = _zd - _dm
            if _strat:
                _cum  = _ll_cumtrapz0(_phif[_dmask] / _zsd, _zd)
                _cum0 = np.interp(_zf[0], _zd, _cum)          # anchor at fit z₀
                _absc = np.log(_zf[0] - _dm) + (_cum - _cum0)
            else:
                _absc = np.log(_zsd)
            _ufit = (1.0/_kap)*_absc + _B
            _zref = (_yg/norm_l_in(case))[_dmask]
            _uref = _ufit * _uc / norm_ustar(case)
            _zown_d = _zd                    # z⁺ in own inner units (y·u★_case/ν)
            _uown_d = _ufit                  # u⁺ in own units (already u/u★_case)
    return {'kappa': _kap, 'd': _dm, 'z0m': _z0m, 'B': _B,
            'r2': (_r2 if _ok else np.nan),
            'law': ('Obukhov stratified' if _strat else 'neutral MOST'),
            'Ri_mean': float(np.mean(_Ri)), 'Ri_max': float(np.max(_Ri)),
            'u_star': _uc, 'Fr': _Fr, 'z_ref': _zref, 'u_ref': _uref,
            'z_own': _zown_d, 'u_own': _uown_d}

def _modloglaw_fit_case(case):
    """Obukhov (1971) MODIFIED log-law fit for one case (mirrors the per-run block
    in PhAvg_rotated.py, applied here across the Froude ladder):

        u⁺(z⁺) = (v*/k) psi(z⁺/L1⁺) + offset        (k = OBU_KAPPA = 0.4)

    GATED on SIM_FR: fitted ONLY for the stratified runs (finite Fr).  For the
    NEUTRAL run (Fr = ∞) it is skipped — psi → ln(z⁺) there, so the modified law
    collapses onto the classical log law already fitted by _loglaw_fit_case.

    Free parameters: v* (in this case's own u★ units — v*≈1 ⇔ the profile-implied
    friction velocity equals the Method-2 u★), L1⁺ (dynamic-turbulence scale in
    wall units; + stable / − unstable) and an additive offset (roughness/intercept).

    Fit window: from _LL_ZMIN up to ≈ the BL top δ⁺ — the curvature that pins L1
    lives at z⁺ ~ L1⁺, well above the neutral [_LL_ZMIN,_LL_ZMAX] window — capped
    at the data.  Fitted in the case's OWN inner units, exactly like the classical
    fit, and the display curve is also returned on the shared reference axis.

    Returns None when the case has no profile / is neutral / has too few points;
    otherwise a dict, with 'ok': False and 'err' if curve_fit itself failed."""
    _upr = gv('u_plus_rot', case)
    _yg  = gv('y', case)
    if _upr is None or _yg is None:
        return None
    _Fr = SIM_FR.get(case, np.inf)
    if not np.isfinite(_Fr):                 # neutral → modified law undefined
        return {'ok': False, 'skipped': True, 'Fr': _Fr,
                'err': 'neutral run (Fr=inf)'}
    # Same per-case inner scale as gy_in/gx_in/_loglaw_fit_case (norm_ustar).
    _uc  = norm_ustar(case)
    _zown = _yg * _uc / case_nu(case)        # z⁺ in THIS case's own inner units
    _uown = _upr / _uc                       # u⁺ in own units

    # δ⁺ = u★²/(f ν)  (f = 1 here), same expression as PhAvg_rotated.py.
    # ν is THIS case's viscosity (Reynolds ladder), not the global config nu.
    _nuc = case_nu(case)
    _delta_plus = float(_uc**2 / (f * _nuc)) if (f != 0 and _nuc != 0) \
        else float(np.nanmax(_zown))
    _fin  = np.isfinite(_uown) & np.isfinite(_zown)
    if not np.any(_fin):
        return None
    _hi   = min(float(np.nanmax(_zown[_fin])),
                max(3.0*_LL_ZMAX, 0.6*_delta_plus))
    _mask = (_zown >= _LL_ZMIN) & (_zown <= _hi) & _fin
    if np.count_nonzero(_mask) < 4:
        return {'ok': False, 'skipped': False, 'Fr': _Fr, 'err': '<4 pts',
                'z_lo': _LL_ZMIN, 'z_hi': _hi}
    # Seed the stable branch; curve_fit migrates to the unstable branch on its own
    # if the profile is better matched there (the surface buoyancy-flux sign is
    # not consulted, exactly as in PhAvg_rotated.py).
    _fit = fit_modified_loglaw(_zown[_mask], _uown[_mask])
    if _fit is None or not _fit.get('ok'):
        return {'ok': False, 'skipped': False, 'Fr': _Fr,
                'err': (_fit.get('err', '<4 pts') if _fit else 'no data'),
                'z_lo': _LL_ZMIN, 'z_hi': _hi}
    # Display curve over the fit window, in own units AND on the shared ref axis
    # (plotted u = u⁺_own·u★_case/ustr_s1  vs  z⁺_ref = y/l_in) — same mapping the
    # classical fit uses, so the two overlays are directly comparable.
    _zd    = _zown[_mask]
    _ufit  = obu_wind_profile(_zd, _fit['v_star'], _fit['L1'], _fit['offset'])
    return {'ok': True, 'skipped': False, 'Fr': _Fr, 'u_star': _uc,
            'v_star': _fit['v_star'], 'L1_plus': _fit['L1'],
            'offset': _fit['offset'], 'r2': _fit['r2'], 'perr': _fit['perr'],
            'delta_plus': _delta_plus, 'z_lo': _LL_ZMIN, 'z_hi': _hi,
            'z_own': _zd, 'u_own': _ufit,
            'z_ref': (_yg/norm_l_in(case))[_mask], 'u_ref': _ufit * _uc / norm_ustar(case)}

# Veer as a BOUNDED angle arctan2(w_rot,u_rot) in degrees (∈[-180,180]) directly
# from the rotated velocity components — the pickled `inst_alpha` is the RATIO
# w_plus_rot/u_plus_rot (a tangent) which diverges wherever u_plus_rot crosses
# zero (e.g. the strongly-stratified Fr=0.0015 run reverses near the surface).
# SIGN CONVENTION: w_plus_rot is pickled with the DISPLAY flip -⟨W_rot⟩ (see
# saveresults.py), so _veer_deg returns POSITIVE angles for the near-wall Ekman
# veer.  The D1 surface-veer plot (Ch6) uses the raw pickled AvgPhW instead and
# is therefore PHYSICAL (negative near the wall) — same quantity, opposite sign.
def _veer_deg(case):
    _u = gv('u_plus_rot', case); _w = gv('w_plus_rot', case)
    if _u is None or _w is None:
        return None
    return np.degrees(np.arctan2(_w, _u))

# _ch6set records reduced numbers into _ch6 (initialised in the plotRes block).
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

def _d2(f, c, ax):
    return np.gradient(np.gradient(f, c, axis=ax), c, axis=ax)
def _dxz(f, xc, yc):
    return np.gradient(np.gradient(f, xc, axis=1), yc, axis=0)

def _panels_zeta(compute_fn, title, cmap, savename, zmax_plus=None):
    # Default z+ cap resolved at CALL time (Z_PLUS_CONTOUR_MAX is set in the
    # plotRes block) — avoids a def-time dependency now that this lives at the
    # top of the file.  Both call sites pass zmax_plus explicitly anyway.
    if zmax_plus is None:
        zmax_plus = Z_PLUS_CONTOUR_MAX
    _av = []
    for cn, lb in zip(SIM_NAMES, SIM_LABELS):
        _r = compute_fn(cn)
        if _r is not None:
            _av.append((cn, lb, _r[0], _r[1]))
    if not _av:
        print(f'  [Ch6] no data for {savename}')
        return
    nrows, ncols = _panel_grid_shape(len(_av))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.6 * ncols, 5.0 * nrows),
                             squeeze=False, constrained_layout=True)
    _axflat = axes.ravel()
    # Per-panel symmetric scale + colorbar (own legend per panel).
    for _i, (cn, lb, _f, _zp) in enumerate(_av):
        ax = _axflat[_i]
        _vmax = float(np.nanmax(np.abs(_f))) or 1.0
        _mesh = ax.pcolormesh(gx_in(cn), _zp, _f, cmap=cmap,
                              vmin=-_vmax, vmax=_vmax, shading='auto')
        ax.set_ylim(0, zmax_plus); ax.set_title(lb, fontsize=9); ax.set_xlabel(r'$x^+$')
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(r'$\zeta^+$ (terrain-following)')
        fig.colorbar(_mesh, ax=ax, shrink=0.9, pad=0.02)
    for _j in range(len(_av), nrows * ncols):
        _axflat[_j].axis('off')
    fig.suptitle(title)
    fig.savefig(_figdir +savename, dpi=FIG_DPI, bbox_inches='tight'); plt.show()

def _tf_disp(cn):
    _du = gv('DispVelU', cn); _dv = gv('DispVelV', cn); _dw = gv('DispVelW', cn)
    _yc = gv('y', cn)
    if _du is None or _yc is None:
        return None
    _mag = np.sqrt(_du ** 2
                   + (_dv ** 2 if _dv is not None else 0.0)
                   + (_dw ** 2 if _dw is not None else 0.0))
    _f, _z = terrain_follow_remap(_mag, _yc, _yc[_surf_rows(cn)])
    return _f, _z / norm_l_in(cn)

def _tf_tauzx(cn):
    _U = gv('AvgPhU', cn); _ruv = gv('rey_uv', cn); _yc = gv('y', cn)
    _uvd = gv('UV_disp', cn)                       # dispersive ũṽ (None if absent)
    if _U is None or _ruv is None or _yc is None:
        return None
    # Momentum shear flux = full Reynolds stress ⟨u'v'⟩ = turbulent ⟨u''v''⟩ (rey_uv)
    # + dispersive ũṽ (UV_disp).  Was rey_uv (turbulent) alone.
    _rey = _ruv if _uvd is None else (_ruv + _uvd)
    # ν is THIS case's viscosity (Reynolds ladder) — the viscous part of the
    # stress is not a shared constant here as it was on the Froude ladder.
    _tau = case_nu(cn) * np.gradient(_U, _yc, axis=0) - _rey
    _f, _z = terrain_follow_remap(_tau, _yc, _yc[_surf_rows(cn)])
    return _f, _z / norm_l_in(cn)

def _load_interm_npz(case, which):
    # Intermittency .npz can live in a different dir than the case's pickle
    # (see INTERM_DIRS): the neutral run's γ comes from the separate same-grid
    # Fr = ∞ simulation.  Fall back to SIM_DIRS when there is no override.
    _d = INTERM_DIRS.get(case, SIM_DIRS.get(case))
    if _d is None:
        return None
    _f = _d + ('intermittency_%s.npz' % which)
    if not os.path.exists(_f):
        return None
    try:
        _z = np.load(_f, allow_pickle=True)
    except Exception as _e:
        print('[interm] could not read %s: %s' % (_f, _e))
        return None
    _names = ([str(_n) for _n in _z['field_names']]
              if 'field_names' in _z.files else ['gamma'])
    _out = {'axis_h': np.asarray(_z['axis_h'], float),
            'axis_v': np.asarray(_z['axis_v'], float),
            'meta':   str(_z['meta']) if 'meta' in _z.files else ''}
    for _n in _names:
        if _n in _z.files:
            _out[_n] = np.asarray(_z[_n], float)
    return _out

def _interm_cases(field, which):
    """[(label, color, ls, axis_h+, axis_v+, plane), …] for active cases whose
    intermittency_<which>.npz carries `field`.

    The axes are returned ALREADY IN INNER UNITS, divided here by that case's
    norm_l_in — the case name is in scope at this point but not in the tuple
    the caller unpacks.  (results.py divided by the single global l_in at the
    plotting site instead; on a Reynolds ladder the divisor is per-case.)"""
    _out = []
    for c in CASES:
        n = c['name']
        if n not in ACTIVE_CASES:
            continue
        d = _load_interm_npz(n, which)
        if d is None or field not in d:
            continue
        _li = norm_l_in(n)
        _out.append((c['label'], c['color'], c['ls'],
                     d['axis_h'] / _li, d['axis_v'] / _li, d[field]))
    return _out

def _interm_field_panels(field, which, cmap, suptitle, savename):
    """Side-by-side field(x⁺, z⁺) panels across active cases from the .npz."""
    _rows = _interm_cases(field, which)
    if not _rows:
        print('[interm] no %s data for %r — panels skipped.' % (which, field))
        return
    _isg  = field.startswith('gamma')          # γ / γ_b keep the fixed 0…1 scale
    _zmax = _contour_zmax(use_inner=True)
    npan  = len(_rows)
    nrows, ncols = _panel_grid_shape(npan)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.6 * ncols, 5.0 * nrows),
                             squeeze=False, constrained_layout=True)
    _axflat = axes.ravel()
    # γ is an intermittency FRACTION -> its 0…1 scale is meaningful and kept
    # fixed for every panel; all other fields get an own per-panel symmetric
    # scale so per-case structure is not washed out by a global min/max.
    for _i, (lbl, _clr, _ls, _xh, _xv, _pl) in enumerate(_rows):
        ax = _axflat[_i]
        if _isg:
            _vmin, _vmax = 0.0, 1.0
        else:
            _a = max(abs(np.nanmin(_pl)), abs(np.nanmax(_pl))) or 1.0
            _vmin, _vmax = -_a, _a
        # _xh/_xv already arrive in inner units from _interm_cases (per-case divisor).
        _pcm = ax.pcolormesh(_xh, _xv, _pl, cmap=cmap,
                             vmin=_vmin, vmax=_vmax, shading='auto')
        if _isg:
            ax.contour(_xh, _xv, np.nan_to_num(_pl),
                       levels=[0.5], colors='cyan', linewidths=1.0)
        ax.set_ylim(0, _zmax)
        ax.set_title(lbl, fontsize=9)
        ax.set_xlabel(r'$x^+$')
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(r'$z^+$')
        _cb = fig.colorbar(_pcm, ax=ax, orientation='vertical', shrink=0.9, pad=0.02)
        _cb.set_label(field, fontsize=8); _cb.ax.tick_params(labelsize=7)
    for _j in range(npan, nrows * ncols):
        _axflat[_j].axis('off')
    fig.suptitle(suptitle, fontsize=11)
    fig.savefig(_figdir + savename, dpi=FIG_DPI, bbox_inches='tight'); plt.show()
    print('Saved: ' + _figdir + savename)

def _interm_profile(fields, which, suptitle, savename, xlabel):
    """x-averaged (fluid-only) profile(s) vs z⁺, one line per active case.
    `fields` = list of (npz-key, linestyle, legend-suffix)."""
    import warnings
    fig, ax = plt.subplots(figsize=FIG_SIZE_TALL, dpi=FIG_DPI)
    _any = False
    for c in CASES:
        n = c['name']
        if n not in ACTIVE_CASES:
            continue
        d = _load_interm_npz(n, which)
        if d is None:
            continue
        for _key, _lsty, _suf in fields:
            if _key not in d:
                continue
            with warnings.catch_warnings():        # all-nan columns → nan, no spam
                warnings.simplefilter('ignore', category=RuntimeWarning)
                _prof = np.nanmean(d[_key], axis=1)
            # per-case inner-unit divisor (Reynolds ladder), not the global l_in
            ax.plot(_prof, d['axis_v'] / norm_l_in(n), color=c['color'],
                    linestyle=_lsty, label=c['label'] + _suf)
            _any = True
    if not _any:
        plt.close(fig)
        print('[interm] no %s data for %r — profile skipped.'
              % (which, [f[0] for f in fields]))
        return
    ax.set_ylim(0, _row_to_height(limity, use_inner=True))
    ax.set_xlabel(xlabel); ax.set_ylabel(r'$z^+$')
    ax.set_title(suptitle); ax.grid(True, ls='--', lw=0.5)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    fig.savefig(_figdir + savename, dpi=FIG_DPI, bbox_inches='tight'); plt.show()
    print('Saved: ' + _figdir + savename)

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
            print('           inst planes (from pickle) : %s'
                  % ', '.join('%s=%s' % (k, v) for k, v in _pv['inst'].items()))
        if _pv.get('inst_skip'):
            print('           inst planes MISSING from pickle: %s'
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
    # Every case that could not be drawn on some figure, and why (see
    # _note_missing / _case_availability).
    _report_missing_summary()
    print('\n--- tally ---')
    print('  cases with pickle (diagnosed): %s' % (_plotted or 'none'))
    print('  cases skipped (no pickle)    : %s' % (_skipped or 'none'))
    print('  smooth references drawn      : %s'
          % (', '.join(SMOOTH_KEYS) if SMOOTH_KEYS else 'none'))
    print('#' * 78)

def case_nu(case):
    """Kinematic viscosity OF THIS CASE.  On the Froude ladder results.py could
    use the single global `nu` everywhere because Re was fixed at 500; here it
    differs by case (nu = 1/(0.5*Re_D^2)), so every expression forming a case's
    inner units, viscous stress or delta+ must go through this.  Falls back to
    the global config nu (Re=500) for an unknown case.  'Sm_Neu' (the flat
    smooth reference) is registered in SIM_NU below with the viscosity of
    WHICHEVER smooth NetCDF was selected, so it is correct at any Reynolds
    number, not only 500."""
    return SIM_NU.get(case, nu)

def _find_smooth_nc(_path):
    """Locate one smooth reference: the path as written first, then its
    <parent>/<file> tail under the examples root and each declared case
    directory (legacy layout).  None if it is nowhere on disk."""
    _tail = '/'.join(_path.rstrip('/').split('/')[-2:])
    for _cand in [_path] + [_r + _tail for _r in [_base] + list(SIM_DIRS.values())]:
        if os.path.exists(_cand):
            return _cand
    return None


def _load_case_pkl(_pkl):
    """Load one sim pickle, dropping unused heavy fields and downcasting float64
    arrays to float32 to bound memory (see the block comment above)."""
    with open(_pkl, 'rb') as _fh:
        _raw = pickle.load(_fh)
    _kept = {}
    for _k, _v in _raw.items():
        if _k in _DROP_KEYS:
            continue
        if isinstance(_v, np.ndarray) and _v.dtype == np.float64:
            _v = _v.astype(np.float32)
        _kept[_k] = _v
    del _raw                                   # free the full pickle immediately
    return _kept


def _find_case_elsewhere(_dir):
    """Hint for a case whose declared directory holds no sim1_results.pkl.

    The Reynolds ladder declares its cases FLAT under the examples root
    (`<root>/EkRe500FrInf/`), but the same runs are also kept one level down in
    a grid-size folder (`<root>/1056x672x1056/EkRe500FrInf/`, the layout
    results.py uses) — in which case the data really is on disk and only the
    cwdN path is wrong.  Search the root, one and two levels deep, for a
    directory with the SAME basename that does carry a pickle, and return its
    path so the loader can point at it.  Read-only, depth-capped, and purely
    advisory: nothing is loaded from it, the cwdN lines stay the single source
    of truth."""
    _want = os.path.basename(_dir.rstrip('/'))
    _hits = []
    try:
        for _l1 in sorted(os.scandir(_base), key=lambda _e: _e.name):
            if not _l1.is_dir():
                continue
            if _l1.name == _want and os.path.exists(
                    os.path.join(_l1.path, 'sim1_results.pkl')):
                _hits.append(_l1.path + '/')
            try:
                for _l2 in sorted(os.scandir(_l1.path), key=lambda _e: _e.name):
                    if (_l2.is_dir() and _l2.name == _want and os.path.exists(
                            os.path.join(_l2.path, 'sim1_results.pkl'))):
                        _hits.append(_l2.path + '/')
            except OSError:
                continue
    except OSError:
        return []
    return [_h for _h in _hits if _h.rstrip('/') != _dir.rstrip('/')]

###############################################################################
# PER-CASE WALL UNITS  (Reynolds-ladder specific)
#
# Every normalization in this file is driven by the USER INPUT dict USTAR_REF
# (top of file — that is the knob you edit).  For each case:
#
#     u*_ref = USTAR_REF[case]                 velocity / stress scale
#     l_in   = nu(case) / u*_ref               length scale (one wall unit)
#
# With the defaults each case sits in its own wall units; set the USTAR_REF
# entries equal to put every case on one common velocity yardstick.  There is
# no separate "mode" — the dict IS the control, so a single set of figures is
# produced and lands directly in fig_Re/.
#
# The measured Method-2 u* (pickled u_star2) is NOT used for normalization; it
# remains the PHYSICAL friction velocity of each case and is still used for
# physical quantities (delta+ markers, Re_tau, the own-units log-law fit).
# Keeping the two apart is deliberate: the normalization must be a value YOU
# control and can equate across cases, while the measured u* must stay whatever
# the simulation actually produced.
###############################################################################
def _ustar_measured(case):
    """(u*, source): the case's PHYSICAL friction velocity, READ FROM THE PICKLE.

    Returns ``sims[case]['u_star']`` — the scalar PhAvg_rotated.py measured and
    pickled, u_star = max_z u_star2_c(z) (the integrate->cavg Method-2 momentum-
    integral profile).  This script must NOT derive a friction velocity of its
    own: no plateau_value() on the pickled u_star2 profile, no crest value, no
    mean.  Stage b owns the definition; stage c only reads it.

    This is also the inner-scale plot normalization: norm_ustar(case) returns it,
    so every surface-layer figure is in the case's own wall units.

    Fallback (legacy pickle with no 'u_star' key): the CLAUDE.md physics table."""
    _u = sims.get(case, {}).get('u_star')
    if _u is not None:
        _val = float(np.asarray(_u, dtype=float).ravel()[0])
        if np.isfinite(_val) and _val > 0:
            return _val, "measured (pickled u_star = max(u_star2_c))"
    _fb = SIM_USTAR_FALLBACK.get(case, np.nan)
    _note = "CLAUDE.md fallback — pickle carries no 'u_star'"
    if SIM_RE.get(case, 500.0) > 500.0:
        _note += ' — ESTIMATE (smooth-wall value +20 % for orography), not measured'
    return float(_fb), _note

def norm_ustar(case):          # [U*-NORM] inner-scale velocity/stress normalizer
    """VELOCITY / stress normalization scale for this case = the case's OWN
    MEASURED Method-2 plateau friction velocity (case_ustar / _ustar_measured).

    Every inner-scale (surface-layer) figure is therefore drawn in the case's
    own wall units — u+ = u/u*_case, tau/u*_case^2, z+ = z/norm_l_in(case) —
    rather than on a hand-entered reference.

    USE_USTAR_REF flips this back to the USER INPUT USTAR_REF entry (the old
    behaviour: set those values equal to force one common velocity scale)."""
    if USE_USTAR_REF:
        return USTAR_REF.get(case, u_star)
    _u = SIM_USTAR_MEASURED.get(case)
    if _u is not None and np.isfinite(_u) and _u > 0:
        return float(_u)
    return USTAR_REF.get(case, u_star)          # smooth ref / missing pickle

def norm_l_in(case):
    """LENGTH normalization (one wall unit) for this case: nu(case)/norm_ustar(case),
    i.e. the wall unit of the case's OWN measured u*.  Note nu is fixed by the
    case's Reynolds number, so the length scale differs between Reynolds numbers
    even when the velocity scales happen to agree — that residual IS the
    Reynolds effect (see the outer-unit z^- figures for a view without it)."""
    return case_nu(case) / norm_ustar(case)

def case_ustar(case):
    """PHYSICAL (measured) friction velocity of this case — see _ustar_measured.
    This is NOT the plot normalization; use norm_ustar(case) for that."""
    return SIM_USTAR_MEASURED.get(case, u_star)


def _load_one_smooth(_key, _path):
    """Load ONE flat smooth reference into a dict of its `*_s` quantities.

    Smooth case computed by the SINGLE shared loader (functions.load_smooth_case),
    identical to PhAvg.py — the two scripts can no longer diverge.  Uses PhAvg's
    validated formulas: TKE_s = 0.5*(Rxx+Ryy+Rzz); Coriolis cor_yx_s = -(W_s-G_z)
    with scalar geostrophic G_z = max(W_s); Fornberg du_dy_s.

    nu / Re_lambda are THIS FILE's own (case_nu(key) — set from its Reynolds
    number), not the global Re=500 pair: they set the loader's viscous stress and
    its inner axis y_s_p = y*u*/nu, which would be wrong by a factor of 4 on the
    Re=1000 reference."""
    _nu_s    = case_nu(_key)
    _relam_s = 1.0 / _nu_s
    _sm = load_smooth_case(_path, x, _nu_s, _relam_s)
    _d = {'sy': _sm['sy'], 'nys': _sm['nys'],
          'U_s': _sm['U_s'], 'V_s': _sm['V_s'], 'W_s': _sm['W_s'],
          'su': _sm['su'], 'sw': _sm['sw'], 'alpha_s': _sm['alpha_s'],
          'alpha_str_s': _sm['alpha_str_s'],
          'y_s': _sm['y_s'], 'y_in_s': _sm['y_s_p'],   # this file uses y_in_s
          'rU_s': _sm['rU_s'], 'rV_s': _sm['rV_s'], 'rW_s': _sm['rW_s'],
          'rP_s': _sm['rP_s'],
          # mean scalar ⟨s⟩ (Boussinesq solution); ≡0 in the neutral reference
          'rs_s': _sm['rs_s'],
          'G_x_s': _sm['G_x_s'], 'G_z_s': _sm['G_z_s'], 'G_s': _sm['G_s'],
          'U_s_p': _sm['U_s_p'], 'W_s_p': _sm['W_s_p'],
          'GblU_s': _sm['GblU_s'], 'GblW_s': _sm['GblW_s'],
          'Rxx_s': _sm['Rxx_s'], 'Rxy_s': _sm['Rxy_s'], 'Rxz_s': _sm['Rxz_s'],
          'Ryy_s': _sm['Ryy_s'], 'Ryz_s': _sm['Ryz_s'], 'Rzz_s': _sm['Rzz_s'],
          'TKE_s': _sm['TKE_s'],
          'cor_yx_s': _sm['cor_yx_s'], 'I_corr_yx_s': _sm['I_corr_yx_s'],
          'du_dy_s': _sm['du_dy_s'], 'visc_yx_s': _sm['visc_yx_s'],
          'tau_yx_s': _sm['tau_yx_s'],
          'cor_yz_s': _sm['cor_yz_s'], 'I_corr_yz_s': _sm['I_corr_yz_s'],
          'dw_dy_s': _sm['dw_dy_s'], 'visc_yz_s': _sm['visc_yz_s'],
          'tau_yz_s': _sm['tau_yz_s'],
          'AVG_TKE_V_s': _sm['AVG_TKE_V_s'],
          'AVG_TKE_V_s_i': _sm['AVG_TKE_V_s_i']}
    # ustr_s1 is the normalizer for every curve OF THIS REFERENCE (u/ustr_s1,
    # stress/ustr_s1**2, outer y_s/ustr_s1).  It follows the USER INPUT entry so
    # the smooth overlay obeys USTAR_REF exactly like the valley cases do.
    # ustr_s1_stored is the reference's PHYSICAL friction velocity and keeps
    # setting its physical delta+.  A flat .nc may carry no `FrictionVelocity`
    # (the loader then returns None): fall back to its Method-2 plateau, then to
    # the registry value, so it is ALWAYS a finite number.
    _u_st = _sm['ustr_s1']
    if _u_st is None or not np.isfinite(_u_st):
        _u_st = _sm.get('ustr_M2_plateau_s')
        print('  %s: .nc stores no FrictionVelocity — using the Method-2 '
              'plateau u*=%s' % (_key, _u_st))
    if _u_st is None or not np.isfinite(_u_st):
        _u_st = SMOOTH_USTAR_FALLBACK.get(_key, u_star)
    _d['ustr_s1_stored'] = float(_u_st)
    _d['ustr_s1'] = float(USTAR_REF.get(_key, _d['ustr_s1_stored']))

    # Derived quantities (TKE advection field), from the loader outputs
    _d['sx']       = np.linspace(0, 1.08, _d['rU_s'].shape[1])
    _d['TKE_s_dx'] = np.gradient(_d['TKE_s'], _d['sx'],  axis=1)
    _d['TKE_s_dy'] = np.gradient(_d['TKE_s'], _d['y_s'], axis=0)
    _d['Adv_s']    = _d['rU_s'] * _d['TKE_s_dx'] + _d['rV_s'] * _d['TKE_s_dy']
    # Smooth-case "dispersive" proxies (P06-P08).  The flat-wall .nc has NO
    # streamwise axis — its second axis is TIME (the averaging records), which
    # sx maps to a pseudo-x.  So these are the deviation of each time record from
    # the time-mean profile (a temporal stand-in for the true spatial dispersive
    # field, which is identically zero on a flat wall).  keepdims=True: rU_s is
    # (ny, nt), mean over the pseudo-x axis is (ny,), must broadcast back (ny,1).
    # Disp_V_s is ~machine-zero (rV_s≡0); Disp_U_s/Disp_W_s carry real structure.
    _d['Disp_U_s'] = _d['rU_s'] - _d['rU_s'].mean(axis=1, keepdims=True)
    _d['Disp_V_s'] = _d['rV_s'] - _d['rV_s'].mean(axis=1, keepdims=True)
    _d['Disp_W_s'] = _d['rW_s'] - _d['rW_s'].mean(axis=1, keepdims=True)
    # Style of this reference (bound alongside its data so every plot call picks
    # up the right colour/linestyle/label without a lookup).
    _st = SMOOTH_MAP[_key]
    _d['SMOOTH_COLOR']  = _st['color'];  _d['SMOOTH_LS']     = _st['ls']
    _d['SMOOTH_LABEL']  = _st['label'];  _d['SMOOTH_MARKER'] = _st['marker']
    _d['SMOOTH_KEY']    = _key
    _d['_nu'] = _nu_s
    return _d


def _bind_smooth(_key):
    """Bind the flat `*_s` module names (and SMOOTH_COLOR/LS/LABEL/MARKER) to
    one loaded smooth reference.  This is what lets a plot block written for a
    single reference draw all of them: `for _ in smooth_refs():` rebinds these
    names once per reference around the unchanged block body."""
    _d = smooths.get(_key)
    if _d is None:
        return False
    globals().update({_n: _d[_n] for _n in _SMOOTH_FLAT if _n in _d})
    return True


def smooth_refs():
    """Iterate over EVERY smooth reference that is declared, found and loaded,
    binding it to the flat `*_s` names before each iteration.  Yields the
    reference key.  Yields NOTHING when no reference is available, so

        for _ in smooth_refs():
            <block>

    is a drop-in replacement for the old `if _smooth_loaded:` guard.  The PRIMARY
    reference is re-bound when the loop ends — in a `finally`, so the flat names
    are restored even if the loop is abandoned or the body raises."""
    try:
        for _k in SMOOTH_KEYS:
            if _bind_smooth(_k):
                yield _k
    finally:
        _bind_smooth(_SM_PRIMARY)


###############################################################################
###############################################################################
####  2.  USER PARAMETERS AND VARIABLE DECLARATION
#
#  THE ONLY SECTION YOU NORMALLY EDIT.  Every knob this script exposes is here:
#  which cases are read, the physical constants, the unit/axis mode, the plot
#  windows, the on/off switches.  Nothing in this section is computed FROM the
#  data -- anything derived belongs in section 3.
#
#  Contents: config.py fallbacks -> unit/axis mode (Z_NORM) -> physics constants
#  -> run controls -> data sources (case dirs, NetCDF references) -> figure and
#  axis settings.
#
#  INDEX OF EVERY KNOB IN THIS SECTION
#    data sources     cwd1 cwd2 [cwd3]  _CASE_REGISTRY
#                     nc1 nc2  _SMOOTH_REGISTRY  INTERM_DIRS
#    case switches    PLOT_SMOOTH  DISABLE_CASES  PLOT_FR_LADDER
#    units / axes     Z_NORM  USE_USTAR_REF  _Z_LABELS  _X_LABELS
#    physics          Re  Re_lambda  nu  f  alpha  u_star  kappa  OBU_KAPPA
#                     Gx  Gz  l_in  l_out  Re_tau  wall_units  limity  ...
#    run controls     plotRes  postprocess  load_arrays  animate  cal_Avg
#                     save_avg  verify_TimeAvg  load_ncfiles  plot_only
#    figures / axes   FIG_SUBDIR  _IBM_COLOR  Z_PLUS_MAX  Z_MINUS_MAX
#                     Z_PLUS_CONTOUR_MAX  _LL_ZMIN  _LL_ZMAX
#    config.py        FIG4_PAPER_SPANWISE_SIGN  PR_T  RI_B_BINS  rough-ladder
#                     (imported with an inline fallback, so this script also
#                      runs in a results-only deployment that has no config.py)
###############################################################################
###############################################################################


# Display handedness of the SPANWISE (τ_zy) shear-stress panels — mirrors
# config.fig4_paper_spanwise_sign.  Our tlab f-sign gives the closing τ_zy budget
# as C_zy<0 / R_zy>0 (mirror of K&A 2024); True negates the τ_zy panel for DISPLAY
# so Coriolis reads positive like the paper (physical closure / u* unaffected — this
# only scales the plotted τ_zy curves).  results.py may be deployed WITHOUT config.py
# (setup.sh --results-only), so fall back to the documented default True.
try:
    from config import fig4_paper_spanwise_sign as FIG4_PAPER_SPANWISE_SIGN
except Exception:
    FIG4_PAPER_SPANWISE_SIGN = True

# Rough Re=1000 stable-ladder LOG-LAW overlay (config-gated; same WITHOUT-config.py
# fallback as above).  Directory holds the ri00.00 → ri18.78 avg_all.nc files and
# lives on the machine with the data; the loader returns [] when it is absent.
try:
    from config import (plot_ref_rough_ladder as REF_ROUGH_LADDER,
                        rough_ladder_dir as ROUGH_LADDER_DIR,
                        rough_ladder_pattern as ROUGH_LADDER_PATTERN,
                        rough_ladder_ustar as ROUGH_LADDER_USTAR,
                        nu_rough as NU_ROUGH)
except Exception:
    REF_ROUGH_LADDER    = False
    ROUGH_LADDER_DIR    = '/home/shreyad95/Documents/PhD/Code/Re1000/'
    ROUGH_LADDER_PATTERN = 'ri*_avg.nc'
    ROUGH_LADDER_USTAR  = 0.0618
    NU_ROUGH            = 2e-6

# MOST neutral turbulent Prandtl number for the cross-case local-similarity
# (φ_h) overlay (Goal 5 / P87); same WITHOUT-config.py fallback as above.
try:
    from config import Pr_t as PR_T
except Exception:
    PR_T = 0.85

# Ansorge (2017) stability-class bin edges (weak | intermediate | strong) used by
# the cross-case stability axis (Goal 1 / P72), mirroring PhAvg_rotated.py's R5.
# WITHOUT-config.py fallback = the config default (0.05, 0.15).
try:
    from config import Ri_B_bins as RI_B_BINS
except Exception:
    RI_B_BINS = (0.05, 0.15)

###############################################################################
# Z_NORM — WALL-NORMAL AXIS OF THE 2-D SIDE-BY-SIDE PANELS.  USER INPUT.
#
# The valley is the SAME OBJECT IN PHYSICAL SPACE in every case: eps0.1 is
# generated once and carried to each new grid by MyPyLib/transfer_eps.py, and
# MyPyLib/compare_eps_geometry.py confirms crest / floor / depth / width / solid
# area agree to better than ONE wall-normal cell across 1024x832x1024,
# 1152x816x1152 and 1728x784x1728.  What differs between the panels is the AXIS,
# not the geometry:
#
#   'inner'  z+ = z/l_in(case),  l_in = nu(case)/u*(case)   [default, unchanged]
#            The physically correct surface-layer scaling for a Reynolds ladder
#            (log layers collapse, the higher Re shows a longer log region), but
#            a FIXED physical valley then draws at h+ = h*u*/nu, i.e.
#               Re_D = 500 -> h+ ~ 25      750 -> h+ ~ 57      1000 -> h+ ~ 96
#            so the topography deliberately does NOT coincide between panels.
#
#   'outer'  z- = z/delta(case),  delta = u*(case)/f  (the Ekman depth).
#            The matching OUTER velocity scale is the geostrophic wind G, not
#            u*, so an outer figure plots u/G, <u'u'>/G^2, ... (G = 1 here, so
#            those are the raw stored values).  h/delta ~ 0.045 in every case,
#            i.e. the valley very nearly coincides, and the OUTER region of the
#            profiles is what collapses on this scaling.
#
#   'topo'   zeta = z/h(case), h = that case's own crest height (case_hill_h).
#            The valley occupies exactly zeta = 1 in EVERY panel, on every grid
#            and at every Reynolds number, so the flow above identical
#            topography can be compared shape-for-shape.  Wall units are then
#            gone from the axis — use it for geometry-referenced comparison, not
#            for the log law.
#
# Flip this ONE line; nothing else needs editing.  It affects the 2-D panels
# (_case_grid / _contour_zmax / axis labels + the black orography, which is
# carried onto the same axis as the field) and the crest marker _mark_h.  The
# 1-D figures come in MATCHED PAIRS — an inner (z+, u+) view and an outer
# (z-, u/G) view of the same quantity — and are unaffected by this switch;
# their scaling is chosen per figure through norm_vel()/z_label().
###############################################################################
Z_NORM = 'topo'          # 'inner' (z+) | 'outer' (z-) | 'topo' (z/h)

# The wall-normal crop of the 2-D panels is NOT a per-mode constant: it is ONE
# physical height (Z_PLUS_CONTOUR_MAX reference wall units, set in the plotRes
# block) re-expressed in whichever units Z_NORM selects — see _contour_zmax.
# Three hand-entered caps would drift apart and silently make the modes show
# different parts of the flow; deriving them keeps the comparison honest.

if Z_NORM not in ('inner', 'outer', 'topo'):
    raise SystemExit("results_Re.py: Z_NORM must be 'inner', 'outer' or 'topo',"
                     " got %r" % (Z_NORM,))

###############################################################################
# THE UNIT SYSTEM — ONE definition, used by EVERY figure.
#
# There are exactly two physical scalings on this ladder, and each is a matched
# PAIR of a length and a velocity.  Mixing halves of the two pairs is what stops
# profiles collapsing, so nothing below is allowed to choose them independently:
#
#   INNER (wall / viscous units)          OUTER (Ekman / geostrophic units)
#   ------------------------------        --------------------------------
#   delta_nu = nu(case)/u_tau(case)       delta = u_tau(case)/f
#   z+ = z/delta_nu = z*u_tau/nu          z- = z/delta = z*f/u_tau
#   u+ = u_i/u_tau                        u- = u_i          (i.e. /G, and G=1)
#   <u_i'u_j'>/u_tau^2                    <u_i'u_j'>        (i.e. /G^2)
#   (anything of velocity power n)/u_tau^n              unchanged
#
# So in OUTER units NOTHING of velocity dimension is divided by u_tau — the
# geostrophic wind is the velocity scale and it is 1 by construction.  u_tau
# survives only inside the LENGTH delta.  (This is the classic Ekman outer
# scaling: the outer region collapses in (z/delta, u/G), the surface layer
# collapses in (z+, u+).)
#
# u_tau ITSELF: ONE number per case, `case_ustar(case)` = the PICKLED scalar
# sims[case]['u_star'] = max_z u_star2_c(z), measured by PhAvg_rotated.py.  It is
# READ, never re-derived here (no plateau_value on the pickled u_star2 profile,
# no crest value).  Every case is normalised by ITS OWN value.  Before this block
# existed the inner figures used a plateau while _z_out used u_star2[ghill] (the
# CREST value) — 0.0663 vs 0.0640 on the Re=500 case, a 3.4 % mismatch that
# displaced every outer curve and prevented the collapse the outer scaling is
# supposed to show; deriving u* upstream once removes that failure mode for good.
###############################################################################
_Z_MODES = ('inner', 'outer', 'topo')

# ── Axis LABELS: the label must follow the scaling, never be typed by hand ────
# A figure that switches scale but keeps a hardcoded r'$z^+$' / r'$/u_*^2$' is
# mislabelled, which is worse than being on the wrong scale (it is invisible).
_Z_LABELS = {'inner': r'$z^+$', 'outer': r'$z^-$', 'topo': r'$z/h$'}
_X_LABELS = {'inner': r'$x^+$', 'outer': r'$x/\delta$', 'topo': r'$x/h$'}

###############################################################################
#  Obukhov (1971) stability-corrected surface-layer wind profile
#  "Turbulence in an Atmosphere with a Non-Uniform Temperature",
#   Bound.-Layer Meteorol. 2, 7-29.
# -----------------------------------------------------------------------------
#  PORTED VERBATIM from MyPyLib/PhAvg_rotated.py (keep the two in sync).  There
#  the same machinery fits ONE run; here it is applied per case so the modified
#  law can be compared ACROSS the Froude ladder.  This is the paper-faithful
#  Section-6 parametric profile, fitted ALONGSIDE (not replacing) the neutral log
#  law and the Xi-integral OLS fit already in _loglaw_fit_case below.
#
#  All quantities are NON-DIMENSIONAL (the DNS has g = 1, f = 1).
#
#  Parametric substitution (eq 39, unified stable + unstable branches):
#      xi  = z / L1 = 1/u' - u'^3            (u' = auxiliary parameter)
#      eta = Ri/Ri_cr = 1 - u'^4
#      phi(Ri) = sqrt(1 - eta) = u'^2       (eq 38)
#    u' in (0,1]   -> xi >= 0  STABLE   (Ri > 0, phi < 1, mixing suppressed)
#    u' in [1,inf) -> xi <= 0  UNSTABLE (Ri < 0, phi > 1, mixing enhanced)
#  Wind gradient (eq 22):  sqrt(phi)*k*z*dv/dz = v*  ->  dv/dz = v*/(k z u'),
#    v(z) = (v*/k) * psi(xi),   psi(xi) = int dxi / (xi u').
###############################################################################
OBU_KAPPA = 0.4        # paper's von Karman constant — the ONE fixed empirical
# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS CONSTANTS AND VARIABLE DECLARATION
# ─────────────────────────────────────────────────────────────────────────────

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

# Selective replot: regenerate ONLY the two shear-stress momentum-balance figures
# (P46 tau_zx + P47 tau_zy) and exit, instead of running the whole ~45-figure block.
#   RESULTS_ONLY=shear python results.py      # only P46 + P47
#   python results.py                         # '' -> full run (default, unchanged)
# Accepted values: shear / tau / p46 / p47 / p46p47.  Hardcode the default below if
# you prefer a plain toggle over the environment variable.
plot_only = os.environ.get('RESULTS_ONLY', '').strip().lower()
_ONLY_SHEAR = plot_only in ('shear', 'tau', 'p46', 'p47', 'p46p47')

# ─────────────────────────────────────────────────────────────────────────────
# DATA SOURCES — where the cases and the smooth references are read from
# ─────────────────────────────────────────────────────────────────────────────

# Parameter decleration
# cwd = the directory in which THIS results.py lives.  It is meant to sit in the
# central "examples" tree that holds every simulation directory, so all case
# paths are taken relative to it (os.path.dirname(__file__) follows the symlink's
# own location, not its MyPyLib target, so the script is relocatable: move the
# whole examples tree and the paths still resolve).
cwd = str(os.path.dirname(__file__) + '/' )
# ═══════════════════════════════════════════════════════════════════════════
# REYNOLDS LADDER (this file) — NEUTRAL valley runs, Fr = ∞ for ALL of them.
# ═══════════════════════════════════════════════════════════════════════════
# This is the Reynolds-number counterpart of results.py's Froude ladder: the
# stratification is held FIXED at neutral and the Reynolds number is varied, so
# every difference between the cases is a Reynolds effect (see the per-case
# physics block below — nu and u* now differ BY CASE, which the Froude-ladder
# results.py never had to handle because Re was fixed at 500).
#
# EXPECTED LAYOUT (everything is relative to the examples root = the directory
# this file sits in, so the whole tree can be moved):
#
#   <root>/EkRe500FrInf/    Re_D = 500   SIMULATION dir  (sim1_results.pkl + grid)
#   <root>/EkRe750FrInf/    Re_D = 750   SIMULATION dir
#   <root>/EkRe1000FrInf/   Re_D = 1000  SIMULATION dir
#   <root>/Re500/*.nc       flat smooth-wall NetCDF REFERENCE, Re_D = 500
#   <root>/Re1000/*.nc      flat smooth-wall NetCDF REFERENCE, Re_D = 1000
#
# The trailing '/' on each case dir is REQUIRED — every consumer builds paths as
# `<dir> + 'sim1_results.pkl'`.  A case dir must carry a pickle written by
# PhAvg_rotated.py (stage b); the loader below reports explicitly whether it
# found a simulation dir, a NetCDF-only dir, or nothing.
_base = cwd                                      # examples root = where this file sits

# ── DATA SOURCES — one line per Reynolds number ──────────────────────────────
# COMMENT A LINE OUT TO DROP THAT CASE.  A commented source is an ABSENT source:
# the case then disappears from every table, legend, figure and console block
# and NOTHING downstream fails.  The registry below picks up only the cwdN names
# that actually exist (globals()), so nothing else in this file has to change —
# do not add the case name anywhere else by hand.
cwd1 = _base + 'EkRe500FrInf/'          # Neutral Fr = ∞, Re_D = 500  (valley present)
cwd2 = _base + 'EkRe750FrInf/'          # Neutral Fr = ∞, Re_D = 750  (valley present)
# cwd3 = _base + 'EkRe1000FrInf/'       # Neutral Fr = ∞, Re_D = 1000 (valley present)

###############################################################################
# CASE REGISTRY — the SINGLE source of truth for the whole ladder.
#
# One entry per possible case; 'var' names the source variable above.  An entry
# whose 'var' is not defined (its line is commented out) is simply DECLARED
# ABSENT: it contributes to no dict, no legend and no loop.  Everything the rest
# of the file used to hold in separate hand-maintained dicts — SIM_DIRS, CASES,
# ACTIVE_CASES, USTAR_REF, SIM_FR, SIM_RE, SIM_NU, SIM_USTAR_FALLBACK — is
# derived from this table below, so the three data-source lines above are the
# only place a case is switched on or off.
#
#   Re          : Re_D  (nu = 1/(0.5*Re_D^2) — the per-case viscosity)
#   Fr          : wall-law switch (np.inf = neutral -> classical ln(z+ - d+))
#   ustar_ref   : USER-INPUT normalization fallback u* (see USTAR_REF below)
#   ustar_fall  : CLAUDE.md physics-table u*, used only if the pickle carries no
#                 measured u_star2 (Re=750/1000 entries are ESTIMATES: the
#                 smooth-wall value +20 % for orography, not measurements)
###############################################################################
_CASE_REGISTRY = (
    {'name': 're500_oro',  'var': 'cwd1', 'Re': 500.0,  'Fr': np.inf,
     'label': r'$Re_D=500$',  'color': '#1565C0', 'ls': '--', 'marker': 's',
     'ustar_ref': 0.0618, 'ustar_fall': 0.068},
    {'name': 're750_oro',  'var': 'cwd2', 'Re': 750.0,  'Fr': np.inf,
     'label': r'$Re_D=750$',  'color': '#C62828', 'ls': '-.', 'marker': '^',
     'ustar_ref': 0.0561, 'ustar_fall': 0.067},
    {'name': 're1000_oro', 'var': 'cwd3', 'Re': 1000.0, 'Fr': np.inf,
     'label': r'$Re_D=1000$', 'color': '#2E7D32', 'ls': ':',  'marker': 'D',
     'ustar_ref': 0.0531, 'ustar_fall': 0.06372},
)

###############################################################################
# MASTER PLOT CONTROL
# PLOT_SMOOTH toggles the flat smooth-wall NetCDF reference.  ACTIVE_CASES is
# DERIVED from the data sources (cwd1/cwd2/cwd3): every declared case is active,
# so a case is switched off by commenting its source line — not by editing a
# second list here that could silently disagree with it.  DISABLE_CASES is the
# escape hatch for keeping a source line but hiding that case from the figures.
###############################################################################
PLOT_SMOOTH   = True     # smooth-wall NetCDF reference (flat wall, Fr = inf)
DISABLE_CASES = set()    # e.g. {'re750_oro'} to hide a declared case

###############################################################################
# SMOOTH-WALL NetCDF REFERENCES — flat-wall benchmarks, one line per file.
#
# These are SMOOTH (flat, no valley) cases — the counterpart of the EkRe*FrInf
# VALLEY cases above, not another valley run.  They are stored as tlab
# horizontally-averaged avg_all.nc (ri00.00 = neutral) in the EXAMPLES ROOT,
# beside the case directories:
#     <examples root>/Re500/ri00.00_re0500_...nc     smooth, Re_D = 500
#     <examples root>/Re1000/ri00.00_re1000_...nc    smooth, Re_D = 1000
# (the path used to be hardwired inside cwd1, i.e. EkRe500FrInf/Re500/..., which
# does not exist — the smooth curve was therefore silently missing from EVERY
# figure and the smooth-only names it defines, e.g. ustr_s1_stored, were never
# created.  That is the NameError at the hodograph block.)
#
# ── SAME CONTROL AS THE VALLEY CASES: COMMENT A LINE OUT TO DROP THAT ────────
# ── REFERENCE.  A commented source is an ABSENT source. ─────────────────────
# EVERY declared reference is loaded AND DRAWN: each figure that carries a
# smooth benchmark now carries one curve per declared reference, each in its
# OWN wall units (its own nu = 1/(0.5*Re_D^2) and its own u*).  Nothing else in
# the file has to be touched to add or remove one.
#
# The FIRST declared reference is the PRIMARY one: it backs the flat `*_s`
# module names (y_in_s, Rxy_s, ustr_s1, …) that anything not iterating over the
# references still reads, and it is the 'Sm_Neu' entry of USTAR_REF / SIM_NU.
#
# Each path is searched as given first, then by basename inside the examples
# root and each declared case directory (legacy layout).  A declared file that
# is nowhere on disk is reported and skipped — it does not stop the run.
###############################################################################
nc1 = _base + 'Re500/ri00.00_re0500_2048x0192x2048_20110615_avg_all.nc'    # smooth, Re_D = 500
nc2 = _base + 'Re1000/ri00.00_re1000_3072x0512x6144_20130520_avg_all.nc'   # smooth, Re_D = 1000

###############################################################################
# SMOOTH REGISTRY — mirrors the case registry.  'var' names the source line
# above; an entry whose variable is not defined (line commented out) is absent.
#   Re     : Re_D of the smooth file -> its nu = 1/(0.5*Re_D^2)
#   ustar  : fallback u*, used ONLY if the .nc stores no FrictionVelocity
#            (Re=500: 0.0618 stored; Re=1000: 0.0531, CLAUDE.md table)
#   colour : greys, so the smooth benchmarks stay visually distinct from the
#            coloured valley cases; the linestyle separates the Reynolds numbers
###############################################################################
_SMOOTH_REGISTRY = (
    {'name': 'Sm_Neu_500',  'var': 'nc1', 'Re': 500.0,  'ustar': 0.0618,
     'label': r'Smooth ($Re_D=500$, flat)',  'color': '#636363',
     'ls': '-', 'marker': 'o'},
    {'name': 'Sm_Neu_1000', 'var': 'nc2', 'Re': 1000.0, 'ustar': 0.0531,
     'label': r'Smooth ($Re_D=1000$, flat)', 'color': '#9E9E9E',
     'ls': (0, (5, 2)), 'marker': 'v'},
)

# ── ACTIVE normalization: each case's OWN MEASURED friction velocity ─────────
# DEFAULT (False): norm_ustar(case) returns that case's MEASURED Method-2
# plateau u* (from its pickled u_star2 profile), so every inner-scale /
# surface-layer figure is in the case's own true wall units.  USTAR_REF is then
# used only as a FALLBACK — for the smooth NetCDF reference, and for any case
# whose pickle carries no u_star2.
#
# Set True to restore the old behaviour: normalize by the hand-entered USTAR_REF
# entries above (equate them to force one common velocity yardstick).
USE_USTAR_REF = False

# ── Froude-ladder machinery: DISABLED on this Reynolds ladder ────────────────
# results.py's cross-case aggregation plots every diagnostic against the bulk
# Richardson number Ri_B (stability axis, dispersive share vs Ri_B, Obukhov
# L+ vs Ri_B, similarity departure vs Ri_B, buoyancy-flux splits, the Fr-vs-Psi
# coupling ladder).  With BOTH cases neutral, Ri_B = 0 and every one of those
# views collapses onto a single point — meaningless, not merely sparse.  The
# code is PRESERVED VERBATIM below (nothing deleted) and gated off by this one
# flag; set it True if a stratified case is ever added to this ladder.
PLOT_FR_LADDER = False

# Per-case override for the intermittency .npz directory ONLY.  A case absent
# from this map falls back to its SIM_DIRS path.  No overrides on this ladder.
INTERM_DIRS = {}


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE + AXIS SETTINGS  (were buried inside the plotting block; they are
# settings, so they live here.  The plotting section only reads them.)
# ─────────────────────────────────────────────────────────────────────────────
FIG_SUBDIR = 'fig_Re/'        # output folder, relative to the examples root
_IBM_COLOR = 'black'          # fill colour of the solid (IBM) region

# 1-D PROFILE axis windows.  These are two deliberately DIFFERENT views of the
# same profile, not one window in two units: z+ <= 200 is the surface layer,
# z- <= 4 is the whole Ekman layer.
Z_PLUS_MAX  = 200.0           # inner-unit (z+) figures
Z_MINUS_MAX = 4.0             # outer-unit (z-) figures

# 2-D PANEL window — the SINGLE control for every Z_NORM mode.  It is given in
# REFERENCE wall units and converted by _contour_zmax / _zplus_to_mode, so all
# modes crop at the same PHYSICAL height.  800 ~ 29 % of the domain; raise it to
# widen every mode at once (2725 ~ the full domain).
Z_PLUS_CONTOUR_MAX = 800.0

# Log-law fit window for THIS flow, in z+ (shared across cases so the fitted
# kappa / d+ / z0m+ are compared on one interval).
_LL_ZMIN, _LL_ZMAX = 45.0, 125.0

# Obukhov (1971) lookup-table bounds: the range of u' the psi(xi) table spans
# (xi in ~[-100, +1e4]).  Only widen these if a fit reports xi out of range.
OBU_U_LO, OBU_U_HI = 1.0e-4, 4.5

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE STYLE — one place for what used to be ~240 literals repeated through
# the plotting section (dpi=FIG_DPI x105, fontsize=LEGEND_FONTSIZE x44, ncol=LEGEND_NCOL x35, figsize x36).
# ─────────────────────────────────────────────────────────────────────────────
FIG_DPI          = 300          # every savefig / figure
FIG_SIZE         = (8, 6)       # the standard 1-D profile figure
FIG_SIZE_WIDE    = (10, 6)      # multi-term budget figures (more legend room)
FIG_SIZE_HODO    = (7, 6)       # hodographs (near-square)
FIG_SIZE_TALL    = (6, 7)       # figures with z on the VERTICAL axis
LEGEND_FONTSIZE  = 7            # legends, colorbar ticks and other small text
LEGEND_NCOL      = 2            # legend columns

# ─────────────────────────────────────────────────────────────────────────────
# PROFILE ROW CAPS — how many wall-normal ROWS of a profile a figure draws.
#
# CAVEAT, deliberately made visible by naming them: these are ROW COUNTS, not
# physical heights.  A fixed row index is a DIFFERENT height on every grid
# (row 430 is z = 0.0139 on the 832-row Re=500 grid but z = 0.0227 on the
# 784-row Re=750 grid), so on a Reynolds ladder these crops are not physically
# equivalent between cases.  They are kept as-is to preserve the existing
# figures; prefer an axis limit in z+/z-/z-over-h when adding a new figure.
# ─────────────────────────────────────────────────────────────────────────────
N_ROWS_SMOOTH_TAU = 160         # smooth-wall tau-budget panels
N_ROWS_SMOOTH_TKE = 130         # smooth-wall TKE profile
N_ROWS_ORO        = 430         # orographic profiles (u*, production)
N_ROWS_ORO_ADV    = 450         # orographic advection at landmarks
N_ROWS_ORO_RSL    = 157         # roughness-sublayer zoom


###############################################################################
###############################################################################
####  3.  CALCULATIONS AND CALLS TO FUNCTIONS
#
#  Everything the script DOES before drawing: read the grid and eps geometry,
#  load each case pickle and the smooth NetCDF references, resolve the per-case
#  friction velocity, build the derived case tables, print the run summary.
#
#  These statements run in order and depend on each other -- reordering them
#  breaks the run.  They leave their results in module-level names the plots
#  read.
###############################################################################
###############################################################################

# Announce it up front: the ONLY visible difference between a run that shows the
# topography at equal height in every panel and one that does not is this line,
# so print which file is live and which mode it selected.  (A rerun that "looks
# unchanged" is almost always a rerun still in 'inner'.)
_Z_NORM_NOTE = {
    'inner': ('2-D panels on z+ = z/delta_nu(case), delta_nu = nu/u*.  The valley is the',
              'SAME physical object in every case, but h+ = h*u*/nu ~ 25 / 57 / 96 at',
              'Re_D = 500 / 750 / 1000, so it DRAWS ~2.3x taller at Re_D=750 than 500.',
              "Set Z_NORM = 'topo' if you want the topography to coincide."),
    'outer': ('2-D panels on z- = z/delta(case), delta = u*/f (Ekman depth).  Velocity',
              'scale is G, not u*, so these panels are in geostrophic units.  The valley',
              'sits at h/delta ~ 0.045 in every case, i.e. very nearly coincident.',
              ''),
    'topo':  ('2-D panels on z/h: the valley is at z/h = 1 in EVERY panel, on every grid',
              'and at every Reynolds number.  Wall units are OFF these axes — use',
              "Z_NORM = 'inner' for the log law and 'outer' for the Ekman outer region.",
              ''),
}
print('=' * 78)
print('results_Re.py  Z_NORM = %r   (%s)' % (Z_NORM, os.path.realpath(__file__)))
for _ln in _Z_NORM_NOTE[Z_NORM]:
    if _ln:
        print('  ' + _ln)
print('=' * 78)

###############################################################################
# WHY A CASE IS NOT IN A FIGURE — stated explicitly, never silent.
#
# A declared case can fall out of a 2-D comparison figure at FOUR stages, and
# only the first two used to say anything:
#   1. its data-source line (cwdN) is commented out        -> reported at load
#   2. its directory carries no sim1_results.pkl           -> reported at load
#   3. it is hidden by DISABLE_CASES                       -> was SILENT
#   4. its pickle carries no such FIELD (stale / partial)  -> was SILENT: the
#      `_avail` filter in plot2D_allRe just skipped it, so a case vanished from
#      that figure — and from every figure reading that field — without a word.
#      THIS is how a case whose data "is available" still misses the plots.
# `_CASE_DROP_REASON` records (1)-(3) as they happen; `_case_availability()`
# adds (4) per field, so every figure can state which cases it drew, which it
# could not, and why.  `_MISSING_LOG` accumulates them for the end-of-run
# summary (_report_missing_summary), printed by _print_run_summary().
###############################################################################
_CASE_DROP_REASON = {}     # case -> why it carries no plottable data at all
_MISSING_LOG      = []     # (figure, field, case, reason)
_MISSING_SEEN     = set()  # (case, reason) already printed, for `once=True`

_OWN_GRID_CACHE = {}
                       # constant of the modified fit (config.kappa = 0.42 and the
                       # neutral fit FITS kappa in _LL_KBND; the paper/Table-III
                       # uses 0.4).  v* scales as 1/k, so a different k only
                       # rescales the reported v*, not L1.

# --- monotone lookup table over u' spanning xi in ~[-100, +1e4] --------------
_OBU_U_LO, _OBU_U_HI = OBU_U_LO, OBU_U_HI   # user params (section 2)
_OBU_U = np.concatenate([
    np.linspace(_OBU_U_HI, 1.0, 4000, endpoint=False),   # unstable side (u'>1)
    np.linspace(1.0, _OBU_U_LO, 12000)])                 # stable side  (u'<=1)
_OBU_XI = 1.0 / _OBU_U - _OBU_U**3                        # ascending along grid

# psi_hat(xi) = psi(xi) - ln|xi|  is finite through xi -> 0.
#   d psi_hat / du' = (1 + 3u'^4)(u' - 1) / (u'^2 (1 - u'^4)) ; limit -1/2 at u'=1
with np.errstate(divide='ignore', invalid='ignore'):
    _OBU_DPSIH = (1.0 + 3.0*_OBU_U**4)*(_OBU_U - 1.0) / (_OBU_U**2 * (1.0 - _OBU_U**4))
_OBU_DPSIH[~np.isfinite(_OBU_DPSIH)] = -0.5
_OBU_I0 = int(np.argmin(np.abs(_OBU_U - 1.0)))
_OBU_PSIH = np.zeros_like(_OBU_U)
_OBU_PSIH[1:] = np.cumsum(0.5*(_OBU_DPSIH[1:] + _OBU_DPSIH[:-1])*np.diff(_OBU_U))
_OBU_PSIH -= _OBU_PSIH[_OBU_I0]                           # psi_hat(xi=0) = 0


# Obukhov (1971) Table III — xi, eta(=Ri/Ri_cr), psi.  The printed xi=1.5 row
# (psi = 5.230) breaks the monotone psi(xi) and is a transcription typo; dropped.
_OBU_TBL3 = np.array([
 (0.05,0.055,1.600),(0.10,0.102,2.370),(0.15,0.144,2.742),(0.20,0.189,3.065),
 (0.25,0.231,3.320),(0.30,0.278,3.500),(0.35,0.320,3.662),(0.40,0.359,3.803),
 (0.45,0.398,3.928),(0.50,0.435,4.045),(0.55,0.470,4.157),(0.60,0.502,4.258),
 (0.65,0.533,4.360),(0.70,0.565,4.450),(0.75,0.597,4.560),(0.80,0.626,4.608),
 (0.85,0.650,4.695),(0.90,0.677,4.769),(0.95,0.700,4.839),(1.00,0.723,4.908),
 (1.1,0.76,5.03),(1.2,0.80,5.16),(1.3,0.84,5.29),(1.4,0.86,5.41),(1.6,0.90,5.63),
 (1.7,0.92,5.74),(1.8,0.93,5.85),(1.9,0.94,5.95),(2.0,0.95,6.06),(2.1,0.96,6.16),
 (2.2,0.96,6.27),(2.3,0.97,6.37),(2.4,0.97,6.47),(2.5,0.98,6.57),(2.6,0.98,6.68),
 (2.7,0.98,6.78),(2.8,0.98,6.88),(2.9,0.99,6.99),(3.0,0.99,7.09),(3.5,0.99,7.60),
 (4.0,1.00,8.10),(4.5,1.00,8.60),(5.0,1.00,9.10),(5.5,1.00,9.60),(6.0,1.00,10.10)])
# A source counts as DECLARED only if its cwdN variable exists and is a string.
_DECLARED = [_e for _e in _CASE_REGISTRY
             if isinstance(globals().get(_e['var']), str)]
_ABSENT   = [_e for _e in _CASE_REGISTRY
             if _e['name'] not in {_d['name'] for _d in _DECLARED}]
if not _DECLARED:
    raise SystemExit('results_Re.py: every data source (cwd1/cwd2/cwd3) is '
                     'commented out — nothing to post-process.')

SIM_DIRS            = {_e['name']: globals()[_e['var']] for _e in _DECLARED}
SIM_RE              = {_e['name']: _e['Re']             for _e in _DECLARED}
SIM_FR              = {_e['name']: _e['Fr']             for _e in _DECLARED}
SIM_USTAR_FALLBACK  = {_e['name']: _e['ustar_fall']     for _e in _DECLARED}
# nu = 1/(0.5*Re_D^2)  (tlab convention: nu = 1/Re_lambda, Re_lambda = 0.5*Re_D^2)
#   Re_D = 500 -> 8.0e-6,  750 -> 3.5556e-6,  1000 -> 2.0e-6
SIM_NU              = {_k: 1.0 / (0.5 * _v ** 2) for _k, _v in SIM_RE.items()}

print('-' * 78)
print('REYNOLDS LADDER — data sources')
for _e in _DECLARED:
    print('  declared : %-12s Re_D=%-6.0f %s  (%s)'
          % (_e['name'], _e['Re'], SIM_DIRS[_e['name']], _e['var']))
for _e in _ABSENT:
    print('  absent   : %-12s Re_D=%-6.0f (%s commented out — case dropped '
          'from every plot/table)' % (_e['name'], _e['Re'], _e['var']))
    # Recorded so any figure that wanted this case can say WHY it is not there
    # (see _case_drop_reason / _case_availability).
    _CASE_DROP_REASON[_e['name']] = ('data source `%s` is commented out at the '
                                     'top of this file' % _e['var'])

# Reference grid + IBM geometry: the central examples/ root holds no grid of its
# own, so they come from the FIRST DECLARED case directory that actually carries
# a `grid` file (previously hardwired to cwd1, which broke as soon as cwd1 was
# commented out).  Every per-case calculation still uses that case's OWN pickled
# grid/eps; this reference only sets the shared axes and the valley geometry
# (nx, ny, hill_hgt) of the 2-D orographic panels.
_REF_DIR = next((_d for _d in SIM_DIRS.values() if os.path.exists(_d + 'grid')),
                None)
if _REF_DIR is None:
    raise SystemExit('results_Re.py: no declared case directory carries a '
                     '`grid` file — checked %s' % list(SIM_DIRS.values()))
print('  reference grid / IBM geometry from: %s' % _REF_DIR)
print('-' * 78)
x, y, z = read_grid(_REF_DIR)

nx = np.size(x)
ny = np.size(y)
nz = np.size(z)
    
try:
    eps = np.load(_REF_DIR + 'eps_save.npy')  # IBM indicator from the reference case grid
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
ACTIVE_CASES  = {_n for _n in SIM_DIRS if _n not in DISABLE_CASES}
for _n in DISABLE_CASES:
    _CASE_DROP_REASON[_n] = 'hidden by DISABLE_CASES (remove it there to draw)'

# Declared = the source line exists; found = the file is also on disk.
_SM_DECLARED = [_e for _e in _SMOOTH_REGISTRY
                if isinstance(globals().get(_e['var']), str)]
SMOOTH_NC = {}                       # name -> resolved path, in registry order
for _e_s in _SM_DECLARED:
    _p_s = _find_smooth_nc(globals()[_e_s['var']])
    if _p_s is not None:
        SMOOTH_NC[_e_s['name']] = _p_s
    else:
        print('  smooth %-12s NOT FOUND on disk (%s) — reference skipped'
              % (_e_s['name'], globals()[_e_s['var']]))
for _e_s in _SMOOTH_REGISTRY:
    if _e_s['name'] not in {_d['name'] for _d in _SM_DECLARED}:
        print('  smooth %-12s absent (%s commented out — reference dropped from '
              'every plot/table)' % (_e_s['name'], _e_s['var']))

SMOOTH_MAP  = {_e['name']: _e for _e in _SMOOTH_REGISTRY}
SMOOTH_KEYS = list(SMOOTH_NC)                       # every reference to be drawn
_SM_PRIMARY = SMOOTH_KEYS[0] if SMOOTH_KEYS else 'Sm_Neu'
_nc_smooth  = SMOOTH_NC.get(_SM_PRIMARY)            # the primary file (or None)
_smooth_re  = SMOOTH_MAP[_SM_PRIMARY]['Re'] if SMOOTH_KEYS else None
# Each smooth reference is a case in its own right for the normalization tables:
# it carries the viscosity of ITS Reynolds number (never the global Re=500 nu).
# 'Sm_Neu' is kept as an ALIAS of the primary so any lookup by that literal name
# still resolves.
for _k_s, _e_s in ((_k, SMOOTH_MAP[_k]) for _k in SMOOTH_KEYS):
    SIM_NU[_k_s] = 1.0 / (0.5 * float(_e_s['Re']) ** 2)
SIM_NU['Sm_Neu'] = SIM_NU.get(_SM_PRIMARY, nu)
SMOOTH_USTAR_FALLBACK = {_e['name']: _e['ustar'] for _e in _SMOOTH_REGISTRY}
if SMOOTH_KEYS:
    print('  smooth references drawn : %s' % ', '.join(
        '%s (Re_D=%.0f)' % (_k, SMOOTH_MAP[_k]['Re']) for _k in SMOOTH_KEYS))
    print('  smooth PRIMARY (backs the flat *_s names): %s  %s'
          % (_SM_PRIMARY, _nc_smooth))
else:
    print('  smooth reference : NONE declared/found — all smooth overlays skipped')
print('-' * 78)

###############################################################################
# ░░  USER INPUT — NORMALIZATION REFERENCE FRICTION VELOCITY  ░░
#
# THE knob for this file.  One friction velocity per case, used to normalize
# EVERY plot of that case:
#
#     z+  = z / l_in(case),   where  l_in(case) = nu(case) / USTAR_REF[case]
#     u+  = u / USTAR_REF[case]
#     tau = tau / USTAR_REF[case]**2
#
# ── DEFAULTS: each case in its OWN wall units ────────────────────────────────
# Each entry defaults to that case's SMOOTH-WALL reference friction velocity,
# so every case is drawn in its own true inner units.  This is the physically
# correct scaling for comparing Reynolds numbers: the log layers should
# COLLAPSE, with the higher Re showing a LONGER log region (larger Re_tau).
#     Re=500  : 0.0618 — smooth Re=500 .nc stored FrictionVelocity (== ustr_s1)
#     Re=750  : 0.0561 — smooth Re=750,  CLAUDE.md "Physics constants" table
#     Re=1000 : 0.0531 — smooth Re=1000, CLAUDE.md "Physics constants" table
#
# ── TO COMPARE ON ONE COMMON REFERENCE ───────────────────────────────────────
# SET THE VALUES EQUAL (e.g. put 0.0618 — or 0.0561 — in every entry).  All
# cases then share one velocity yardstick and the plots show the raw,
# UNCOLLAPSED Reynolds effect on a single scale.
#
# NOTE on what "equal" does and does not equate: setting these equal equates the
# VELOCITY scale (u+, tau/u*^2) exactly.  The LENGTH scale l_in = nu/u* still
# differs between the cases, because nu is fixed by the Reynolds number itself
# (nu = 1/(0.5*Re_D^2): 8.0e-6 at Re=500, 3.556e-6 at Re=750, 2.0e-6 at
# Re=1000) — that residual
# difference IS the Reynolds effect and cannot be normalized away.  To place two
# different Re on a literally identical z+ axis you would have to abandon wall
# units; the outer-unit figures (z^- panels) already provide that view.
#
# 'Sm_Neu' is the flat smooth NetCDF reference; its entry follows WHICHEVER
# smooth file was selected above (Re=500 → 0.0618, Re=1000 → 0.0531), so the
# smooth overlay never silently sits on a different axis than the valley cases.
#
# THE VALUES LIVE IN THE CASE REGISTRY (`ustar_ref`, top of the file) so that a
# commented-out data source cannot leave a stale entry behind.  Edit them there,
# or override an individual case here — the dict is a plain dict:
#     USTAR_REF['re750_oro'] = 0.0618      # e.g. one common velocity yardstick
###############################################################################
USTAR_REF = {_e['name']: _e['ustar_ref'] for _e in _DECLARED}
# One entry per DECLARED smooth reference (each on its own u*), plus the
# 'Sm_Neu' alias of the primary for any lookup by that literal name.
for _k_s in SMOOTH_KEYS:
    USTAR_REF[_k_s] = float(SMOOTH_MAP[_k_s]['ustar'])
USTAR_REF['Sm_Neu'] = float(USTAR_REF.get(_SM_PRIMARY, 0.0618))
###############################################################################

###############################################################################
# CASES — smooth reference first, then the DECLARED valley runs by increasing Re
# (styling comes straight from the case registry, so a commented-out source can
# never leave a dangling legend entry).  Fr = inf for every valley case: this
# ladder varies Reynolds number ONLY.
# Coordinate convention: simulation uses engineering coords
#   u/x = streamwise, v/y = wall-normal, w/z = spanwise.
#   Plots label the wall-normal axis as z+ (meteorological convention).
#
# Per-case Froude number (SIM_FR, built from the registry) is the switch that
# selects the wall-law form fitted to each simulation's velocity profile
# (mirrors config.Fr, which is per-run there):
#   Fr = np.inf (neutral)  → classical Monin–Obukhov (1954) log law of the wall,
#   Fr finite  (stratified)→ Obukhov (1971) stability-corrected law.
# Every case here is neutral, so all fits take the classical ln(z+ - d+) branch
# — bit-for-bit the same path results.py takes for its neutral case.  The
# Obukhov branch stays in the file but is unreachable on this ladder.
#
# PER-CASE PHYSICS (SIM_RE / SIM_NU / case_nu, built from the registry above):
# results.py could use the single global nu / u_star / l_in from config.py
# because all six of its cases shared Re = 500.  Here Re VARIES, so viscosity
# and inner length scale are per-case:  nu = 1/(0.5*Re_D^2),  l_in = nu/u*.
#
# u* SOURCE (in order of preference, resolved per case below):
#   1. The case's OWN measured Method-2 friction velocity, read from its pickle
#      (`u_star2`, plateau/crest) — the only correct u* in a rotating Ekman
#      layer (Kostelecky & Ansorge 2024 eq. 4.2).  Preferred whenever present.
#   2. The CLAUDE.md physics-table fallback (registry `ustar_fall`), used only
#      when the pickle has no u_star2.  The Re=750 / Re=1000 entries are
#      ESTIMATES (smooth-wall value + 20 % for orography), NOT measurements —
#      replace them once those flows are measured.  The console reports which
#      source was used for each case.
###############################################################################
CASES = ([{'name': _k, 'label': SMOOTH_MAP[_k]['label'],
           'color': SMOOTH_MAP[_k]['color'], 'ls': SMOOTH_MAP[_k]['ls'],
           'marker': SMOOTH_MAP[_k]['marker']} for _k in SMOOTH_KEYS]
         + [{'name': _e['name'], 'label': _e['label'], 'color': _e['color'],
             'ls': _e['ls'], 'marker': _e['marker']} for _e in _DECLARED])
CASE_STYLE = {_c['name']: _c for _c in CASES}

SIM_USTAR_MEASURED = {}   # filled after the pickles load (see the resolver below)

###############################################################################
# Cross-case statistics log.  Tee ALL subsequent stdout (data-loading messages,
# the per-case wall-unit table, the Ch.6 diagnostics and the end-of-run summary)
# to sim_stats_Re.log in the examples root.
#   * distinct from the per-case sim_stats.log PhAvg_rotated.py writes inside
#     each case directory (different directories, no clobber), AND
#   * distinct from results.py's cross-case sim_stats.log, which lives in this
#     SAME examples root — hence the _Re suffix: the Froude-ladder log and this
#     Reynolds-ladder log must not overwrite each other.
# Rewritten fresh ('w') on each run by start_stats_log.
###############################################################################
import datetime as _dt
start_stats_log(cwd + 'sim_stats_Re.log')
print('=' * 78)
print('results_Re.py — REYNOLDS-LADDER post-processing log   %s'
      % _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print('Examples root : %s' % cwd)
print('Case dirs     : %s' % {k: v for k, v in SIM_DIRS.items()})
print('Smooth ref    : %s' % (_nc_smooth or 'none found'))
print('Stratification: Fr = inf (NEUTRAL) for every case — Reynolds varies only')
print('=' * 78)

###############################################################################
# Load pickles — one per rough-wall simulation
#
# MEMORY: each sim1_results.pkl bundles ~70 dense 2-D fields (ny x nx).  On the
# stratified grid (1056x672) that is ~5.7 MB/field, ~400 MB/case, and every case
# is held in `sims` at once — several GB total, which slows and crashes the run.
# Two safe reductions are applied AS EACH pickle is loaded (see _load_case_pkl):
#   1. Drop the 2-D fields that results.py provably never reads (_DROP_KEYS —
#      PhAvg-only compute intermediates: ghost-filled interpolation planes, the
#      full stress tensor / mean-mean & cross dispersive components, unused
#      derivative and wavenumber fields, stencil selectors).  Verified by grep:
#      none of these names appears anywhere else in this file.
#   2. Downcast float64 arrays to float32 — halves the RAM of every retained
#      field.  Post-processing precision is unaffected (float32 ~ 7 sig figs;
#      the console tables report ~4, gradients/integrals are well within it).
# The full pickle dict is freed immediately after the pruned copy is built, so
# peak memory is ~one full pickle + the pruned cases, not all full pickles.
###############################################################################
_DROP_KEYS = frozenset({
    # Ghost-filled interpolation planes for the compact-derivative scheme
    'AvgPhU_i', 'AvgPhU_j', 'AvgPhV_i', 'AvgPhV_j', 'AvgPhW_i', 'AvgPhW_j',
    # Full phase-averaged stress tensor (only the triple-decomposition pieces
    # below are consumed) and the mean-mean (_G) contribution
    'AvgStrUU', 'AvgStrUV', 'AvgStrUW', 'AvgStrVV', 'AvgStrVW', 'AvgStrWW',
    'UU_G', 'UV_G', 'UW_G', 'VV_G', 'VW_G', 'WW_G',
    # NOTE: UW_disp, VW_disp, rey_uw are deliberately NOT dropped — the per-component
    # stress-family maps (Total / Reynolds / Turbulent / Dispersive, 6 components
    # each) need the FULL turbulent (rey_*) and dispersive (*_disp) tensors.
    # Unused velocity / dispersive-velocity derivative fields
    'du_dx', 'dv_dy', 'dw_dx',
    'dud_dy', 'dud_dx', 'dvd_dy', 'dvd_dx', 'dwd_dx', 'dwd_dy',
    # Gravity-wave wavenumber fields (recomputed on demand where needed)
    'm_dispV', 'k_dispV', 'm_dispU', 'k_dispU', 'km_dispV',
    # Global-geostrophic velocity fields and unused time derivatives
    'VelGblU', 'VelGblV', 'VelGblW', 'du_dt', 'ds_dt',
    # Compact-scheme stencil selector arrays and the interior/interface masks
    'case_v', 'case_h', 'case_v_g', 'case_v_itrp', 'case_h_itrp',
    'mask_v', 'mask_intr',
})


sims = {}
# Per-case provenance for the end-of-run summary (filled during loading below).
_prov = {_n: {'pickle': False, 'per_case_grid': False,
              'inst': {}, 'inst_skip': {}} for _n in SIM_DIRS}
for _name, _d in SIM_DIRS.items():
    _pkl = _d + 'sim1_results.pkl'
    if os.path.exists(_pkl):
        sims[_name] = _load_case_pkl(_pkl)
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
        # Diagnostic for the Ekman18/Re750/ layout ambiguity flagged above: that
        # directory is a sibling of the NetCDF reference dir Ekman18/Re500/, so
        # say plainly whether we found a simulation dir, a NetCDF dir, or nothing.
        # The same sentence is stored in _CASE_DROP_REASON so every figure that
        # wanted this case can repeat it instead of dropping the case silently.
        if os.path.isdir(_d):
            _ncs = [f for f in os.listdir(_d) if f.endswith('.nc')]
            if _ncs:
                print(f'  {_name}: {_d} exists but holds NetCDF ({_ncs[0]}), not a '
                      f'sim1_results.pkl.  This ladder expects a SIMULATION dir '
                      f'(run PhAvg_rotated.py there); a NetCDF reference would '
                      f'need loading through functions.load_ekman_nc_case instead.')
                _CASE_DROP_REASON[_name] = (
                    '%s holds NetCDF (%s) but no sim1_results.pkl — this ladder '
                    'needs a SIMULATION dir (run PhAvg_rotated.py there)'
                    % (_d, _ncs[0]))
            else:
                print(f'  {_name}: {_d} exists but carries no sim1_results.pkl — '
                      f'run stage b (PhAvg_rotated.py) in it.')
                _CASE_DROP_REASON[_name] = (
                    '%s exists but carries no sim1_results.pkl — run stage b '
                    '(PhAvg_rotated.py) in it' % _d)
                # The single most common cause: the .npy from stage a are there
                # but stage b was never run, so there is nothing to plot FROM.
                _npys = [f for f in os.listdir(_d) if f.endswith('.npy')]
                if _npys:
                    print(f'  {_name}: NOTE — {len(_npys)} .npy files ARE present '
                          f'in {_d} (stage a output).  Stage c plots only from '
                          f'sim1_results.pkl, so run PhAvg_rotated.py there to '
                          f'turn them into one.')
                    _CASE_DROP_REASON[_name] += (
                        ' (%d stage-a .npy ARE present there — only the stage-b '
                        'pickle is missing)' % len(_npys))
        else:
            print(f'  {_name}: directory {_d} does not exist.')
            _CASE_DROP_REASON[_name] = (
                'directory %s does not exist — fix the cwdN path at the top of '
                'this file, or place the case there' % _d)
        # The data may simply be somewhere else in the tree (grid-size subfolder
        # layout).  Say so — this is the difference between "no data" and "the
        # path is wrong", and it is the usual cause of a case that you KNOW you
        # have silently missing from every figure.
        _alt = _find_case_elsewhere(_d)
        if _alt:
            print('  %s: FOUND a sim1_results.pkl for this case name at %s'
                  % (_name, ', '.join(_alt)))
            print('     → the data IS on disk; point the cwdN line for %s at it '
                  '(this file does not follow it automatically — the cwdN lines '
                  'are the single source of truth).' % _name)
            _CASE_DROP_REASON[_name] += (
                '.  BUT a pickle for this case name EXISTS at %s — update the '
                'cwdN path' % ', '.join(_alt))

# A declared case whose pickle is missing is treated exactly like a commented-out
# source: it is removed from ACTIVE_CASES, so it cannot appear as a phantom
# legend entry / empty curve on any figure.  The warning above already said why.
_NO_PKL = {_n for _n in SIM_DIRS if _n not in sims}
if _NO_PKL:
    ACTIVE_CASES -= _NO_PKL
    print('  → dropped from every figure (no pickle): %s' % sorted(_NO_PKL))

# Fallback wall unit per case from the USER INPUT dict (used when USE_USTAR_REF
# is True, or for a case with no measured u*).  'Sm_Neu' (the flat smooth
# reference) carries the viscosity of the smooth file actually selected.
_L_IN = {_c: case_nu(_c) / _u for _c, _u in USTAR_REF.items() if _u}

# PHYSICAL (measured) friction velocity — must be resolved FIRST, because it is
# now the ACTIVE inner-scale normalization (see norm_ustar / USE_USTAR_REF).
print('-' * 78)
print('REYNOLDS LADDER — measured (physical) Method-2 friction velocity')
for _name in SIM_DIRS:
    _u, _src = _ustar_measured(_name)
    SIM_USTAR_MEASURED[_name] = _u
    print('    %-12s u*=%.5f  Re_tau=%8.1f   %s'
          % (_name, _u, _u ** 2 / case_nu(_name), _src))

# ACTIVE normalization table — what the inner-scale figures actually use.
_ACTIVE_USTAR = {}
for _c in [_k['name'] for _k in CASES if _k['name'] in USTAR_REF]:
    _um = SIM_USTAR_MEASURED.get(_c)
    if (not USE_USTAR_REF) and _um is not None and np.isfinite(_um) and _um > 0:
        _ACTIVE_USTAR[_c] = (float(_um), 'measured')
    else:
        _ACTIVE_USTAR[_c] = (float(USTAR_REF[_c]), 'USTAR_REF')
print('  normalization in use: %s'
      % ('USER INPUT USTAR_REF (USE_USTAR_REF=True)' if USE_USTAR_REF
         else "each case's OWN measured u* (USE_USTAR_REF=False)"))
print('  %-12s %6s %12s %11s %14s %10s'
      % ('case', 'Re_D', 'nu', 'u*_norm', 'l_in = nu/u*', 'source'))
for _c, (_uv, _usrc) in _ACTIVE_USTAR.items():
    print('  %-12s %6s %12.4e %11.5f %14.4e %10s'
          % (_c, ('%.0f' % SIM_RE[_c]) if _c in SIM_RE
             else ('%.0f(sm)' % SMOOTH_MAP[_c]['Re'] if _c in SMOOTH_MAP
                   else '(sm)'),
             case_nu(_c), _uv, case_nu(_c) / _uv, _usrc))
_uvals = {round(_v[0], 10) for _c, _v in _ACTIVE_USTAR.items() if _c in ACTIVE_CASES}
if len(_uvals) == 1:
    print('  → one common velocity scale (u*=%.4f) across active cases.  Plots '
          'show the raw, uncollapsed Reynolds effect; the length scale l_in '
          'still differs through nu (that IS the Re effect).' % _uvals.pop())
else:
    print('  → each case in its OWN wall units '
          '(log layers should collapse; higher Re = longer log region).')
print('-' * 78)

###############################################################################
# PLOT READINESS — WHY A CASE DOES OR DOES NOT APPEAR IN THE *_allRe FIGURES.
#
# Every shared z+/z- figure loops over the active cases and SKIPS any case for
# which `gy_in(case)` is None or the plotted field is absent:
#
#       _upr = gv('u_plus_rot', case);  _yi = gy_in(case)
#       if _upr is None or _yi is None:
#           continue                     <-- silent drop, no message
#
# so a case with a stale pickle (no bundled grid, different ny than the
# reference) used to vanish from EVERY comparison figure without a word, while
# the case that happens to own the reference grid still drew.  This table makes
# that visible: it reports, per case, whether the pickle loaded, which grid its
# profiles are placed on (pickle / its own `grid` file / the reference grid),
# and which of the core plotted fields are missing.  A case marked SKIPPED here
# is exactly a case you will not see in the figures — and the reason is stated.
###############################################################################
_CORE_FIELDS = ('u_plus_rot', 'w_plus_rot', 'u_star2', 'TKE', 'rey_uv',
                'du_dy', 'AvgPhU')
print('-' * 78)
print('PLOT READINESS — which cases the *_allRe figures can actually draw')
print('  %-12s %-7s %-10s %-16s %s'
      % ('case', 'pickle', 'ny(prof)', 'z+ axis from', 'missing core fields'))
_READY, _NOT_READY = [], []
for _name in SIM_DIRS:
    _sd   = sims.get(_name, {})
    _pk   = 'yes' if _sd else 'NO'
    _prof = _sd.get('u_plus_rot')
    _ny   = ('%d' % len(_prof)) if _prof is not None else '-'
    _yg, _src = _case_axis_y(_name)
    _miss = [_f for _f in _CORE_FIELDS if _sd.get(_f) is None]
    print('  %-12s %-7s %-10s %-16s %s'
          % (_name, _pk, _ny, _src if _yg is not None else 'NONE',
             ','.join(_miss) if _miss else '(none)'))
    if _yg is None or _prof is None:
        _NOT_READY.append((_name, _src if _yg is None else 'no u_plus_rot'))
    else:
        _READY.append(_name)
for _name, _why in _NOT_READY:
    print('  → %s will be SKIPPED by the shared z+/z- figures: %s' % (_name, _why))
    if _name in sims and sims[_name].get('y') is None:
        print('    fix: re-run stage b (PhAvg_rotated.py) in %s so the pickle '
              'bundles its own grid, or put that case\'s `grid` file in the '
              'directory.' % SIM_DIRS[_name])
print('CASES PLOTTED: %s%s'
      % (sorted(set(_READY) & ACTIVE_CASES) or 'NONE',
         (' + smooth ' + ', '.join(SMOOTH_KEYS)) if SMOOTH_KEYS else ''))
print('-' * 78)

###############################################################################
# Load EVERY declared smooth-wall reference (flat NetCDF) — the benchmark curves.
#
# `SMOOTH_KEYS` / `SMOOTH_NC` were resolved with the source lines above (nc1,
# nc2, …).  Each reference is loaded into `smooths[key]` as a self-contained
# dict of that file's `*_s` quantities, and the FLAT module names (y_in_s,
# Rxy_s, ustr_s1, SMOOTH_COLOR, …) are BOUND to one reference at a time by
# `_bind_smooth`.  Plot blocks iterate with `for _ in smooth_refs():`, which
# rebinds those names per reference — so one block body draws every reference.
#
# EVERY flat name is PRE-SEEDED first, so a missing (or switched-off) smooth
# reference can never raise NameError further down: the scalars fall back to the
# USTAR_REF entry and the arrays to None, while `_smooth_loaded = False` keeps
# all smooth curves out of the figures.
###############################################################################
# The complete set of flat names one smooth reference owns (pre-seed + rebind).
_SMOOTH_FLAT = (
    'sy', 'nys', 'U_s', 'V_s', 'W_s', 'su', 'sw', 'alpha_s',
    'alpha_str_s', 'y_s', 'y_in_s', 'rU_s', 'rV_s', 'rW_s', 'rP_s',
    'rs_s', 'G_x_s', 'G_z_s', 'G_s', 'U_s_p', 'W_s_p', 'GblU_s',
    'GblW_s', 'Rxx_s', 'Rxy_s', 'Rxz_s', 'Ryy_s', 'Ryz_s', 'Rzz_s',
    'TKE_s', 'cor_yx_s', 'I_corr_yx_s', 'du_dy_s', 'visc_yx_s',
    'tau_yx_s', 'cor_yz_s', 'I_corr_yz_s', 'dw_dy_s', 'visc_yz_s',
    'tau_yz_s', 'AVG_TKE_V_s', 'AVG_TKE_V_s_i', 'sx', 'TKE_s_dx',
    'TKE_s_dy', 'Adv_s', 'Disp_U_s', 'Disp_V_s', 'Disp_W_s',
    'ustr_s1', 'ustr_s1_stored',
    # style of the reference currently bound (used by every smooth plot call)
    'SMOOTH_COLOR', 'SMOOTH_LS', 'SMOOTH_LABEL', 'SMOOTH_MARKER', 'SMOOTH_KEY',
)
_smooth_loaded = False
for _k_s in _SMOOTH_FLAT:
    globals().setdefault(_k_s, None)
ustr_s1        = float(USTAR_REF.get('Sm_Neu', u_star))   # smooth normalizer
ustr_s1_stored = ustr_s1                                  # smooth PHYSICAL u*


smooths = {}                      # key -> that reference's flat quantities
for _k_s in list(SMOOTH_KEYS):
    try:
        smooths[_k_s] = _load_one_smooth(_k_s, SMOOTH_NC[_k_s])
        print('Loaded smooth reference: %-12s %s  (Re_D=%.0f, nu=%.4e, '
              'u*_stored=%.5f, u*_norm=%.5f)'
              % (_k_s, SMOOTH_NC[_k_s], SMOOTH_MAP[_k_s]['Re'],
                 smooths[_k_s]['_nu'], smooths[_k_s]['ustr_s1_stored'],
                 smooths[_k_s]['ustr_s1']))
    except Exception as _e_ld:
        # One unreadable reference must not take the whole run down: report it
        # and carry on with the others (same policy as a missing pickle).
        print('Warning: smooth reference %s could not be loaded (%s) — skipped'
              % (_k_s, _e_ld))
SMOOTH_KEYS = [_k for _k in SMOOTH_KEYS if _k in smooths]   # only what loaded
if not PLOT_SMOOTH:
    SMOOTH_KEYS = []                                        # master off switch
if SMOOTH_KEYS and _SM_PRIMARY not in SMOOTH_KEYS:
    _SM_PRIMARY = SMOOTH_KEYS[0]                            # primary must exist


# Bind the PRIMARY reference so the flat names are populated for everything that
# does not iterate (helpers, tables, the 2-D panel builder's default).
_smooth_loaded = bool(SMOOTH_KEYS) and _bind_smooth(_SM_PRIMARY)
if not SMOOTH_KEYS:
    print('Warning: no smooth reference available — every smooth overlay is '
          'skipped.  Declared sources: %s'
          % ([_e['var'] for _e in _SMOOTH_REGISTRY] or 'none'))
# NOTE: there is no single "_ustar_ref" any more — every smooth curve is scaled
# by the u* of the reference being drawn (`ustr_s1`, rebound per reference by
# smooth_refs()), and every valley curve by norm_ustar(case).

###############################################################################
# Derived quantities stored back into each sim dict
###############################################################################
for _name, _sd in sims.items():
    if 'rey_uv' in _sd and 'du_dy' in _sd:
        # Production P = -⟨u'v'⟩·∂⟨u⟩/∂z is a single elementwise product of two
        # already-pickled fields (cheap combining, not a derivative build) — kept
        # here so it need not round-trip through the pickle.
        _sd['P'] = -_sd['rey_uv'] * _sd['du_dy']
    # TKE-advection fields (dTKE_dx, dTKE_dy, Adv) are computed ONCE in
    # PhAvg_rotated.py (stage b, compact-derivative scheme) and pickled; stage c
    # (this file) only READS them — no derivative field is rebuilt in results.py
    # (pipeline rule).  A legacy pickle lacking 'Adv' is reported and its
    # TKE-advection panels are skipped, not recomputed.
    if 'TKE' in _sd and 'Adv' not in _sd:
        print(f'Note: {_name} pickle carries no Adv/dTKE_d* (legacy/stale — '
              f'regenerate with the current PhAvg_rotated.py); its TKE-advection '
              f'panels will be skipped.')

###############################################################################
# Instantaneous fluctuation planes (inst_u/v/w/scal) — READ from the pickle.
# u'ᵢ = uᵢ − ⟨uᵢ⟩ₓ (one x–y plane) is the ONLY pipeline quantity derived from a
# RAW record (flow.*/scal.*).  That raw-record read now lives in PhAvg_rotated.py
# (stage b), which pickles inst_*; stage c (this file) therefore NEVER opens
# flow.*/scal.* — it just notes which inst_* each pickle carries, for the P19–P22
# snapshot panels and the end-of-run summary.  A legacy pickle predating these
# keys simply lacks them and the affected snapshot panel is skipped.
###############################################################################
for _iname in SIM_DIRS:
    _sd = sims.get(_iname)
    if not _sd:
        continue
    for _ikey in ('inst_u', 'inst_v', 'inst_w', 'inst_scal'):
        if _sd.get(_ikey) is not None:
            _prov[_iname]['inst'][_ikey] = 'pickled'
        else:
            _prov[_iname]['inst_skip'][_ikey] = 'absent (regenerate pickle)'


###############################################################################
###############################################################################
####  4.  PLOTS
#
#  All figure generation, gated by the `plotRes` switch in section 2.  This
#  section only READS what sections 2 and 3 produced; it computes no new
#  physical quantity (see the pipeline load-balancing rule in CLAUDE.md).
###############################################################################
###############################################################################


# Plot results
#
# ONE pass, exactly as results.py.  The normalization is not a "mode" here — it
# is the USER INPUT dict USTAR_REF (top of file): each case is normalized by its
# own USTAR_REF entry, and setting the entries equal puts every case on one
# common velocity scale.  So a single set of figures is produced, straight into
# fig_Re/.
if (1 == plotRes):
    import matplotlib.lines as mlines
    from matplotlib.lines import Line2D
    import os as _os

    ###########################################################################
    # Per-simulation style objects — CASES is the single source of truth.
    # Direct lookup: CASE_MAP['re500_oro']['color'], CASE_MAP['Sm_Neu']['ls'], …
    # Derived lists (SIM_NAMES, SIM_COLORS, …) kept for loop compatibility.
    ###########################################################################
    CASE_MAP  = {c['name']: c for c in CASES}
    # `_smooth` is the PRIMARY smooth reference's style; SMOOTH_COLOR / _LS /
    # _LABEL / _MARKER are NOT frozen here — `_bind_smooth` rebinds them to
    # whichever reference `smooth_refs()` is currently yielding, so a plot block
    # written for one smooth curve draws every declared reference in its own
    # colour.  Between iterations they hold the primary reference (bound at load).
    _smooth   = CASE_MAP.get(_SM_PRIMARY, {'name': 'Sm_Neu', 'label': 'Smooth',
                                           'color': '#636363', 'ls': '-',
                                           'marker': 'o'})
    _smooths  = [CASE_MAP[k] for k in SMOOTH_KEYS]      # every reference drawn
    _sims     = [c for c in CASES
                 if c['name'] not in SMOOTH_MAP and c['name'] in ACTIVE_CASES]

    SIM_NAMES      = [c['name']   for c in _sims]
    SIM_LABELS     = [c['label']  for c in _sims]
    SIM_COLORS     = [c['color']  for c in _sims]
    SIM_LINESTYLES = [c['ls']     for c in _sims]
    SIM_MARKERS    = [c['marker'] for c in _sims]
    # Axis windows, figure directory, IBM colour and the log-law fit region are
    # USER PARAMETERS — declared in the USER PARAMETERS section, not here.
    # ── Figure output: EVERY figure from this file goes in fig_Re/ ─────────
    # Kept distinct from results.py's fig/ so the Froude-ladder figures are
    # never overwritten by this ladder.  Every savefig in the body below is
    # written as `_figdir + 'NAME.png'`, so this one definition covers them all.
    _figroot = cwd + FIG_SUBDIR
    _figdir  = _figroot
    _os.makedirs(_figroot, exist_ok=True)
    print('Figures → %s' % _figroot)

    # Global line-quality settings
    plt.rcParams.update({'lines.linewidth': 1.5, 'font.size': 10})

    # Helpers gv/gy_in/geps/gmask0/geps_f/gx_in/ghill/_xprof/all_handles/
    # sim_handles/_mark_h/_mark_heights_v → FUNCTION DEFINITIONS section (top).

    ###########################################################################
    # Shared axis conventions for the z-resolved profiles (Req 8)
    #   inner units : z+ = y / l_in  (single reference l_in), capped at z+ = 200
    #   outer units : z- = y / u_star2(h)  (per-case),        capped at z- = 4
    # Z_PLUS_MAX / Z_MINUS_MAX are the axis limits requested for the two views.
    ###########################################################################
    # (Z_PLUS_MAX / Z_MINUS_MAX are USER PARAMETERS — see that section.)

    # Helpers _z_out/_oro_layer_idx/_smo_layer_idx → FUNCTION DEFINITIONS (top).

    ###########################################################################
    # SHEAR-STRESS MOMENTUM BALANCE — P46 (tau_zx) + P47 (tau_zy), inner units.
    #
    # Defined HERE (right after the case/axis setup) rather than inline in
    # SECTION 3 so the RESULTS_ONLY=shear flag can draw just these two figures
    # and exit, without executing the other ~45.  A full run calls it from its
    # original SECTION 3 position, so the figure order is unchanged.
    #
    # 🔒 LOCKED — STANDARD shear-stress budget (CLAUDE.md "Standard shear-stress
    # budget formulation"; verified against Kostelecky & Ansorge fig-4).  DO NOT
    # MODIFY the P46/P47 (inner) or P48/P49 (outer) tau_zx/tau_zy blocks below.
    #     Viscous  V = +visc_*      Reynolds R = -(turbulent + dispersive)   (gold)
    #     Temporal   = +dudt/+dwdt  Total = C + V + R + temporal              (black)
    #     Coriolis:  C_zx = -I_corr_yx  BUT  C_zy = +I_corr_yz  (Levi-Civita
    #                ε_{ik3}: the two Coriolis terms carry OPPOSITE signs — this
    #                is the only difference between the tau_zx and tau_zy panels,
    #                and it is what keeps each Total height-constant).
    #     DISPLAY: the SPANWISE (tau_zy) panels are negated by _SP for paper
    #                handedness (FIG4_PAPER_SPANWISE_SIGN, mirroring
    #                config.fig4_paper_spanwise_sign / PhAvg_rotated.py); this scales
    #                only the plotted tau_zy curves — closure and u* are untouched.
    #
    # The Reynolds shear stress is drawn as the SINGLE combined curve
    # (turbulent + dispersive = rey_flux_yx / rey_flux_yz of PhAvg_rotated.py).
    # The magenta/cyan turbulent-vs-dispersive split is deliberately NOT drawn
    # here — it made the figure unreadable.  The outer-unit twins further down
    # still show the split, which is why the shared _term_handles keeps them.
    ###########################################################################
    def _plot_shear_stress_balance():
        # Reduced, LOCAL handles: no Turbulent/Dispersive (not drawn here).  The
        # module-level _term_handles/_total_handle are left untouched for the
        # outer-unit plots, which do still draw the split.
        _rey_handles = [
            Line2D([0], [0], color='steelblue',   ls='-', lw=1.5, label='Coriolis'),
            Line2D([0], [0], color='firebrick',   ls='-', lw=1.5, label='Viscous'),
            Line2D([0], [0], color='gold',        ls='-', lw=1.5, label='Reynolds'),
            Line2D([0], [0], color='saddlebrown', ls='-', lw=1.5, label='Temporal'),
        ]
        _tot_handle = Line2D([0], [0], color='black', ls='-', lw=1.5,
                             label=r'Total $\Sigma$')
        # Boundary-layer region markers.  Colour already encodes the stress TERM,
        # so the markers must be black — marking every case would be unreadable.
        # Only ONE representative valley curve (the first case that actually draws
        # a Reynolds curve) + the smooth curve carry them, on the gold Reynolds
        # curve.  Small size keeps the figure uncluttered.
        _MK_SIZE = 3.5

        # ---- 3a. Shear stress tau_zx — streamwise/wall-normal (INNER units) ----
        plt.figure(figsize=FIG_SIZE_WIDE, dpi=FIG_DPI)
        for _ in smooth_refs():
            _rey_s = -np.mean(Rxy_s, axis=1)      # flat wall: Reynolds ≡ turbulent
            _tot_s = -I_corr_yx_s + np.mean(visc_yx_s, axis=1) + _rey_s
            plt.plot(y_in_s[:N_ROWS_SMOOTH_TAU], -I_corr_yx_s[:N_ROWS_SMOOTH_TAU]/ustr_s1**2,
                     color='steelblue', linestyle=SMOOTH_LS, linewidth=1.5)
            plt.plot(y_in_s[:N_ROWS_SMOOTH_TAU], np.mean(visc_yx_s, axis=1)[:N_ROWS_SMOOTH_TAU]/ustr_s1**2,
                     color='firebrick', linestyle=SMOOTH_LS, linewidth=1.5)
            plt.plot(y_in_s[:N_ROWS_SMOOTH_TAU], _rey_s[:N_ROWS_SMOOTH_TAU]/ustr_s1**2,
                     color='gold', linestyle=SMOOTH_LS, linewidth=1.5)
            plt.plot(y_in_s[:N_ROWS_SMOOTH_TAU], _tot_s[:N_ROWS_SMOOTH_TAU]/ustr_s1**2,
                     color='black', linestyle=SMOOTH_LS, linewidth=1.5)
            mark_layers(y_in_s[:N_ROWS_SMOOTH_TAU], _rey_s[:N_ROWS_SMOOTH_TAU]/ustr_s1**2, _smo_layer_idx(),
                        filled=False, color='black', size=_MK_SIZE)
        _marked = False                        # only the first valley curve is marked
        for case, ls in zip(SIM_NAMES, SIM_LINESTYLES):
            _Ic = gv('I_corr_yx', case)
            _vx = gv('visc_yx',   case)
            _rv = gv('rey_uv',    case)
            _dv = gv('UV_disp',   case)
            _dt = gv('dudt',      case)
            _yi = gy_in(case)
            if _Ic is None or _yi is None:
                continue
            _yn = _yi[:limity]
            plt.plot(_yn, -_Ic[:limity]/norm_ustar(case)**2, color='steelblue', linestyle=ls)
            _tot = -np.asarray(_Ic, dtype=float)
            _vxp = _xprof(case, _vx)
            if _vxp is not None:
                plt.plot(_yn, _vxp[:limity]/norm_ustar(case)**2, color='firebrick', linestyle=ls)
                _tot = _tot + np.asarray(_vxp, dtype=float)
            # Reynolds = turbulent ⟨u''v''⟩ (rey_uv) + dispersive ũṽ (UV_disp),
            # drawn as ONE curve; both still enter the Total.
            _rvp = _xprof(case, _rv)
            _dvp = _xprof(case, _dv)
            if _rvp is not None:
                _tot = _tot - np.asarray(_rvp, dtype=float)
            if _dvp is not None:
                _tot = _tot - np.asarray(_dvp, dtype=float)
            if _rvp is not None and _dvp is not None:
                _rey = -(np.asarray(_rvp) + np.asarray(_dvp))/norm_ustar(case)**2
                plt.plot(_yn, _rey[:limity], color='gold', linestyle=ls)
                if not _marked:
                    mark_layers(_yn, _rey[:limity], _oro_layer_idx(case),
                                filled=True, color='black', size=_MK_SIZE)
                    _marked = True
            if _dt is not None:
                plt.plot(_yn, _dt[:limity]/norm_ustar(case)**2, color='saddlebrown', linestyle=ls)
                _tot = _tot + np.asarray(_dt, dtype=float)
            plt.plot(_yn, _tot[:limity]/norm_ustar(case)**2, color='black', linestyle=ls, linewidth=1.5)
        plt.legend(handles=_rey_handles + [_tot_handle] + all_handles(),
                   fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL, loc='upper right')
        _mark_h('v')
        add_marker_legend(oro=True, smooth=_smooth_loaded, case_lines=True,
                          shade_case=True, smooth_ls=SMOOTH_LS, smooth_color=SMOOTH_COLOR)
        plt.xlim(0, Z_PLUS_MAX)
        # y-scale left free (autoscaled) so curves dipping below 0 are shown.
        plt.xlabel(r'$z^+$')
        plt.ylabel(r'$\langle\bar{\tau}_{zx}\rangle^+$')
        plt.title(r'Shear stress $\tau_{zx}$ — all Re (neutral) ($z^+\leq200$)')
        plt.grid(True)
        plt.savefig(_figdir+'P46_MomBal_tauyx_allRe.png', dpi=FIG_DPI)
        plt.show()

        # ---- 3b. Shear stress tau_zy — spanwise/wall-normal (INNER units) ----
        # 🔒 LOCKED — STANDARD budget (see CLAUDE.md "Standard shear-stress budget
        # formulation").  Same construction as tau_zx, but the spanwise Coriolis
        # carries the OPPOSITE sign (Levi-Civita ε_{ik3}): C_zy = +I_corr_yz (vs
        # −I_corr_yx for tau_zx); R_zy = −(turb+disp).  DISPLAY: the whole τ_zy panel
        # is negated by _SP for paper handedness (mirrors PhAvg_rotated.py /
        # fig4_smooth_standalone.py); the physical closure and u* are unaffected.
        _SP = -1.0 if FIG4_PAPER_SPANWISE_SIGN else 1.0
        plt.figure(figsize=FIG_SIZE_WIDE, dpi=FIG_DPI)
        for _ in smooth_refs():
            _rey_sz = -np.mean(Ryz_s, axis=1)     # R_zy = -⟨v'w'⟩
            _tot_sz = I_corr_yz_s + np.mean(visc_yz_s, axis=1) + _rey_sz
            plt.plot(y_in_s[:N_ROWS_SMOOTH_TAU], _SP*I_corr_yz_s[:N_ROWS_SMOOTH_TAU]/ustr_s1**2,
                     color='steelblue', linestyle=SMOOTH_LS, linewidth=1.5)
            plt.plot(y_in_s[:N_ROWS_SMOOTH_TAU], _SP*np.mean(visc_yz_s, axis=1)[:N_ROWS_SMOOTH_TAU]/ustr_s1**2,
                     color='firebrick', linestyle=SMOOTH_LS, linewidth=1.5)
            plt.plot(y_in_s[:N_ROWS_SMOOTH_TAU], _SP*_rey_sz[:N_ROWS_SMOOTH_TAU]/ustr_s1**2,
                     color='gold', linestyle=SMOOTH_LS, linewidth=1.5)
            plt.plot(y_in_s[:N_ROWS_SMOOTH_TAU], _SP*_tot_sz[:N_ROWS_SMOOTH_TAU]/ustr_s1**2,
                     color='black', linestyle=SMOOTH_LS, linewidth=1.5)
            mark_layers(y_in_s[:N_ROWS_SMOOTH_TAU], _SP*_rey_sz[:N_ROWS_SMOOTH_TAU]/ustr_s1**2, _smo_layer_idx(),
                        filled=False, color='black', size=_MK_SIZE)
        _marked = False                        # only the first valley curve is marked
        for case, ls in zip(SIM_NAMES, SIM_LINESTYLES):
            _Iz  = gv('I_corr_yz', case)
            _vz  = gv('visc_yz',   case)
            _rw  = gv('rey_vw',    case)
            _dvw = gv('VW_disp',   case)
            _dw  = gv('dwdt',      case)
            _yi = gy_in(case)
            if _Iz is None or _yi is None:
                continue
            _yn = _yi[:limity]
            plt.plot(_yn, _SP*_Iz[:limity]/norm_ustar(case)**2, color='steelblue', linestyle=ls)
            _tot = np.asarray(_Iz, dtype=float)
            _vzp = _xprof(case, _vz)
            if _vzp is not None:
                plt.plot(_yn, _SP*_vzp[:limity]/norm_ustar(case)**2, color='firebrick', linestyle=ls)
                _tot = _tot + np.asarray(_vzp, dtype=float)
            # Reynolds = turbulent ⟨v''w''⟩ (rey_vw) + dispersive ṽw̃ (VW_disp).
            _rwp = _xprof(case, _rw)
            _dwp = _xprof(case, _dvw)
            if _rwp is not None:
                _tot = _tot - np.asarray(_rwp, dtype=float)
            if _dwp is not None:
                _tot = _tot - np.asarray(_dwp, dtype=float)
            if _rwp is not None and _dwp is not None:
                _rey = -(np.asarray(_rwp) + np.asarray(_dwp))/norm_ustar(case)**2
                plt.plot(_yn, _SP*_rey[:limity], color='gold', linestyle=ls)
                if not _marked:
                    mark_layers(_yn, _SP*_rey[:limity], _oro_layer_idx(case),
                                filled=True, color='black', size=_MK_SIZE)
                    _marked = True
            if _dw is not None:
                plt.plot(_yn, _SP*_dw[:limity]/norm_ustar(case)**2, color='saddlebrown', linestyle=ls)
                _tot = _tot + np.asarray(_dw, dtype=float)
            plt.plot(_yn, _SP*_tot[:limity]/norm_ustar(case)**2, color='black', linestyle=ls, linewidth=1.5)
        plt.legend(handles=_rey_handles + [_tot_handle] + all_handles(),
                   fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL, loc='upper right')
        _mark_h('v')
        add_marker_legend(oro=True, smooth=_smooth_loaded, case_lines=True,
                          shade_case=True, smooth_ls=SMOOTH_LS, smooth_color=SMOOTH_COLOR)
        plt.xlim(0, Z_PLUS_MAX)
        # y-scale left free (autoscaled), exactly as for tau_zx.
        plt.xlabel(r'$z^+$')
        plt.ylabel(r'$\langle\bar{\tau}_{zy}\rangle^+$')
        plt.title(r'Shear stress $\tau_{zy}$ — all Re (neutral) ($z^+\leq200$)')
        plt.grid(True)
        plt.savefig(_figdir+'P47_MomBal_tauyz_allRe.png', dpi=FIG_DPI)
        plt.show()

    # Selective replot: draw only P46 + P47 and stop, so these two can be
    # iterated on without running the other ~45 figures (RESULTS_ONLY=shear).
    if _ONLY_SHEAR:
        _plot_shear_stress_balance()
        raise SystemExit(0)

    ###########################################################################
    # Helper: side-by-side 2D pcolormesh panels for all available Fr.
    # A single shared colorbar is placed in an explicit dedicated axes at the
    # far right.  Colour limits are the global max/min across ALL panels.
    ###########################################################################
    _DIVERGING_CMAPS = {'RdBu_r', 'coolwarm', 'seismic', 'PiYG', 'bwr', 'RdYlBu'}

    # Helpers _case_grid/_row_to_height → FUNCTION DEFINITIONS section (top).

    # (Z_PLUS_CONTOUR_MAX is a USER PARAMETER — see that section.  It is the
    #  single control for the 2-D panel window in every Z_NORM mode.)

    ###########################################################################
    # WALL-NORMAL SCALE REPORT — printed every run so the number on a panel
    # axis can always be traced back to the length it was divided by.
    #
    # It exists because a cropped panel is easy to misread as the full domain:
    # the 2-D panels show only Z_PLUS_CONTOUR_MAX REFERENCE wall units of
    # height (~29 % of the domain), so their top tick is the CROP, not the
    # domain top.  Both are printed below, in the active units.
    ###########################################################################
    print('-' * 78)
    print('WALL-NORMAL SCALE REPORT   (Z_NORM = %r  ->  axis %s)'
          % (Z_NORM, _z_axis_label()))
    print('  divisor per case = norm_z_len(case, %r):  '
          'inner delta_nu=nu/u_tau | outer delta=u_tau/f | topo h' % Z_NORM)
    print('  %-12s %10s %12s %12s %12s %11s %11s'
          % ('case', 'u_tau', 'divisor', 'domain top', 'panel top',
             'h (crest)', 'u_tau src'))
    for _c in SIM_NAMES:
        _yc, _ = _case_axis_y(_c)
        if _yc is None:
            print('  %-12s (no grid — skipped)' % _c)
            continue
        _dv = norm_z_len(_c, Z_NORM)
        print('  %-12s %10.6f %12.5e %12.4f %12.4f %11.4f %11s'
              % (_c, norm_ustar(_c), _dv, float(_yc[-1]) / _dv,
                 _contour_zmax(True), case_hill_pos(_c, Z_NORM),
                 'pickle' if sims.get(_c, {}).get('u_star') is not None
                 else 'FALLBACK'))
    print('  panel crop = %.1f reference wall units = %.6f physical '
          '= %.1f%% of the domain'
          % (Z_PLUS_CONTOUR_MAX, _contour_zmax(False),
             100.0 * _contour_zmax(False) / float(y[-1])))
    print('  NOTE: the panel top above is the CROP, not the domain top — a 2-D '
          'panel never')
    print('        shows the full domain.  Raise Z_PLUS_CONTOUR_MAX to widen '
          'every mode at once.')
    print('-' * 78)
    # Helpers _contour_zmax/_clip_rows → FUNCTION DEFINITIONS section (top).

    # Helpers _smooth_field_2d/_panel_grid_shape/plot2D_allRe
    # → FUNCTION DEFINITIONS section (top of file).

    # One colour for the IBM solid in every cross-case plot, so the body always
    # reads the same.  Matches the canonical single-case convention (PlotField.
    # plot2D_div legend "Solid IBM elements").
    # (_IBM_COLOR is a USER PARAMETER — see that section.)

    # Helpers _shade_ibm/plot2D_div_allcases → FUNCTION DEFINITIONS (top).

    ###########################################################################
    # SECTION 0 — NEW FIGURES (drawn first so they can be produced on their own
    # by running just this block against an already-loaded namespace).
    #
    #   P29b  wind turning angle α(x⁺,z⁺) = atan2(⟨v⟩,⟨u⟩)   [met. labels;
    #         engineering arrays AvgPhW = spanwise, AvgPhU = streamwise]
    #   P24d2 second wall-normal derivative ∂²⟨u⟩/∂z², whose zero isoline is the
    #         inflection line of the streamwise profile.  Read together with the
    #         existing P24de (∂⟨u⟩/∂z = 0 → separation / reattachment): the
    #         inflection line marks where the shear stops steepening and starts
    #         relaxing, i.e. where the separated shear layer detaches from the
    #         surface, which is far easier to localise than the shear zero itself.
    ###########################################################################

    # ── P29b: 2-D wind turning angle ────────────────────────────────────────
    # arctan2 (not arctan of the ratio) so the angle stays bounded in [-180,180]
    # where ⟨u⟩ reverses inside the valley — the same fix already applied to the
    # 1-D P30 veer profile.  Solid cells are NaN'd so they neither set the colour
    # limits nor produce a spurious α = atan2(0,0) = 0 in the recirculation shade.
    for _cn in SIM_NAMES:
        _sd = sims.get(_cn)
        if _sd is None:
            continue
        _phu, _phw = gv('AvgPhU', _cn), gv('AvgPhW', _cn)
        if _phu is None or _phw is None:
            continue
        _ang = np.arctan2(_phw, _phu)          # RADIANS (bounded [-pi, pi])
        _sd['veer_2d'] = np.where(geps(_cn) >= 0.5, np.nan, _ang)

    plot2D_allRe('veer_2d',
                 r'Wind turning angle $\alpha=\arctan(\langle v\rangle/\langle u\rangle)$ — all Re (neutral)',
                 'RdBu_r', 'P29b_TurningAngle2D_allRe.png',
                 include_smooth=False, shared_scale=True,
                 cbar_label=r'$\alpha$ (rad)',
                 overlay_contours=True, n_contours=12, contour_fmt='%.2f')

    for _cn in SIM_NAMES:
        sims.get(_cn, {}).pop('veer_2d', None)

    # ── P24d2: inflection line ∂²⟨u⟩/∂z² = 0 ────────────────────────────────
    # du_dy is pickled per case; differentiate it once more with the same IBM-aware
    # Fornberg stencils used for the first derivative, on THIS case's own y grid
    # (the stratified runs are on a different wall-normal grid than the neutral one).
    # Inner scaling: ∂²u⁺/∂z⁺² = (l_in²/u_*)·∂²u/∂z² = (nu²/u_*³)·∂²u/∂z².
    # Cap = z+ <= 200 re-expressed in the ACTIVE axis units (same physical
    # height in every mode — see _zplus_to_mode).
    _zI_lim_in = _zplus_to_mode(200.0)

    _d2_panels = []
    _d2_fig = 'P24d2_inflection_d2udz2_allRe.png'
    for _cn_a, _why_a, _case_lvl in _case_availability('du_dy')[1]:
        _note_missing(_d2_fig, 'du_dy', _cn_a, _why_a, once=_case_lvl)
    for _cname, _clbl in zip(SIM_NAMES, SIM_LABELS):
        _dudy = gv('du_dy', _cname)
        if _dudy is None:
            continue
        _epsc = geps(_cname)
        if _dudy.shape != _epsc.shape:
            print(f'P24d2: {_cname} du_dy shape {_dudy.shape} != eps {_epsc.shape}; skipped.')
            continue
        _yg = sims[_cname].get('y', y)
        # ∂²⟨u⟩/∂z² is computed once in PhAvg_rotated.py (stage b, compact D2Y
        # scheme) and pickled; stage c reads it (pipeline rule: no derivative
        # rebuilt here).  A legacy pickle predating the d2u_dy2 key falls back to a
        # one-off diffu_dy(du_dy) so old pickles still plot.
        # NB: name this _d2fld, NOT _d2 — _d2 is the module-level second-derivative
        # helper (see FUNCTION DEFINITIONS); rebinding it here would clobber the
        # function for every later block that calls it (e.g. D18 Poisson sources).
        _d2fld = gv('d2u_dy2', _cname)
        if _d2fld is None or getattr(_d2fld, 'shape', None) != _dudy.shape:
            _nyc, _nxc = _dudy.shape
            _d2fld = diffu_dy(_dudy, _nyc, _nxc, _epsc, _yg)
        _xc, _yc, _xo, _yo = _case_grid(_cname, use_inner=True)
        _jl = _clip_rows(_yc, _zI_lim_in)
        _d2_panels.append((_clbl, _xc, _yc[:_jl], _d2fld[:_jl, :] * (nu**2 / norm_ustar(_cname)**3),
                           _xo, _yo, _epsc[:_jl, :]))
        _d2fld = None

    if _d2_panels:
        plot2D_div_allcases(
            _d2_panels,
            r'$(\partial^2\langle u\rangle/\partial z^2)\,\nu^2/u_*^3$',
            (r'Inflection of the streamwise profile — $\partial^2\langle u\rangle'
             r'/\partial z^2$, all cases (black isoline: $=0$ inflection)'),
            'P24d2_inflection_d2udz2_allRe.png', cmap='RdBu_r',
            xname=_x_axis_label(), yname=_z_axis_label(), ylim_top=_zI_lim_in,
            zero_contour=True, vmax_pct=92)
    _d2_panels = None

    ###########################################################################
    # SECTION 1 — 2D SIDE-BY-SIDE COLORMAPS (rough-wall cases, all available Fr)
    # x-axis = x+ (streamwise); y-axis = z+ (wall-normal, meteorological label).
    # Re = 500 for all cases.
    ###########################################################################

    plot2D_allRe('AvgPhU',   r'Ph-avg $\langle\bar{u}\rangle$ — all Re (neutral)',              'Reds',  'P01_PhAvgU_allRe.png',
                 shared_scale=True, overlay_contours=True, n_contours=12 )
    
    plot2D_allRe('AvgPhV',   r'Ph-avg wall-normal $\langle\bar{w}\rangle$ — all Re (neutral)',  'RdBu_r',  'P02_PhAvgV_allRe.png',
                 include_smooth=True, shared_scale=True,   # smooth rV_s is ~0 (machine noise), no real data
                 overlay_contours=True, n_contours=10)
    
    plot2D_allRe('AvgPhW',   r'Ph-avg spanwise $\langle\bar{v}\rangle$ — all Re (neutral)',     'RdBu_r',  'P03_PhAvgW_allRe.png',
                 shared_scale=True, overlay_contours=True, n_contours=12)
    
    plot2D_allRe('AvgP',     r'Ph-avg pressure $\langle\bar{p}\rangle$ — all Re (neutral)',     'PiYG',    'P04_AvgP_allRe.png', 
                 overlay_contours=True, n_contours=12)
    
    plot2D_allRe('AvgScal',  r'Ph-avg potential temperature $\langle\bar{\theta}\rangle$ — all Re (neutral)', 'inferno', 'P05_PotTemp_allRe.png',
                 shared_scale=True, cbar_label=r'$\langle\overline{\theta}\rangle$ (buoyancy $b$)', overlay_contours=True, n_contours=12)
    
    plot2D_allRe('DispVelU', r'Dispersive streamwise $\tilde{u}$ — all Re (neutral)',           'RdBu_r',  'P06_DispU_allRe.png',
                 shared_scale=True, overlay_contours=True, n_contours=12)
    
    plot2D_allRe('DispVelV', r'Dispersive wall-normal $\tilde{w}$ — all Re (neutral)',          'RdBu_r',  'P07_DispV_allRe.png',
                 shared_scale=True, overlay_contours=True, n_contours=12)
    
    plot2D_allRe('DispVelW', r'Dispersive spanwise $\tilde{v}$ — all Re (neutral)',             'RdBu_r',  'P08_DispW_allRe.png',
                 shared_scale=True, overlay_contours=True, n_contours=12)
    
    # Raw turbulent kinetic energy k = ½⟨u_i'u_i'⟩ (NOT wall-normalised — the z+/x+
    # axes use the single 0.0618 reference l_in, but the field is raw, shared scale).
    plot2D_allRe('TKE',      r'Turbulent kinetic energy — all Re (neutral)',                   'hot_r',   'P09_TKE_allRe.png',
                 cbar_label=r"$k=\frac{1}{2}\,\overline{u_i'u_i'}$ (raw)", shared_scale=True, overlay_contours=True, n_contours=12)
    
    plot2D_allRe('disp_vortz', r'Dispersive vorticity $\tilde{\omega}_y$ — all Re (neutral)',   'coolwarm','P10_disp_vortz_allRe.png', ylim=200,
                 overlay_contours=True, n_contours=12)
    plot2D_allRe('vort_z',   r'Ph-avg vorticity $\langle\bar{\omega}_y\rangle$ — all Re (neutral)', 'coolwarm','P11_vort_z_allRe.png', ylim=200,
                 overlay_contours=True, n_contours=12)
    
    # ── Stress-tensor families (symmetric tensor → 6 independent components each) ──
    # The pickle stores the TURBULENT stress rey_* (⟨u''_i u''_j⟩) and the DISPERSIVE
    # stress *_disp (ũ_iũ_j).  Reconstruct per case the REYNOLDS stress
    # ⟨u'_i u'_j⟩ = turbulent + dispersive and the TOTAL momentum flux
    # ⟨u_i u_j⟩ = ⟨u_i⟩⟨u_j⟩ + turbulent, then plot all FOUR families × 6 components.
    # (rey_* was previously plotted as "Reynolds stress" — it is the TURBULENT part.)
    # Met labels: AvgPhU=⟨u⟩, AvgPhV=⟨w⟩ (wall-normal), AvgPhW=⟨v⟩ (spanwise).
    _SCOMP = {'uu': ('AvgPhU', 'AvgPhU'), 'uv': ('AvgPhU', 'AvgPhV'),
              'uw': ('AvgPhU', 'AvgPhW'), 'vv': ('AvgPhV', 'AvgPhV'),
              'vw': ('AvgPhV', 'AvgPhW'), 'ww': ('AvgPhW', 'AvgPhW')}
    _DISP_KEY = {'uu': 'UU_disp', 'uv': 'UV_disp', 'uw': 'UW_disp',
                 'vv': 'VV_disp', 'vw': 'VW_disp', 'ww': 'WW_disp'}

    def _build_stress(_ek):
        """Materialise the derived reyn_<ek> / tot_<ek> in every case dict.

        Returns the (case, key) pairs created so the caller can free them once
        the figures that read them are drawn: each array is ny*nx float64
        (~6 MB), and 6 components x 2 derived families x N cases would otherwise
        stay resident for the whole run."""
        _added = []
        _ia, _ib = _SCOMP[_ek]
        for _cn in SIM_NAMES:
            _sd = sims.get(_cn)
            if _sd is None:
                continue
            _turb = gv('rey_%s' % _ek, _cn)
            _disp = gv(_DISP_KEY[_ek], _cn)
            _a = gv(_ia, _cn); _b = gv(_ib, _cn)
            if _turb is not None and _disp is not None:
                _sd['reyn_%s' % _ek] = _turb + _disp              # Reynolds ⟨u'_i u'_j⟩
                _added.append((_cn, 'reyn_%s' % _ek))
            if _turb is not None and _a is not None and _b is not None:
                _sd['tot_%s' % _ek] = _a * _b + _turb             # Total ⟨u_i u_j⟩
                _added.append((_cn, 'tot_%s' % _ek))
        return _added

    # engineering component -> meteorological display label (v↔w swap)
    _MET = {'uu': 'uu', 'uv': 'uw', 'uw': 'uv', 'vv': 'ww', 'vw': 'wv', 'ww': 'vv'}
    # (key builder, family label, math fmt, include_smooth, shared_scale, n_contours)
    # The smooth flat-wall .nc carries ONLY the Reynolds stresses ⟨u'_iu'_j⟩ — no
    # triple decomposition — so it appears on the Total and Reynolds families and
    # is EXCLUDED from the Turbulent and Dispersive ones (see _smooth_field_2d).
    _STRESS_FAMILIES = [
        (lambda ek: 'tot_%s'  % ek, 'Total momentum',    r'$\langle %s%s\rangle$',
         True,  True,  10),
        (lambda ek: 'reyn_%s' % ek, 'Reynolds stress',   r"$\langle %s'%s'\rangle$",
         True,  True,  10),
        (lambda ek: 'rey_%s'  % ek, 'Turbulent stress',  r"$\langle %s''%s''\rangle$",
         False, False, 12),
        (lambda ek: _DISP_KEY[ek],  'Dispersive stress', r'$\widetilde{%s}\widetilde{%s}$',
         False, True,  10),
    ]
    # Figure-number prefix per stress family (P12-P15); the 6 components of a
    # family share its number, as the P37-42 shear-stress block does.
    _STRESS_PNUM = {'Total': 'P12', 'Reynolds': 'P13',
                    'Turbulent': 'P14', 'Dispersive': 'P15'}
    # Component-outer / family-inner so the two derived families for a component
    # are built once, used by both figures, then released before the next one.
    for _ek in ('uu', 'uv', 'uw', 'vv', 'vw', 'ww'):
        _added = _build_stress(_ek)
        _ml   = _MET[_ek]
        _cmap = 'RdBu_r' if _ek in ('uv', 'uw', 'vw') else 'hot_r'   # shear diverging; normals ≥0
        for _vkey, _flabel, _nfmt, _ism, _ish, _nctr in _STRESS_FAMILIES:
            _fam   = _flabel.split()[0]
            _title = '%s %s — all Re (neutral)' % (_flabel, _nfmt % (_ml[0], _ml[1]))
            _save  = '%s_%s_R%s_allRe.png' % (_STRESS_PNUM[_fam], _fam, _ml)
            plot2D_allRe(_vkey(_ek), _title, _cmap, _save,
                         include_smooth=_ism, shared_scale=_ish,
                         overlay_contours=True, n_contours=_nctr)
        for _cn, _k in _added:
            sims[_cn].pop(_k, None)      # free reyn_<ek> / tot_<ek>
        _added = None

    # Mean-flow (product-of-phase-average) stresses ⟨u_i⟩⟨u_j⟩ — the mean×mean
    # term of the momentum flux, the counterpart to the turbulent ⟨u_i''u_j''⟩ and
    # dispersive ũ_iũ_j stresses (turbulent + dispersive = the Reynolds stress).
    # Formed here from the pickled phase-averaged
    # velocity fields (met labels: AvgPhU=⟨u⟩, AvgPhV=⟨w⟩ wall-normal,
    # AvgPhW=⟨v⟩ spanwise) and stored per case for plot2D_allRe's gv() lookup.
    for _cn in SIM_NAMES:
        _sd = sims.get(_cn)
        if _sd is None:
            continue
        _phu, _phv, _phw = gv('AvgPhU', _cn), gv('AvgPhV', _cn), gv('AvgPhW', _cn)
        if _phu is not None and _phv is not None:
            _sd['PhUV_mean'] = _phu * _phv        # shear ⟨u⟩⟨w⟩
        if _phu is not None:
            _sd['PhUU_mean'] = _phu * _phu        # normal ⟨u⟩⟨u⟩
        if _phv is not None:
            _sd['PhVV_mean'] = _phv * _phv        # normal ⟨w⟩⟨w⟩ (wall-normal)
        if _phw is not None:
            _sd['PhWW_mean'] = _phw * _phw        # normal ⟨v⟩⟨v⟩ (spanwise)

    # Shear ⟨u⟩⟨w⟩ can change sign → diverging map; the normals are ≥0 → 'hot_r'
    # (matching the dispersive-stress panels).  include_smooth=False on the two
    # terms carrying the ~0 flat-wall wall-normal mean (⟨w⟩), as for P02.
    plot2D_allRe('PhUV_mean', r'Mean-flow stress $\langle u\rangle\langle w\rangle$ — all Re (neutral)', 'RdBu_r', 'P85_PhUV_mean_allRe.png',
                 include_smooth=False, shared_scale=True, overlay_contours=True, n_contours=12)
    plot2D_allRe('PhUU_mean', r'Mean-flow stress $\langle u\rangle\langle u\rangle$ — all Re (neutral)', 'hot_r',  'P86_PhUU_mean_allRe.png',
                 shared_scale=True, overlay_contours=True, n_contours=12)
    plot2D_allRe('PhVV_mean', r'Mean-flow stress $\langle w\rangle\langle w\rangle$ — all Re (neutral)', 'hot_r',  'P87_PhVV_mean_allRe.png',
                 include_smooth=False, shared_scale=True, overlay_contours=True, n_contours=12)
    plot2D_allRe('PhWW_mean', r'Mean-flow stress $\langle v\rangle\langle v\rangle$ — all Re (neutral)', 'hot_r',  'P88_PhWW_mean_allRe.png',
                 shared_scale=True, overlay_contours=True, n_contours=12)

    # The four mean-flow products are cheap to rebuild from the pickled velocity
    # fields; drop them now rather than carry ~4 x N x 6 MB through the rest of
    # the script.
    for _cn in SIM_NAMES:
        for _k in ('PhUV_mean', 'PhUU_mean', 'PhVV_mean', 'PhWW_mean'):
            sims.get(_cn, {}).pop(_k, None)

    ###########################################################################
    # SECTION 1b — 2D INSTANTANEOUS PLANE COLORMAPS (all available Fr)
    # First x-y plane of flow.* / scal.* binary files; turbulent fluctuation
    # (subtract x-averaged y-profile) zeroed over solid region.
    ###########################################################################
    # No flat-wall analog exists for an instantaneous plane → include_smooth=False.
    plot2D_allRe('inst_u',    r"Inst. $u' = u - \langle u\rangle_x$ — all Re (neutral)",               'RdBu_r', 'P19_inst_u_allRe.png', 530, False,
                 include_smooth=False, shared_scale=True, overlay_contours=True, n_contours=10)
    plot2D_allRe('inst_v',    r"Inst. wall-normal $w' = w - \langle w\rangle_x$ — all Re (neutral)",    'RdBu_r', 'P20_inst_v_allRe.png', 530, False,
                 include_smooth=False, shared_scale=True, overlay_contours=True, n_contours=10)
    plot2D_allRe('inst_w',    r"Inst. spanwise $v' = v - \langle v\rangle_x$ — all Re (neutral)",       'RdBu_r', 'P21_inst_w_allRe.png', 530, False,
                 include_smooth=False, shared_scale=True, overlay_contours=True, n_contours=10)
    plot2D_allRe('inst_scal', r"Inst. $\theta' = \theta - \langle\theta\rangle_x$ — all Re (neutral)", 'RdBu_r', 'P22_inst_scal_allRe.png', 700, False,
                 overlay_contours=True, n_contours=12)

    # (Req 1) Neutral-only streamline/vorticity figures removed — this script
    # only produces combined all-simulation plots; the single-case streamline
    # maps live in PhAvg_rotated.py's fig/ (all plots now share one fig/ folder).

    # TKE shear production — smooth (if loaded) + all active rough cases (subplots)
    # Smooth panel mirrors the advection panel P24: the flat-wall .nc IS x-homogeneous,
    # so the shear production −⟨u'w'⟩ ∂⟨u⟩/∂z is built from the loaded Rxy_s and the
    # x-mean shear du_dy_s (broadcasts ny×1 over the nt pseudo-x columns sx), the SAME
    # apples-to-apples form the rough cases (sims[case]['P']) and the 1D plot P63 use.
    # (The .nc also stores the exact TKE production `Prd`; we use −Rxy_s·du_dy_s here to
    #  match the rough-case definition rather than the full Pxx+Pyy+Pzz trace.)
    _prod_panels = []
    _zmax_lim = _contour_zmax(use_inner=False)
    # P = -rey_uv * du_dy is formed in the "Derived quantities" block above, so a
    # case missing it is really missing rey_uv or du_dy — report it either way.
    for _cn_a, _why_a, _case_lvl in _case_availability('P')[1]:
        _note_missing('P23_TKEprod_allRe.png', 'P', _cn_a, _why_a, once=_case_lvl)
    for _ in smooth_refs():
        _P_s2d = -Rxy_s * du_dy_s
        _prod_panels.append((SMOOTH_LABEL,
                             sx, y_s[:limity_range], _P_s2d[:limity_range, :],
                             np.array([]), np.array([])))
    for _cname, _clbl in zip(SIM_NAMES, SIM_LABELS):
        _P_c = sims.get(_cname, {}).get('P')
        if _P_c is not None:
            _xc, _yc, _xo, _yo = _case_grid(_cname, use_inner=False)
            _jl = _clip_rows(_yc, _zmax_lim)
            _prod_panels.append((_clbl, _xc, _yc[:_jl], _P_c[:_jl, :], _xo, _yo,
                                 geps(_cname)[:_jl, :]))
    if _prod_panels:
        plot2D_div_allcases(
            _prod_panels,
            r'$-\overline{u^\prime w^\prime}\,\partial\langle\bar{u}\rangle/\partial z$',
            r'TKE production — all cases', 'P23_TKEprod_allRe.png',
            shared_scale=True, overlay_contours=True, n_contours=10)
    _prod_panels = None

    # TKE advection — smooth (if loaded) + all active rough cases (subplots)
    _adv_panels = []
    for _cn_a, _why_a, _case_lvl in _case_availability('Adv')[1]:
        _note_missing('P24_TKEadv_allRe.png', 'Adv', _cn_a, _why_a, once=_case_lvl)
    for _ in smooth_refs():
        _adv_panels.append((SMOOTH_LABEL,
                            sx, y_s[:limity_range], Adv_s[:limity_range, :],
                            np.array([]), np.array([])))
    for _cname, _clbl in zip(SIM_NAMES, SIM_LABELS):
        _Adv_c = sims.get(_cname, {}).get('Adv')
        if _Adv_c is not None:
            _xc, _yc, _xo, _yo = _case_grid(_cname, use_inner=False)
            _jl = _clip_rows(_yc, _zmax_lim)
            _adv_panels.append((_clbl, _xc, _yc[:_jl], _Adv_c[:_jl, :], _xo, _yo,
                                geps(_cname)[:_jl, :]))
    if _adv_panels:
        plot2D_div_allcases(
            _adv_panels,
            r'$u\,\partial k/\partial x + w\,\partial k/\partial z$',
            r'TKE advection — all cases', 'P24_TKEadv_allRe.png')

    # (Req 1) Neutral-only dv/dx and resultant-velocity-magnitude maps removed
    # (single-case figures; kept only in PhAvg_rotated.py).

    # ADVERSE PRESSURE GRADIENT near the IBM — all cases, z+ <= 200.
    # The adverse pressure gradient is the streamwise gradient dP/dx (> 0 adverse /
    # decelerating, < 0 favorable); it is datum-independent and identical to
    # d(DispP)/dx.  Companion: wall-normal dP/dz+ (= engineering dP/dy).  The
    # spanwise gradient is identically 0 in the spanwise+phase-averaged field.
    # Fresh pickles carry dP_dx/dP_dy (saveresults.py); older ones don't, so we
    # fall back to a centred-difference of the pickled AvgP on each case's own
    # grid.  Both normalised to wall units nu/u*^3 with the single-reference u*.
    # z cap for the pressure-gradient zoom, in the units of the ACTIVE axis
    _zP_lim_in = _zplus_to_mode(200.0)

    # Helper _gradP_case (applies nu/norm_ustar(cn)^3) → FUNCTION DEFINITIONS (top).

    for _comp, _sym, _tag in (('x', r'\partial x', 'dPdx_APG'),
                              ('y', r'\partial z', 'dPdz')):
        _gp_panels = []
        # _gradP_case falls back to a centred difference of AvgP, so the panel is
        # only lost when BOTH dP_d* and AvgP are absent — report against AvgP.
        for _cn_a, _why_a, _case_lvl in _case_availability('AvgP')[1]:
            _note_missing(f'P24bc_{_tag}_allRe.png', 'dP_d%s/AvgP' % _comp,
                          _cn_a, _why_a, once=_case_lvl)
        for _cname, _clbl in zip(SIM_NAMES, SIM_LABELS):
            _gfld = _gradP_case(_cname, _comp)
            if _gfld is None:
                continue
            _xc, _yc, _xo, _yo = _case_grid(_cname, use_inner=True)   # inner units
            _jl = _clip_rows(_yc, _zP_lim_in)
            _gp_panels.append((_clbl, _xc, _yc[:_jl], _gfld[:_jl, :], _xo, _yo,
                               geps(_cname)[:_jl, :]))
        if _gp_panels:
            _adv = (_comp == 'x')
            plot2D_div_allcases(
                _gp_panels,
                rf'$(\partial\langle\bar p\rangle/{_sym})\,\nu/u_*^3$',
                (r'Adverse pressure gradient $\partial\langle\bar p\rangle/\partial x$'
                 r' — all cases (red: adverse, blue: favorable)') if _adv else
                (r'Wall-normal pressure gradient $\partial\langle\bar p\rangle/\partial z$'
                 r' — all cases'),
                f'P24bc_{_tag}_allRe.png', cmap='RdBu_r',
                xname=_x_axis_label(), yname=_z_axis_label(), ylim_top=_zP_lim_in,
                vmax_pct=98, zero_contour=True)

    # FLOW SEPARATION near the IBM — all cases, z+ <= 200.
    # Separation is where the near-wall streamwise shear ∂⟨u⟩/∂z (met.; engineering
    # ∂u/∂y) vanishes and reverses: wall value +→- downstream = separation point,
    # -→+ = reattachment, and the negative band = reversed / recirculating flow.
    # The kinematic partner of the adverse-pressure-gradient plot above: the APG
    # is what drives separation.  Companion field: spanwise shear ∂⟨v⟩/∂z (eng.
    # ∂w/∂y) for the 3-D signature.  du_dy/dw_dy are pickled; scaled to ∂u⁺/∂z⁺
    # by nu/u*² (single-reference u*).  Contours carry the ∂/∂z = 0 isoline.
    _zS_lim_in = _zplus_to_mode(200.0)

    for _key, _sym, _rev, _tag in (('du_dy', r'\partial u', True,  'sep_dudz_APG'),
                                   ('dw_dy', r'\partial v', False, 'sep_dvdz')):
        _sh_panels = []
        for _cn_a, _why_a, _case_lvl in _case_availability(_key)[1]:
            _note_missing(f'P24de_{_tag}_allRe.png', _key, _cn_a, _why_a,
                          once=_case_lvl)
        for _cname, _clbl in zip(SIM_NAMES, SIM_LABELS):
            _f = gv(_key, _cname)
            if _f is None:
                continue
            _xc, _yc, _xo, _yo = _case_grid(_cname, use_inner=True)
            _jl = _clip_rows(_yc, _zS_lim_in)
            _sh_panels.append((_clbl, _xc, _yc[:_jl],
                               _f[:_jl, :] * (nu / norm_ustar(_cname)**2), _xo, _yo,
                               geps(_cname)[:_jl, :]))
        if _sh_panels:
            plot2D_div_allcases(
                _sh_panels,
                rf'$({_sym}/\partial z)\,\nu/u_*^2$',
                (r'Streamwise shear $\partial\langle u\rangle/\partial z$ — all cases'
                 r' (blue: reversed $\to$ separated)') if _rev else
                (r'Spanwise shear $\partial\langle v\rangle/\partial z$ — all cases'),
                f'P24de_{_tag}_allRe.png', cmap='RdBu_r',
                xname=_x_axis_label(), yname=_z_axis_label(), ylim_top=_zS_lim_in,
                zero_contour=True, vmax_pct=92)

    # Surface skin friction ∂⟨u⟩/∂z|_wall(x⁺) — all cases overlaid.
    # The definitive separation-point comparison: each curve's zero-crossing marks
    # where that case separates (+→-) / reattaches (-→+).  Wall value taken at the
    # first fluid cell above each column (per-case eps_hgt), scaled by nu/u*².
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=FIG_DPI)
    _any_sf = False
    for _cname, _clbl, _col in zip(SIM_NAMES, SIM_LABELS, SIM_COLORS):
        _dudy = gv('du_dy', _cname)
        _eh   = gv('eps_hgt', _cname)
        if _dudy is None or _eh is None:
            continue
        _nxc  = _dudy.shape[1]
        _js   = np.minimum(_eh, _dudy.shape[0] - 1)
        _txw  = (nu / norm_ustar(_cname)**2) * _dudy[_js, np.arange(_nxc)]  # ∂u⁺/∂z⁺|_wall(x)
        ax.plot(gx_in(_cname), _txw, color=_col, lw=1.3, label=_clbl)
        _any_sf = True
    if _any_sf:
        ax.axhline(0, color='k', lw=0.8, ls=':')
        ax.set_xlabel(r'$x^+$')
        ax.set_ylabel(r'$\partial\langle u\rangle/\partial z|_{\rm wall}\;\nu/u_*^2$')
        ax.set_title(r'Surface skin friction along IBM — separation where curve crosses 0')
        ax.legend(fontsize=8, ncol=LEGEND_NCOL)
        ax.grid(True)
        plt.tight_layout()
        _out = _figdir + 'P24f_sep_skinfric_allRe.png'
        fig.savefig(_out, dpi=FIG_DPI, bbox_inches='tight')
        print(f'Saved: {_out}')
    plt.close(fig)

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

    # Helper _lyr_idx → FUNCTION DEFINITIONS section (top).

    _LYR_IDX   = _lyr_idx(y_in)
    # Layer index windows per smooth reference (each has its own z+ axis, so a
    # single shared set of indices would mis-window every non-primary curve).
    _LYR_IDX_S_BY = {}
    for _ in smooth_refs():
        _LYR_IDX_S_BY[SMOOTH_KEY] = _lyr_idx(y_in_s)
    _LYR_IDX_S = _LYR_IDX_S_BY.get(_SM_PRIMARY, [(0, 0)] * 4)

    # Helpers _autoscale_y/_autoscale_x/_save_layers_x/_save_layers_y
    # → FUNCTION DEFINITIONS section (top).

    # Log-law references
    def _u_most_s():
        """Neutral MOST log law on the CURRENTLY BOUND smooth reference's inner
        axis.  A function, not a stored array: each reference has its own y_in_s,
        so it must be evaluated inside the smooth_refs() loop that draws it."""
        _um    = (1/0.43)*np.log(y_in_s) + 4.9
        _um[0] = 0
        return _um
    u_most = _u_most_s() if _smooth_loaded else None   # primary (kept for compat)
    u_most_v    = (1/0.43)*np.log(y_in) + 4.7
    u_most_v[0] = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Per-simulation log-law CURVE FIT (Froude-dependent), mirroring the fit in
    # PhAvg_rotated.py so every case gets its own wall law here in the central
    # comparison script:
    #   Fr = ∞ (neutral)     → Monin–Obukhov (1954) law of the wall
    #                          u⁺ = (1/κ) ln(z⁺ − d⁺) + B ,   B = −(1/κ) ln z₀ₘ⁺
    #   Fr finite (stratified) → Obukhov (1971) stability-corrected law
    #                          u⁺ = (1/κ) Ξ(z⁺) + B ,
    #        Ξ(z⁺) = ∫ (1 − Ri/Ri_cr)^(−1/4)/(z⁺ − d⁺) dz⁺   ( → ln(z⁺−d⁺) as Ri→0 )
    #   with Ri = (∂⟨b⟩/∂z)/(∂⟨ū⟩/∂z)² measured from THIS case's own profiles
    #   (AvgScal is the buoyancy b; ū is the rotated mean).  Fit method matches
    #   PhAvg_rotated: OLS of u⁺ vs the abscissa, d⁺ grid-searched, κ constrained,
    #   best R² kept.  The fit is done in each case's OWN inner units (its Method-2
    #   crest u★ — the physical wall scale, so κ lands in the wall-law band); the
    #   fitted curve is then mapped onto the shared single-reference z⁺ axis for
    #   overlay on P25.  Constants mirror config.py.
    # (_LL_ZMIN / _LL_ZMAX are USER PARAMETERS — see that section.)
    _LL_KBND           = (0.40, 0.44)    # config.kappa_bounds
    _LL_RICR           = 0.25            # config.Ri_cr (Miles–Howard)
    # Display window for the fitted line — kept to the fit window so the drawn
    # wall law does NOT extrapolate/bend across the plot (the stratified Ξ abscissa
    # is not a straight log, so a wide display window pollutes the figure).
    _LL_DISP_LO, _LL_DISP_HI = _LL_ZMIN, _LL_ZMAX

    # Helpers _ll_cumtrapz0/_loglaw_fit_case → FUNCTION DEFINITIONS section (top).
    # (They read the _LL_* fit constants defined just above, at call time.)

    # Compute every active case's fit once (used for both the printout and the
    # curve overlay on P25).
    _ll_fits = {case: _loglaw_fit_case(case) for case in SIM_NAMES}

    # Obukhov (1971) MODIFIED log-law (paper-faithful, nonlinear curve_fit) —
    # stratified cases only; the neutral run returns skipped=True.  Fitted here
    # per case, exactly as PhAvg_rotated.py fits it per run.
    _mod_fits = {case: _modloglaw_fit_case(case) for case in SIM_NAMES}

    # Rough Re=1000 STABLE LADDER (ri00.00 → ri18.78) — LOG-LAW overlay only, drawn
    # in each case's OWN inner units (z⁺ = y·u*/ν_rough, u⁺ = ⟨ū⟩/u*).  Gated on the
    # config flag REF_ROUGH_LADDER; the loader reads only mean rU + stored
    # FrictionVelocity per file (memory-light) and returns [] when the data dir is
    # absent.  Colour = Ri gradient (viridis); overlaid on P25 and P25b below.
    _rough_ladder = (load_rough_ladder_loglaw(ROUGH_LADDER_DIR, NU_ROUGH,
                                              ROUGH_LADDER_PATTERN,
                                              u_star_default=ROUGH_LADDER_USTAR)
                     if REF_ROUGH_LADDER else [])
    _ladder_colors = (plt.cm.viridis(np.linspace(0.12, 0.92, len(_rough_ladder)))
                      if _rough_ladder else [])
    _lgh_ladder = ([Line2D([0], [0], color=plt.cm.viridis(0.5), linestyle='-',
                           linewidth=1.2,
                           label=r'rough Re1000 stable ladder (own $u_\star$)')]
                   if _rough_ladder else [])

    # Smooth flat-wall reference (neutral, Fr = ∞): fit the SAME neutral law to
    # its plotted profile.  U_s_p is already in u⁺ units and z⁺ = y_in_s, so no
    # rescaling is needed (unlike the rough cases, whose u_plus_rot is in G units).
    # One fit per smooth reference, keyed by its name (_llf_s() returns the fit
    # of whichever reference is currently bound).
    _ll_fits_s = {}
    def _llf_s():
        return _ll_fits_s.get(SMOOTH_KEY)
    for _ in smooth_refs():
        _ll_fit_smooth = None
        _s_u = np.mean(U_s_p, axis=1)
        _s_m = (y_in_s >= _LL_ZMIN) & (y_in_s <= _LL_ZMAX)
        if np.count_nonzero(_s_m) >= 3:
            _s_zf, _s_uf = y_in_s[_s_m], _s_u[_s_m]
            _k, _dm, _z0, _B, _r2 = 0.41, 0.0, 0.068, np.nan, -np.inf
            for _d in np.linspace(0.0, 0.9*_LL_ZMIN, 1001):
                _zd = _s_zf - _d
                if np.any(_zd <= 0):
                    break
                _sl, _ic, _rv, *_ = linregress(np.log(_zd), _s_uf)
                if _sl <= 0:
                    continue
                _kk = 1.0/_sl
                if not (_LL_KBND[0] <= _kk <= _LL_KBND[1]):
                    continue
                if _rv**2 > _r2:
                    _r2, _k, _dm, _B = _rv**2, _kk, _d, _ic
                    _z0 = np.exp(-_ic/_sl)
            if np.isfinite(_r2) and np.isfinite(_B):
                # Extend the fitted line across the wide display window so it is
                # visible where it peels off the data (U_s_p is already u⁺, z⁺=y_in_s).
                _s_dm = ((y_in_s >= _LL_DISP_LO) & (y_in_s <= _LL_DISP_HI)
                         & ((y_in_s - _dm) > 1e-9))
                _s_zd = y_in_s[_s_dm]
                # Smooth profile is already in its OWN inner units (z⁺=y_in_s,
                # U_s_p=u⁺ scaled by ustr_s1 = smooth u★), so own-unit curve ≡ ref.
                _s_ufit = (1.0/_k)*np.log(_s_zd-_dm)+_B
                _ll_fit_smooth = {'kappa': _k, 'd': _dm, 'z0m': _z0, 'B': _B,
                                  'r2': _r2, 'law': 'neutral MOST',
                                  'Ri_mean': 0.0, 'Ri_max': 0.0, 'u_star': ustr_s1,
                                  'Fr': np.inf, 'z_ref': _s_zd,
                                  'u_ref': _s_ufit,
                                  'z_own': _s_zd, 'u_own': _s_ufit}
        _ll_fits_s[SMOOTH_KEY] = _ll_fit_smooth

    # ── Print the fitted values as output (one row per simulation) ────────────
    print('=' * 78)
    print('LOG-LAW VELOCITY-PROFILE FIT  (per simulation, Froude-dependent)')
    print('  Fr=inf     -> Monin-Obukhov (1954)  u+ = (1/k) ln(z+ - d+) + B')
    print('  Fr finite  -> Obukhov (1971)        u+ = (1/k) Xi(z+) + B   '
          '(Xi -> ln as Ri->0)')
    print('  fit window z+ in [%.0f, %.0f] (own inner units),  k in [%.2f, %.2f]'
          % (_LL_ZMIN, _LL_ZMAX, _LL_KBND[0], _LL_KBND[1]))
    print('-' * 78)
    print('  %-14s %-12s %6s %7s %9s %7s %6s %10s %9s'
          % ('case', 'law', 'kappa', 'd+', 'z0m+', 'B', 'R2', '<Ri>', 'Ri_max'))
    for _ in smooth_refs():
        if _llf_s() is not None:
            _fs = _llf_s()
            print('  %-14s %-12s %6.4f %7.2f %9.5f %7.3f %6.4f %+10.4f %+9.4f'
                  % (SMOOTH_KEY, _fs['law'], _fs['kappa'], _fs['d'], _fs['z0m'],
                     _fs['B'], _fs['r2'], _fs['Ri_mean'], _fs['Ri_max']))
        else:
            print('  %-14s %-12s  FIT NOT SUCCESSFUL '
                  '(no valid kappa in [%.2f,%.2f] over z+ in [%.0f,%.0f]) '
                  '-- not plotted'
                  % (SMOOTH_KEY, 'neutral MOST', _LL_KBND[0], _LL_KBND[1],
                     _LL_ZMIN, _LL_ZMAX))
    for case in SIM_NAMES:
        _f = _ll_fits.get(case)
        if _f is None:
            print('  %-14s  (no profile / grid pickled — skipped)' % case)
            continue
        if not np.isfinite(_f['r2']):
            print('  %-14s %-12s  FIT NOT SUCCESSFUL '
                  '(no valid kappa in [%.2f,%.2f] over z+ in [%.0f,%.0f]) '
                  '-- not plotted'
                  % (case, _f['law'], _LL_KBND[0], _LL_KBND[1],
                     _LL_ZMIN, _LL_ZMAX))
            continue
        print('  %-14s %-12s %6.4f %7.2f %9.5f %7.3f %6.4f %+10.4f %+9.4f'
              % (case, _f['law'], _f['kappa'], _f['d'], _f['z0m'],
                 _f['B'], _f['r2'], _f['Ri_mean'], _f['Ri_max']))
    print('=' * 78)

    # ── Obukhov (1971) MODIFIED log-law fit (per simulation, stratified only) ──
    print('=' * 78)
    print('MODIFIED LOG-LAW FIT — Obukhov (1971), Sec. 6  (nonlinear curve_fit)')
    print('  u+(z+) = (v*/k) psi(z+/L1+) + offset      (k = %.2f, fixed at the '
          'paper value)' % OBU_KAPPA)
    print('  v* is in each case\'s OWN u* units (v*~1 <=> profile u* = Method-2 u*)')
    print('  L1+ > 0 stable, < 0 unstable;  psi -> ln(z+) as L1+ -> inf '
          '(neutral log law)')
    print('  fit window z+ in [%.0f, min(3*%.0f, 0.6*delta+)] (own inner units)'
          % (_LL_ZMIN, _LL_ZMAX))
    print('-' * 78)
    print('  %-14s %8s %12s %9s %7s %10s %12s'
          % ('case', 'v*/u*', 'L1+', 'offset', 'R2', 'delta+', 'z+ window'))
    _any_mod = False
    for case in SIM_NAMES:
        _m = _mod_fits.get(case)
        if _m is None:
            print('  %-14s  (no profile / grid pickled — skipped)' % case)
            continue
        if _m.get('skipped'):
            print('  %-14s  skipped — neutral run (Fr=inf); the modified law '
                  'reduces to the classical log law' % case)
            continue
        if not _m.get('ok'):
            print('  %-14s  FIT NOT SUCCESSFUL (%s) -- not plotted'
                  % (case, _m.get('err', 'unknown')))
            continue
        _any_mod = True
        print('  %-14s %8.4f %+12.3e %9.3f %7.4f %10.1f  [%5.0f,%6.0f]'
              % (case, _m['v_star'], _m['L1_plus'], _m['offset'], _m['r2'],
                 _m['delta_plus'], _m['z_lo'], _m['z_hi']))
    if _any_mod:
        # Table-III self-test (unit-independent) — provenance for the modified law.
        validate_obukhov_tableIII(verbose=True)
    else:
        print('  (no stratified case fitted — Table-III self-test skipped)')
    print('=' * 78)

    # 2a. Log-law velocity profile (u+ and w+ vs z+) — INNER units.
    # Solid lines = streamwise (u+), faded (alpha=0.4) = spanwise (w+).
    # Vertical dashed lines mark the per-case BL thickness δ⁺ = u_*(h) × u_star/ν.
    # (Req 3) PhAvg-style on-curve BL-layer markers (functions.mark_layers):
    # symbol = sublayer ('o' viscous, 's' canopy, '^' log start, 'D' log top,
    # 'X' crest h), colour = case; filled for the valley cases, hollow for smooth.
    # X-axis upper limit = max BL height among all simulations + 500 (inner units).
    # BL height = friction velocity: smooth δ⁺ = u★_s²/ν, each valley case
    # δ⁺ = u_*(h)·u_star/ν on the shared single-reference z+ axis.
    _delta_all = []
    for _ in smooth_refs():
        _delta_all.append((ustr_s1_stored / norm_l_in(SMOOTH_KEY)))
    for _c in SIM_NAMES:
        _u2 = gv('u_star2', _c)
        if _u2 is not None:
            # δ = u_tau/f (BL height = friction velocity) placed on the ACTIVE
            # inner axis, i.e. delta+ = delta/delta_nu.  u_tau is the case's
            # Method-2 PLATEAU (case_ustar), the same value every other figure
            # uses — not u_star2[ghill], the crest value, which differs by ~3 %
            # and put this marker on a different u* than the curves beside it.
            _delta_all.append(delta_out(_c) / delta_nu(_c))
    _xmax_loglaw = (max(_delta_all) + 500.0) if _delta_all else Z_PLUS_MAX

    plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    for _ in smooth_refs():
        _u_sm = np.mean(U_s_p, axis=1)
        plt.plot(y_in_s, _u_sm,  color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
        plt.plot(y_in_s, -np.mean(W_s_p, axis=1), color=SMOOTH_COLOR, linestyle=SMOOTH_LS, alpha=0.4)
        _delta_smooth = (ustr_s1_stored / norm_l_in(SMOOTH_KEY))
        plt.axvline(x=_delta_smooth, color=SMOOTH_COLOR, linestyle='--', linewidth=1.0, alpha=0.8)
        mark_layers(y_in_s, _u_sm, _smo_layer_idx(), filled=False, color=SMOOTH_COLOR)
        if _llf_s() is not None:
            plt.plot(_llf_s()['z_ref'], _llf_s()['u_ref'],
                     color=SMOOTH_COLOR, linestyle=(0, (6, 2)), linewidth=1.0,
                     alpha=0.5, zorder=6)
    for case, clr, ls, mrkr in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_MARKERS):
        _upr  = gv('u_plus_rot', case)
        _wpr  = gv('w_plus_rot', case)
        _us2  = gv('u_star2',    case)
        _yi   = gy_in(case)
        if _upr is None or _yi is None:
            continue
        _uref_c = norm_ustar(case)          # THIS case's own measured u* (inner scale)
        plt.plot(_yi, _upr/_uref_c, color=clr, linestyle=ls)
        plt.plot(_yi, _wpr/_uref_c, color=clr, linestyle=ls, alpha=0.4)
        _mk = _oro_layer_idx(case)
        if _mk is not None:
            mark_layers(_yi, _upr/_uref_c, _mk, filled=True, color=clr)
        # Overlay THIS case's own Froude-dependent wall-law fit over its fit window
        # (thin dotted, case colour); the fitted κ/d⁺/z₀ₘ⁺ are printed above.
        _ff = _ll_fits.get(case)
        if _ff is not None and _ff['z_ref'] is not None:
            plt.plot(_ff['z_ref'], _ff['u_ref'], color=clr, linestyle=(0, (6, 2)),
                     linewidth=1.0, alpha=0.5, zorder=6)
        # Overlay the Obukhov (1971) MODIFIED log-law fit (dash-dot, case colour).
        # Stratified cases only — the neutral run has no modified law to draw.
        _mf = _mod_fits.get(case)
        if _mf is not None and _mf.get('ok') and _mf.get('z_ref') is not None:
            plt.plot(_mf['z_ref'], _mf['u_ref'], color=clr, linestyle=(0, (4, 1, 1, 1)),
                     linewidth=1.2, alpha=0.85, zorder=7)
        # BL height = the friction velocity (grid rule: y_BL = u★).  On the case's
        # OWN inner axis this is δ⁺ = u_*(h)/l_in(case), using THIS case's Method-2
        # crest friction velocity u_*(h) = u_star2[ghill] — so each case's BL line
        # sits at its own height rather than the common Re_tau.
        if _us2 is not None:
            # δ_phys = u_*(h) on the ACTIVE inner axis (`* u_star / nu` was the
            # fixed-Re shorthand for `/ l_in`; per-case on a Reynolds ladder).
            _delta_case = delta_out(case) / delta_nu(case)
            plt.axvline(x=_delta_case, color=clr, linestyle='--', linewidth=1.0, alpha=0.8)
    for _ in smooth_refs():
        plt.plot(y_in_s, _u_most_s(), color='black', linestyle='--', linewidth=1.0, alpha=0.6)
    # Rough Re=1000 stable-ladder log-law overlay (own inner units; colour = Ri).
    for _rc, _rcol in zip(_rough_ladder, _ladder_colors):
        plt.plot(_rc['z_plus'], _rc['u_plus'], color=_rcol, linestyle='-',
                 linewidth=0.8, alpha=0.75, zorder=2)
    _mark_h('v')
    plt.xscale('log')
    plt.xlim(_axis_start_zplus(), _xmax_loglaw)
    plt.xlabel(r'$z^+$')
    plt.ylabel(r'$\langle\bar{u}_i\rangle^+$')
    _lgh_2a = (all_handles()
               + [Line2D([0],[0], color='k', ls='-',   lw=1.5, label=r'$u^+$ (solid)'),
                  Line2D([0],[0], color='k', ls='-',   lw=1.5, alpha=0.4, label=r'$v^+$ (faded)'),
                  Line2D([0],[0], color='k', ls='--',  lw=1.0, alpha=0.6, label='Log-law'),
                  Line2D([0],[0], color='k', ls=(0,(6,2)), lw=1.0, alpha=0.5, label=r'Wall-law fit ($z^+\!\in[45,125]$)'),
                  Line2D([0],[0], color='k', ls=(0,(4,1,1,1)), lw=1.2, alpha=0.85,
                         label=r'Obukhov (1971) mod. log-law (stratified)'),
                  Line2D([0],[0], color='k', ls='--',  lw=1.0, alpha=0.8, label=r'$\delta_o$ per case')]
               + _lgh_ladder)
    plt.legend(handles=_lgh_2a, fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
    add_marker_legend(oro=True, smooth=_smooth_loaded)
    plt.grid(True, which='both', linestyle='--', linewidth=0.4)
    plt.title(r'Log-law velocity profile — all Re (neutral) '
              r'($z^+\leq\delta^+_{\max}+500$)')
    plt.savefig(_figdir+'P25_Velocity_LogLaw_allRe.png', dpi=FIG_DPI)
    _save_layers_x(_figdir+'P25_Velocity_LogLaw_allRe', r'Log-law velocity profile — all Re (neutral)', is_log=True)
    plt.show()

    # 2a (own u★). Same log-law velocity profile, but every case is
    # normalised by its OWN Method-2 friction velocity u★_case = u_star2(h) instead
    # of the single 0.0618 neutral reference used in P25.  So here
    #   z⁺ = y·u★_case/ν      u⁺ = u_plus_rot / u★_case      δ⁺ = u★_case²/ν
    # and the overlaid wall-law fit uses the fit's own-unit curve (z_own,u_own).
    # This is the collapsed / self-scaled view; P25 stays the shared-yardstick
    # view.  The smooth reference is already in its own units (u★ = ustr_s1).
    _delta_own = []
    for _ in smooth_refs():
        _delta_own.append((ustr_s1_stored / norm_l_in(SMOOTH_KEY)))
    for _c in SIM_NAMES:
        _u2 = gv('u_star2', _c)
        if _u2 is not None:
            _uc_c = float(case_ustar(_c))
            if np.isfinite(_uc_c) and _uc_c > 0:
                _delta_own.append(_uc_c**2 / case_nu(_c))
    _xmax_own = (max(_delta_own) + 500.0) if _delta_own else Z_PLUS_MAX

    plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    for _ in smooth_refs():
        # Smooth profile is already in its own inner units (z⁺=y_in_s, u⁺=U_s_p).
        _u_sm = np.mean(U_s_p, axis=1)
        plt.plot(y_in_s, _u_sm,  color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
        plt.plot(y_in_s, -np.mean(W_s_p, axis=1), color=SMOOTH_COLOR, linestyle=SMOOTH_LS, alpha=0.4)
        plt.axvline(x=(ustr_s1_stored / norm_l_in(SMOOTH_KEY)), color=SMOOTH_COLOR, linestyle='--', linewidth=1.0, alpha=0.8)
        mark_layers(y_in_s, _u_sm, _smo_layer_idx(), filled=False, color=SMOOTH_COLOR)
        if _llf_s() is not None:
            plt.plot(_llf_s()['z_own'], _llf_s()['u_own'],
                     color=SMOOTH_COLOR, linestyle=(0, (6, 2)), linewidth=1.0,
                     alpha=0.5, zorder=6)
    for case, clr, ls, mrkr in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_MARKERS):
        _upr  = gv('u_plus_rot', case)
        _wpr  = gv('w_plus_rot', case)
        _us2  = gv('u_star2',    case)
        _yg   = gv('y', case)
        if _upr is None or _yg is None or _us2 is None:
            continue
        # THIS case's own u_tau = the Method-2 plateau of its pickled u_star2
        # (case_ustar).  Was u_star2[ghill] (the crest value), which made this
        # figure's "own units" a different u* from every other inner figure.
        _uc = float(case_ustar(case))
        if not (np.isfinite(_uc) and _uc > 0):
            continue
        _zown = _yg * _uc / case_nu(case)         # own inner-unit wall-normal coord
        plt.plot(_zown, _upr/_uc, color=clr, linestyle=ls)
        plt.plot(_zown, _wpr/_uc, color=clr, linestyle=ls, alpha=0.4)
        _mk = _oro_layer_idx(case)
        if _mk is not None:
            mark_layers(_zown, _upr/_uc, _mk, filled=True, color=clr)
        # This case's own-unit wall-law fit (thin dotted, case colour).
        _ff = _ll_fits.get(case)
        if _ff is not None and _ff.get('z_own') is not None:
            plt.plot(_ff['z_own'], _ff['u_own'], color=clr, linestyle=(0, (6, 2)),
                     linewidth=1.0, alpha=0.5, zorder=6)
        # Obukhov (1971) modified log-law, own units (stratified cases only).
        _mf = _mod_fits.get(case)
        if _mf is not None and _mf.get('ok') and _mf.get('z_own') is not None:
            plt.plot(_mf['z_own'], _mf['u_own'], color=clr, linestyle=(0, (4, 1, 1, 1)),
                     linewidth=1.2, alpha=0.85, zorder=7)
        # BL height δ⁺ = u★_case²/ν in own units (ν per case on this ladder).
        plt.axvline(x=_uc**2 / case_nu(case), color=clr, linestyle='--',
                    linewidth=1.0, alpha=0.8)
    for _ in smooth_refs():
        plt.plot(y_in_s, _u_most_s(), color='black', linestyle='--', linewidth=1.0, alpha=0.6)
    # Rough Re=1000 stable-ladder log-law overlay (own inner units; colour = Ri).
    for _rc, _rcol in zip(_rough_ladder, _ladder_colors):
        plt.plot(_rc['z_plus'], _rc['u_plus'], color=_rcol, linestyle='-',
                 linewidth=0.8, alpha=0.75, zorder=2)
    plt.xscale('log')
    plt.xlim(_axis_start_zplus(), _xmax_own)
    plt.xlabel(r'$z^+ = z\,u_{\star,\mathrm{case}}/\nu$')
    plt.ylabel(r'$\langle\bar{u}_i\rangle / u_{\star,\mathrm{case}}$')
    _lgh_2b = (all_handles()
               + [Line2D([0],[0], color='k', ls='-',   lw=1.5, label=r'$u^+$ (solid)'),
                  Line2D([0],[0], color='k', ls='-',   lw=1.5, alpha=0.4, label=r'$v^+$ (faded)'),
                  Line2D([0],[0], color='k', ls='--',  lw=1.0, alpha=0.6, label='Log-law'),
                  Line2D([0],[0], color='k', ls=(0,(6,2)), lw=1.0, alpha=0.5, label=r'Wall-law fit ($z^+\!\in[45,125]$)'),
                  Line2D([0],[0], color='k', ls=(0,(4,1,1,1)), lw=1.2, alpha=0.85,
                         label=r'Obukhov (1971) mod. log-law (stratified)'),
                  Line2D([0],[0], color='k', ls='--',  lw=1.0, alpha=0.8, label=r'$\delta_o$ per case')]
               + _lgh_ladder)
    plt.legend(handles=_lgh_2b, fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
    add_marker_legend(oro=True, smooth=_smooth_loaded)
    plt.grid(True, which='both', linestyle='--', linewidth=0.4)
    plt.title(r'Log-law velocity profile — all Re (neutral) '
              r'(each case in its OWN $u_\star$ units)')
    plt.savefig(_figdir+'P25b_Velocity_LogLaw_allRe_ownustar.png', dpi=FIG_DPI)
    plt.show()

    # 2a (outer). Log-law velocity profile — OUTER units.  MATCHED PAIR of
    # scales (see THE UNIT SYSTEM): length z- = y/delta, delta = u_tau(case)/f;
    # velocity u- = u_i/G, and G = 1, so the curves are the raw stored means.
    # The velocity was previously divided by u_star2(h) here — an INNER velocity
    # scale on an OUTER abscissa, which is precisely the mix that stops the
    # outer region collapsing.  z- capped at 4.  Layer markers are array indices
    # and so are axis-agnostic.
    plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    for _ in smooth_refs():
        _zs   = y_s / (ustr_s1 / float(f))      # delta_s = u*_s/f
        _svs  = norm_vel_smooth(1, 'outer')     # = 1.0 (geostrophic units)
        _u_sm = np.mean(U_s_p, axis=1) / _svs   # per reference, not the leftover
        plt.plot(_zs, _u_sm,  color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
        plt.plot(_zs, -np.mean(W_s_p, axis=1)/_svs, color=SMOOTH_COLOR, linestyle=SMOOTH_LS, alpha=0.4)
        mark_layers(_zs, _u_sm, _smo_layer_idx(), filled=False, color=SMOOTH_COLOR)
    for case, clr, ls in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES):
        _upr = gv('u_plus_rot', case); _wpr = gv('w_plus_rot', case)
        _zo, _dlt = _z_out(case)
        if _upr is None or _zo is None:
            continue
        _vs = norm_vel(case, 1, 'outer')        # = 1.0 (geostrophic units)
        plt.plot(_zo, _upr/_vs, color=clr, linestyle=ls)
        plt.plot(_zo, _wpr/_vs, color=clr, linestyle=ls, alpha=0.4)
        _mk = _oro_layer_idx(case)
        if _mk is not None:
            mark_layers(_zo, _upr/_vs, _mk, filled=True, color=clr)
    _mark_h('v', mode='outer')
    plt.xlim(0, Z_MINUS_MAX)
    plt.xlabel(z_label('outer'))
    plt.ylabel(vel_label(r'$\langle\bar{u}_i\rangle$', 1, 'outer'))
    plt.legend(handles=_lgh_2a, fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
    add_marker_legend(oro=True, smooth=_smooth_loaded)
    plt.grid(True, linestyle='--', linewidth=0.4)
    plt.title(r'Log-law velocity profile — outer units ($z^-\leq4$) — all Re (neutral)')
    plt.savefig(_figdir+'P26_Velocity_LogLaw_allRe_outer.png', dpi=FIG_DPI)
    plt.show()

    # 2b. Roughness sublayer velocity profile (log-log, rough-wall cases only)
    # Only the first 10 points are plotted (RSL fit near the wall); evaluating the
    # exponential over the full z+ range (~800) overflows to inf, so bound it here.
    u_roughnesslayer = 0.1018*np.exp(1.3165*y_in[:10])
    plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    for case, clr, ls, lbl in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_LABELS):
        _upr = gv('u_plus_rot', case)
        _yi  = gy_in(case)
        if _upr is None or _yi is None:
            continue
        plt.plot(_yi[:N_ROWS_ORO_RSL], (_upr/norm_ustar(case))[:N_ROWS_ORO_RSL], color=clr, linestyle=ls, label=lbl)
    plt.plot(y_in[:10], u_roughnesslayer[:10], color='black', linestyle='--', alpha=0.5, label='RSL fit')
    _mark_h('v')
    plt.xscale('log')
    plt.xlabel(r'$z^+$')
    plt.ylabel(r'$\langle\bar{u}\rangle^+$')
    plt.legend(fontsize=8)
    plt.grid(True, which='both', linestyle='--', linewidth=0.4)
    plt.title(r'Roughness sublayer velocity — rough-wall cases (neutral)')
    plt.savefig(_figdir+'P27_Velocity_RSL_allRe.png', dpi=FIG_DPI)
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
    # (the smooth BL thickness in inner units is computed per reference where it
    # is actually drawn — the dead `_Re_tau_s` that used to sit here was the
    # NameError site whenever the smooth file was missing)
    _fig_hodo, _ax_hodo = plt.subplots(figsize=FIG_SIZE_HODO, dpi=FIG_DPI)
    _mkw = dict(zorder=5, markeredgewidth=0.8)
    _hodo_data = []   # stores (un, wn) 1-D arrays for each available rough case

    # (key, u_norm, w_norm, per-reference layer indices) for every smooth
    # reference — reused by the layer-zoom windows and the outer hodograph
    # below, so neither depends on a value left behind by the loop.
    _hodo_smooth = []
    for _ in smooth_refs():
        _Us_ref = GblU_s[-1]                           # geostrophic u-component
        _Ws_ref = GblW_s[-1]                           # geostrophic w-component
        _G_ref  = np.sqrt(_Us_ref**2 + _Ws_ref**2)    # geostrophic wind magnitude
        _un_s   = GblU_s / _G_ref
        _wn_s   = -GblW_s / _G_ref
        _ax_hodo.plot(_un_s, _wn_s, color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
        mark_layers(_un_s, _wn_s, _smo_layer_idx(), filled=False, color=SMOOTH_COLOR)
        _hodo_smooth.append((SMOOTH_KEY, _un_s, _wn_s, SMOOTH_COLOR,
                             _LYR_IDX_S_BY.get(SMOOTH_KEY, [(0, 0)] * 4)))

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
        # (Req 4) PhAvg-style BL-layer markers placed ON the (u,w) curve — same
        # symbols as the log-law plot ('o' viscous, 's' canopy, '^' log start,
        # 'D' log top, 'X' crest h); colour = case.  Placed by array index, so
        # EVERY case's hodograph now carries the full marker set (the old scheme
        # marked only h/3h/δ_o by size and was easy to miss / drop).
        _mk = _oro_layer_idx(case)
        if _mk is not None:
            mark_layers(_un, _wn, _mk, filled=True, color=clr)

    _ax_hodo.set_xlabel(r'$u_{\mathrm{rot}}\,/\,G$')
    _ax_hodo.set_ylabel(r'$v_{\mathrm{rot}}\,/\,G$')
    # Legend: case (line) handles + the layer-marker footnote key.
    _lgh_hodo = all_handles()
    _ax_hodo.legend(handles=_lgh_hodo, fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
    add_marker_legend(ax=_ax_hodo, oro=True, smooth=_smooth_loaded)
    _ax_hodo.grid(True)
    _ax_hodo.set_title(r'Hodograph — all Re (neutral)')
    _fig_hodo.savefig(_figdir+'P28_Hodograph_allRe.png', dpi=FIG_DPI, bbox_inches='tight')
    # Layer-zoomed hodographs: restrict visible window to each z+ range
    for _ilyr, (_ln, _lt, (_i0, _i1)) in enumerate(zip(
            _LYR_NAMES, _LYR_TITLES, _LYR_IDX)):
        _uall, _wall = [], []
        for _ks, _un_sk, _wn_sk, _cs, _lyr_s in _hodo_smooth:
            _i0_s, _i1_s = _lyr_s[_ilyr]          # this reference's own indices
            if _i1_s > _i0_s:
                _uall.extend(_un_sk[_i0_s:_i1_s].tolist())
                _wall.extend(_wn_sk[_i0_s:_i1_s].tolist())
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
        _ax_hodo.set_title(f'Hodograph — all Re (neutral)\n{_lt}', fontsize=9)
        _fig_hodo.savefig(_figdir+f'P28_Hodograph_allRe_{_ln}.png', dpi=FIG_DPI, bbox_inches='tight')
    _ax_hodo.autoscale()
    _ax_hodo.set_title(r'Hodograph — all Re (neutral)')
    plt.show()

    # 2c (outer). Hodograph — OUTER units.  A hodograph has no wall-normal
    # axis; what makes it "outer" is the VELOCITY scale, and that is the
    # geostrophic wind G (already applied below as _u_rot/_G_ref_c).  The title
    # used to claim u_star2(h) — the INNER scale — which mislabelled a figure
    # that was in fact already correct.  Markers locate h, 3h, δ_o.
    _fig_hodo_out, _ax_hodo_out = plt.subplots(figsize=FIG_SIZE_HODO, dpi=FIG_DPI)
    for _ks, _un_sk, _wn_sk, _cs, _lyr_s in _hodo_smooth:
        _bind_smooth(_ks)                       # so _smo_layer_idx() matches _ks
        _ax_hodo_out.plot(_un_sk, _wn_sk, color=_cs, linestyle=SMOOTH_LS)
        mark_layers(_un_sk, _wn_sk, _smo_layer_idx(), filled=False, color=_cs)
    _bind_smooth(_SM_PRIMARY)
    for case, clr, ls, mrkr in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_MARKERS):
        _u_rot = gv('u_plus_rot', case)
        _w_rot = gv('w_plus_rot', case)
        _us2   = gv('u_star2', case)
        if _u_rot is None or _w_rot is None or _us2 is None:
            continue
        _G_ref_c = np.sqrt(_u_rot[-1]**2 + _w_rot[-1]**2)
        _un_o    = _u_rot / _G_ref_c
        _wn_o    = _w_rot / _G_ref_c
        _ax_hodo_out.plot(_un_o, _wn_o, color=clr, linestyle=ls)
        # (Req 4) same PhAvg-style layer markers on every curve.
        _mk = _oro_layer_idx(case)
        if _mk is not None:
            mark_layers(_un_o, _wn_o, _mk, filled=True, color=clr)
    _ax_hodo_out.set_xlabel(r'$u_{\mathrm{rot}}\,/\,G$')
    _ax_hodo_out.set_ylabel(r'$v_{\mathrm{rot}}\,/\,G$')
    _ax_hodo_out.legend(handles=_lgh_hodo, fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
    add_marker_legend(ax=_ax_hodo_out, oro=True, smooth=_smooth_loaded)
    _ax_hodo_out.grid(True)
    _ax_hodo_out.set_title(r'Hodograph — outer units ($u_i/G$) — all Re (neutral)')
    _fig_hodo_out.savefig(_figdir+'P29_Hodograph_allRe_outer.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.show()

    # 2d. Wind turning angle vs z+ (rough-wall cases) — INNER units.
    # BUG FIX (Req 5): the pickled `inst_alpha` is the RATIO w_plus_rot/u_plus_rot
    # (a tangent), which diverges wherever u_plus_rot crosses zero — e.g. the
    # strongly-stratified Fr=0.0015 run reverses near the surface, so w/u blew up
    # to ±3000°.  Compute the veer as the BOUNDED angle arctan2(w_rot,u_rot) in
    # degrees (∈[-180,180]) directly from the rotated velocity components.
    # Helper _veer_deg → FUNCTION DEFINITIONS section (top).

    plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    for case, clr, ls, lbl in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_LABELS):
        _ang = _veer_deg(case)
        _yi = gy_in(case)
        if _ang is None or _yi is None:
            continue
        plt.plot(_yi[1:], _ang[1:], color=clr, linestyle=ls, label=lbl)
    _mark_h('v')
    plt.xscale('log')
    plt.xlim(_axis_start_zplus(), Z_PLUS_MAX)
    plt.xlabel(r'$z^+$')
    plt.ylabel(r'$\alpha\;(\mathrm{deg})$')
    plt.legend(handles=sim_handles(), fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
    plt.grid(True, which='both', linestyle='--', linewidth=0.4)
    plt.title(r'Wind turning angle — rough-wall cases (neutral) ($z^+\leq200$)')
    plt.savefig(_figdir+'P30_TurningAngle_allRe.png', dpi=FIG_DPI)
    _save_layers_x(_figdir+'P30_TurningAngle_allRe', r'Wind turning angle — rough-wall cases (neutral)', is_log=True)
    plt.show()

    # 2d (outer). Wind turning angle vs z- = y/u_star2(h) (Req 8).
    plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    for _ in smooth_refs():
        _ang_s = np.degrees(np.arctan2(-np.mean(W_s_p, axis=1), np.mean(U_s_p, axis=1)))
        plt.plot(y_s / (ustr_s1 / float(f)), _ang_s, color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
    for case, clr, ls, lbl in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_LABELS):
        _ang = _veer_deg(case)
        _zo, _dlt = _z_out(case)
        if _ang is None or _zo is None:
            continue
        # The veer is an ANGLE — dimensionless, so it carries no velocity scale
        # in either unit system; only the abscissa changes between the pair.
        plt.plot(_zo[1:], _ang[1:], color=clr, linestyle=ls, label=lbl)
    _mark_h('v', mode='outer')
    plt.xlim(0, Z_MINUS_MAX)
    plt.xlabel(z_label('outer'))
    plt.ylabel(r'$\alpha\;(\mathrm{deg})$')
    plt.legend(handles=all_handles(), fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
    plt.grid(True, linestyle='--', linewidth=0.4)
    plt.title(r'Wind turning angle — outer units ($z^-\leq4$) — all Re (neutral)')
    plt.savefig(_figdir+'P31_TurningAngle_allRe_outer.png', dpi=FIG_DPI)
    plt.show()

    # 2e. TKE vertical profile — all 6 cases (INNER units, z+ <= 200)
    plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    for _ in smooth_refs():
        plt.plot(y_in_s[:N_ROWS_SMOOTH_TKE], np.mean(TKE_s, axis=1)[:N_ROWS_SMOOTH_TKE]/ustr_s1**2,
                 color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
    for case, clr, ls in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES):
        _tke = gv('TKE', case)
        _yi  = gy_in(case)
        if _tke is None or _yi is None:
            continue
        plt.plot(_yi[:460], np.mean(_tke, axis=1)[:460]/norm_ustar(case)**2,
                 color=clr, linestyle=ls)
    _mark_h('v')                       # valley-crest height h (always within z+<=200)
    # Boundary-layer-height (delta) markers: the smooth delta_s and per-case
    # delta_o are ~ u*^2/nu ~ 450-480 in z+, i.e. OFF this z+<=200 axis, so the
    # guarded helper skips them instead of drawing a line far outside the plot.
    # Only height markers that fit the scale limit are rendered.
    _delta_marks = []
    for _ in smooth_refs():
        _delta_marks.append(((ustr_s1_stored / norm_l_in(SMOOTH_KEY)),
                             r'$\delta_s$', SMOOTH_COLOR))
    for case, clr in zip(SIM_NAMES, SIM_COLORS):
        _us2 = gv('u_star2', case)
        if _us2 is not None:
            _delta_marks.append((delta_out(case) / delta_nu(case),
                                 r'$\delta_o$', clr))
    _mark_heights_v(_delta_marks)
    plt.xlim(0, Z_PLUS_MAX)
    plt.xlabel(r'$z^+$')
    plt.ylabel(r'$k/u_*^2$')
    plt.legend(handles=all_handles(), fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
    plt.grid(True, which='both', linestyle='--', linewidth=0.4)
    plt.title(r'TKE vertical profile — all Re (neutral) ($z^+\leq200$)')
    plt.savefig(_figdir+'P32_TKE_vertical_allRe.png', dpi=FIG_DPI)
    _save_layers_x(_figdir+'P32_TKE_vertical_allRe', r'TKE vertical profile — all Re (neutral)')
    plt.show()

    # 2e (outer). TKE vertical profile — OUTER units: z- = y/delta and k/G^2
    # (k is a velocity^2, so it takes the OUTER velocity scale G, not u_tau).
    plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    for _ in smooth_refs():
        plt.plot(y_s / (ustr_s1 / float(f)),
                 np.mean(TKE_s, axis=1)/norm_vel_smooth(2, 'outer'),
                 color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
    for case, clr, ls in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES):
        _tke = gv('TKE', case)
        _zo, _dlt = _z_out(case)
        if _tke is None or _zo is None:
            continue
        plt.plot(_zo, np.mean(_tke, axis=1)/norm_vel(case, 2, 'outer'),
                 color=clr, linestyle=ls)
    _mark_h('v', mode='outer')
    plt.xlim(0, Z_MINUS_MAX)
    plt.xlabel(z_label('outer'))
    plt.ylabel(vel_label(r'$k$', 2, 'outer'))
    plt.legend(handles=all_handles(), fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
    plt.grid(True, linestyle='--', linewidth=0.4)
    plt.title(r'TKE vertical profile — outer units ($z^-\leq4$) — all Re (neutral)')
    plt.savefig(_figdir+'P33_TKE_vertical_allRe_outer.png', dpi=FIG_DPI)
    plt.show()

    # 2f. TKE horizontal (column-integrated) distribution (x/h axis, no z-twin)
    plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    # Smooth reference now comes from the shared loader (was wrongly read as zeros
    # from the rough-wall pickle via _n0.get('AVG_TKE_V_s_i')).
    # ABSCISSA = x/h, NOT x+.  Every case spans the SAME physical L_x over the
    # SAME valley, but x+ = x/l_in(case) stretches that domain by the l_in ratio
    # (~2.3x between Re_D=500 and 750), so on an x+ axis the cases cover
    # different ranges and the single IBM silhouette below could not match any
    # of them.  h is identical in physical space on every grid (see
    # compare_eps_geometry.py), so x/h puts all cases and the topography on one
    # common abscissa.  The smooth reference has no valley; it is drawn on the
    # reference h for the same reason as the 2-D 'topo' panels.
    _xh_max = 0.0
    for _ in smooth_refs():
        _xs_h = x / float(y[hill_hgt])
        plt.plot(_xs_h, AVG_TKE_V_s_i/ustr_s1**2,
                 color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
        _xh_max = max(_xh_max, float(_xs_h[-1]))
    for case, clr, ls, lbl in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_LABELS):
        _atv = gv('AVG_TKE_V', case)
        if _atv is None:
            continue
        _xc = gv('x', case)
        _xc = x if _xc is None else _xc
        _xh = _xc / case_hill_h(case)
        _xh_max = max(_xh_max, float(_xh[-1]))
        plt.plot(_xh, _atv/norm_ustar(case)**2, color=clr, linestyle=ls, label=lbl)
    # IBM silhouette on that same x/h abscissa (display amplitude unchanged).
    _xsil = np.linspace(0.0, _xh_max if _xh_max > 0 else 1.0, 512)
    _hill_line = (y[hill_hgt]/u_star)*(1 + np.cos(2*np.pi*_xsil/_xsil[-1]))
    plt.fill_between(_xsil, _hill_line, color='black', alpha=1.0, label='IBM solid')
    plt.xlabel(r'$x/h$')
    plt.ylabel(r'$\langle k\rangle_z / u_*^2$')
    plt.legend(handles=sim_handles(), fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
    plt.grid(True)
    plt.title(r'Horizontal TKE distribution — rough-wall cases (neutral)')
    plt.savefig(_figdir+'P34_TKE_horizontal_allRe.png', dpi=FIG_DPI)
    plt.show()

    # 2g. Friction velocity profile u*(z+) — rough cases (INNER, z+<=200)
    plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    for case, clr, ls, lbl in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_LABELS):
        _us2 = gv('u_star2', case)
        _yi  = gy_in(case)
        if _us2 is None or _yi is None:
            continue
        plt.plot(_us2[:N_ROWS_ORO], _yi[:N_ROWS_ORO], color=clr, linestyle=ls, label=lbl)
    _mark_h('h')
    plt.ylim(0, Z_PLUS_MAX)
    plt.xlabel(r'$u_*(z)$')
    plt.ylabel(r'$z^+$')
    plt.legend(handles=sim_handles(), fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
    plt.grid(True)
    plt.title(r'Friction velocity profile — rough-wall cases (neutral) ($z^+\leq200$)')
    plt.savefig(_figdir+'P35_FrictionVelocity_allRe.png', dpi=FIG_DPI)
    _save_layers_y(_figdir+'P35_FrictionVelocity_allRe', r'Friction velocity profile — rough-wall cases (neutral)')
    plt.show()

    # 2g (outer). Friction velocity profile u*(z-) — OUTER units:
    # z- = y/delta on the y-axis, u*(z)/G on the x-axis (u* is a velocity, so it
    # takes the outer velocity scale G = 1 -> the raw profile).
    plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    for case, clr, ls, lbl in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_LABELS):
        _us2 = gv('u_star2', case)
        _zo, _dlt = _z_out(case)
        if _us2 is None or _zo is None:
            continue
        plt.plot(_us2/norm_vel(case, 1, 'outer'), _zo, color=clr, linestyle=ls, label=lbl)
    _mark_h('h', mode='outer')
    plt.ylim(0, Z_MINUS_MAX)
    plt.xlabel(vel_label(r'$u_*(z)$', 1, 'outer'))
    plt.ylabel(z_label('outer'))
    plt.legend(handles=sim_handles(), fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
    plt.grid(True)
    plt.title(r'Friction velocity profile — outer units ($z^-\leq4$) — all Re (neutral)')
    plt.savefig(_figdir+'P36_FrictionVelocity_allRe_outer.png', dpi=FIG_DPI)
    plt.show()

    # 2h. Reynolds and dispersive normal stress profiles (uu, vv, ww) —
    # all 6 cases.  The loop emits 6 figures: uu/vv/ww each INNER (z+<=200) then
    # OUTER (z-<=4).  Solid lines = TURBULENT stress ⟨u''_i u''_j⟩ (rey_*); faded
    # (alpha=0.4) = dispersive stress ũ_iũ_j.  (Their sum = the Reynolds stress.)
    _eps_f = np.where(np.mean(mask0, axis=1) > 0, np.mean(mask0, axis=1), np.nan)
    for _key_r, _key_d, _key_sm, _lbl_r, _lbl_d in [
        ('rey_uu', 'UU_disp', 'Rxx_s', r"$\overline{u''u''}$", r'$\tilde{u}\tilde{u}$'),
        ('rey_vv', 'VV_disp', 'Ryy_s', r"$\overline{w''w''}$", r'$\tilde{w}\tilde{w}$'),
        ('rey_ww', 'WW_disp', 'Rzz_s', r"$\overline{v''v''}$", r'$\tilde{v}\tilde{v}$'),
    ]:
        plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
        _sm = globals().get(_key_sm)
        for _ in smooth_refs():
            _sm_r = globals().get(_key_sm)     # re-fetch: rebound per reference
            if _sm_r is None:
                continue
            plt.semilogy(y_in_s, np.mean(_sm_r, axis=1)/ustr_s1**2,
                         color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
        for case, clr, ls in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES):
            _r = gv(_key_r, case)
            _d = gv(_key_d, case)
            _yi = gy_in(case)
            if _r is None or _yi is None:
                continue
            plt.semilogy(_yi, np.mean(_r*gmask0(case), axis=1)/geps_f(case)/norm_ustar(case)**2,
                         color=clr, linestyle=ls)
            if _d is not None:
                plt.semilogy(_yi, np.mean(_d, axis=1)/norm_ustar(case)**2,
                             color=clr, linestyle=ls, alpha=0.4)
        _mark_h('v', ha='left')
        plt.xlim(_hill_pos_min(), Z_PLUS_MAX)
        plt.xlabel(r'$z^+$')
        plt.ylabel(_lbl_r + ', ' + _lbl_d + r' / $u_*^2$')
        _leg_n = ([Line2D([0],[0], color='k', ls='-', lw=2,   label=_lbl_r + r' (solid)'),
                   Line2D([0],[0], color='k', ls='-', lw=0.8, alpha=0.4, label=_lbl_d + r' (faded)')]
                  + all_handles())
        plt.legend(handles=_leg_n, fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
        plt.grid(True, which='both', linestyle='--', linewidth=0.4)
        plt.title('Normal stress: ' + _lbl_r + ' and ' + _lbl_d + r' — all Re (neutral) ($z^+\leq200$)')
        plt.savefig(_figdir+'P37-42_Stress_'+_key_r+'_allRe.png', dpi=FIG_DPI)
        _save_layers_x(_figdir+'P37-42_Stress_'+_key_r+'_allRe',
                       'Normal stress: ' + _lbl_r + ' and ' + _lbl_d + r' — all Re (neutral)')
        plt.show()

        # (Req 8) OUTER-units twin of the figure above: z- = y/delta and
        # <u_i'u_j'>/G^2 (a velocity^2 takes the OUTER velocity scale G, not
        # u_tau — see THE UNIT SYSTEM), z- <= 4.
        plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
        for _ in smooth_refs():
            _sm_r = globals().get(_key_sm)     # re-fetch: rebound per reference
            if _sm_r is None:
                continue
            plt.semilogy(y_s/(ustr_s1/float(f)),
                         np.mean(_sm_r, axis=1)/norm_vel_smooth(2, 'outer'),
                         color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
        for case, clr, ls in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES):
            _r = gv(_key_r, case)
            _d = gv(_key_d, case)
            _zo, _dlt = _z_out(case)
            if _r is None or _zo is None:
                continue
            _vs2 = norm_vel(case, 2, 'outer')
            plt.semilogy(_zo, np.mean(_r*gmask0(case), axis=1)/geps_f(case)/_vs2,
                         color=clr, linestyle=ls)
            if _d is not None:
                plt.semilogy(_zo, np.mean(_d, axis=1)/_vs2,
                             color=clr, linestyle=ls, alpha=0.4)
        _mark_h('v', mode='outer')
        plt.xlim(0, Z_MINUS_MAX)
        plt.xlabel(z_label('outer'))
        plt.ylabel(vel_label(_lbl_r + ', ' + _lbl_d, 2, 'outer'))
        plt.legend(handles=_leg_n, fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
        plt.grid(True, which='both', linestyle='--', linewidth=0.4)
        plt.title('Normal stress: ' + _lbl_r + ' and ' + _lbl_d + r' — outer units ($z^-\leq4$)')
        plt.savefig(_figdir+'P37-42_Stress_'+_key_r+'_allRe_outer.png', dpi=FIG_DPI)
        plt.show()

    # 2i. Turbulent + dispersive shear stress uv — all 6 cases (INNER)
    # Solid lines = TURBULENT stress ⟨u''w''⟩ (rey_uv); faded (alpha=0.4) = dispersive
    # stress ũw̃ (their sum = the Reynolds shear stress); z+<=200.
    plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    for _ in smooth_refs():
        plt.plot(y_in_s, np.mean(Rxy_s, axis=1)/ustr_s1**2,
                 color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
    for case, clr, ls in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES):
        _r = gv('rey_uv', case)
        _d = gv('UV_disp', case)
        _yi = gy_in(case)
        if _r is None or _yi is None:
            continue
        plt.plot(_yi, np.mean(_r*gmask0(case), axis=1)/geps_f(case)/norm_ustar(case)**2,
                 color=clr, linestyle=ls)
        if _d is not None:
            plt.plot(_yi, np.mean(_d, axis=1)/norm_ustar(case)**2,
                     color=clr, linestyle=ls, alpha=0.4)
    _mark_h('v')
    plt.xlim(0, Z_PLUS_MAX)
    plt.xlabel(r'$z^+$')
    plt.ylabel(r"$\overline{u''w''},\;\tilde{u}\tilde{w}\;/ u_*^2$")
    _leg_uv = ([Line2D([0],[0], color='k', ls='-', lw=2,   label=r"$\overline{u''w''}$ (solid)"),
                Line2D([0],[0], color='k', ls='-', lw=0.8, alpha=0.4, label=r'$\tilde{u}\tilde{w}$ (faded)')]
               + all_handles())
    plt.legend(handles=_leg_uv, fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
    plt.grid(True)
    plt.title(r"Shear stress: turbulent $\overline{u''w''}$ and dispersive $\tilde{u}\tilde{w}$ — all Re (neutral)")
    plt.savefig(_figdir+'P43_Stress_uv_allRe.png', dpi=FIG_DPI)
    plt.show()

    # 2i (outer). Reynolds + dispersive shear stress uv — OUTER units:
    # z- = y/delta per case; stress /G^2 (the outer velocity scale).
    plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    for _ in smooth_refs():
        _y_out_s = y_s / (ustr_s1 / float(f))
        plt.plot(_y_out_s, np.mean(Rxy_s, axis=1)/norm_vel_smooth(2, 'outer'),
                 color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
    for case, clr, ls in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES):
        _r   = gv('rey_uv',  case)
        _d   = gv('UV_disp', case)
        _y_out, _dlt = _z_out(case)
        if _r is None or _y_out is None:
            continue
        _vs2 = norm_vel(case, 2, 'outer')
        plt.plot(_y_out, np.mean(_r*gmask0(case), axis=1)/geps_f(case)/_vs2,
                 color=clr, linestyle=ls)
        if _d is not None:
            plt.plot(_y_out, np.mean(_d, axis=1)/_vs2,
                     color=clr, linestyle=ls, alpha=0.4)
    _mark_h('v', mode='outer')
    plt.xlim(0, Z_MINUS_MAX)
    plt.xlabel(z_label('outer'))
    plt.ylabel(vel_label(r"$\overline{u''w''},\;\tilde{u}\tilde{w}$", 2, 'outer'))
    _leg_uv_out = ([Line2D([0],[0], color='k', ls='-', lw=2,   label=r"$\overline{u''w''}$ (solid)"),
                    Line2D([0],[0], color='k', ls='-', lw=0.8, alpha=0.4, label=r'$\tilde{u}\tilde{w}$ (faded)')]
                   + all_handles())
    plt.legend(handles=_leg_uv_out, fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
    plt.grid(True)
    plt.title(r"Shear stress: turbulent $\overline{u''w''}$ and dispersive $\tilde{u}\tilde{w}$ — outer units, all Re (neutral)")
    plt.savefig(_figdir+'P44_Stress_uv_allRe_outer.png', dpi=FIG_DPI)
    plt.show()

    # 2j. p'v' pressure transport — only plotted if pdvd2D is in the pickle.
    # Gate on ANY active case carrying it (it used to test only the first case,
    # by name, so the panel vanished whenever that case was switched off).
    if any(gv('pdvd2D', _c) is not None for _c in SIM_NAMES):
        plt.figure(figsize=(6, 8), dpi=FIG_DPI)
        for case, clr, ls, lbl in zip(SIM_NAMES, SIM_COLORS,
                                      SIM_LINESTYLES, SIM_LABELS):
            _pv = gv('pdvd2D', case)
            _yi = gy_in(case)
            if _pv is None or _yi is None:
                continue
            plt.semilogy(np.mean(_pv, axis=1)/norm_ustar(case)**3, _yi,
                         color=clr, linestyle=ls, label=lbl)
        _mark_h('h')
        plt.xlabel(r"$\langle p'w'\rangle / u_*^3$")
        plt.ylabel(r'$z^+$')
        plt.legend(handles=sim_handles(), fontsize=LEGEND_FONTSIZE)
        plt.grid(True, which='both', ls='--', alpha=0.5)
        plt.title(r"Pressure transport $\langle p'w'\rangle$ — all Re (neutral)")
        plt.savefig(_figdir+'P45_PressureTransport.png', dpi=FIG_DPI)
        plt.show()

    ###########################################################################
    # SECTION 3 -- MOMENTUM BALANCE (all 5 Fr, zoomed to y+ <= 200)
    ###########################################################################

    # Shared term colour handles for the OUTER-unit tau_yx / tau_yz plots below,
    # which still draw the turbulent-vs-dispersive split.  The inner-unit P46/P47
    # build their own reduced handles inside _plot_shear_stress_balance() (they
    # show only the combined Reynolds curve), so do NOT trim this list.
    # Double encoding: term by colour, case by linestyle (see all_handles() for case key).
    _term_handles = [
        Line2D([0],[0], color='steelblue',   ls='-', lw=1.5, label='Coriolis'),
        Line2D([0],[0], color='firebrick',   ls='-', lw=1.5, label='Viscous'),
        Line2D([0],[0], color='magenta',     ls='-', lw=1.5, label='Turbulent'),
        Line2D([0],[0], color='cyan',        ls='-', lw=1.5, label='Dispersive'),
        Line2D([0],[0], color='gold',        ls='-', lw=1.5, label='Reynolds'),
        Line2D([0],[0], color='saddlebrown', ls='-', lw=1.5, label='Temporal'),
    ]

    # (Req 6) Black "Total" handle — the sum of all shear-stress terms (as in
    # PhAvg_rotated.py PLOT 32r / PlotField.plot_fig4_budget).  Used by the
    # outer-unit tau_yx plot, which is the only one below that draws a total curve.
    _total_handle = Line2D([0], [0], color='black', ls='-', lw=1.5, label=r'Total $\Sigma$')

    # 3a/3b. Shear stress tau_zx (P46) + tau_zy (P47), inner units, all cases.
    # Body lives in _plot_shear_stress_balance() (defined near the top of this
    # block) so RESULTS_ONLY=shear can regenerate just these two figures.
    # Colour = stress term; linestyle = case; black = Total; the Reynolds stress
    # is one combined curve (turbulent + dispersive).  No zoomed layer PNGs.
    _plot_shear_stress_balance()

    ###########################################################################
    # SECTION 3 (outer units) — MOMENTUM BALANCE
    # OUTER scaling pair (see THE UNIT SYSTEM): length z- = y/delta with
    # delta = u_tau(case)/f, velocity scale G (= 1), so a stress (velocity^2) is
    # plotted as tau/G^2 — the RAW stored value.  It was previously divided by
    # u_star2(h)^2, an INNER velocity scale on an OUTER abscissa.  The smooth
    # reference follows the same pair via norm_vel_smooth.
    # 🔒 The SIGN combination of every term below is the LOCKED standard budget
    # (C = -I_corr_yx / +I_corr_yz, V = +visc, R = -<u'v'>); only the DISPLAY
    # normalisation changed here — no sign, no term, no closure was touched.
    ###########################################################################

    # 3a (outer). Shear stress tau_yx — OUTER units z- = y/delta, tau/G^2.
    # (Req 6/8) renamed to "shear stress"; free y-scale; black total; z- <= 4.
    plt.figure(figsize=FIG_SIZE_WIDE, dpi=FIG_DPI)
    for _ in smooth_refs():
        _y_out_s = y_s / (ustr_s1 / float(f))
        _svs2    = norm_vel_smooth(2, 'outer')
        plt.plot(_y_out_s, -I_corr_yx_s/_svs2,
                 color='steelblue',   linestyle=SMOOTH_LS, linewidth=1.5)
        plt.plot(_y_out_s, np.mean(visc_yx_s, axis=1)/_svs2,
                 color='firebrick',   linestyle=SMOOTH_LS, linewidth=1.5)
        plt.plot(_y_out_s, -np.mean(Rxy_s, axis=1)/_svs2,
                 color='gold',        linestyle=SMOOTH_LS, linewidth=1.5)   # flat wall: Reynolds ≡ turbulent
        _tot_s = (-I_corr_yx_s + np.mean(visc_yx_s, axis=1)
                  - np.mean(Rxy_s, axis=1))
        plt.plot(_y_out_s, _tot_s/_svs2,
                 color='black', linestyle=SMOOTH_LS, linewidth=1.5)
    for case, ls in zip(SIM_NAMES, SIM_LINESTYLES):
        _Ic  = gv('I_corr_yx',   case)
        _vx  = gv('visc_yx',     case)
        _rv  = gv('rey_uv',      case)
        _dv  = gv('UV_disp',     case)
        _dt  = gv('dudt',        case)
        _yc  = gv('y',           case)
        _y_out, _dlt = _z_out(case)
        if _Ic is None or _y_out is None:
            continue
        _us2_hgt = norm_vel(case, 1, 'outer')   # = 1.0 -> tau/G^2 below
        plt.plot(_y_out, -_Ic/_us2_hgt**2,  color='steelblue',  linestyle=ls)
        _tot = -np.asarray(_Ic, dtype=float)
        if _vx is not None:
            plt.plot(_y_out,  _vx/_us2_hgt**2, color='firebrick',  linestyle=ls)
            _tot = _tot + np.asarray(_vx, dtype=float)
        # turbulent (rey_uv) + dispersive (UV_disp) = Reynolds shear stress.
        _rvp = _xprof(case, _rv) if _rv is not None else None
        _dvp = _xprof(case, _dv) if _dv is not None else None
        if _rvp is not None:
            plt.plot(_y_out, -_rvp/_us2_hgt**2, color='magenta', linestyle=ls)   # turbulent
            _tot = _tot - np.asarray(_rvp, dtype=float)
        if _dvp is not None:
            plt.plot(_y_out, -_dvp/_us2_hgt**2, color='cyan',    linestyle=ls)   # dispersive
            _tot = _tot - np.asarray(_dvp, dtype=float)
        if _rvp is not None and _dvp is not None:
            plt.plot(_y_out, -(np.asarray(_rvp) + np.asarray(_dvp))/_us2_hgt**2,
                     color='gold', linestyle=ls)                                 # Reynolds = turb+disp
        if _dt is not None:
            plt.plot(_y_out,  _dt/_us2_hgt**2, color='saddlebrown', linestyle=ls)
            _tot = _tot + np.asarray(_dt, dtype=float)
        plt.plot(_y_out, _tot/_us2_hgt**2, color='black', linestyle=ls, linewidth=1.5)
    plt.legend(handles=_term_handles + [_total_handle] + all_handles(),
               fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL, loc='upper right')
    _mark_h('v', mode='outer')
    plt.xlim(0, Z_MINUS_MAX)
    plt.xlabel(z_label('outer'))
    plt.ylabel(vel_label(r'$\langle\bar{\tau}_{zx}\rangle$', 2, 'outer'))
    plt.title(r'Shear stress $\tau_{zx}$ — outer units ($z^-\leq4$) — all Re (neutral)')
    plt.grid(True)
    plt.savefig(_figdir+'P48_MomBal_tauyx_allRe_outer.png', dpi=FIG_DPI)
    plt.show()

    # 3b (outer). Shear stress tau_yz — OUTER units z- = y/delta, tau/G^2.
    # 🔒 LOCKED — STANDARD signs (see CLAUDE.md), mirroring the tau_zx outer twin but
    # with the spanwise Coriolis C_zy = +I_corr_yz (Levi-Civita) and R_zy = -(turb+disp).
    # DISPLAY: the whole τ_zy panel is negated by _SP for paper handedness (same flag
    # as PhAvg_rotated.py / the inner P47 twin); physical closure & u* unaffected. z- <= 4.
    _SP = -1.0 if FIG4_PAPER_SPANWISE_SIGN else 1.0
    plt.figure(figsize=FIG_SIZE_WIDE, dpi=FIG_DPI)
    for _ in smooth_refs():
        _y_out_s = y_s / (ustr_s1 / float(f))
        _svs2    = norm_vel_smooth(2, 'outer')
        plt.plot(_y_out_s, _SP*I_corr_yz_s/_svs2,
                 color='steelblue',   linestyle=SMOOTH_LS, linewidth=1.5)
        plt.plot(_y_out_s, _SP*np.mean(visc_yz_s, axis=1)/_svs2,
                 color='firebrick',   linestyle=SMOOTH_LS, linewidth=1.5)
        plt.plot(_y_out_s, _SP*-np.mean(Ryz_s, axis=1)/_svs2,
                 color='gold',        linestyle=SMOOTH_LS, linewidth=1.5)   # flat wall: Reynolds ≡ turbulent
    for case, ls in zip(SIM_NAMES, SIM_LINESTYLES):
        _Iz  =  gv('I_corr_yz',   case)   # C_zy = +I_corr_yz (standard)
        _vz  =  gv('visc_yz',     case)
        _rw  =  gv('rey_vw',      case)
        _dvw =  gv('VW_disp',     case)
        _dw  =  gv('dwdt',        case)
        _yc  =  gv('y',           case)
        _y_out, _dlt = _z_out(case)
        if _Iz is None or _y_out is None:
            continue
        _us2_hgt = norm_vel(case, 1, 'outer')   # = 1.0 -> tau/G^2 below
        plt.plot(_y_out,  _SP*_Iz/_us2_hgt**2, color='steelblue',  linestyle=ls)
        if _vz is not None:
            plt.plot(_y_out, _SP*_xprof(case, _vz)/_us2_hgt**2,
                     color='firebrick',  linestyle=ls)
        # turbulent (rey_vw) + dispersive (VW_disp) = Reynolds shear stress.
        _rwp = _xprof(case, _rw)  if _rw  is not None else None
        _dwp = _xprof(case, _dvw) if _dvw is not None else None
        if _rwp is not None:
            plt.plot(_y_out, _SP*-_rwp/_us2_hgt**2, color='magenta', linestyle=ls)   # turbulent  R=-turb
        if _dwp is not None:
            plt.plot(_y_out, _SP*-_dwp/_us2_hgt**2, color='cyan',    linestyle=ls)   # dispersive R=-disp
        if _rwp is not None and _dwp is not None:
            plt.plot(_y_out, _SP*-(np.asarray(_rwp) + np.asarray(_dwp))/_us2_hgt**2,
                     color='gold', linestyle=ls)                                 # Reynolds = -(turb+disp)
        if _dw is not None:
            plt.plot(_y_out,  _SP*_dw/_us2_hgt**2, color='saddlebrown', linestyle=ls)
    plt.legend(handles=_term_handles + all_handles(), fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL, loc='upper right')
    _mark_h('v', mode='outer')
    plt.xlim(0, Z_MINUS_MAX)
    plt.xlabel(z_label('outer'))
    plt.ylabel(vel_label(r'$\langle\bar{\tau}_{zy}\rangle$', 2, 'outer'))
    plt.title(r'Shear stress $\tau_{zy}$ — outer units ($z^-\leq4$) — all Re (neutral)')
    plt.grid(True)
    plt.savefig(_figdir+'P49_MomBal_tauyz_allRe_outer.png', dpi=FIG_DPI)
    plt.show()

    ###########################################################################
    # SECTION 4 -- NEW SCIENTIFICALLY INTERESTING PLOTS
    ###########################################################################

    # 4a. Form drag vs skin friction partition (bar chart, no z-axis)
    _dlbls, _form_d, _skin_d = [], [], []
    for case, lbl in zip(SIM_NAMES, SIM_LABELS):
        _pd = gv('P_drag', case)
        _fx = gv('Fyx',    case)
        if _pd is None or _fx is None:
            continue
        _dlbls.append(lbl)
        _uc2 = norm_ustar(case)**2             # THIS case's own inner stress scale
        _form_d.append(float(_pd)/_uc2)
        _skin_d.append(float(_fx)/_uc2)
    if _dlbls:
        _xp = np.arange(len(_dlbls))
        _bw = 0.35
        plt.figure(figsize=(8, 5), dpi=FIG_DPI)
        plt.bar(_xp - _bw/2, _form_d, _bw, color='steelblue', label='Form drag')
        plt.bar(_xp + _bw/2, _skin_d, _bw, color='firebrick',  label='Skin friction')
        plt.xticks(_xp, _dlbls, fontsize=8)
        plt.ylabel(r'Drag $/ u_*^2$')
        plt.legend()
        plt.grid(True, axis='y')
        plt.title('Form drag vs skin friction partition -- all Re')
        plt.savefig(_figdir+'P50_DragPartition_allRe.png', dpi=FIG_DPI)
        plt.show()

    # 4b. Dispersive kinetic energy (DKE) profile — rough cases (INNER)
    plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    for case, clr, ls, lbl in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_LABELS):
        _du = gv('DispVelU', case)
        _dv = gv('DispVelV', case)
        _dw = gv('DispVelW', case)
        _yi = gy_in(case)
        if _du is None or _yi is None:
            continue
        _dke = (0.5*(np.mean(_du**2, axis=1) + np.mean(_dv**2, axis=1)
                     + np.mean(_dw**2, axis=1)) / norm_ustar(case)**2)
        plt.semilogy(_yi, _dke, color=clr, linestyle=ls, label=lbl)
    _mark_h('v')
    plt.xlim(0, Z_PLUS_MAX)
    plt.xlabel(r'$z^+$')
    plt.ylabel(r'$\frac{1}{2}\langle\tilde{u}_i\tilde{u}_i\rangle / u_*^2$')
    plt.legend(handles=sim_handles(), fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
    plt.grid(True, which='both', linestyle='--', linewidth=0.4)
    plt.title(r'Dispersive kinetic energy (DKE) — rough-wall cases (neutral) ($z^+\leq200$)')
    plt.savefig(_figdir+'P51_DKE_allRe.png', dpi=FIG_DPI)
    _save_layers_x(_figdir+'P51_DKE_allRe', r'Dispersive kinetic energy (DKE) — rough-wall cases (neutral)')
    plt.show()

    # 4b (outer). DKE — OUTER units z- = y/delta, DKE/G^2 (velocity^2), z-<=4
    plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    for case, clr, ls, lbl in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_LABELS):
        _du = gv('DispVelU', case); _dv = gv('DispVelV', case); _dw = gv('DispVelW', case)
        _zo, _dlt = _z_out(case)
        if _du is None or _zo is None:
            continue
        _dke = (0.5*(np.mean(_du**2, axis=1) + np.mean(_dv**2, axis=1)
                     + np.mean(_dw**2, axis=1)) / norm_vel(case, 2, 'outer'))
        plt.semilogy(_zo, _dke, color=clr, linestyle=ls, label=lbl)
    _mark_h('v', mode='outer')
    plt.xlim(0, Z_MINUS_MAX)
    plt.xlabel(z_label('outer'))
    plt.ylabel(vel_label(r'$\frac{1}{2}\langle\tilde{u}_i\tilde{u}_i\rangle$', 2, 'outer'))
    plt.legend(handles=sim_handles(), fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
    plt.grid(True, which='both', linestyle='--', linewidth=0.4)
    plt.title(r'Dispersive kinetic energy (DKE) — outer units ($z^-\leq4$) — all Re (neutral)')
    plt.savefig(_figdir+'P52_DKE_allRe_outer.png', dpi=FIG_DPI)
    plt.show()

    # 4c. TKE shear production P(z+)/u*3 — rough cases (INNER, z+<=200)
    plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    for case, clr, ls, lbl in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_LABELS):
        _rv = gv('rey_uv', case)
        _dd = gv('du_dy',  case)
        _yi = gy_in(case)
        if _rv is None or _dd is None or _yi is None:
            continue
        _P1d = -(_xprof(case, _rv)) * _xprof(case, _dd) / norm_ustar(case)**3
        plt.plot(_yi[:N_ROWS_ORO], _P1d[:N_ROWS_ORO], color=clr, linestyle=ls, label=lbl)
    _mark_h('v')
    plt.xlim(0, Z_PLUS_MAX)
    plt.xlabel(r'$z^+$')
    plt.ylabel(r"$-\langle\overline{u'w'}\rangle\,\partial\langle\bar{u}\rangle/\partial z\;/ u_*^3$")
    plt.legend(handles=sim_handles(), fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
    plt.grid(True)
    plt.title(r'TKE shear production — rough-wall cases (neutral) ($z^+\leq200$)')
    plt.savefig(_figdir+'P53_TKEproduction_allRe.png', dpi=FIG_DPI)
    _save_layers_x(_figdir+'P53_TKEproduction_allRe', r'TKE shear production — rough-wall cases (neutral)')
    plt.show()

    # 4c (outer). TKE shear production — OUTER units z- = y/delta.  P has
    # dimension velocity^3/length; the outer velocity scale is G, so P/G^3 is
    # what pairs with z-.  (The residual 1/length is common to every case
    # because the domain is identical, so the comparison is unaffected.)
    plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    for case, clr, ls, lbl in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_LABELS):
        _rv = gv('rey_uv', case); _dd = gv('du_dy', case)
        _zo, _dlt = _z_out(case)
        if _rv is None or _dd is None or _zo is None:
            continue
        _P1d = -(_xprof(case, _rv)) * _xprof(case, _dd) / norm_vel(case, 3, 'outer')
        plt.plot(_zo, _P1d, color=clr, linestyle=ls, label=lbl)
    _mark_h('v', mode='outer')
    plt.xlim(0, Z_MINUS_MAX)
    plt.xlabel(z_label('outer'))
    plt.ylabel(vel_label(r"$-\langle\overline{u'w'}\rangle\,\partial\langle\bar{u}\rangle/\partial z$", 3, 'outer'))
    plt.legend(handles=sim_handles(), fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
    plt.grid(True)
    plt.title(r'TKE shear production — outer units ($z^-\leq4$) — all Re (neutral)')
    plt.savefig(_figdir+'P54_TKEproduction_allRe_outer.png', dpi=FIG_DPI)
    plt.show()

    # 4d. Streamwise advection at orographic landmarks — all Re (INNER)
    _loc_colors = {'top': 'magenta', 'lf': 'red', 'bottom': 'black', 'rf': 'blue'}
    _loc_labels = {'top': 'Valley top', 'lf': 'Left flank',
                   'bottom': 'Valley bottom', 'rf': 'Right flank'}
    plt.figure(figsize=FIG_SIZE_TALL, dpi=FIG_DPI)
    for case, ls in zip(SIM_NAMES, SIM_LINESTYLES):
        _yi = gy_in(case)
        if _yi is None:
            continue
        for loc, clr in _loc_colors.items():
            _cv = gv('conv_'+loc, case)
            if _cv is None:
                continue
            plt.plot(_cv[:N_ROWS_ORO_ADV]/norm_ustar(case)**3, _yi[:N_ROWS_ORO_ADV], color=clr, linestyle=ls)
    _lh = [Line2D([0],[0], color=c, ls='-', label=_loc_labels[loc])
           for loc, c in _loc_colors.items()]
    plt.legend(handles=_lh + sim_handles(), fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
    _mark_h('h')
    plt.ylim(0, Z_PLUS_MAX)
    plt.xlabel(r'$u_j\,\partial u_i/\partial x_j\;/ u_*^3$')
    plt.ylabel(r'$z^+$')
    plt.grid(True, linestyle=':')
    plt.title(r'Streamwise advection at orographic landmarks — rough cases (neutral) ($z^+\leq200$)')
    plt.savefig(_figdir+'P55_Advection_landmarks_allRe.png', dpi=FIG_DPI)
    _save_layers_y(_figdir+'P55_Advection_landmarks_allRe',
                   r'Streamwise advection at orographic landmarks — rough-wall cases (neutral)')
    plt.show()

    # 4d (outer). Streamwise advection — OUTER units z- = y/delta on the
    # y-axis, value /G^3 (velocity^3), z- <= 4 (Req 8).
    plt.figure(figsize=FIG_SIZE_TALL, dpi=FIG_DPI)
    for case, ls in zip(SIM_NAMES, SIM_LINESTYLES):
        _zo, _dlt = _z_out(case)
        if _zo is None:
            continue
        _vs3 = norm_vel(case, 3, 'outer')
        for loc, clr in _loc_colors.items():
            _cv = gv('conv_'+loc, case)
            if _cv is None:
                continue
            # conv_* profiles start at a landmark-specific row offset (top=94,
            # flanks=flk_hgt, bottom=0; cf. PhAvg.py), so each is SHORTER than the
            # full case y-grid.  Clip both to their common length (index-0 aligned,
            # matching plot 55's _cv[:N_ROWS_ORO_ADV]/_yi[:N_ROWS_ORO_ADV]) so the outer z- ylim crops it.
            _n = min(_cv.shape[0], _zo.shape[0])
            plt.plot(_cv[:_n]/_vs3, _zo[:_n], color=clr, linestyle=ls)
    plt.legend(handles=_lh + sim_handles(), fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
    _mark_h('h', mode='outer')
    plt.ylim(0, Z_MINUS_MAX)
    plt.xlabel(vel_label(r'$u_j\,\partial u_i/\partial x_j$', 3, 'outer'))
    plt.ylabel(z_label('outer'))
    plt.grid(True, linestyle=':')
    plt.title(r'Streamwise advection at orographic landmarks — outer units ($z^-\leq4$) — all Re (neutral)')
    plt.savefig(_figdir+'P56_Advection_landmarks_allRe_outer.png', dpi=FIG_DPI)
    plt.show()

    # ═══════════════════════════════════════════════════════════════════════
    # ░░  SECTION 5 — CHAPTER-6 IMMEDIATELY-ACHIEVABLE DIAGNOSTICS (all Re)  ░░
    # Research.md "Immediately achievable from existing data" (lines 568-610) +
    # the medium-priority items computable from the existing phase-averaged 2-D
    # fields and the first-plane snapshots.  Most plots here are cross-case
    # visualizations (all simulations overlaid / side-by-side panels); the few
    # per-case figures (e.g. the D19 velocity-field maps) and all plots now share
    # the single fig/ folder with PhAvg_rotated.py.
    # Each block is gated: a case missing a required field is skipped, not crashed.
    # Reduced numbers are collected in _ch6 for the end-of-run summary.  Scaling
    # is single-reference (u_star / l_in / Re_tau); grids/eps/geometry per-case.
    # ═══════════════════════════════════════════════════════════════════════
    print('\n' + '=' * 78)
    print('SECTION 5 — Chapter-6 immediately-achievable diagnostics (all Re)')
    print('=' * 78)
    _ch6 = {}
    # Helpers _ch6set/_stations/_surf_rows/_col_at_xplus
    # → FUNCTION DEFINITIONS section (top).  (_ch6set records into _ch6 above.)

    # ── D1. Surface wind-veer angle α_s(x⁺) (immediate #1) ────────
    # Veer = ∠ of the near-wall (first-fluid-cell) wind to the geostrophic
    # (rotated x), per x-column.  All cases overlaid.
    # SIGN CONVENTION: built from the raw pickled AvgPhU/AvgPhW (physical
    # rotated-frame sign), so α_s is NEGATIVE near the surface (Ekman veer left
    # of G) — like the smooth reference rW/FrictionAngle, and OPPOSITE to the
    # hodograph/_veer_deg figures, which use the display-flipped w_plus_rot.
    plt.figure(figsize=(9, 5), dpi=FIG_DPI)
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
    plt.legend(handles=sim_handles(), fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
    plt.grid(True, linestyle='--', linewidth=0.4)
    plt.title(r'Surface wind-veer angle $\alpha_s(x^+)$ — all Re (neutral)')
    plt.savefig(_figdir +'P57_Ch6_veer_surface_allRe.png', dpi=FIG_DPI)
    plt.show()

    # ── D4. Depth-integrated Ekman transport M_y(x⁺)=∫₀^δ⟨U⟩dz (#4) ─
    # P1 falsification test for the τ_zy depth-integral mechanism: M_y should NOT
    # be suppressed over the valley by a factor comparable to the τ_zx reduction
    # (~0.59, Research.md §6.14.5).  All cases overlaid; per-case windward
    # suppression logged + a verdict.
    _TAU_ZX_REDUCTION = 0.59          # reference valley τ_zx reduction (Ch. 6)
    plt.figure(figsize=(9, 5), dpi=FIG_DPI)
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
    plt.legend(handles=sim_handles(), fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
    plt.grid(True, linestyle='--', linewidth=0.4)
    plt.title(r'Depth-integrated Ekman transport $M_y(x^+)$ — all Re (neutral)')
    plt.savefig(_figdir +'P58_Ch6_My_x_allRe.png', dpi=FIG_DPI)
    plt.show()

    # ── D5. Form-drag windward/lee split (immediate #5) ───────────
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
        plt.figure(figsize=(8, 5), dpi=FIG_DPI)
        plt.bar(_xp - _bw / 2, [r[1] for r in _d5], _bw, color='steelblue', label='windward (P_Lag)')
        plt.bar(_xp + _bw / 2, [r[2] for r in _d5], _bw, color='firebrick', label='lee (P_Front)')
        plt.xticks(_xp, [r[0] for r in _d5], fontsize=8)
        plt.ylabel('Form-drag contribution'); plt.legend(); plt.grid(True, axis='y')
        plt.title('Form-drag windward vs lee split — all Re')
        plt.savefig(_figdir +'P59_Ch6_Dform_windlee_allRe.png', dpi=FIG_DPI)
        plt.show()

    # ── D8. Streamwise momentum budget at a station x⁺≈1050 (#8) ───
    # Recompute C/V/R at one x-column from the 2-D fields (the pickled I_corr_yx
    # etc. are x-averaged 1-D): V=ν∂⟨u⟩/∂z, R=−⟨u'v'⟩, C=−∫⟨v⟩dz (g2≈0 rotated).
    # All cases overlaid; smooth reference (x-independent) for context.
    plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    for case, clr, ls in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES):
        _U = gv('AvgPhU', case); _V = gv('AvgPhV', case); _ruv = gv('rey_uv', case)
        _yc = gv('y', case)
        if _U is None or _V is None or _ruv is None or _yc is None:
            continue
        _i = _col_at_xplus(case, 1050.0)
        _V_visc = case_nu(case) * np.gradient(_U[:, _i], _yc)
        _R = -_ruv[:, _i]                                # turbulent ⟨u''v''⟩ at this column
        _uvd = gv('UV_disp', case)
        _Rd = -_uvd[:, _i] if _uvd is not None else np.zeros_like(_R)   # dispersive ũṽ
        _C = -vIntegral(_V[:, _i], _U.shape[0], _yc)
        _zc = gy_in(case)
        _u2 = norm_ustar(case) ** 2
        plt.plot(_zc, _C / _u2, color='steelblue',  linestyle=ls)
        plt.plot(_zc, _V_visc / _u2, color='firebrick', linestyle=ls)
        plt.plot(_zc, _R / _u2, color='magenta', linestyle=ls)                # turbulent
        plt.plot(_zc, _Rd / _u2, color='cyan', linestyle=ls)                  # dispersive
        plt.plot(_zc, (_R + _Rd) / _u2, color='gold', linestyle=ls)          # Reynolds = turb+disp
        plt.plot(_zc, (_C + _V_visc + _R + _Rd) / _u2, color='black', linestyle=ls)
    _mark_h('v')
    plt.xlim(0, 200)
    plt.xlabel(r'$z^+$'); plt.ylabel(r'$\tau_{zx}$ budget $/u_*^2$ at $x^+\!\approx\!1050$')
    _d8h = [Line2D([0], [0], color=c, label=l) for c, l in
            [('steelblue', 'Coriolis C'), ('firebrick', 'Viscous V'),
             ('magenta', 'Turbulent'), ('cyan', 'Dispersive'),
             ('gold', 'Reynolds R'), ('black', 'Total T')]]
    plt.legend(handles=_d8h + sim_handles(), fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
    plt.grid(True, linestyle='--', linewidth=0.4)
    plt.title(r'Streamwise momentum budget at $x^+\approx1050$ — all Re')
    plt.savefig(_figdir +'P60_Ch6_MomBudget_x1050_allRe.png', dpi=FIG_DPI)
    plt.show()

    # ── D9. Log-log |⟨W⟩|(z⁺) at the windward peak (immediate #9) ──
    plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    for case, clr, ls, lbl in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_LABELS):
        _W = gv('AvgPhW', case)
        if _W is None:
            continue
        _i = _stations(case)['wind']
        _Wp = np.abs(_W[:, _i]); _zc = gy_in(case)
        plt.loglog(_zc, _Wp, color=clr, linestyle=ls, label=lbl)
        _jpk = int(np.argmax(_Wp))
        _ch6set(case, 'Wwind_peak', (float(_Wp[_jpk]), float(_zc[_jpk])))
    plt.xlabel(r'$z^+$'); plt.ylabel(r'$|\langle\bar{v}\rangle|$ (spanwise, windward)')
    plt.legend(handles=sim_handles(), fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
    plt.grid(True, which='both', linestyle='--', linewidth=0.4)
    plt.title(r'Log-log spanwise $|\langle\bar{v}\rangle|(z^+)$ at windward peak — all Re')
    plt.savefig(_figdir +'P61_Ch6_Wwind_loglog_allRe.png', dpi=FIG_DPI)
    plt.show()

    # ── D10. Lee–windward W symmetry test (immediate #10) ─────────
    plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
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
    plt.xlabel(r'$z^+$'); plt.ylabel(r'$\langle\bar{v}\rangle$ (spanwise; solid=windward, faded=lee)')
    plt.legend(handles=sim_handles(), fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
    plt.grid(True, linestyle='--', linewidth=0.4)
    plt.title(r'Lee vs windward spanwise $\langle\bar{v}\rangle(z^+)$ — all Re')
    plt.savefig(_figdir +'P62_Ch6_W_leewind_allRe.png', dpi=FIG_DPI)
    plt.show()

    # ── D12. TKE production at valley centre vs smooth (#15) ───────
    # P(z⁺) = −⟨u'v'⟩ ∂⟨u⟩/∂z at the valley-centre column; smooth reference is
    # x-mean.  Logs the production-peak height (Ch. 6: ≈17 smooth → ≈34 valley).
    plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    for _ in smooth_refs():
        _Ps = -np.mean(Rxy_s, axis=1) * np.mean(du_dy_s, axis=1)
        plt.semilogx(y_in_s, _Ps / ustr_s1 ** 3, color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
        _ch6set(SMOOTH_KEY, 'TKEprod_peak_z', float(y_in_s[int(np.argmax(_Ps))]))
    for case, clr, ls, lbl in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_LABELS):
        _ruv = gv('rey_uv', case); _dd = gv('du_dy', case)
        if _ruv is None or _dd is None:
            continue
        _i = _stations(case)['floor']
        _P = -_ruv[:, _i] * _dd[:, _i]; _zc = gy_in(case)
        plt.semilogx(_zc, _P / norm_ustar(case) ** 3, color=clr, linestyle=ls, label=lbl)
        _jpk = int(np.argmax(_P))
        _ch6set(case, 'TKEprod_peak_z', float(_zc[_jpk]))
    _mark_h('v')
    plt.xlabel(r'$z^+$'); plt.ylabel(r'$\mathcal{P}/u_*^3$ (valley centre)')
    plt.legend(handles=all_handles(), fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
    plt.grid(True, which='both', linestyle='--', linewidth=0.4)
    plt.title(r'TKE production at valley centre vs smooth — all Re')
    plt.savefig(_figdir +'P63_Ch6_TKEprod_centre_allRe.png', dpi=FIG_DPI)
    plt.show()

    # ── D14. Outer-layer mean-velocity surplus ΔU⁺(z⁺) (Fig 6.19 #3) ─
    # ΔU⁺ = ⟨u⟩⁺_valley − ⟨u⟩⁺_smooth (both on the single-reference axis), per
    # case interpolated onto the smooth z⁺ grid.  Logs max surplus above z⁺≈100.
    for _ in smooth_refs():
        # One figure PER smooth reference (the surplus is defined against a
        # specific flat wall), so the filename/title carry the reference key.
        _Usm = np.mean(U_s_p, axis=1)
        plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
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
                # (recorded against the smooth reference currently bound)
        plt.axhline(0, color='k', lw=0.6)
        _mark_h('v'); plt.xscale('log')
        plt.xlabel(r'$z^+$'); plt.ylabel(r'$\Delta\langle\bar{u}\rangle^+$ (valley − smooth)')
        plt.legend(handles=sim_handles(), fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
        plt.grid(True, which='both', linestyle='--', linewidth=0.4)
        plt.title(r'Outer-layer velocity surplus $\Delta U^+$ — all Re '
                  '(vs %s)' % SMOOTH_LABEL)
        plt.savefig(_figdir + 'P64_Ch6_dUplus_allRe_%s.png' % SMOOTH_KEY, dpi=FIG_DPI)
        plt.show()

    # ── D15. TKE anisotropy: normal-stress components (TKE #5) ─────
    # Turbulent normal stresses ⟨u''u''⟩, ⟨v''v''⟩, ⟨w''w''⟩ (rey_*; x-averaged,
    # intrinsic) vs z⁺, all cases overlaid; smooth reference (flat wall: turbulent
    # ≡ Reynolds).  Logs which component the valley most enhances at peak.
    plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    for _ in smooth_refs():
        plt.plot(y_in_s, np.mean(Rxx_s, axis=1) / ustr_s1 ** 2, color=SMOOTH_COLOR, linestyle='-')
        plt.plot(y_in_s, np.mean(Ryy_s, axis=1) / ustr_s1 ** 2, color=SMOOTH_COLOR, linestyle='--')
        plt.plot(y_in_s, np.mean(Rzz_s, axis=1) / ustr_s1 ** 2, color=SMOOTH_COLOR, linestyle=':')
    for case, clr, ls, lbl in zip(SIM_NAMES, SIM_COLORS, SIM_LINESTYLES, SIM_LABELS):
        _uu = gv('rey_uu', case); _vv = gv('rey_vv', case); _ww = gv('rey_ww', case)
        _yi = gy_in(case)
        if _uu is None or _vv is None or _ww is None or _yi is None:
            continue
        _pu = _xprof(case, _uu) / norm_ustar(case) ** 2
        _pv = _xprof(case, _vv) / norm_ustar(case) ** 2
        _pw = _xprof(case, _ww) / norm_ustar(case) ** 2
        plt.plot(_yi, _pu, color=clr, linestyle='-')
        plt.plot(_yi, _pv, color=clr, linestyle='--')
        plt.plot(_yi, _pw, color=clr, linestyle=':')
        _comp = ['uu', 'vv', 'ww'][int(np.argmax([np.max(_pu), np.max(_pv), np.max(_pw)]))]
        _ch6set(case, 'TKE_dominant_component', _comp)
    _mark_h('v'); plt.xscale('log')
    plt.xlabel(r'$z^+$'); plt.ylabel(r"$\langle u_i''^2\rangle/u_*^2$")
    _d15h = [Line2D([0], [0], color='k', ls=s, label=l) for s, l in
             [('-', r"$u''u''$"), ('--', r"$w''w''$"), (':', r"$v''v''$")]]
    plt.legend(handles=_d15h + sim_handles(), fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL)
    plt.grid(True, which='both', linestyle='--', linewidth=0.4)
    plt.title(r'TKE anisotropy (normal stresses) — all Re')
    plt.savefig(_figdir +'P65_Ch6_TKEanisotropy_allRe.png', dpi=FIG_DPI)
    plt.show()

    # ── D2/D3. Mean & dispersive streamfunction ψ(x⁺,z⁺) (#2,#19) ─
    # ψ = ∫₀^z ⟨u⟩ dz' (wall-anchored).  Computed per case, stored into the sim
    # dict, then drawn as side-by-side panels by the existing plot2D_allRe.
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
    # The MEAN streamfunction figure (former P66) is intentionally not drawn;
    # psi_mean is still computed above because psi_min / psi_disp_ratio (D2/D3)
    # depend on it, and is released below.
    plot2D_allRe('psi_disp', r"Dispersive streamfunction $\psi''(x^+,z^+)$ — all Re",
                 'RdBu_r', 'P67_Ch6_streamfunction_disp_allRe.png', ylim=250,
                 shared_scale=True, overlay_contours=True, n_contours=12)
    for case in SIM_NAMES:
        sims.get(case, {}).pop('psi_mean', None)   # only psi_min (a scalar) is kept
        sims.get(case, {}).pop('psi_disp', None)
    print('  [D2/D3] ψ note: 2-D spanwise-mean projection; the spanwise drift '
          '⟨w̄⟩ (AvgPhW) carries fluid through the apparent recirculation — a '
          'true 3-D closed-orbit test needs spanwise-resolved fields (gated).')
# %%


    # ── D17. Streamwise-resolved Coriolis integrand C(x⁺,z⁺) (#3) ──
    # C(x,z)=∫₀^z(g2−⟨v⟩)dz' (g2≈0 rotated).  R(x,z)=−⟨u''v''⟩ is already the
    # Turbulent-stress panel (Section 1), so only the new C map is added here.
    for case in SIM_NAMES:
        _V = gv('AvgPhV', case); _yc = gv('y', case)
        if _V is None or _yc is None:
            continue
        _m = gmask0(case); _Vm = _V * _m if np.shape(_m) == np.shape(_V) else _V
        sims[case]['C2D'] = -vIntegral_2d(_Vm, _V.shape[0], _yc)
    plot2D_allRe('C2D', r'Streamwise-resolved Coriolis integrand $\mathcal{C}(x^+,z^+)$ — all Re',
                 'RdBu_r', 'P68_Ch6_Coriolis2D_allRe.png', ylim=200,
                 shared_scale=True, overlay_contours=True, n_contours=12)
    for case in SIM_NAMES:
        sims.get(case, {}).pop('C2D', None)
        
    # ─── DISABLED (2026-08-17, user request): D18 is very slow (four 2nd-order
    # derivative fields per case on the full x-z grid).  Re-enable by uncommenting
    # the block below when the pressure decomposition is needed.  The Chapter-6
    # summary print of poisson_fracs (~line 2591) is already guarded by an
    # "if in _ob" test, so it simply stays silent while this is off.
    # # ── D18. Pressure-Poisson source decomposition (medium 3) ──────
    # # ∇²P = −∂²(u_iu_j)/∂x_i∂x_j.  Split the RHS into mean-strain / turbulent /
    # # dispersive sources (turbulent + dispersive = the Reynolds source); store the
    # # total for the panel, log which dominates at the Cp extrema (windward / lee floor).
    # # Helpers _d2/_dxz → FUNCTION DEFINITIONS section (top).
    # for case in SIM_NAMES:
    #     _U = gv('AvgPhU', case); _V = gv('AvgPhV', case)
    #     _xc = gv('x', case); _yc = gv('y', case)
    #     if _U is None or _V is None or _xc is None or _yc is None:
    #         continue
    #     _Smean = -(_d2(_U * _U, _xc, 1) + 2 * _dxz(_U * _V, _xc, _yc) + _d2(_V * _V, _yc, 0))
    #     _uu = gv('rey_uu', case); _uv = gv('rey_uv', case); _vv = gv('rey_vv', case)
    #     _Srey = (-(_d2(_uu, _xc, 1) + 2 * _dxz(_uv, _xc, _yc) + _d2(_vv, _yc, 0))
    #              if (_uu is not None and _uv is not None and _vv is not None)
    #              else np.zeros_like(_Smean))
    #     _Du = gv('UU_disp', case); _Duv = gv('UV_disp', case); _Dvv = gv('VV_disp', case)
    #     _Sdisp = (-(_d2(_Du, _xc, 1) + 2 * _dxz(_Duv, _xc, _yc) + _d2(_Dvv, _yc, 0))
    #               if (_Du is not None and _Duv is not None and _Dvv is not None)
    #               else np.zeros_like(_Smean))
    #     sims[case]['Psource_total'] = _Smean + _Srey + _Sdisp
    #     # Dominant source over the near-wall valley band (z⁺<50) at the floor.
    #     _band = gy_in(case) < 50.0
    #     _i0, _i1 = _stations(case)['wind'], _stations(case)['lee']
    #     _fm = float(np.nanmean(np.abs(_Smean[_band][:, _i0:_i1 + 1])))
    #     _fr = float(np.nanmean(np.abs(_Srey[_band][:, _i0:_i1 + 1])))
    #     _fd = float(np.nanmean(np.abs(_Sdisp[_band][:, _i0:_i1 + 1])))
    #     _tot = _fm + _fr + _fd
    #     if _tot > 0:
    #         _ch6set(case, 'poisson_fracs', (_fm / _tot, _fr / _tot, _fd / _tot))
    #         _dom = ['mean-strain', 'Reynolds', 'dispersive'][int(np.argmax([_fm, _fr, _fd]))]
    #         print(f"  [D18] {case:<12} Poisson source over valley: mean={_fm/_tot:.2f} "
    #               f"rey={_fr/_tot:.2f} disp={_fd/_tot:.2f} → dominant: {_dom}")
    # plot2D_allRe('Psource_total', r'Pressure-Poisson source $-\partial^2(u_iu_j)/\partial x_i\partial x_j$ — all Re',
    #              'RdBu_r', 'P69_Ch6_Poisson_source_allRe.png', ylim=200, overlay_contours=True, n_contours=12)

    # ── D18 (lite). Pressure-Poisson source — READ, never recomputed ────
    # Replacement for the commented-out block above: PhAvg_rotated.py (stage b)
    # now builds Psource_{mean,rey,disp,total} with the compact operators and
    # pickles them, together with poisson_fracs = (mean, rey, disp) shares over
    # the valley band z⁺<50.  Stage c only loads and draws them — no derivative
    # is taken here, so the D18 cost is the one panel, not nine 2nd-derivative
    # passes per case.  Silent no-op on pickles written before those keys
    # existed: re-run PhAvg_rotated.py for a case to light it up.
    for case in SIM_NAMES:
        _pf = gv('poisson_fracs', case)
        if _pf is None or not np.all(np.isfinite(np.asarray(_pf, float))):
            continue
        _fm, _fr, _fd = (float(_v) for _v in _pf)
        _ch6set(case, 'poisson_fracs', (_fm, _fr, _fd))
        _dom = ['mean-strain', 'Reynolds', 'dispersive'][int(np.argmax([_fm, _fr, _fd]))]
        print(f"  [D18] {case:<12} Poisson source over valley: mean={_fm:.2f} "
              f"rey={_fr:.2f} disp={_fd:.2f} → dominant: {_dom}")
    if any(gv('Psource_total', case) is not None for case in SIM_NAMES):
        plot2D_allRe('Psource_total',
                     r'Pressure-Poisson source $-\partial^2(u_iu_j)/\partial x_i\partial x_j$ — all Re',
                     'RdBu_r', 'P69_Ch6_Poisson_source_allRe.png', ylim=200,
                     overlay_contours=True, n_contours=12)
        for case in SIM_NAMES:
            for _pk in ('Psource_total', 'Psource_mean', 'Psource_rey', 'Psource_disp'):
                sims.get(case, {}).pop(_pk, None)
    else:
        print('  [D18] Psource_* not in any pickle — skipped (re-run '
              'PhAvg_rotated.py per case to generate them).')


    # ── D11/D16. Terrain-following maps (immediate #14, #20) ────
    # Re-sample to ζ⁺ = z⁺ − local-surface⁺ so a constant-ζ row sits a constant
    # distance above the local surface, removing the leading-order kinematic
    # crest/valley artefact.  Side-by-side panels, all cases.
    # Helpers _panels_zeta/_tf_disp/_tf_tauzx → FUNCTION DEFINITIONS section (top).

    _panels_zeta(_tf_disp,
                 r'Terrain-following dispersive velocity magnitude $|\tilde{u}_i|(x^+,\zeta^+)$ — all Re',
                 'hot_r', 'P70_Ch6_dispTF_allRe.png', zmax_plus=800)
    _panels_zeta(_tf_tauzx,
                 r'Terrain-following streamwise stress $\nu\partial\bar{u}/\partial z-\overline{u^\prime w^\prime}$ — all Re',
                 'RdBu_r', 'P71_Ch6_tauzxTF_allRe.png', zmax_plus=800)

    # ── D19. 3-component velocity FIELD maps (mirror of PhAvg_rotated.py's named
    #         [PLOT 06]/[PLOT 11] plot_phavg_velocity_3D figures) ──────────────
    # The per-run figure is a 3-panel map of a velocity triad: in-plane (u,v)
    # streamlines over a spanwise-velocity contour, the out-of-plane yaw angle,
    # and the 3-D speed.  It has no cross-case panel form (streamline overlays
    # don't tile cleanly), so we reproduce it here per case for BOTH the
    # phase-averaged MEAN velocity ⟨ū_i⟩ (P91) and the DISPERSIVE velocity ũ_i
    # (P92), each figure NAMED via the title arg + a per-case filename.  A case
    # missing the components is skipped (graceful).
    def _vel3D_field_per_case(keys, title_root, fname_prefix):
        _kU, _kV, _kW = keys
        for case, lbl in zip(SIM_NAMES, SIM_LABELS):
            _U = gv(_kU, case); _V = gv(_kV, case); _W = gv(_kW, case)
            if _U is None or _V is None or _W is None:
                continue
            _xp, _yp, _xo, _yo = _case_grid(case, use_inner=True)
            _t = min(limity, len(_yp), np.shape(_U)[0])
            plot_phavg_velocity_3D(_xp, _yp[:_t],
                                   np.asarray(_U)[:_t, :], np.asarray(_V)[:_t, :],
                                   np.asarray(_W)[:_t, :], geps(case)[:_t, :], 1000,
                                   _xo, _yo,
                                   _figdir +fname_prefix + case + '.png',
                                   title=title_root + '  —  ' + lbl)
    _vel3D_field_per_case(('AvgPhU', 'AvgPhV', 'AvgPhW'),
                          r'Phase-averaged mean velocity field  '
                          r'$\langle\overline{u}_i\rangle(x^+,z^+)$',
                          'P91_MeanVel3D_')
    _vel3D_field_per_case(('DispVelU', 'DispVelV', 'DispVelW'),
                          r'Dispersive velocity field  $\widetilde{u}_i(x^+,z^+)$',
                          'P92_DispVel3D_')

    # [console table, no figure] ── D6. Wall-normal pressure equilibrium ∂P/∂z (#6) ─
    print('\n  [D6] Wall-normal pressure equilibrium  |∂P/∂z|  (IBM band vs outer):')
    print(f"    {'case':<14}{'max|dP/dz| IBM':>18}{'mean|dP/dz| outer':>20}")
    for case, lbl in zip(SIM_NAMES, SIM_LABELS):
        _P = gv('AvgP', case); _yc = gv('y', case)
        if _P is None or _yc is None:
            continue
        _dPdz = np.gradient(_P, _yc, axis=0)
        _zc = gy_in(case)
        # "IBM band" = below 1.5 h of THIS case.  Comparing a per-case z+ axis
        # against the reference crest y_in[hill_hgt] (as before) sampled a band
        # of the wrong physical depth in every case but the reference one — at
        # Re_D=750 the reference crest z+~23 sits well BELOW the real crest
        # z+~57, so the "IBM band" excluded most of the valley.
        _band = _zc < 1.5 * case_hill_zplus(case)
        _outer = _zc > 100.0
        _ibm = float(np.nanmax(np.abs(_dPdz[_band]))) if np.any(_band) else float('nan')
        _out = float(np.nanmean(np.abs(_dPdz[_outer]))) if np.any(_outer) else float('nan')
        _ch6set(case, 'pressure_equil', (_ibm, _out))
        print(f"    {lbl:<14}{_ibm:>18.4e}{_out:>20.4e}")

    # [console table, no figure] ── D7. Surface-pressure streamwise spectrum (#7) ─
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
        _dxp = 2.0 * float(gv('dx', case) or np.nan) / norm_l_in(case)
        _flag = 'grid?' if np.isfinite(_dxp) and _lam < 2.0 * _dxp else 'ok'
        _ch6set(case, 'cp_lambda', float(_lam))
        print(f"    {lbl:<14}{_lam:>12.1f}{_dxp:>18.2f}{_flag:>10}")

    # [console table, no figure] ── D13. Log-law parameters κ, z₀ₘ⁺, d⁺ (#16,#17) ─
    print('\n  [D13] Log-law parameters (κ, z₀ₘ⁺, d⁺):')
    for _ in smooth_refs():
        _Us = np.mean(U_s_p, axis=1)
        _msk = (y_in_s >= 60) & (y_in_s <= 200) & (_Us > 0)
        if np.count_nonzero(_msk) >= 2:
            _sl, _ic, _rv, _pv, _se = linregress(np.log(y_in_s[_msk]), _Us[_msk])
            _ksm = (1.0 / _sl) if _sl != 0 else float('nan')
            _ch6set(SMOOTH_KEY, 'kappa_matched', float(_ksm))
            print(f"    {SMOOTH_KEY} (matched z⁺∈[60,200]): "
                  f"κ={_ksm:.4f}  R²={_rv**2:.4f}")
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

    # [console table, no figure] ── D19. Linear potential-flow Cp vs measured (#med4) ─
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
    # ░░  FROUDE-LADDER ONLY — DISABLED ON THIS REYNOLDS LADDER            ░░
    # ═══════════════════════════════════════════════════════════════════════
    # PRESERVED VERBATIM, NOT DELETED.  Everything in the `else:` branch below
    # (X1 stability axis … X10 local similarity, ~430 lines, P72–P78) plots a
    # diagnostic AGAINST the bulk Richardson number Ri_B, or splits a flux into
    # its buoyancy part.  BOTH cases on this ladder are neutral (Fr = ∞), so
    # Ri_B = 0 for both: every scatter-vs-Ri_B view degenerates to a single
    # point and every buoyancy flux is identically zero.  Those figures would be
    # not just sparse but misleading, so the whole branch is gated off by
    # PLOT_FR_LADDER (set at the top of the file).  Flip that flag to True if a
    # stratified case is ever added here and the block comes back unchanged.
    #
    # Deliberately NOT gated: the intermittency views further below (P79–P84).
    # Ansorge γ is a Reynolds-dependent quantity, meaningful at neutral
    # stratification, and each of those helpers already skips a case whose .npz
    # is absent.  Its buoyancy-conditional fields (γ_b, ⟨b⟩cond) simply will not
    # exist in a neutral run's .npz and are skipped by the same mechanism.
    _FR = SIM_FR                      # was the six-entry Froude map in results.py
    _xc = ([c for c in CASES
            if c['name'] in sims and gv('Ri_B', c['name']) is not None]
           if PLOT_FR_LADDER else [])
    if len(_xc) == 0:
        if not PLOT_FR_LADDER:
            print('[results_Re] cross-case Ri_B aggregation (P72-P78) skipped: '
                  'Froude-ladder view, and both cases here are neutral '
                  '(PLOT_FR_LADDER=False).')
        else:
            print('[results_Re] cross-case research aggregation: no pickles carry '
                  'the research diagnostics yet — run PhAvg_rotated.py per case '
                  'first.')
    else:
        # _figdir is a dir+filename-prefix (see the figure-output block), so
        # makedirs must target the real directory _figroot, not the prefix.
        _figdir_x = _figdir
        _os.makedirs(_figroot, exist_ok=True)
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

        # ── [X1] Stability axis: each Fr at its measured Ri_B + bins ──
        # Cross-case mirror of PhAvg_rotated.py's per-run Research_stability_axis.png
        # (R5): the weak | intermediate | strong bins are the Ansorge (2017) edges
        # config.Ri_B_bins (not hardcoded), labelled bands as in the per-run figure;
        # each Fr case is placed at its measured Ri_B (one marker per case instead of
        # the single run's vertical line).  _hi matches R5's upper limit so the band
        # extents agree between the per-run and cross-case views.
        fig, ax = plt.subplots(figsize=(9, 3.6), dpi=FIG_DPI)
        _b0, _b1 = float(RI_B_BINS[0]), float(RI_B_BINS[1])
        _hi = max(_b1 * 2.0, float(np.nanmax(np.abs(_RiB))) * 1.3, _b1 + 0.05)
        ax.axvspan(0,   _b0, color='green',  alpha=0.12, label='weak')
        ax.axvspan(_b0, _b1, color='orange', alpha=0.12, label='intermediate')
        ax.axvspan(_b1, _hi, color='red',    alpha=0.12, label='strong')
        for i, n in enumerate(_nm):
            ax.scatter(_RiB[i], 0.0, color=_col[i], marker=_mk[i], s=90, zorder=5, label=_lab[i])
        ax.set_yticks([]); ax.set_xlim(0, _hi)
        ax.set_xlabel(r'$Ri_B = B_0\,\delta_{neu}/G^2$')
        ax.set_title('Stability axis: weak | intermediate | strong')
        ax.legend(fontsize=LEGEND_FONTSIZE, ncol=3, loc='upper center')
        fig.savefig(_figdir_x + 'P72_Xcase_stability_axis.png', dpi=FIG_DPI, bbox_inches='tight'); plt.show()

        # ── [X2] Dispersive share vs Ri_B (momentum & buoyancy; Goal 4) ─
        _sm = np.array([_share_BL(n, 'disp_share_mom')  for n in _nm])
        _sb = np.array([_share_BL(n, 'disp_share_buoy') for n in _nm])
        fig, ax = plt.subplots(figsize=FIG_SIZE_HODO, dpi=FIG_DPI)
        ax.plot(_RiB, _sm, 'bo-',  label='momentum')
        ax.plot(_RiB, _sb, 'rs--', label='buoyancy')
        ax.set_xlabel(r'$Ri_B$'); ax.set_ylabel('BL-mean dispersive share')
        ax.set_title('Dispersive share vs $Ri_B$')
        ax.legend(); ax.grid(True, ls='--', lw=0.5)
        fig.savefig(_figdir_x + 'P73_Xcase_dispshare_vs_RiB.png', dpi=FIG_DPI, bbox_inches='tight'); plt.show()

        # ── [X3] Scales & Obukhov vs Ri_B (Goals 3 & 1) ─────────────
        _panels = [('u_star', 'u*'), ('delta', r'$\delta$'), ('Psi', r'$\Psi=L_x/2\delta$'),
                   ('H_delta', r'$H/\delta$'), ('H_plus', r'$H^+$')]
        fig, axs = plt.subplots(2, 3, figsize=(14, 8), dpi=FIG_DPI)
        for axi, (key, ttl) in zip(axs.flat, _panels):
            axi.plot(_RiB, [_scl(n, key) for n in _nm], 'ko-')
            axi.set_xlabel(r'$Ri_B$'); axi.set_title(ttl); axi.grid(True, ls='--', lw=0.5)
        axs.flat[5].plot(_RiB, _Lp, 'ko-'); axs.flat[5].set_title(r'$L_{col}^+$')
        axs.flat[5].axhline(100, color='r', ls=':', label='collapse ~100')
        axs.flat[5].set_xlabel(r'$Ri_B$'); axs.flat[5].legend(fontsize=LEGEND_FONTSIZE); axs.flat[5].grid(True, ls='--', lw=0.5)
        fig.suptitle('Scales & Obukhov length vs $Ri_B$')
        fig.tight_layout()
        fig.savefig(_figdir_x + 'P74_Xcase_scales_vs_RiB.png', dpi=FIG_DPI, bbox_inches='tight'); plt.show()

        # ── [X4] Similarity departure vs Ri_B (Goal 5) ──────────────
        fig, ax = plt.subplots(figsize=FIG_SIZE_HODO, dpi=FIG_DPI)
        ax.plot(_RiB, [_depmean(n, 'phi_m_dep') for n in _nm], 'bo-',  label=r'$\phi_m$')
        ax.plot(_RiB, [_depmean(n, 'phi_h_dep') for n in _nm], 'rs--', label=r'$\phi_h$')
        ax.set_xlabel(r'$Ri_B$'); ax.set_ylabel('RMS departure from MOST (station mean)')
        ax.set_title('Similarity departure vs $Ri_B$')
        ax.legend(); ax.grid(True, ls='--', lw=0.5)
        fig.savefig(_figdir_x + 'P75_Xcase_phidep_vs_RiB.png', dpi=FIG_DPI, bbox_inches='tight'); plt.show()

        # ── [X5] Intermittency collapse vs Ri_B (Goal 6; if γ computed) ─
        _gc = []
        for n in _nm:
            g = gv('gamma_z', n); jt = gv('bl_top_j', n)
            _gc.append(float(np.nanmean(g[:max(int(jt), 1)] if jt is not None else g))
                       if g is not None else np.nan)
        _gc = np.array(_gc)
        if np.any(np.isfinite(_gc)):
            fig, ax = plt.subplots(figsize=(7, 5), dpi=FIG_DPI)
            ax.plot(_RiB, _gc, 'ko-')
            ax.set_xlabel(r'$Ri_B$'); ax.set_ylabel(r'BL-mean intermittency $\gamma$')
            ax.set_title('Intermittency collapse vs $Ri_B$')
            ax.grid(True, ls='--', lw=0.5)
            fig.savefig(_figdir_x + 'P76_Xcase_gamma_vs_RiB.png', dpi=FIG_DPI, bbox_inches='tight'); plt.show()

        # ── [X6] Coriolis–topography COUPLING observables vs Ψ ──────
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
        fig, axs = plt.subplots(2, 2, figsize=(11, 8), dpi=FIG_DPI)
        for axi, (yv, ttl) in zip(axs.flat, _cpl_panels):
            if np.any(np.isfinite(yv)):
                axi.plot(_Psi[_o], yv[_o], '-', color='0.6', lw=1.0, zorder=1)
            for i, n in enumerate(_nm):
                axi.scatter(_Psi[i], yv[i], color=_col[i], marker=_mk[i],
                            s=80, zorder=3, label=_lab[i])
            axi.set_xlabel(r'$\Psi = L_x/(2\delta)$'); axi.set_title(ttl)
            axi.grid(True, ls='--', lw=0.5)
        axs.flat[3].axhline(100, color='r', ls=':', lw=1, label='collapse ~100')
        axs.flat[0].legend(fontsize=LEGEND_FONTSIZE, loc='best')
        fig.suptitle('Coriolis–topography coupling vs $\\Psi$ (first look; '
                     'covaried path, not a scaling law)')
        fig.tight_layout()
        fig.savefig(_figdir_x + 'P77_Xcase_coupling_vs_Psi.png', dpi=FIG_DPI, bbox_inches='tight'); plt.show()

        # ── [X7] Wall-normal WAVE fluxes vs z+ (Goal 7 / R4) ─────────
        # Cross-case mirror of PhAvg_rotated.py's per-run Research_wave_flux.png:
        # wave momentum flux ũṽ (solid) + wave buoyancy flux (dashed, faded) per
        # case on the shared z+ axis, each case's BL top marked; a case whose
        # sponge reflection guard tripped is flagged in the legend.  Uses only the
        # pickled 1-D profiles (wave_mom_flux / wave_buoy_flux / bl_top_j).
        fig, ax = plt.subplots(figsize=(7, 7), dpi=FIG_DPI)
        _anyw = False
        for c in _xc:
            n   = c['name']
            _wm = gv('wave_mom_flux', n)
            _yi = gy_in(n)
            if _wm is None or _yi is None:
                continue
            _jt   = gv('bl_top_j', n)
            _top  = min(int(_jt) + 30 if _jt is not None else len(_yi),
                        len(_yi), len(_wm))
            _rok  = gv('reflection_ok', n)
            _sfx  = '' if (_rok is None or _rok) else ' [reflection!]'
            ax.plot(_wm[:_top], _yi[:_top], color=c['color'], ls='-',
                    label=c['label'] + _sfx)
            _wb = gv('wave_buoy_flux', n)
            if _wb is not None:
                _tb = min(_top, len(_wb))
                ax.plot(_wb[:_tb], _yi[:_tb], color=c['color'], ls='--', alpha=0.6)
            if _jt is not None and int(_jt) < len(_yi):
                ax.axhline(_yi[int(_jt)], color=c['color'], ls=':', lw=0.6, alpha=0.5)
            _anyw = True
        if _anyw:
            ax.set_xlabel('wall-normal wave flux'); ax.set_ylabel(r'$z^+$')
            ax.set_ylim(0, _row_to_height(limity, use_inner=True))
            ax.set_title(r'Wave fluxes vs $z^+$ — solid: momentum $\tilde u\tilde w$, '
                         r'dashed: buoyancy; dotted: BL top')
            ax.grid(True, ls='--', lw=0.5); ax.legend(fontsize=LEGEND_FONTSIZE)
            fig.savefig(_figdir_x + 'P78_Xcase_wave_flux.png', dpi=FIG_DPI, bbox_inches='tight'); plt.show()
        else:
            plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # CROSS-CASE PROFILE MIRRORS of the per-run PhAvg_rotated.py Research_*.png
        # figures.  The scatter-vs-Ri_B views above (P72–P77) collapse each profile
        # to one BL-mean number; these overlay the FULL z⁺ profiles, one colour per
        # case, so the wall-normal structure a single run shows can be compared side
        # by side.  All read only the per-run research keys already pickled; a case
        # missing a key is skipped and an empty figure is dropped (graceful).
        # (Not mirrored here: Research_intermittency_gamma2D — its γ(x,z) field is
        # the npz-based P79/P80 below; Research_intermittency_omega_field — the
        # instantaneous |ω| snapshot is NOT pickled, so it cannot be rebuilt from
        # pickles and stays a per-run-only diagnostic.)
        # ══════════════════════════════════════════════════════════════════════

        # ── [X8] Turbulent vs dispersive FLUX SPLIT vs z⁺ (Goal 4 / R1) ────────
        # Mirror of Research_flux_split.png: x-averaged wall-normal momentum flux
        # (left) and buoyancy flux (right), split into turbulent (solid) and
        # dispersive (dashed) parts.  Buoyancy ≈ 0 for the neutral case (graceful).
        figR, (axRm, axRb) = plt.subplots(1, 2, figsize=(12, 6), dpi=FIG_DPI)
        _anyf = False
        for c in _xc:
            n = c['name']
            _zi = gy_in(n)
            if _zi is None:
                continue
            _ruv = gv('rey_uv_x', n); _duv = gv('UV_disp_x', n)
            if _ruv is not None and _duv is not None:
                _t = min(limity, len(_zi), len(_ruv), len(_duv))
                axRm.plot(np.asarray(_ruv)[:_t], _zi[:_t], color=c['color'], ls='-',
                          label=c['label'])
                axRm.plot(np.asarray(_duv)[:_t], _zi[:_t], color=c['color'], ls='--')
                _anyf = True
            _bt = gv('Bflux_temp', n); _bd = gv('Bflux_disp', n)
            if _bt is not None and _bd is not None:
                _tb = min(limity, len(_zi), len(_bt), len(_bd))
                axRb.plot(np.asarray(_bt)[:_tb], _zi[:_tb], color=c['color'], ls='-',
                          label=c['label'])
                axRb.plot(np.asarray(_bd)[:_tb], _zi[:_tb], color=c['color'], ls='--')
        if _anyf:
            for _ax, _ttl in ((axRm, 'Momentum flux split'),
                              (axRb, r'Buoyancy flux split (neutral $\approx$ 0)')):
                _ax.set_ylim(0, _row_to_height(limity, use_inner=True))
                _ax.set_xlabel('wall-normal flux'); _ax.set_ylabel(r'$z^+$')
                _ax.set_title(_ttl); _ax.grid(True, ls='--', lw=0.5)
            axRm.legend(fontsize=LEGEND_FONTSIZE, title='solid: turbulent   dashed: dispersive')
            figR.tight_layout()
            figR.savefig(_figdir_x + 'P85_Xcase_flux_split.png', dpi=FIG_DPI, bbox_inches='tight'); plt.show()
        else:
            plt.close(figR)

        # ── [X8b] Buoyancy-flux VECTOR components vs z⁺ (Goal 4 / R1b) ─────────
        # Mirror of Research_flux_components.png: the three components of the
        # x-averaged phase-averaged buoyancy flux ⟨u_i'b'⟩(z) — streamwise
        # ⟨u'b'⟩ (left), wall-normal/vertical ⟨v'b'⟩ (centre, = the Bflux of X8),
        # spanwise ⟨w'b'⟩ (right) — each split into turbulent (solid) and
        # dispersive (dashed), one colour per case.  Rotated (geostrophic-aligned)
        # frame, consistent with the per-run figure.  Needs Uflux_*/Wflux_* which
        # are pickled by PhAvg_rotated.py (IO.var_names); a case predating those
        # keys, or the neutral run (buoyancy ≈ 0), is skipped/flat (graceful).
        figC, _axC = plt.subplots(1, 3, figsize=(15, 6), dpi=FIG_DPI, sharey=True)
        _comp = (('Uflux_temp', 'Uflux_disp', r"streamwise $\langle u'b'\rangle$"),
                 ('Bflux_temp', 'Bflux_disp', r"wall-normal $\langle v'b'\rangle$"),
                 ('Wflux_temp', 'Wflux_disp', r"spanwise $\langle w'b'\rangle$"))
        _anyc = False
        for _ax, (_kt, _kd, _ttl) in zip(_axC, _comp):
            for c in _xc:
                n = c['name']; _zi = gy_in(n)
                if _zi is None:
                    continue
                _tmp = gv(_kt, n); _dsp = gv(_kd, n)
                if _tmp is None or _dsp is None:
                    continue
                _t = min(limity, len(_zi), len(_tmp), len(_dsp))
                _ax.plot(np.asarray(_tmp)[:_t], _zi[:_t], color=c['color'], ls='-',
                         label=c['label'])
                _ax.plot(np.asarray(_dsp)[:_t], _zi[:_t], color=c['color'], ls='--')
                _anyc = True
            _ax.axvline(0.0, color='0.6', lw=0.6)
            _ax.set_xlabel(_ttl); _ax.grid(True, ls='--', lw=0.5)
        if _anyc:
            _axC[0].set_ylim(0, _row_to_height(limity, use_inner=True))
            _axC[0].set_ylabel(r'$z^+$')
            _axC[0].legend(fontsize=LEGEND_FONTSIZE, title='solid: turbulent   dashed: dispersive')
            figC.suptitle('Buoyancy-flux vector components vs $z^+$ — all Re '
                          r'(neutral $\approx$ 0)')
            figC.tight_layout()
            figC.savefig(_figdir_x + 'P90_Xcase_flux_components.png', dpi=FIG_DPI, bbox_inches='tight'); plt.show()
        else:
            plt.close(figC)

        # ── [X9] Dispersive FLUX SHARE profiles vs z⁺ (Goal 4 / R2) ────────────
        # Mirror of Research_dispersive_share.png: |disp| / (|disp| + |turb|) for
        # momentum (solid) and buoyancy (dashed).  P73 is the BL-mean of these vs
        # Ri_B; this shows their z-structure.
        fig, ax = plt.subplots(figsize=(7, 7), dpi=FIG_DPI)
        _anys = False
        for c in _xc:
            n = c['name']; _zi = gy_in(n)
            if _zi is None:
                continue
            _sm = gv('disp_share_mom', n); _sb = gv('disp_share_buoy', n)
            if _sm is not None:
                _t = min(limity, len(_zi), len(_sm))
                ax.plot(np.asarray(_sm)[:_t], _zi[:_t], color=c['color'], ls='-',
                        label=c['label']); _anys = True
            if _sb is not None:
                _t = min(limity, len(_zi), len(_sb))
                ax.plot(np.asarray(_sb)[:_t], _zi[:_t], color=c['color'], ls='--')
        if _anys:
            ax.set_xlim(0, 1); ax.set_ylim(0, _row_to_height(limity, use_inner=True))
            ax.set_xlabel('dispersive share  |disp| / (|disp| + |turb|)')
            ax.set_ylabel(r'$z^+$')
            ax.set_title('Dispersive flux share vs $z^+$\nsolid: momentum   dashed: buoyancy')
            ax.grid(True, ls='--', lw=0.5); ax.legend(fontsize=LEGEND_FONTSIZE)
            fig.savefig(_figdir_x + 'P86_Xcase_dispshare_profile.png', dpi=FIG_DPI, bbox_inches='tight'); plt.show()
        else:
            plt.close(fig)

        # ── [X10] Local similarity φ_m, φ_h vs z⁺ by station (Goal 5 / R3) ─────
        # Mirror of Research_similarity_phi.png: φ_m (left col) and φ_h (right col)
        # at windward / floor / lee stations (rows), one colour per case.  P75 is
        # the RMS departure of these from MOST vs Ri_B; this shows the profiles.
        # The φ arrays start at the station's local surface, so the surface row is
        # js = ny − len(φ) and z⁺ = (y − y[js]) / l_in on the shared reference axis.
        # φ_h is NaN for a neutral run (no buoyancy) → its line simply draws nothing.
        _stn_order = ['windward', 'floor', 'lee']
        figP, axP = plt.subplots(3, 2, figsize=(11, 12), dpi=FIG_DPI, sharey='row')
        _anyp = False
        for _r, _st in enumerate(_stn_order):
            for c in _xc:
                n = c['name']
                _pm = gv('phi_m_st', n); _ph = gv('phi_h_st', n); _yc = gv('y', n)
                if not isinstance(_pm, dict) or _yc is None or _st not in _pm:
                    continue
                _phim = np.asarray(_pm[_st], float)
                _yc = np.asarray(_yc, float)
                _js = len(_yc) - len(_phim)
                if _js < 0 or _js >= len(_yc):
                    continue
                _zp = (_yc[_js:_js + len(_phim)] - _yc[_js]) / norm_l_in(n)
                axP[_r, 0].plot(_phim, _zp, color=c['color'], ls='-', label=c['label'])
                _anyp = True
                if isinstance(_ph, dict) and _st in _ph:
                    _phih = np.asarray(_ph[_st], float)
                    _m = min(len(_phih), len(_zp))
                    axP[_r, 1].plot(_phih[:_m], _zp[:_m], color=c['color'], ls='-')
            axP[_r, 0].axvline(1.0,  color='k', ls=':', lw=0.8)   # MOST neutral φ_m = 1
            axP[_r, 1].axvline(PR_T, color='k', ls=':', lw=0.8)   # MOST neutral φ_h = Pr_t
            axP[_r, 0].set_ylabel(_st + '\n' + r'$z^+$ (from local surface)')
            for _cc in (0, 1):
                axP[_r, _cc].set_ylim(0, 500)
                axP[_r, _cc].grid(True, ls='--', lw=0.5)
        axP[0, 0].set_title(r'$\phi_m$  (MOST neutral $=1$)')
        axP[0, 1].set_title(r'$\phi_h$  (MOST neutral $=Pr_t$)')
        axP[2, 0].set_xlabel(r'$\phi_m$'); axP[2, 1].set_xlabel(r'$\phi_h$')
        axP[0, 0].legend(fontsize=LEGEND_FONTSIZE)
        if _anyp:
            figP.suptitle('Local similarity by station — one colour per case')
            figP.tight_layout()
            figP.savefig(_figdir_x + 'P87_Xcase_similarity_phi.png', dpi=FIG_DPI, bbox_inches='tight'); plt.show()
        else:
            plt.close(figP)

        # ── [X11] Intermittency γ(z⁺) overlay (Goal 6 / R6a; if γ computed) ────
        # Mirror of Research_intermittency_gamma.png using the per-run γ(z) pickled
        # by PhAvg_rotated.py (planesK / flow-plane based).  This is INDEPENDENT of
        # the standalone-Intermittency.py .npz used by P82 below; a case whose pickle
        # carries no γ (compute_intermittency=0) is skipped.
        fig, ax = plt.subplots(figsize=FIG_SIZE_TALL, dpi=FIG_DPI)
        _anyg = False
        for c in _xc:
            n = c['name']; _zi = gy_in(n); _g = gv('gamma_z', n)
            if _zi is None or _g is None:
                continue
            _t = min(limity, len(_zi), len(_g))
            ax.plot(np.asarray(_g)[:_t], _zi[:_t], color=c['color'], label=c['label'])
            _anyg = True
        if _anyg:
            ax.set_xlim(0, 1.02); ax.set_ylim(0, _row_to_height(limity, use_inner=True))
            ax.set_xlabel(r'intermittency $\gamma$'); ax.set_ylabel(r'$z^+$')
            ax.set_title(r'Intermittency $\gamma(z^+)$ (per-run) — all Re')
            ax.grid(True, ls='--', lw=0.5); ax.legend(fontsize=LEGEND_FONTSIZE)
            fig.savefig(_figdir_x + 'P88_Xcase_gamma_profile.png', dpi=FIG_DPI, bbox_inches='tight'); plt.show()
        else:
            plt.close(fig)

        # ── [X12] Local intermittency γ(x,z⁺) field panels (Goal 6 / R6b) ──────
        # Mirror of Research_intermittency_gamma2D.png from the pickled per-run
        # gamma_field (ny×nx, planesK / flow-plane based).  Complements the npz-based
        # P79/P80 below: those need the standalone Intermittency.py cluster run, this
        # works straight from the PhAvg_rotated.py pickle (compute_intermittency=1).
        # A case with no γ field is skipped; if none has one the figure is dropped.
        _grows = [c for c in _xc if gv('gamma_field', c['name']) is not None]
        if _grows:
            _zmax = _contour_zmax(use_inner=True)
            _npan = len(_grows)
            _nr, _ncl = _panel_grid_shape(_npan)
            figG, axG = plt.subplots(_nr, _ncl, figsize=(4.8 * _ncl, 4.6 * _nr),
                                     squeeze=False, dpi=FIG_DPI)
            _gflat = axG.ravel()
            for _i, c in enumerate(_grows):
                n = c['name']; _ax = _gflat[_i]
                _xp = gx_in(n); _zp = gy_in(n)
                if _zp is None:
                    _ax.axis('off'); continue
                _gf = np.asarray(gv('gamma_field', n), float)
                _jr = _clip_rows(_zp, _zmax)
                _pcm = _ax.pcolormesh(_xp, _zp[:_jr], _gf[:_jr, :], cmap='hot_r',
                                      vmin=0.0, vmax=1.0, shading='auto')
                _ax.contour(_xp, _zp[:_jr], np.nan_to_num(_gf[:_jr, :]),
                            levels=[0.5], colors='cyan', linewidths=1.0)
                _shade_ibm(_ax, _xp, _zp[:_jr], geps(n)[:_jr, :])
                _ax.set_ylim(0, _zmax); _ax.set_title(c['label'], fontsize=9)
                _ax.set_xlabel(r'$x^+$')
                if _ax.get_subplotspec().is_first_col():
                    _ax.set_ylabel(r'$z^+$')
                _cb = figG.colorbar(_pcm, ax=_ax, shrink=0.9, pad=0.02)
                _cb.set_label(r'$\gamma$', fontsize=8); _cb.ax.tick_params(labelsize=7)
            for _j in range(_npan, _nr * _ncl):
                _gflat[_j].axis('off')
            figG.suptitle(r'Local intermittency $\gamma(x^+,z^+)$ (per-run pickle) — '
                          r'cyan: $\gamma=0.5$', fontsize=11)
            figG.tight_layout()
            figG.savefig(_figdir_x + 'P89_Xcase_gamma_field.png', dpi=FIG_DPI, bbox_inches='tight'); plt.show()

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
    # ░░  INTERMITTENCY FROM STANDALONE .npz  ░░  (Ansorge & Mellado 2016, G6)
    # The full 3-D γ = ⟨H(|ω'|−ω₀)⟩ CANNOT be recomputed here — it needs the raw
    # flow.*.1/2/3 velocity triplet (cluster-only, MyPyLib/Intermittency.py).
    # Instead load the small planes that run writes into each case dir:
    #   intermittency_xy.npz          — spanwise-MEAN γ(x,z)  [+ gamma_b, ⟨b⟩cond]
    #   intermittency_slice_z0000.npz — one spanwise plane (z=0), their fig-2 analog
    # and build the CROSS-CASE views a single run cannot: side-by-side γ(x,z⁺)
    # panels + an x-averaged γ(z⁺) profile overlay (velocity + buoyancy) + the
    # γ-conditional ⟨b⟩ turbulent-vs-quiescent split.  Fully INDEPENDENT of the
    # research pickle (works even when compute_intermittency was never run); a
    # case with no .npz is silently skipped.  Inner units use the single-reference
    # l_in (shared z+ yardstick), consistent with the rest of results.py.
    # ═══════════════════════════════════════════════════════════════════════
    # Helpers _load_interm_npz/_interm_cases/_interm_field_panels/_interm_profile
    # → FUNCTION DEFINITIONS section (top of file).

    # Velocity intermittency γ — spanwise-mean field + z=0 plane (fig-2 analog)
    _interm_field_panels('gamma', 'xy', 'hot_r',
        r'Intermittency $\gamma(x^+,z^+)$ (spanwise-mean) — all Re',
        'P79_Xcase_intermittency_gamma_field.png')
    _interm_field_panels('gamma', 'slice_z0000', 'hot_r',
        r'Intermittency $\gamma(x^+,z^+)$ ($z{=}0$ plane) — all Re',
        'P80_Xcase_intermittency_gamma_slice.png')
    # Buoyancy intermittency γ_b (own threshold b₀) — only if scalar was present
    _interm_field_panels('gamma_b', 'xy', 'hot_r',
        r'Buoyancy intermittency $\gamma_b(x^+,z^+)$ (spanwise-mean) — all Re',
        'P81_Xcase_intermittency_gammab_field.png')
    # x-averaged profiles overlaid across Fr
    _interm_profile([('gamma', '-', '')], 'xy',
        r'Intermittency $\gamma(z^+)$ (spanwise & x mean) — all Re',
        'P82_Xcase_intermittency_gamma_profile.png', r'intermittency $\gamma$')
    _interm_profile([('gamma_b', '-', '')], 'xy',
        r'Buoyancy intermittency $\gamma_b(z^+)$ — all Re',
        'P83_Xcase_intermittency_gammab_profile.png', r'buoyancy intermittency $\gamma_b$')
    # γ-conditional mean buoyancy: turbulent (solid) vs quiescent (dashed)
    _interm_profile([('mean_b_turb', '-', ' (turb)'),
                     ('mean_b_quiet', '--', ' (quiet)')], 'xy',
        r'$\gamma$-conditional mean buoyancy $\langle b\rangle$ — all Re',
        'P84_Xcase_intermittency_bcond_profile.png', r'conditional $\langle b\rangle$')

    # ═══════════════════════════════════════════════════════════════════════
    # ░░  END-OF-RUN SUMMARY  ░░  (detailed; teed to sim_stats.log)
    # Per-case inputs found/skipped, scales & stability, the Chapter-6
    # observations gathered in _ch6, and the honestly-gated/blocked items.
    # ═══════════════════════════════════════════════════════════════════════
    # Helpers _fmt/_print_run_summary → FUNCTION DEFINITIONS section (top of file).

    _print_run_summary()
