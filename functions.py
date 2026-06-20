#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 16:42:14 2026

@author: shreyad95
"""
import os
import struct
import pickle
import numpy as np
import netCDF4 as nc
from scipy.integrate import simpson
from scipy.integrate import trapezoid
from scipy.stats import linregress
from scipy.interpolate import CubicSpline
from scipy.interpolate import griddata
from scipy.interpolate import make_interp_spline
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import uniform_filter1d


###############################################################################
############################## Function defintion #############################

def load_or_compute(names, recompute, compute_fn, save=True, verbose=True,
                    label=None):
    """
    Load a group of cached .npy arrays, or compute them fresh.

    Parameters
    ----------
    names      : list[str]   — base names; files are '<name>.npy'.
    recompute  : bool        — if True, always recompute (ignore caches).
    compute_fn : callable    — returns a tuple/list of arrays in the same
                               order as `names` (used only when computing).
    save       : bool        — save freshly computed arrays to '<name>.npy'.
    verbose    : bool        — print a load/compute message.
    label      : str | None  — human label for the message (defaults to names).

    Returns
    -------
    tuple of arrays, in the order of `names`, so callers can keep flat
    assignments, e.g.  a, b, c = load_or_compute([...], flag, lambda: (...)).
    """
    tag = label or ', '.join(names)
    if (not recompute) and all(os.path.exists(n + '.npy') for n in names):
        if verbose:
            print(f'Loading cached {tag} ...')
        return tuple(np.load(n + '.npy') for n in names)

    if verbose:
        print(f'Computing {tag} ...')
    vals = tuple(compute_fn())
    if len(vals) != len(names):
        raise ValueError(
            f'load_or_compute: compute_fn returned {len(vals)} arrays '
            f'but {len(names)} names were given.')
    if save:
        for n, v in zip(names, vals):
            np.save(n + '.npy', v)
        if verbose:
            print(f'{tag} saved.')
    return vals


def load_smooth_case(nc_path, x_grid, nu, Re_lambda):
    """
    Load and post-process the smooth-wall reference case (flat wall, neutral,
    Re=500) from its NetCDF file, returning every derived quantity in a dict.

    This is the SINGLE source of truth for the smooth reference, shared by
    PhAvg.py and results.py so the two can never diverge.  The formulas are
    PhAvg.py's (the validated pipeline), including the Coriolis convention
    cor_yx_s = -(W_s - G_z) with the scalar geostrophic value G_z = max(W_s).

    Uses this module's own `diffu_dy` (7-point Fornberg, order 1) and
    `vIntegral`, so the result is independent of any local helpers a caller may
    define.

    Parameters
    ----------
    nc_path   : str    — full path to the smooth-case NetCDF.
    x_grid    : array  — streamwise grid onto which AVG_TKE_V_s is interpolated
                         (AVG_TKE_V_s_i = interp(x_grid, x_s, AVG_TKE_V_s)).
    nu        : float  — kinematic viscosity (inner scaling of y).
    Re_lambda : float  — Re_λ (viscous-stress scaling 1/Re_lambda).

    Returns
    -------
    dict with keys: sy, nys, U_s, V_s, W_s, su, sw, alpha_s, ustr_s1,
        alpha_str_s, y_s, y_s_p, rU_s, rV_s, rW_s, G_x_s, G_z_s, G_s, U_s_p,
        W_s_p, GblU_s, GblW_s, Rxx_s, Rxy_s, Ryy_s, Ryz_s, Rzz_s, TKE_s,
        case_v_s, cor_yx_s, I_corr_yx_s, du_dy_s, visc_yx_s, tau_yx_s,
        cor_yz_s, I_corr_yz_s, dw_dy_s, visc_yz_s, tau_yz_s, AVG_TKE_V_s,
        x_s, AVG_TKE_V_s_i.
    """
    s1 = nc.Dataset(nc_path, 'r')
    sy = (s1.variables['y'][:])
    nys = np.size(sy)
    U_s = (s1.variables['fU'][:]).T
    V_s = (s1.variables['fV'][:]).T
    W_s = (s1.variables['fW'][:]).T
    su = np.mean(U_s, axis=1)
    sw = np.mean(W_s, axis=1)
    alpha_s = (sw/su)   # turning angle profile for smooth case
    ustr_s1 = np.mean(s1.variables['FrictionVelocity'][:])
    alpha_str_s = np.mean(s1.variables['FrictionAngle'][:])
    y_s = s1.variables['y'][:]
    y_s_p = (y_s*ustr_s1)/nu
    rU_s = (s1.variables['rU'][:]).T
    rV_s = (s1.variables['rV'][:]).T
    rW_s = (s1.variables['rW'][:]).T
    G_x_s = np.max(U_s)
    G_z_s = np.max(W_s)
    G_s = np.sqrt(G_x_s**2 + G_z_s**2)
    U_s_p = (U_s/ustr_s1)/G_s
    W_s_p = (W_s/ustr_s1)/G_s
    GblU_s = np.mean(rU_s, axis=1)
    GblW_s = np.mean(rW_s, axis=1)
    Rxx_s = (s1.variables['Rxx'][:]).T
    Rxy_s = (s1.variables['Rxy'][:]).T
    Ryy_s = (s1.variables['Ryy'][:]).T
    Ryz_s = (s1.variables['Ryz'][:]).T
    Rzz_s = (s1.variables['Rzz'][:]).T
    TKE_s = 0.5 * (Rxx_s + Ryy_s + Rzz_s)   # TKE = 0.5*(⟨u'u'⟩+⟨v'v'⟩+⟨w'w'⟩)
    case_v_s = np.zeros((nys,1)).astype(int)
    case_v_s[:,:] = 4; case_v_s[0,0] = 1; case_v_s[1,0] = 2; case_v_s[2,0] = 3
    case_v_s[-3,0] = 5; case_v_s[-2,0] = 6; case_v_s[-1,0] = 7
    cor_yx_s = -(W_s - G_z_s)
    I_corr_yx_s = vIntegral(np.mean(cor_yx_s, axis=1), y_s.size, y_s)
    du_dy_s = diffu_dy((np.reshape(GblU_s,(y_s.size,1))), y_s.size, 1, case_v_s, y_s, 1)
    visc_yx_s = (1/Re_lambda) * du_dy_s
    tau_yx_s = I_corr_yx_s + np.mean(visc_yx_s, axis=1) - np.mean(Rxy_s, axis=1)
    cor_yz_s = (U_s - G_x_s)
    I_corr_yz_s = vIntegral(np.mean(cor_yz_s, axis=1), y_s.size, y_s)
    dw_dy_s = diffu_dy((np.reshape(GblW_s,(y_s.size,1))), y_s.size, 1, case_v_s, y_s, 1)
    visc_yz_s = (1/Re_lambda) * dw_dy_s
    tau_yz_s = -I_corr_yz_s + np.mean(visc_yz_s, axis=1) - np.mean(Ryz_s, axis=1)
    AVG_TKE_V_s = np.mean(TKE_s, axis=0)
    x_s = np.linspace(0, 1, 250)
    AVG_TKE_V_s_i = np.interp(x_grid, x_s, AVG_TKE_V_s)
    s1.close()

    return {
        'sy': sy, 'nys': nys, 'U_s': U_s, 'V_s': V_s, 'W_s': W_s,
        'su': su, 'sw': sw, 'alpha_s': alpha_s, 'ustr_s1': ustr_s1,
        'alpha_str_s': alpha_str_s, 'y_s': y_s, 'y_s_p': y_s_p,
        'rU_s': rU_s, 'rV_s': rV_s, 'rW_s': rW_s,
        'G_x_s': G_x_s, 'G_z_s': G_z_s, 'G_s': G_s,
        'U_s_p': U_s_p, 'W_s_p': W_s_p, 'GblU_s': GblU_s, 'GblW_s': GblW_s,
        'Rxx_s': Rxx_s, 'Rxy_s': Rxy_s, 'Ryy_s': Ryy_s, 'Ryz_s': Ryz_s,
        'Rzz_s': Rzz_s, 'TKE_s': TKE_s, 'case_v_s': case_v_s,
        'cor_yx_s': cor_yx_s, 'I_corr_yx_s': I_corr_yx_s, 'du_dy_s': du_dy_s,
        'visc_yx_s': visc_yx_s, 'tau_yx_s': tau_yx_s,
        'cor_yz_s': cor_yz_s, 'I_corr_yz_s': I_corr_yz_s, 'dw_dy_s': dw_dy_s,
        'visc_yz_s': visc_yz_s, 'tau_yz_s': tau_yz_s,
        'AVG_TKE_V_s': AVG_TKE_V_s, 'x_s': x_s, 'AVG_TKE_V_s_i': AVG_TKE_V_s_i,
    }


bias_patterns = {
    "1": [0,1,2,3,4,5,6],
    "2": [-1,0,1,2,3,4,5],
    "3": [-2,-1,0,1,2,3,4],
    "4":  [-3,-2,-1,0,1,2,3],
    "5": [-4,-3,-2,-1,0,1,2],
    "6": [-5,-4,-3,-2,-1,0,1],
    "7": [-6,-5,-4,-3,-2,-1,0]
}

def fornberg_weights(x_stencil, x0, m):
    n = len(x_stencil)
    c = np.zeros((n, m+1))
    c1 = 1.0
    c4 = x_stencil[0] - x0
    c[0, 0] = 1.0

    for i in range(1, n):
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = x_stencil[i] - x0

        for j in range(i):
            c3 = x_stencil[i] - x_stencil[j]
            c2 *= c3

            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k*c[i-1, k-1] - c5*c[i-1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i-1, 0] / c2

            for k in range(mn, 0, -1):
                c[j, k] = (c4*c[j, k] - k*c[j, k-1]) / c3
            c[j, 0] = c4*c[j, 0] / c3

        c1 = c2

    return c[:, m]

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

def epsVolume(eps,ny,nx, hill_hgt):
    eps_vol = np.zeros((ny,nx))
    
    for j in range (hill_hgt+1):
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
                if i != 0 and i != nx-1:
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

def diff_cases(eps,nx,ny):
    case_h = np.zeros((ny,nx)).astype(int)
    case_v = np.zeros((ny,nx)).astype(int)
    # For horizontal cases
    # Case1 : Forward derivatives. current point is solid The fluid is on the next 6 points
    # Case2 : Forward2 there is 1 fluid point behind and 5 fluid points ahead
    # Case3 : Forward3 there are 2 fluid points behind and and 4 fluid points ahead
    # Case4 : Central there are 3 fluid points behind and 3 fluid points ahead
    # Case5 : Backward there are 4 fluid points behind and 2 ahead and after which there is solid
    # Case6 : Backward there are 5 fluid points behind and 1 ahead and after which there is solid
    # Case7 : Backward there are 6 fluid points behind and 0 ahead and after which there is solid
    for j in range(ny):
        for i in range(nx):
            #### Solid #####
            if eps[j,i] == 1:
                if  i == 0:
                    case_h[j,i] = 4
                elif i == 1:
                    if eps[j,i+1] == 0:
                        case_h[j,i] = 4
                    else:
                        case_h[j,i] = 4
                elif i == 2:
                    if eps[j,i+1] == 0:
                        case_h[j,i] = 4
                    else:
                        case_h[j,i] = 4
                elif ((i > 2) and (i < nx-3)):
                    lhs = sum(eps[j,i-3:i])
                    rhs = sum(eps[j,i+1:i+4])
                    if (lhs == rhs):
                        case_h[j,i] = 4
                    elif (lhs == 3) and (rhs == 0):
                        case_h[j,i] = 1
                    elif (lhs == 3) and (rhs == 1):
                        case_h[j,i] = 6
                    elif (lhs == 3) and (rhs == 2):
                        case_h[j,i] = 5 
                    elif (lhs == 0) and (rhs == 3):
                        case_h[j,i] = 7
                    elif (lhs == 1) and (rhs == 3):
                        case_h[j,i] = 2
                    elif (lhs == 2) and (rhs == 3):
                        case_h[j,i] = 3
                    
                elif (i == nx-3):
                    if eps[j,i+1] == 0:
                        case_h[j,i] = 4
                    else:
                        case_h[j,i] = 4
                elif (i == nx-2):
                    case_h[j,i] = 4
                elif (i == nx-1):
                    case_h[j,i] = 4
                else:
                    case_h[j,i] = 4
            #### FLuid ####
            elif eps[j,i] == 0:
                if  i == 0:
                    case_h[j,i] = 4
                elif i == 1:
                    case_h[j,i] = 4
                
                elif i == 2:
                    if eps[j,i-1] == 1:
                        case_h[j,i] = 4
                    else:
                        case_h[j,i] = 4
                    
                elif ((i > 2) and (i < nx-3)):
                    lhs = sum(eps[j,i-3:i])
                    rhs = sum(eps[j,i+1:i+4])
                    if (lhs == rhs):
                        case_h[j,i] = 4
                    elif (lhs == 3 and lhs != rhs):
                        case_h[j,i] = 2
                    elif (lhs == 2 and lhs != rhs):
                        case_h[j,i] = 3
                    elif (lhs == 1 and lhs != rhs):
                        case_h[j,i] = 4
                    elif (rhs == 3 and lhs != rhs):
                        case_h[j,i] = 6
                    elif (rhs == 2 and lhs != rhs):
                        case_h[j,i] = 5
                    elif ((lhs == 1) or (rhs == 1)):
                        case_h[j,i] = 4
                elif (i == nx-3):
                    if (eps[j,i+1] == 1):
                        case_h[j,i] = 4
                    else:
                        case_h[j,i] = 4
                elif (i == nx-2):
                    case_h[j,i] = 4
                elif (i == nx-1):
                    case_h[j,i] = 4
                else:
                    case_h[j,i] = 4
            # =============================================================== #    
            #### Verical case
            # =============================================================== #   
            if (j == 0):
                case_v[j,i] = 1
            elif (j == 1):
                if (eps[j,i] == 1 and eps[j+1,i] == 0):
                    case_v[j,i] = 1
                else:
                    case_v[j,i] = 2
            elif (j == 2):
                if (eps[j,i] == 1 and eps[j+1,i] == 0):
                    case_v[j,i] = 1
                elif ((eps[j,i] == 1) and (eps[j-1,i] == 1) and (eps[j-2,i] == 1)):
                    case_v[j,i] = 3
                elif ((eps[j,i] == 0) and (eps[j-1,i] == 0) and (eps[j-2,i] == 0)):
                    case_v[j,i] = 3
                else:
                    case_v[j,i] = 2
            elif (j == 3):
                bottom = sum(eps[j-3:j, i])
                top = sum(eps[j+1:j+4, i])
                if (eps[j,i] == 1 and eps[j+1,i] == 0):
                    case_v[j,i] = 1
                elif (bottom == top):
                    case_v[j,i] = 4
                elif (bottom == 2):
                    case_v[j,i] = 3
                else:
                    case_v[j,i] = 4
            elif (j < ny-3):
                bottom = sum(eps[j-3:j, i])
                top = sum(eps[j+1:j+4, i])
                # Inside Solids 
                if (eps[j,i] == 1): 
                    if ((top == 0) and (bottom > 0)):
                        case_v[j,i] = 1
                        
                    elif ((top == 1) and (bottom > 1)):
                        if (j > 4):
                            case_v[j,i] = 6
                        else:
                            case_v[j,i] = 5
                    
                    elif ((top == 2) and (bottom > 1)):
                        if (j > 3):
                            case_v[j,i] = 5
                        else:
                            case_v[j,i] = 4
                    # Equal number of points on both sides
                    elif (top == bottom ):
                        case_v[j,i] = 4
                    
                    elif ((top > 1) and (bottom == 0)):
                        case_v[j,i] = 4
                    
                    elif ((top > 1) and (bottom == 1)):
                        if (j+5 < nx):
                            case_v[j,i] = 2
                        else:
                            case_v[j,i] = 4
                    
                    elif ((top > 1) and (bottom == 2)):
                        if (j+4 < nx):
                            case_v[j,i] = 3
                        else:
                            case_v[j,i] = 4
                # Inside fluid            
                elif (eps[j,i] == 0):
                    if ((bottom == 1) or (bottom == 0)):
                        case_v[j,i] = 4
                        
                    elif (bottom == 2):
                        case_v[j,i] = 3
                    
                    elif (bottom == 3):
                        case_v[j,i] = 2
                    
                    elif (bottom == top):
                        case_v[j,i] = 4
                    
                    else:
                        case_v[j,i] = 0
            # Top boundary
            elif (j == ny-3):
                if (eps[j,i] == 1 and eps[j-1,i] == 0):
                    case_v[j,i] = 7
                else:
                    case_v[j,i] = 5
            elif (j == ny-2):
                if (eps[j,i] == 1 and eps[j-1,i] == 0):
                    case_v[j,i] = 7
                else:
                    case_v[j,i] = 6
            elif (j == ny-1):
                case_v[j,i] = 7
            else:
                case_v[j,i] = 0          
    # for i in range (nx):
    #     for j in range (ny):
    #         if j == 0:
    #             case_v[j,i] = 1
    #         if j == 1:
    #             case_v[j,i] = 2
    #         if j == 2:
    #             case_v[j,i] = 3
    #         if j >= 3 and j <= ny-4:
    #             case_v[j,i] = 4
    #         if j == ny-3:
    #             case_v[j,i] = 5
    #         if j == ny-2:
    #             case_v[j,i] = 6
    #         if j == ny-1:
    #             case_v[j,i] = 7
    # case_h[:,:] = 4
    return case_v, case_h

def interpolate_component(x, y, Nx, Ny, eps, field,
                          ghost_depth=5, n_anchor=4, smooth_width=5):
    """
    Mirror-ghost strategy:
      s         = first solid cell  (BC, val=0, never overwritten)
      e         = last  solid cell  (BC, val=0, never overwritten)
      s+1       = -field[s-1]   (negative mirror of adjacent fluid)
      e-1       = -field[e+1]   (negative mirror of adjacent fluid)
      s+2..e-2  = PCHIP interior, anchored on [...fluid, BC, mirror, mirror, BC, fluid...]

    PCHIP now sees a smooth sign-changing sequence around each zero, so
    it does not collapse to a flat-zero segment.
    Mirrors are only set when the solid is wide enough (e-s >= 3).
    Interior is only filled when e-s >= 4.
    """
    dy_bottom = y[1] - y[0]
    # eps[0, :] = 0  # floor row always fluid

    # ------------------------------------------------------------------
    # 1. Ghost cells below the domain
    # ------------------------------------------------------------------
    ghost_fld = np.zeros((ghost_depth, field.shape[1]))
    ghost_eps = np.zeros((ghost_depth, eps.shape[1]))
    full_fld  = np.concatenate((ghost_fld, field),   axis=0)
    full_eps  = np.concatenate((ghost_eps, eps), axis=0)

    y_ghost = np.linspace(y[0] - ghost_depth * dy_bottom, y[0] - dy_bottom, ghost_depth)
    y_full  = np.concatenate((y_ghost, y))

    full_fld_x          = full_fld.copy()
    full_fld_y          = full_fld.copy()
    interior_solid_mask = np.zeros_like(full_eps, dtype=bool)

    # ------------------------------------------------------------------
    # 2. Horizontal interpolation (X-direction)
    # ------------------------------------------------------------------
    for j in range(full_fld.shape[0]):
        row_eps = full_eps[j, :].copy()
        row_fld = full_fld[j, :].copy()
        f_min, f_max = -row_fld.max(), row_fld.max()

        # s = first solid cell (+1 shift), e = last solid cell
        starts = np.where((row_eps[:-1] == 0) & (row_eps[1:] == 1))[0] 
        ends   = np.where((row_eps[:-1] == 1) & (row_eps[1:] == 0))[0]

        if starts.size == 0:
            continue

        needs_roll = (ends.size > 0) and (starts[0] > ends[0])
        offset = 0
        if needs_roll:
            offset  = int(Nx / 2)
            row_eps = np.roll(row_eps, -offset)
            row_fld = np.roll(row_fld, -offset)
            starts  = np.where((row_eps[:-1] == 0) & (row_eps[1:] == 1))[0] + 1
            ends    = np.where((row_eps[:-1] == 1) & (row_eps[1:] == 0))[0]

        for s, e in zip(starts, ends):
            # Bounds guard: need fluid neighbours outside solid for mirrors
            if s < 1 or e + 1 >= row_fld.shape[0]:
                continue

            width = e - s  # 0 → 1 cell, 1 → 2 cells, etc.

            # --- Step 1: place mirror ghost values ---
            # Only when solid is wide enough that s+1 and e-1 are distinct
            # and do not coincide with the BC cells themselves.
            if width >= 3:                              # at least 4 solid cells
                row_fld[s + 1] = -row_fld[s - 1]      # left mirror
                row_fld[e - 1] = -row_fld[e + 1]      # right mirror

            # --- Step 2: interpolate interior s+2 .. e-2 ---
            if width < 4:                              # nothing between the mirrors
                continue

            x_fill = x[s + 2 : e - 1]                # interior indices
            if x_fill.size == 0:
                continue

            left_idx  = np.where(row_eps[:s]   == 0)[0][-n_anchor:]
            right_idx = np.where(row_eps[e+1:] == 0)[0][:n_anchor] + (e + 1)

            if left_idx.size < 1 or right_idx.size < 1:
                continue

            # Anchor order: fluid | BC=0 | mirror | ... gap ... | mirror | BC=0 | fluid
            x_anc = np.concatenate([
                x[left_idx],
                x[[s, s + 1]],          # BC then mirror
                x[[e - 1, e]],          # mirror then BC
                x[right_idx],
            ])
            f_anc = np.concatenate([
                row_fld[left_idx],
                row_fld[[s, s + 1]],    # 0, -f[s-1]
                row_fld[[e - 1, e]],    # -f[e+1], 0
                row_fld[right_idx],
            ])

            try:
                row_fld[s + 2 : e - 1] = np.clip(
                    PchipInterpolator(x_anc, f_anc)(x_fill), f_min, f_max
                )
            except Exception:
                continue

        if needs_roll:
            row_fld = np.roll(row_fld, offset)
        full_fld_x[j, :] = row_fld

    # ------------------------------------------------------------------
    # 3. Vertical interpolation (Y-direction)
    # ------------------------------------------------------------------
    for i in range(full_fld.shape[1]):
        col_eps = full_eps[:, i]
        col_fld = full_fld[:, i].copy()
        f_min, f_max = -col_fld.max(), col_fld.max()

        # s_idx: last fluid cell before solid  (col_eps[s_idx]=0, [s_idx+1]=1)
        # e_idx: last solid cell               (col_eps[e_idx]=1, [e_idx+1]=0)
        starts = np.where((col_eps[:-1] == 0) & (col_eps[1:] == 1))[0]
        ends   = np.where((col_eps[:-1] == 1) & (col_eps[1:] == 0))[0]

        for s_idx in starts:
            valid_ends = ends[ends > s_idx]
            if valid_ends.size == 0:
                continue
            e_idx = valid_ends[0]

            s = s_idx + 1   # first solid cell  (BC, val=0)
            e = e_idx       # last  solid cell  (BC, val=0)

            if s_idx < 0 or e_idx + 1 >= col_fld.shape[0]:
                continue

            width = e - s

            # --- Step 1: mirror ghost values ---
            if width >= 3:
                col_fld[s + 1] = -col_fld[s_idx]       # mirror fluid below
                col_fld[e - 1] = -col_fld[e_idx + 1]   # mirror fluid above

            # --- Step 2: interpolate interior s+2 .. e-2 ---
            if width == 2:
                if (e == s + 1):
                    col_fld[e - 1] = -col_fld[e + 1]   # mirror fluid above
                continue
            
            if width == 1:
                continue

            gap    = slice(s + 2, e - 1)
            y_fill = y_full[s + 2 : e - 1]
            if y_fill.size == 0:
                continue

            interior_solid_mask[gap, i] = True

            bot_idx = np.where(col_eps[:s]   == 0)[0][-n_anchor:]
            top_idx = np.where(col_eps[e+1:] == 0)[0][:n_anchor] + (e + 1)

            if bot_idx.size < 1 or top_idx.size < 1:
                continue

            y_anc = np.concatenate([
                y_full[bot_idx],
                y_full[[s, s + 1]],     # BC then mirror
                y_full[[e - 1, e]],     # mirror then BC
                y_full[top_idx],
            ])
            f_anc = np.concatenate([
                col_fld[bot_idx],
                col_fld[[s, s + 1]],    # 0, -f[s_idx]
                col_fld[[e - 1, e]],    # -f[e_idx+1], 0
                col_fld[top_idx],
            ])

            try:
                col_fld[gap] = np.clip(
                    PchipInterpolator(y_anc, f_anc)(y_fill), f_min, f_max
                )
            except Exception:
                continue

        full_fld_y[:, i] = col_fld

    # ------------------------------------------------------------------
    # 4. Smooth only interior solid cells; restore fluid + BC + mirror cells
    # ------------------------------------------------------------------
    for j in range(full_fld_y.shape[0]):
        if interior_solid_mask[j, :].sum() < 3:
            continue
        backup = full_fld_y[j, ~interior_solid_mask[j, :]].copy()
        full_fld_y[j, :] = uniform_filter1d(
            full_fld_y[j, :], size=smooth_width, mode='wrap'
        )
        full_fld_y[j, ~interior_solid_mask[j, :]] = backup
    
    # ------------------------------------------------------------------
    # 5. Crop ghost cells
    # ------------------------------------------------------------------
    eps_sum = np.sum(eps, axis=0).astype(int);
    eps_sum[eps_sum > 0] -= 1
    fld_x, fld_y = full_fld_x[ghost_depth:, :], full_fld_y[ghost_depth:, :]
    fld_x[eps_sum, np.arange(Nx)] = 0; fld_y[eps_sum, np.arange(Nx)] = 0
    return fld_x, fld_y
                
def get_weights(x, i, derivative_order, bias_type):
    offsets = bias_patterns[bias_type]
    idx = [i + o for o in offsets]
    x_stencil = x[idx]
    return fornberg_weights(x_stencil, x[i], derivative_order), idx

def diffu_dy(field, ny, nx, case_v, y, order):    # ny is number of points in vertical
    du = np.zeros((ny,nx))
    for i in range (nx):
        for j in range (ny):
            if (case_v[j,i] != 0):
                w, jdx = get_weights(y, j, order, str(case_v[j,i]))
                du[j,i] = np.dot(w, field[jdx, i])
            else:
                w = np.zeros((7))
                du[j,i] = 0
    return du

def diffu_dx(field, ny, nx, case_h, x, order):    # ny is number of points in vertical
    du = np.zeros((ny,nx))
    for j in range (ny):
        for i in range (nx):
            if (case_h[j,i] != 0):
                w, idx = get_weights(x, i, order, str(case_h[j,i]))
                idx = np.linspace(i-3,i+3,7).astype(int)
                idx = (idx%nx).astype(int)
                du[j,i] = np.dot(w, field[j, idx])
            else:
                du[j,i] = 0
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
        I[j] = I[j-1] + 0.5*(varaible[j] + varaible[j-1])*(y[j]-y[j-1])
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

def calculate_roughness_function(y_smooth, u_smooth, y_rough, u_rough, utau_smooth, utau_rough, nu, kappa=0.41):
    """
    Calculates the roughness function Delta U+ by comparing smooth and rough log-laws.
    """
    # 1. Convert to wall units (+)
    y_plus_s = y_smooth * utau_smooth / nu
    u_plus_s = u_smooth / utau_smooth
    
    y_plus_r = y_rough * utau_rough / nu
    u_plus_r = u_rough / utau_rough
    
    # 2. Define the Log-Law Region (Standard range is 30 < y+ < 0.2*delta)
    # We'll filter for y+ between 50 and 500 for a stable fit
    mask_s = (y_plus_s > 50) & (y_plus_s < 500)
    mask_r = (y_plus_r > 50) & (y_plus_r < 500)
    
    # 3. Fit: U+ = (1/kappa) * ln(y+) + B
    # Smooth Wall Fit
    slope_s, intercept_s, _, _, _ = linregress(np.log(y_plus_s[mask_s]), u_plus_s[mask_s])
    B_smooth = intercept_s
    
    # Rough Wall Fit
    slope_r, intercept_r, _, _, _ = linregress(np.log(y_plus_r[mask_r]), u_plus_r[mask_r])
    B_rough = intercept_r
    
    # 4. Calculate Delta U+ (The downward shift)
    delta_u_plus = B_smooth - B_rough
    
    return delta_u_plus, B_smooth, B_rough

def avg_c(eps, field, axis):
    n = np.shape(field)[axis]
    eps_y = np.sum((1-eps), axis=axis)
    eps_y = np.where(eps_y < n, eps_y+1, eps_y)
    fld = np.sum(field, axis=axis)
    mean_c = fld/eps_y
    return mean_c

def valley_profile(field_2d, loc, y_in, flk_hgt, lf_ind, rf_ind):
    '''Extract vertical profile at a valley location.
    loc: 'top', 'lf', 'bottom', 'rf'
    Returns (profile, y_coord) pair.
    '''
    if loc == 'top':
        return np.mean(field_2d[94:, 0:5], axis=1), y_in[94:]
    elif loc == 'lf':
        return np.mean(field_2d[flk_hgt:, lf_ind], axis=1), y_in[flk_hgt:]
    elif loc == 'bottom':
        return np.mean(field_2d[:, 507:517], axis=1), y_in[:]
    elif loc == 'rf':
        return np.mean(field_2d[flk_hgt:, rf_ind], axis=1), y_in[flk_hgt:]


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

def save_sim(sim_id, data_dict):
    filename = f"results_{sim_id}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"Saved: {filename}")


def read_plane(filepath, imax, jmax, n_kplanes, nvars, var_idx, kplane_idx):
    """Return a 2-D (imax × jmax) array for one variable from a raw float32 planesK file."""
    expected_bytes = imax * jmax * n_kplanes * nvars * 4
    actual_bytes = os.path.getsize(filepath)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{filepath}: expected {expected_bytes} B "
            f"({imax}×{jmax}×{n_kplanes}×{nvars}×4), got {actual_bytes}. "
            f"Check N_KPLANES / NVARS."
        )
    data = np.fromfile(filepath, dtype=np.float32)
    block_size = imax * jmax * n_kplanes
    var_block  = data[var_idx * block_size : (var_idx + 1) * block_size]
    plane_3d   = var_block.reshape((imax, jmax, n_kplanes), order='F')
    return plane_3d[:, :, kplane_idx]   # shape (imax, jmax)


def read_all_planes(filepath, imax, jmax, n_kplanes, nvars, kplane_idx):
    """Read all nvars variables from one planesK file; return list of (ny, nx) arrays."""
    expected_bytes = imax * jmax * n_kplanes * nvars * 4
    actual_bytes   = os.path.getsize(filepath)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{filepath}: expected {expected_bytes} B "
            f"({imax}×{jmax}×{n_kplanes}×{nvars}×4), got {actual_bytes}."
        )
    data       = np.fromfile(filepath, dtype=np.float32)
    block_size = imax * jmax * n_kplanes
    planes = []
    for vi in range(nvars):
        var_block = data[vi * block_size : (vi + 1) * block_size]
        plane_3d  = var_block.reshape((imax, jmax, n_kplanes), order='F')
        planes.append(plane_3d[:, :, kplane_idx].T)   # (nx,ny) → (ny,nx)
    return planes   # list of nvars arrays each (ny, nx)


# def update_frame(frame):
#     # path =
#     pl_id = frame_ids[frame]  # Get the specific plane ID from the list
#     field = read_plane(path, Nx, Ny, pl_id)  # Read the field for this plane
#     im.set_data(field)  # Update the data for the image
#     return [im]


def power_law_model(z, a, n):
    """Near-wall power law: u⁺ = a · z⁺^n."""
    return a * z ** n


def fmt_val(v, fmt='.5f', na_str='n/a'):
    """Format a scalar for the parameter table; return na_str if None or NaN."""
    if v is None:
        return na_str
    if isinstance(v, float) and np.isnan(v):
        return na_str
    return format(v, fmt)


def print_summary_table(title, rows, width=74, label_w=46):
    """
    Print a consolidated, aligned key-value table to the console.

    Parameters
    ----------
    title : str
    rows  : list — each item is either
              ('section', text)            -> a sub-header / divider line, or
              (label, value)               -> formatted with default '.5f', or
              (label, value, fmt)          -> formatted with `fmt`, or
              (label, value, fmt, unit)    -> value then a trailing unit string.
            `value` may be None/NaN (shown as 'n/a').

    Uses fmt_val so NaN/None are handled gracefully.  Pure output — no return.
    """
    bar = '=' * width
    print('\n' + bar)
    print('  ' + title)
    print(bar)
    for row in rows:
        if row[0] == 'section':
            print('-' * width)
            if len(row) > 1 and row[1]:
                print('  ' + str(row[1]))
                print('-' * width)
            continue
        label = row[0]
        value = row[1]
        fmt   = row[2] if len(row) > 2 else '.5f'
        unit  = row[3] if len(row) > 3 else ''
        cell  = fmt_val(value, fmt) + (f' {unit}' if unit else '')
        print(f'  {label:<{label_w}s}{cell:>{width - label_w - 2}s}')
    print(bar)


###############################################################################
# Plot annotation helpers (valley-crest height + boundary-layer sublayers)
#
# Boundary-layer sublayer marker convention (inner units z+ = y·u*/ν).
# Discrete markers — same symbol = same level in EVERY plot:
#   'o'  viscous sublayer top          z+ ≈ 5      (both cases)
#   's'  canopy top / roughness start  (orographic only) = peak of dispersive
#                                       stress uv_t (x-averaged), where it stops
#                                       rising and starts to dissipate
#   '^'  log-layer start               smooth z+ ≈ 30 ;  orographic z+ ≈ 75
#   'D'  log-layer top                 smooth z+ ≈ 100;  orographic z+ ≈ 200
#   'X'  valley crest h (index 94)     used only where z+ is NOT an axis
#                                       (e.g. hodograph); elsewhere h is a line
# Fill encodes the wall type:
#   filled marker  = orographic (valley) curve
#   hollow marker  = smooth-wall curve
# Layer structure:
#   Smooth     : viscous(≤5) | buffer(5–30)         | log(30–100)
#   Orographic : viscous(≤5) | canopy(5–canopy_top) | roughness(canopy_top–75)
#                | log(75–200)
###############################################################################

# symbol -> human-readable layer-boundary name (for legends / reference)
LAYER_MARKER_NAMES = {
    'o': r'viscous top ($z^+\!\approx\!5$)',
    's': r'canopy/roughness (oro)',
    '^': r'log start ($z^+\!\approx\!30/75$)',
    'D': r'log top ($z^+\!\approx\!100/200$)',
    'X': r'valley crest $h$',
}


def mark_h(pos, orient='v', label=True, lblpos=0.04, ha='right', ax=None,
           color='black', linestyle='--', linewidth=0.8):
    """Mark a reference height with a dashed line labelled 'h'.

    pos    : coordinate value of the line — e.g. y_in[hill_hgt], the valley
             crest height (z+ = y_in[94]) in inner units.
    orient : 'v' draws a vertical line (use when z+ is on the x-axis);
             'h' draws a horizontal line (use when z+ is on the y-axis).
    The 'h' label is placed in axes-fraction coordinates along the line so it
    renders correctly on linear or log axes and after axis re-scaling.
    """
    import matplotlib.pyplot as plt
    a = ax if ax is not None else plt.gca()
    if orient == 'v':
        a.axvline(x=pos, color=color, linestyle=linestyle, linewidth=linewidth)
        if label:
            a.text(pos, lblpos, r'$h$', rotation=90, va='bottom', ha=ha,
                   fontsize=9, transform=a.get_xaxis_transform())
    else:
        a.axhline(y=pos, color=color, linestyle=linestyle, linewidth=linewidth)
        if label:
            a.text(lblpos, pos, r'$h$', va='bottom', ha='left',
                   fontsize=9, transform=a.get_yaxis_transform())


def mark_layers(xdata, ydata, idx_dict, ax=None, filled=True, color='black',
                size=9, zorder=6, edgewidth=0.8):
    """Place discrete boundary-layer markers on a curve.

    xdata, ydata : the x- and y-data arrays of the curve to mark on (e.g. z+ and
                   u+, or the (u, w) components of a hodograph).  The markers are
                   drawn at (xdata[idx], ydata[idx]) so they sit ON the curve.
    idx_dict     : mapping {marker-symbol: index}, e.g.
                   {'o': i_visc, 's': i_canopy, '^': i_logstart, 'D': i_logtop}.
    filled       : True  -> filled marker  (orographic / valley curve);
                   False -> hollow marker  (smooth-wall curve).
    """
    import matplotlib.pyplot as plt
    a = ax if ax is not None else plt.gca()
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)
    n = min(xdata.size, ydata.size)
    for sym, idx in idx_dict.items():
        if idx is None or idx < 0 or idx >= n:
            continue
        a.plot(xdata[idx], ydata[idx], linestyle='none', marker=sym,
               markersize=size,
               markerfacecolor=(color if filled else 'none'),
               markeredgecolor=color, markeredgewidth=edgewidth, zorder=zorder)


def mark_layers_multi(xdata, ydata_list, idx_dict, **kwargs):
    """Apply mark_layers to several curves that share the same x-data.

    Use this so EVERY curve of a case carries its layer markers, e.g. all the
    valley term-curves of a momentum budget (shared x = y_inner) or all the
    smooth term-curves (shared x = y_s_p).
    """
    for ydata in ydata_list:
        mark_layers(xdata, ydata, idx_dict, **kwargs)


def layer_legend_handles(symbols=('o', 's', '^', 'D'), oro=True, smooth=True,
                         markersize=5):
    """Return Line2D legend handles describing the layer-marker symbols.

    markersize keeps the legend glyphs small (the on-curve markers drawn by
    mark_layers are deliberately larger).  Pass the result to ax.legend(), or
    use add_marker_legend() to drop a fine-print legend at the bottom.
    """
    from matplotlib.lines import Line2D
    h = []
    for sym in symbols:
        h.append(Line2D([0], [0], ls='none', marker=sym, color='black',
                        markerfacecolor='black', markeredgecolor='black',
                        markersize=markersize,
                        label=LAYER_MARKER_NAMES.get(sym, sym)))
    if oro:
        h.append(Line2D([0], [0], ls='none', marker='o', color='black',
                        markerfacecolor='black', markeredgecolor='black',
                        markersize=markersize, label='valley (filled)'))
    if smooth:
        h.append(Line2D([0], [0], ls='none', marker='o', color='black',
                        markerfacecolor='none', markeredgecolor='black',
                        markersize=markersize, label='smooth (hollow)'))
    return h


def add_marker_legend(ax=None, oro=True, smooth=True, fontsize=6, markersize=5):
    """Add a separate, fine-print legend explaining the layer markers at the
    BOTTOM of the plot, without disturbing the plot's existing (line) legend.

    The on-curve markers are large for visibility; this legend uses small glyphs
    (markersize) and small text (fontsize) so it reads as a discreet footnote.
    It sits inside the axes (lower-centre, light semi-transparent box) so it is
    always captured by savefig regardless of tight_layout / bbox settings.
    Call it AFTER the main legend has been created on the same axes.
    """
    import matplotlib.pyplot as plt
    a = ax if ax is not None else plt.gca()
    main = a.get_legend()                 # main (line) legend already on the axes
    handles = layer_legend_handles(oro=oro, smooth=smooth, markersize=markersize)
    leg = a.legend(handles=handles, loc='lower center', ncol=min(3, len(handles)),
                   fontsize=fontsize, handletextpad=0.3, columnspacing=0.9,
                   frameon=True, framealpha=0.7, facecolor='white',
                   edgecolor='none')
    if main is not None:
        a.add_artist(main)                # restore the main legend a.legend() displaced
    return leg
