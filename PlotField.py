#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:50:09 2024

@author: shreyad95
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from scipy.integrate import solve_ivp
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
from matplotlib.colors import SymLogNorm
from matplotlib.colors import Normalize
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

# --- Plotting function ---
def plot_ns_budget(ns_dict, component_label, y_inner, limity, flk_hgt, savedir):
    '''
    Single plot: colours = NS terms, linestyles = valley locations.
    '''
    term_colors = {
        'Advection':       'black',
        'Pressure grad.':  'green',
        'Viscous':         'blue',
        'Reynolds stress': 'orange',
        'Coriolis':        'red',
    }
    loc_styles = {
        'top':    '-',
        'lf':     '--',
        'bottom': ':',
        'rf':     '-.',
    }
    loc_labels = {
        'top':    'Top',
        'lf':     'LF',
        'bottom': 'Bot',
        'rf':     'RF',
    }
    loc_jstart = {
        'top':    94,
        'lf':     flk_hgt,
        'bottom': 0,
        'rf':     flk_hgt,
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    for term, col in term_colors.items():
        for loc, ls in loc_styles.items():
            j0 = loc_jstart[loc]
            if term not in ns_dict[loc]:
                continue
            prof = ns_dict[loc][term]
            n_pts = prof.size
            y_loc = y_inner[j0:j0+n_pts]
            clip = min(limity - j0, n_pts) if limity > j0 else n_pts
            if clip <= 0:
                continue
            ax.plot(prof[:clip], y_loc[:clip], color=col, linestyle=ls, linewidth=0.9)

    # Custom legend: two groups
    from matplotlib.lines import Line2D
    present_terms = set().union(*(ns_dict[loc].keys() for loc in ns_dict))
    term_handles = [Line2D([0],[0], color=c, ls='-', lw=1.5, label=t)
                    for t, c in term_colors.items() if t in present_terms]
    loc_handles  = [Line2D([0],[0], color='grey', ls=ls, lw=1.5, label=ll)
                    for loc, (ls, ll) in zip(loc_styles.keys(),
                    zip(loc_styles.values(), loc_labels.values()))]
    leg1 = ax.legend(handles=term_handles, loc='upper right', fontsize=7,
                     title='Term', title_fontsize=8, framealpha=0.8)
    ax.add_artist(leg1)
    ax.legend(handles=loc_handles, loc='upper left', fontsize=7,
              title='Location', title_fontsize=8, framealpha=0.8)

    ax.set_xlabel(rf'${component_label}$-momentum')
    ax.set_ylabel(r'$z^{+}$')
    ax.set_title(f'NS budget — ${component_label}$')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, f'NS_budget_{component_label}.png'), dpi=300)
    plt.show()

def plot_2Dfield(x, y, field_2D, field_label, title_prefix, savename, xfill, yfill):
    Gy, Gx = np.meshgrid(y, x)  # Adjusted to match the order of field_2D
    
    # Plot the 2D field on the grid
    plt.figure(figsize=(8, 6))
    plt.contourf(Gx, Gy, field_2D.T, cmap='viridis', levels=1000)  # Use contourf for filled contour plot
    plt.colorbar(label=field_label)  # Add colorbar to show field values
    plt.fill(xfill, yfill, facecolor='black')
    
    # Create a Rectangle patch with label
    black_patch = mpatches.Rectangle((0, 0), 1, 1, facecolor='black', label='Solid IBM elements')
    
    # Add legend with the Rectangle patch
    plt.legend(handles=[black_patch])
    
    plt.xlabel(r'$x^{+}$')
    plt.ylabel(r'$z^{+}$')
    plt.title(title_prefix)
    plt.grid(True)
    plt.savefig(savename, dpi=300)
    plt.show()

    
def plot2D_div(x, y, field_2D, field_label, title_prefix, xname, yname, savename, xfill, yfill, resolution):
    # Create meshgrid
    Y, X = np.meshgrid(y, x)
    
    # Determine colorbar limits
    if np.max(field_2D) > abs(np.min(field_2D)):
        ll = -np.max(field_2D)
        ul = np.max(field_2D)
    else:
        ll = np.min(field_2D)
        ul = -np.min(field_2D)
     
    # Plot the data with contourf
    plt.figure(figsize=(8, 6))
    contourf = plt.contourf(X, Y, field_2D.T, cmap='seismic', levels=resolution, vmin=ll, vmax=ul)
 
    # Overlay zero values with black
    plt.fill(xfill, yfill, facecolor='black')
    
    # Add colorbar with labeled ticks
    ticks = np.linspace(ll, ul, 11)  # Adjust number of ticks as needed
    plt.colorbar(contourf, label=field_label, ticks=ticks)
    
    # Create a Rectangle patch with label
    black_patch = mpatches.Rectangle((0, 0), 1, 1, facecolor='black', label='Solid IBM elements')
    
    # Add legend with the Rectangle patch
    plt.legend(handles=[black_patch])
    
    # Set plot title and axis labels
    plt.title(title_prefix)
    plt.xlabel(xname)
    plt.ylabel(yname)
    
    # Save and show plot
    plt.savefig(savename, dpi=300)
    plt.show()
    
def plot2D_div_smooth(x, y, field_2D, field_label, title_prefix, xname, yname, savename, xfill, yfill, resolution):
    # Create meshgrid
    Y, X = np.meshgrid(y, x)
    
    # Determine colorbar limits
    if np.max(field_2D) > abs(np.min(field_2D)):
        ll = -np.max(field_2D)
        ul = np.max(field_2D)
    else:
        ll = np.min(field_2D)
        ul = -np.min(field_2D)
     
    # Plot the data with contourf
    plt.figure(figsize=(8, 6))
    contourf = plt.contourf(X, Y, field_2D.T, cmap='seismic', levels=resolution, vmin=ll, vmax=ul)
    
    # Add colorbar with labeled ticks
    ticks = np.linspace(ll, ul, 11)  # Adjust number of ticks as needed
    plt.colorbar(contourf, label=field_label, ticks=ticks)
    
    # Create a Rectangle patch with label
    black_patch = mpatches.Rectangle((0, 0), 1, 1, facecolor='black', label='Solid IBM elements')
    
    # Add legend with the Rectangle patch
    plt.legend(handles=[black_patch])
    
    # Set plot title and axis labels
    plt.title(title_prefix)
    plt.xlabel(xname)
    plt.ylabel(yname)
    
    # Save and show plot
    plt.savefig(savename, dpi=300)
    plt.show()
    
def plotanimate(x, y, field_2D, field_label, title_prefix, xname, yname, savename, xfill, yfill, resolution):
    Y, X = np.meshgrid(y, x)
    
    ll, ul = 0, 0.02
    levels = np.linspace(ll, ul, resolution)
    
    plt.figure(figsize=(12, 6), dpi=300)
    contourf = plt.contourf(X, Y, field_2D.T, cmap='Reds', levels=levels, vmin=ll, vmax=ul)

    plt.fill(xfill, yfill, facecolor='black')
    
    cbar = plt.colorbar(contourf)
    cbar.set_ticks([])
    
    plt.title(title_prefix)
    plt.xlabel(xname)
    plt.ylabel(yname)

    # Save with correct size and no padding
    plt.savefig(savename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close figure to prevent memory issues
    
def plotanimatelog(x, y, field_2D, field_label, title_prefix, xname, yname, savename, xfill, yfill, resolution):
    Y, X = np.meshgrid(y, x)
    
    # Apply logarithmic transformation (avoid log(0) by adding a small offset if necessary)
    field_2D_log = np.log(field_2D + 1e-10)  # Small value added to avoid log(0)
    
    ll, ul = np.log(1e-10), np.log(np.max(field_2D))  # Adjust the color levels according to log scale
    levels = np.linspace(ll, ul, resolution)
    
    plt.figure(figsize=(12, 6), dpi=300)
    contourf = plt.contourf(X, Y, field_2D_log.T, cmap='Reds', levels=levels, vmin=ll, vmax=ul)

    plt.fill(xfill, yfill, facecolor='black')
    
    cbar = plt.colorbar(contourf)
    cbar.set_ticks([])

    plt.title(title_prefix)
    plt.xlabel(xname)
    plt.ylabel(yname)

    # Save with correct size and no padding
    plt.savefig(savename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close figure to prevent memory issues
    
def plot2D_cont_log(x, y, field_2D, field_label, title_prefix, xname, yname, savename, xfill, yfill, resolution):
    """Plots a 2D contour with a **logarithmic** color scale while handling zero and negative values."""

    # Create meshgrid
    Y, X = np.meshgrid(y, x)

    # Set symmetric colorbar limits
    ll, ul = -911, 911  

    # Define symmetric logarithmic normalization (handles positive, negative, and zero)
    norm = SymLogNorm(linthresh=1, linscale=1, vmin=ll, vmax=ul, base=10)  

    # Define logarithmically spaced contour levels (ensuring balance in positive and negative)
    levels = np.concatenate([
        -np.logspace(np.log10(abs(ll)), np.log10(1), resolution//2, endpoint=False), 
         np.logspace(np.log10(1), np.log10(ul), resolution//2)
    ])  

    # Plot the data with contourf using the log normalization
    plt.figure(figsize=(8, 6))
    contourf = plt.contourf(X, Y, field_2D.T, cmap='seismic', levels=levels, norm=norm)

    # Overlay zero values with black (solid objects)
    plt.fill(xfill, yfill, facecolor='black')

    # Add colorbar with **log-spaced ticks**
    cbar = plt.colorbar(contourf, label=field_label)
    cbar.set_ticks([-911, -100, -10, -1, 0, 1, 10, 100, 911])  # Adjust tick spacing if needed

    # Add legend for solid IBM elements
    black_patch = mpatches.Rectangle((0, 0), 1, 1, facecolor='black', label='Solid IBM elements')
    plt.legend(handles=[black_patch])

    # Set plot title and axis labels
    plt.title(title_prefix)
    plt.xlabel(xname)
    plt.ylabel(yname)

    # Save and show plot
    plt.savefig(savename, dpi=300)
    plt.show()
    
def plot2D_cont(x, y, field_2D, field_label, title_prefix, xname, yname, savename, xfill, yfill, resolution):
    """Plots a 2D contour with a fixed symmetric colorbar range (-911 to +911) and uniform colorbar ticks."""

    # Create meshgrid
    Y, X = np.meshgrid(y, x)

    # Set hardcoded colorbar limits
    ll, ul = -911, 911  

    # Define contour levels manually to enforce the scale
    levels = np.linspace(ll, ul, resolution)  

    # Plot the data with contourf
    plt.figure(figsize=(8, 6))
    contourf = plt.contourf(X, Y, field_2D.T, cmap='seismic', levels=levels, vmin=ll, vmax=ul)

    # Overlay zero values with black (solid objects)
    plt.fill(xfill, yfill, facecolor='black')

    # Add colorbar with **fixed** tick range from -911 to 911
    cbar = plt.colorbar(contourf, label=field_label)
    cbar.set_ticks(np.linspace(ll, ul, 11))  # **Forces uniform tick intervals**
    contourf.set_clim(ll, ul)  # **Correct method to enforce colorbar limits**

    # Add legend for solid IBM elements
    black_patch = mpatches.Rectangle((0, 0), 1, 1, facecolor='black', label='Solid IBM elements')
    plt.legend(handles=[black_patch])

    # Set plot title and axis labels
    plt.title(title_prefix)
    plt.xlabel(xname)
    plt.ylabel(yname)

    # Save and show plot
    plt.savefig(savename, dpi=300)
    plt.show()


def plot2D_div_log(x, y, field_2D, field_label, title_prefix, xname, yname, savename, xfill, yfill, resolution):
    """Plots 2D field with a logarithmic color scale, handling both positive and negative values separately."""
    
    # Create meshgrid
    Y, X = np.meshgrid(y, x)

    # Separate positive and negative values
    field_positive = np.where(field_2D > 0, field_2D, np.nan)  # Keep only positive values
    field_negative = np.where(field_2D < 0, -field_2D, np.nan)  # Flip sign for negative values

    # Determine logarithmic normalization
    norm = mcolors.LogNorm(vmin=np.nanpercentile(abs(field_2D), 1), vmax=np.nanpercentile(abs(field_2D), 99))

    # Create figure
    plt.figure(figsize=(8, 6))

    # Plot positive values with Reds
    if np.any(~np.isnan(field_positive)):
        plt.contourf(X, Y, field_positive.T, cmap='Reds', norm=norm, levels=resolution)

    # Plot negative values with Blues
    if np.any(~np.isnan(field_negative)):
        plt.contourf(X, Y, field_negative.T, cmap='Blues', norm=norm, levels=resolution)

    # Overlay zero values with black (solid objects)
    plt.fill(xfill, yfill, facecolor='black')

    # Add colorbar
    cbar = plt.colorbar(label=field_label)
    
    # Add legend for solid IBM elements
    black_patch = mpatches.Rectangle((0, 0), 1, 1, facecolor='black', label='Solid IBM elements')
    plt.legend(handles=[black_patch])

    # Titles and labels
    plt.title(title_prefix)
    plt.xlabel(xname)
    plt.ylabel(yname)

    # Save and show plot
    plt.savefig(savename, dpi=300)
    plt.show()
    
def plot2D_log(x, y, field_2D, field_label, title_prefix, xname, yname, savename, xfill, yfill, resolution):
    # Create meshgrid
    Y, X = np.meshgrid(y, x)

    # Set threshold for linear vs. log scaling
    linthresh = 100  # Adjust based on data distribution

    # Apply symmetric log normalization
    norm = mcolors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=np.min(field_2D), vmax=np.max(field_2D))

    # Plot the data with contourf
    plt.figure(figsize=(8, 6))
    contourf = plt.contourf(X, Y, field_2D.T, cmap='seismic', levels=resolution, norm=norm)

    # Overlay zero values with black
    plt.fill(xfill, yfill, facecolor='black')

    # Add colorbar
    plt.colorbar(contourf, label=field_label)

    # IBM region legend
    black_patch = mpatches.Rectangle((0, 0), 1, 1, facecolor='black', label='Solid IBM elements')
    plt.legend(handles=[black_patch])

    # Titles and labels
    plt.title(title_prefix)
    plt.xlabel(xname)
    plt.ylabel(yname)

    # Save and show plot
    plt.savefig(savename, dpi=300)
    plt.show()

def quiver_div(x, y, u, v, w, speed, skipX, skipY, field_label, title_prefix, savename, xfill, yfill, log_flag):
    # Create meshgrid
    Y, X = np.meshgrid(y, x)

    # Determine colorbar limits for speed
    speed_min = np.min(speed)
    speed_max = np.max(speed)

    # Plot the speed field with contourf
    plt.figure(figsize=(10, 6))
    speed = np.sqrt((u**2 + v**2 + w**2))
    if (log_flag == 1):
        speed = np.log(speed)
    speed = np.where(np.isinf(speed), 0, speed)
    contourf = plt.contourf(X, Y, speed.T, cmap='plasma', levels=100, vmin=speed_min, vmax=speed_max)

    # Overlay the quiver plot for velocity vectors
    # plt.quiver(X, Y, u.T, v.T, scale=50, color='black')  # Adjust scale as needed

    plt.quiver( X[::skipX, ::skipY].T, Y[::skipX, ::skipY].T, u[::skipY, ::skipX], v[::skipY, ::skipX].T, color='black', headwidth=2, scale=200, headlength=5)

    # Overlay zero values with black (e.g., IBM solid region)
    plt.fill(xfill, yfill, facecolor='black')

    # Add colorbar for speed
    plt.colorbar(contourf, label=field_label)

    # Create a Rectangle patch for the filled region
    black_patch = mpatches.Rectangle((0, 0), 1, 1, facecolor='black', label='Solid IBM elements')

    # Add legend with the Rectangle patch
    plt.legend(handles=[black_patch])

    # Set plot title and axis labels
    plt.title(title_prefix)
    plt.xlabel(r'$x^{+}$')
    plt.ylabel(r'$z^{+}$')

    # Save and show plot
    plt.savefig(savename, dpi=300)
    plt.show()

def plot_2Dlogfield(x, y, field_2D, field_label, title_prefix, savename, xfill, yfill):
    tolerance = 1e-6  # Adjust the tolerance as needed
    mask = np.isclose(field_2D, 0, atol=tolerance)
    masked_field_data = np.ma.masked_where(mask, field_2D)
    Gy, Gx = np.meshgrid(y, x)  # Adjusted to match the order of field_2D
    
    # Plot the 2D field on the grid
    plt.figure(figsize=(8, 6))
    
    # Determine levels dynamically based on field values
    cmap = plt.cm.seismic
    if (np.min(field_2D) >= 0 ):
        levels = np.linspace(0, np.max(field_2D), 1000)
    elif (np.min(field_2D) <= 0 ):
        if (abs(np.min(field_2D)) > abs(np.max(field_2D))):
            levels = np.linspace(0, -np.min(field_2D), 1000)
        else:
            levels = np.linspace(0, np.max(field_2D), 1000)
        
    # Plot contourf with custom colormap and dynamic levels
    plt.contourf(Gx, Gy, masked_field_data.T, cmap=cmap, levels=levels, extend='both')
    plt.fill(xfill, yfill, facecolor='black')
    # Add colorbar to show field values
    plt.colorbar(label=field_label)
    
    plt.xlabel(r'$L/\delta$')
    plt.ylabel(r'$h/L$')
    plt.title(title_prefix)
    plt.grid(True)
    plt.savefig(savename, dpi=300)
    plt.show()


def plot_3Dfield(x, y, z, field_3D, field_label, title_prefix, savename, slice_axis='z', slice_index=None):
    """
    Plot 3D fields along specified slice axis.

    Parameters:
        x (array): Array representing coordinates along the x-axis.
        y (array): Array representing coordinates along the y-axis.
        z (array): Array representing coordinates along the z-axis.
        field_3D (array): 3D array representing the field to be plotted.
        field_label (str): Label for the colorbar representing the field values.
        title_prefix (str): Prefix for the plot title.
        savename (str): File path to save the plot (including file extension).
        slice_axis (str): Axis along which to slice the 3D field ('x', 'y', or 'z'). Default is 'z'.
        slice_index (int): Index of the slice along the chosen axis. Default is None (middle slice).

    Returns:
        None
    """
    # Determine the slice index
    if slice_index is None:
        if slice_axis == 'x':
            slice_index = len(x) // 2
        elif slice_axis == 'y':
            slice_index = len(y) // 2
        else:
            slice_index = len(z) // 2
    
    # Slice the 3D field along the specified axis
    if slice_axis == 'x':
        field_slice = field_3D[slice_index, :, :]
        slice_coord = x[slice_index]
        xlabel = r'$L/\delta$'
        ylabel = r'$h/L$'
    elif slice_axis == 'y':
        field_slice = field_3D[:, slice_index, :]
        slice_coord = y[slice_index]
        xlabel = r'$L/\delta$'
        ylabel = r'$h/L$'
    else:
        field_slice = field_3D[:, :, slice_index]
        slice_coord = z[slice_index]
        xlabel = r'$L/\delta$'
        ylabel = r'$h/L$'

    # Plot the 2D slice of the field
    plt.figure(figsize=(8, 6))
    plt.contourf(x, y, field_slice.T, cmap='viridis')  # Use contourf for filled contour plot
    plt.colorbar(label=field_label)  # Add colorbar to show field values
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title_prefix} at {slice_axis} = {slice_coord}")
    plt.grid(True)
    plt.savefig(savename)
    plt.show()
    
    
def plot_hodograph(u, v, heights):
    plt.figure(figsize=(8, 6))
    
    # Plot the hodograph using the wind vector components
    plt.plot(u, v, marker='o', linestyle='-', label='Wind vector')
    
    # Annotate with heights
    for i in range(len(heights)):
        plt.text(u[i], v[i], f'{heights[i]}', fontsize=8, color='red')
    
    # Formatting the plot
    plt.xlabel('u (zonal wind component)')
    plt.ylabel('v (meridional wind component)')
    plt.title('Hodograph: Wind turning with height')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.show()
    
    
def plot_TKE(x, y, z, TKE, savename):
    """
    Plot turbulent kinetic energy (TKE).

    Parameters:
        x (array): Array representing coordinates along the x-axis.
        y (array): Array representing coordinates along the y-axis.
        z (array): Array representing coordinates along the z-axis.
        TKE (array): 3D array representing the turbulent kinetic energy.
        savename (str): File path to save the plot (including file extension).

    Returns:
        None
    """
    plot_3Dfield(x, y, z, TKE, 'TKE', 'Turbulent Kinetic Energy', savename, slice_axis='z', slice_index=None)

def plot_dissipation_rate(x, y, z, dissipation_rate, savename):
    """
    Plot dissipation rate.

    Parameters:
        x (array): Array representing coordinates along the x-axis.
        y (array): Array representing coordinates along the y-axis.
        z (array): Array representing coordinates along the z-axis.
        dissipation_rate (array): 3D array representing the dissipation rate.
        savename (str): File path to save the plot (including file extension).

    Returns:
        None
    """
    plot_3Dfield(x, y, z, dissipation_rate, 'Dissipation Rate', 'Dissipation Rate', savename, slice_axis='z', slice_index=None)

def plot2D_div_enhanced(x, y, field_2D, field_label, title_prefix, xname, yname, savename, xfill, yfill, resolution, contour=False):
    # Create meshgrid
    Y, X = np.meshgrid(y, x)
    
    # Handle NaN and -Inf values
    field_2D = np.nan_to_num(field_2D, nan=0.0, neginf=0.0)
    
    # Determine colorbar limits for better contrast
    max_abs_val = np.max(np.abs(field_2D))
    ll, ul = -max_abs_val, max_abs_val  # Symmetric range
    
    # Plot the data with contourf (color fill)
    plt.figure(figsize=(8, 6))
    contourf = plt.contourf(X, Y, field_2D.T, cmap='coolwarm', levels=resolution, vmin=ll, vmax=ul)
    
    # Add contour lines if enabled
    if contour:
        contour_line_levels = np.linspace(ll, ul, resolution // 10)  # Fewer contour lines
        contour_lines = plt.contour(X, Y, field_2D.T, levels=contour_line_levels, colors='yellow', linewidths=0.1)
        plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
    
    # Overlay zero values with black
    plt.fill(xfill, yfill, facecolor='black')
    
    # Add colorbar with labeled ticks
    ticks = np.linspace(ll, ul, 11)
    plt.colorbar(contourf, label=field_label, ticks=ticks)
    
    # Create a Rectangle patch with label
    black_patch = mpatches.Rectangle((0, 0), 1, 1, facecolor='black', label='Solid IBM elements')
    
    # Add legend with the Rectangle patch
    plt.legend(handles=[black_patch])
    
    # Set plot title and axis labels
    plt.title(title_prefix)
    plt.xlabel(xname)
    plt.ylabel(yname)
    
    # Save and show plot
    plt.savefig(savename, dpi=300)
    plt.show()

# def plot2D_streamlines(x, y, U, V, field_label, title_prefix, xname, yname, savename, xfill, yfill, resolution):
#     """ Plots streamlines using interpolation for a non-uniform grid in y. """

#     # Create a uniform grid in the y-direction while keeping x unchanged
#     y_uniform = np.linspace(y.min(), y.max(), resolution)  # Create uniform y-grid
#     X, Y = np.meshgrid(x, y_uniform, indexing='xy')  # Generate meshgrid

#     # Interpolate U and V onto the uniform grid
#     U_interp = RectBivariateSpline(x, y, U.T)  # Transpose to match (x, y) order
#     V_interp = RectBivariateSpline(x, y, V.T)

#     U_uniform = U_interp(x, y_uniform).T  # Transpose back to original shape
#     V_uniform = V_interp(x, y_uniform).T

#     # Compute speed for colormap
#     speed = np.sqrt(U_uniform**2 + V_uniform**2)
#     cmap = 'seismic' if np.min(speed) < 0 else 'Reds'

#     # Plot streamlines
#     plt.figure(figsize=(8, 6))
#     plt.streamplot(X, Y, U_uniform, V_uniform, color=speed, cmap=cmap, linewidth=0.7, density=1.5)

#     # Overlay zero values with black (solid objects)
#     plt.fill(xfill, yfill, facecolor='black')

#     # Titles and labels
#     plt.title(title_prefix)
#     plt.xlabel(xname)
#     plt.ylabel(yname)

#     # Save and show plot
#     plt.savefig(savename, dpi=300)
#     plt.show()


def plot2D_streamlines(x, y, U, V, field_label, title_prefix, xname, yname, savename, xfill, yfill, resolution):
    """ Plots streamlines with a contour overlay for speed on a non-uniform y grid while avoiding IBM regions. """

    # Create a uniform grid in the y-direction while keeping x unchanged
    y_uniform = np.linspace(y.min(), y.max(), resolution)  # Create uniform y-grid
    X, Y = np.meshgrid(x, y_uniform, indexing='xy')  # Generate meshgrid

    # Interpolate U and V onto the uniform grid
    U_interp = RectBivariateSpline(y, x, U)  # Flip input order (y, x) for RectBivariateSpline
    V_interp = RectBivariateSpline(y, x, V)

    U_uniform = U_interp(y_uniform, x)
    V_uniform = V_interp(y_uniform, x)

    # Compute speed magnitude
    speed = np.sqrt(U_uniform**2 + V_uniform**2)

    # Define colormap based on min/max values
    cmap = 'seismic' if np.min(speed) < 0 else 'Reds'

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # **Plot Contours of Speed**
    contour = ax.contourf(X, Y, speed, levels=resolution, cmap=cmap)

    # **Plot Streamlines**
    ax.streamplot(X, Y, U_uniform, V_uniform, color=speed, cmap=cmap, linewidth=0.7, density=1.5)

    # **Overlay black IBM region**
    ax.fill(xfill, yfill, facecolor='black', zorder=3)  # zorder ensures it's on top

    # **Titles and Labels**
    ax.set_title(title_prefix)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)

    # **Colorbar for speed**
    cbar = fig.colorbar(contour, ax=ax, label=field_label)

    # **Save and Show Plot**
    plt.savefig(savename, dpi=300)
    plt.show()
    
def plot2D_equipotential(x, y, phi, title, xname, yname, savename,
                          xfill, yfill, resolution=1000, n_levels=2):
    """
    Plot equipotential lines of a scalar potential field phi.

    Directly adapted from plot2D_streamlines_vorticityX:
      - Same interpolation onto a uniform y-grid via RectBivariateSpline
      - Same IBM solid fill (xfill / yfill polygon, black, on top)
      - Same save / show pattern
      - Replaces streamplot + vorticity contourf with:
          contourf  → smooth background colour-fill of phi
          contour   → black equipotential lines
          clabel    → inline level labels (auto-scaled to O(1))

    Scale management
    ----------------
    Automatically detects the order of magnitude of phi (e.g. 1e-6) and
    multiplies by the inverse power of 10 before plotting.  All tick labels,
    contour labels, and the colourbar label are expressed in those scaled
    units (e.g. "phi [x10^-6]"), so the numbers on screen are always O(1)
    regardless of the raw field magnitude.

    Parameters
    ----------
    x          : 1-D array  (nx,)   x-coordinates (uniform or non-uniform)
    y          : 1-D array  (ny,)   y-coordinates (possibly stretched)
    phi        : 2-D array  (ny, nx)  scalar potential field
    title      : str   figure title
    xname      : str   x-axis label
    yname      : str   y-axis label
    savename   : str   full output path, e.g. 'fig/phi_equipotential.png'
    xfill      : 1-D array   x-coordinates of the IBM / solid polygon
    yfill      : 1-D array   y-coordinates of the IBM / solid polygon
    resolution : int   number of uniform y-points for interpolation (default 500)
    n_levels   : int   number of equipotential contour lines (default 25)

    Example
    -------
    eps_hgt  = np.sum(eps, axis=0).astype(int)
    hill_top = np.where(eps_hgt > 0, y[np.maximum(eps_hgt-1, 0)], 0.0)
    xfill    = np.concatenate([[x[0]], x, [x[-1]]])
    yfill    = np.concatenate([[0],    hill_top, [0]])

    plot2D_equipotential(
        x, y[:300], phi[:300, :],
        title=r'Equipotential lines of $\\phi$',
        xname='x', yname='y',
        savename='fig/phi_equipotential.png',
        xfill=xfill, yfill=yfill,
        resolution=500, n_levels=25,
    )
    """
    # ── 1. Interpolate phi onto a uniform y-grid ───────────────────────────
    y_uniform  = np.linspace(y.min(), y.max(), resolution)
    phi_interp = RectBivariateSpline(y, x, phi)
    phi_uniform = phi_interp(y_uniform, x)
    X, Y = np.meshgrid(x, y_uniform, indexing='xy')

    # ── 2. Auto-scale to O(1) ──────────────────────────────────────────────
    finite_vals = phi_uniform[np.isfinite(phi_uniform)]
    mean_abs    = np.abs(finite_vals).mean()
    if mean_abs > 0:
        exp       = int(np.floor(np.log10(mean_abs)))
        scale     = 10 ** (-exp)          # e.g. 1e6 for phi ~ 1e-6
    else:
        scale, exp = 1.0, 0

    phi_scaled = phi_uniform * scale
    vmin_s = float(np.nanmin(phi_scaled))
    vmax_s = float(np.nanmax(phi_scaled))

    # ── 3. Contour levels: strictly increasing, inside data range ─────────
    # Trim one step from each end so the outermost level never equals the
    # data extremum (avoids a zero-width contour band at the boundary).
    levels = np.linspace(vmin_s, vmax_s, n_levels + 2)[1:-1]

    # ── 4. Choose colourmap ────────────────────────────────────────────────
    # RdBu_r if phi has mixed sign (rare for potential), Blues_r if all negative,
    # Reds if all positive.
    if vmin_s < 0 < vmax_s:
        cmap_fill = 'RdBu_r'
    elif vmax_s <= 0:
        cmap_fill = 'Blues_r'
    else:
        cmap_fill = 'Reds'

    # ── 5. Figure ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    # Background: smooth colour-fill (same as contourf in the original)
    cf = ax.contourf(X, Y, phi_scaled,
                     levels=np.linspace(vmin_s, vmax_s, 120),
                     cmap=cmap_fill, alpha=1.0, zorder=2)

    # Equipotential lines in black
    cs = ax.contour(X, Y, phi_scaled,
                    levels=levels,
                    colors='grey', linewidths=0.6, zorder=3)

    # Inline labels every ~6 levels so the plot stays readable
    label_levels = levels[::max(1, n_levels // 6)]
    ax.clabel(cs, levels=label_levels,
              fmt=lambda v: f'{v:.2f}',
              fontsize=4, inline=True, inline_spacing=3, zorder=4)

    # IBM / solid region on top — identical to the original function
    ax.fill(xfill, yfill, facecolor='black', zorder=4)

    # ── 6. Colourbar with scaled label ────────────────────────────────────
    cbar = fig.colorbar(cf, ax=ax, pad=0.01, shrink=0.95)
    if exp != 0:
        cbar_label = fr'$\phi$ [$\times 10^{{{exp}}}$]'
    else:
        cbar_label = r'$\phi$'
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # ── 7. Labels, save, show ─────────────────────────────────────────────
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xname, fontsize=9)
    ax.set_ylabel(yname, fontsize=9)
    ax.tick_params(labelsize=8)

    plt.savefig(savename, dpi=300, format='png', transparent=False,
                bbox_inches='tight')
    plt.show()

    
# def plot2D_streamlines_vorticity(x, y, U, V, vorticity, field_label, title_prefix, xname, yname, savename, xfill, yfill, resolution):
#     """Plots streamlines over vorticity contours while avoiding IBM regions."""

#     # Create a uniform grid in the y-direction while keeping x unchanged
#     y_uniform = np.linspace(y.min(), y.max(), resolution)  # Create uniform y-grid
#     X, Y = np.meshgrid(x, y_uniform, indexing='xy')  # Generate meshgrid

#     # Interpolate U, V, and vorticity onto the uniform grid
#     U_interp = RectBivariateSpline(y, x, U)
#     V_interp = RectBivariateSpline(y, x, V)
#     vort_interp = RectBivariateSpline(y, x, vorticity)

#     U_uniform = U_interp(y_uniform, x)
#     V_uniform = V_interp(y_uniform, x)
#     vort_uniform = vort_interp(y_uniform, x)

#     # Choose streamline color (e.g., U component)
#     streamline_color = U_uniform  
#     cmap_streamlines = 'seismic'  # Red = right, Blue = left

#     # **Plotting**
#     fig, ax = plt.subplots(figsize=(8, 6))

#     # **Compute min/max vorticity for legend**
#     vmin, vmax = np.min(vorticity), np.max(vorticity)
    
#     # **Plot vorticity contours**
#     levels = np.linspace(vmin, vmax, 50)  # Ensure full vorticity range is captured
#     contour = ax.contourf(X, Y, vort_uniform, levels=levels, cmap='coolwarm', alpha=0.6)  # Slight transparency
#     cbar = fig.colorbar(contour, ax=ax)  # <--- Now explicitly displayed
    
#     # Update colorbar label with min/max values
#     cbar.set_label(f"Vorticity")  

#     # **Plot streamlines on top**
#     ax.streamplot(X, Y, U_uniform, V_uniform, color=streamline_color, cmap=cmap_streamlines, linewidth=0.7, density=1.5)

#     # **Overlay black IBM region**
#     ax.fill(xfill, yfill, facecolor='black', zorder=3)  # Ensures it's on top

#     # **Titles and Labels**
#     ax.set_title(title_prefix)
#     ax.set_xlabel(xname)
#     ax.set_ylabel(yname)

#     # **Save and Show Plot**
#     plt.savefig(savename, dpi=300)
#     plt.show()

def plot2D_streamlines_vorticity(x, y, U, V, vorticity, eps, field_label, title_prefix, xname, yname, savename, xfill, yfill, resolution): 
    """Plots streamlines over vorticity contours while avoiding IBM (solid) regions.
    
    Parameters:
      - x, y: Original grid coordinates.
      - U, V: Dispersive velocity components (streamwise and normal).
      - vorticity: Vorticity field.
      - eps: Indicator variable where the solid (IBM) region is 1.
      - field_label, title_prefix, xname, yname, savename: Plot labeling and saving parameters.
      - xfill, yfill: Coordinates for filling (plotting) the solid region.
      - resolution: Number of grid points for the uniform y-grid.
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import RectBivariateSpline
    
    # Create a uniform grid in the y-direction while keeping x unchanged
    y_uniform = np.linspace(y.min(), y.max(), resolution)  # Create uniform y-grid
    X, Y = np.meshgrid(x, y_uniform, indexing='xy')  # Generate meshgrid

    # Interpolate U, V, vorticity, and eps onto the uniform grid
    U_interp    = RectBivariateSpline(y, x, U)
    V_interp    = RectBivariateSpline(y, x, V)
    vort_interp = RectBivariateSpline(y, x, vorticity)
    eps_interp  = RectBivariateSpline(y, x, eps)
    
    U_uniform    = U_interp(y_uniform, x)
    V_uniform    = V_interp(y_uniform, x)
    vort_uniform = vort_interp(y_uniform, x)
    eps_uniform  = eps_interp(y_uniform, x)
    
    # Create a mask from eps: assume eps==1 indicates the solid region.
    # You can adjust the threshold if eps is not exactly 1.
    mask = (eps_uniform >= 1)
    
    # Apply the mask: set velocity and vorticity values in the solid to NaN.
    # This prevents the streamline routine from "connecting" through the solid.
    U_uniform_masked    = np.copy(U_uniform)
    V_uniform_masked    = np.copy(V_uniform)
    vort_uniform_masked = np.copy(vort_uniform)
    
    U_uniform_masked[mask]    = np.nan
    V_uniform_masked[mask]    = np.nan
    vort_uniform_masked[mask] = np.nan
    
    # Choose streamline color (for example, using the U component)
    streamline_color = U_uniform_masked  
    cmap_streamlines = 'seismic'  # Red = right, Blue = left

    # **Plotting**
    fig, ax = plt.subplots(figsize=(8, 6))

    # Compute min/max vorticity for contour levels (ignoring NaNs)
    vmin = np.nanmin(vort_uniform_masked)
    vmax = np.nanmax(vort_uniform_masked)
    
    # Plot vorticity contours
    levels = np.linspace(vmin, vmax, 50)  # Ensure full vorticity range is captured
    contour = ax.contourf(X, Y, vort_uniform_masked, levels=levels, cmap='coolwarm', alpha=0.6)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Vorticity")
    
    # Plot streamlines on top using the masked velocity fields
    ax.streamplot(X, Y, U_uniform_masked, V_uniform_masked, color=streamline_color, 
                  cmap=cmap_streamlines, linewidth=0.7, density=1.5)
    
    # Overlay the solid (IBM) region explicitly
    ax.fill(xfill, yfill, facecolor='black', zorder=3)  # Ensures it's on top
    
    # Titles and Labels
    ax.set_title(title_prefix)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    
    # Save and show plot
    plt.savefig(savename, dpi=300)
    plt.show()

def plot2D_streamlines_vorticityX(x, y, U, V, vorticity, field_label, title_prefix, xname, yname, savename, xfill, yfill, resolution):
    """Plots streamlines over fully opaque vorticity contours while avoiding IBM regions."""

    # Create a uniform grid in the y-direction while keeping x unchanged
    y_uniform = np.linspace(y.min(), y.max(), resolution)  # Create uniform y-grid
    X, Y = np.meshgrid(x, y_uniform, indexing='xy')  # Generate meshgrid

    # Interpolate U, V, and vorticity onto the uniform grid
    U_interp = RectBivariateSpline(y, x, U)
    V_interp = RectBivariateSpline(y, x, V)
    vort_interp = RectBivariateSpline(y, x, vorticity)

    U_uniform = U_interp(y_uniform, x)
    V_uniform = V_interp(y_uniform, x)
    vort_uniform = vort_interp(y_uniform, x)

    # Choose streamline color (e.g., U component)
    streamline_color = U_uniform  
    cmap_streamlines = 'seismic'  # Red = right, Blue = left

    # **Plotting**
    fig, ax = plt.subplots(figsize=(8, 6))

    # **Compute min/max vorticity for legend**
    vmin, vmax = np.min(vorticity), np.max(vorticity)

    # **Plot vorticity contours (fully opaque)**
    levels = np.linspace(vmin, vmax, 50)
    contour = ax.contourf(X, Y, vort_uniform, levels=levels, cmap='coolwarm', alpha=1, zorder=2)  # Fully opaque
    cbar = fig.colorbar(contour, ax=ax)  
    cbar.set_label(f"Vorticity [{vmin:.2e}, {vmax:.2e}]")  

    # **Plot streamlines on top (thinner to improve visibility)**
    ax.streamplot(X, Y, U_uniform, V_uniform, color=streamline_color, 
                  cmap=cmap_streamlines, linewidth=0.5, density=1.5, zorder=3)

    # **Overlay black IBM region**
    ax.fill(xfill, yfill, facecolor='black', zorder=4)  # IBM region always on top

    # **Titles and Labels**
    ax.set_title(title_prefix)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)

    # **Save and Show Plot (ensures no unwanted transparency)**
    plt.savefig(savename, dpi=300, format='png', transparent=False)
    plt.show()

def plot_phavg_velocity_3D(x, y, U, V, W, eps, resolution, xfill, yfill, savename,
                           title=''):
    """
    Three-panel figure visualising a 3-component velocity field (U, V, W) on a 2D (x,y) domain.

    Panel 1 — In-plane (U,V) streamlines; background colour = spanwise W.
    Panel 2 — Out-of-plane yaw angle arctan2(W, sqrt(U²+V²)) in degrees.
    Panel 3 — Total 3D speed sqrt(U²+V²+W²) with (U,V) streamlines overlaid.

    The overlay is a filled-contour (spanwise component) + streamline (in-plane
    vector field) view of the SAME velocity triad; ``title`` names WHICH field is
    shown — e.g. the phase-averaged MEAN velocity ⟨ū_i⟩ or the DISPERSIVE velocity
    ũ_i — since the two are otherwise identical apart from the input arrays.

    Parameters
    ----------
    x, y       : 1-D arrays  grid coordinates (inner-scaled; either axis may be
                 stretched — both are resampled onto a uniform mesh below,
                 because streamplot() only accepts an equally-spaced grid)
    U          : (ny, nx)  streamwise velocity
    V          : (ny, nx)  wall-normal velocity
    W          : (ny, nx)  spanwise velocity
    eps        : (ny, nx)  IBM indicator (1 = solid)
    resolution : int  number of points for the uniform y-grid used by streamplot
    xfill, yfill : IBM polygon coordinates for the solid fill
    savename   : str  output file path
    title      : str  optional figure-level name (suptitle) identifying the field
    """
    # ── grid regularisation for streamplot() ──────────────────────────────
    # matplotlib's streamplot builds a Grid(X, Y) that demands a PERFECTLY
    # regular mesh: it raises "'x'/'y' values must be equally spaced" unless
    # np.allclose(np.diff(axis), span/(n-1)) — rtol 1e-5, so even a mildly
    # stretched (or non-finite) axis is rejected.  Only y used to be resampled
    # here, and x was handed through raw, so a case whose streamwise axis is not
    # exactly uniform — or whose inner-unit normalisation produced NaN/inf —
    # crashed the whole figure inside matplotlib.  Resample BOTH axes onto their
    # own uniform grid: for the tlab periodic x (dx = Lx/nx) x_uniform == x to
    # round-off, so an already-uniform case draws exactly as before.
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    _why = None
    if x.size < 4 or y.size < 4:
        _why = 'needs >=4 points per axis for the spline (nx=%d, nz=%d)' % (x.size, y.size)
    elif not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        _why = ('non-finite grid coordinates (%d NaN/inf in x, %d in y) — a NaN '
                'or zero normalisation length gives an all-NaN inner axis'
                % (int(np.sum(~np.isfinite(x))), int(np.sum(~np.isfinite(y)))))
    elif not ((np.diff(x) > 0).all() and (np.diff(y) > 0).all()):
        _why = 'grid coordinates are not strictly increasing'
    else:
        for _nm, _f in (('U', U), ('V', V), ('W', W), ('eps', eps)):
            if np.shape(_f) != (y.size, x.size):
                _why = ('%s has shape %s, expected (nz, nx) = (%d, %d)'
                        % (_nm, np.shape(_f), y.size, x.size))
                break
    if _why is not None:
        print('plot_phavg_velocity_3D: %s — %s skipped.'
              % (_why, os.path.basename(str(savename))))
        return

    _dx, _ideal = np.diff(x), (x[-1] - x[0]) / (x.size - 1)
    if not np.allclose(_dx, _ideal):
        # Say so rather than only failing: this is the axis streamplot rejects.
        print('plot_phavg_velocity_3D: streamwise axis is NOT uniform '
              '(dx in [%.6e, %.6e], uniform value %.6e) — resampled onto a '
              'uniform x for %s.'
              % (_dx.min(), _dx.max(), _ideal, os.path.basename(str(savename))))

    x_uniform = np.linspace(x[0], x[-1], x.size)
    y_uniform = np.linspace(y[0], y[-1], resolution)
    X, Y = np.meshgrid(x_uniform, y_uniform, indexing='xy')

    def _interp(field):
        return RectBivariateSpline(y, x, field)(y_uniform, x_uniform)

    U_u   = _interp(U)
    V_u   = _interp(V)
    W_u   = _interp(W)
    eps_u = _interp(eps)

    solid = eps_u >= 0.5
    U_u[solid] = np.nan
    V_u[solid] = np.nan
    W_u[solid] = np.nan

    spd_inplane = np.sqrt(U_u**2 + V_u**2)
    spd_3D      = np.sqrt(U_u**2 + V_u**2 + W_u**2)
    yaw_angle   = np.degrees(np.arctan2(W_u, spd_inplane))

    def _sym_norm(data):
        vmax = np.nanmax(np.abs(data))
        vmax = vmax if vmax > 0 else 1.0
        return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    ibm_patch = mpatches.Rectangle((0, 0), 1, 1, fc='black', label='IBM solid')
    fig, axes = plt.subplots(3, 1, figsize=(10, 14))

    # Panel 1: spanwise W as background, (U,V) streamlines
    ax = axes[0]
    cf1 = ax.contourf(X, Y, W_u, levels=100, cmap='RdBu_r', norm=_sym_norm(W_u))
    ax.streamplot(X, Y, U_u, V_u, color='black', linewidth=0.5, density=1.5, zorder=3)
    ax.fill(xfill, yfill, facecolor='black', zorder=4)
    fig.colorbar(cf1, ax=ax, label=r'$\langle\overline{W}\rangle$ (spanwise)')
    ax.set_xlabel(r'$x^+$')
    ax.set_ylabel(r'$z^+$')
    ax.set_title(r'In-plane $(U,V)$ streamlines; colour = spanwise $W$')
    ax.legend(handles=[ibm_patch], fontsize=7)

    # Panel 2: yaw angle (out-of-plane deviation)
    ax = axes[1]
    cf2 = ax.contourf(X, Y, yaw_angle, levels=100, cmap='seismic', norm=_sym_norm(yaw_angle))
    ax.fill(xfill, yfill, facecolor='black', zorder=4)
    fig.colorbar(cf2, ax=ax, label=r'Yaw angle $\theta_W$ (°)')
    ax.set_xlabel(r'$x^+$')
    ax.set_ylabel(r'$z^+$')
    ax.set_title(r'Out-of-plane yaw angle $\theta_W = \arctan\!\left(W / \sqrt{U^2+V^2}\right)$')
    ax.legend(handles=[ibm_patch], fontsize=7)

    # Panel 3: 3D resultant speed + (U,V) streamlines
    ax = axes[2]
    cf3 = ax.contourf(X, Y, spd_3D, levels=100, cmap='viridis')
    ax.streamplot(X, Y, U_u, V_u, color='white', linewidth=0.5, density=1.5, zorder=3)
    ax.fill(xfill, yfill, facecolor='black', zorder=4)
    fig.colorbar(cf3, ax=ax, label=r'$|\mathbf{U}_{3D}| = \sqrt{U^2+V^2+W^2}$')
    ax.set_xlabel(r'$x^+$')
    ax.set_ylabel(r'$z^+$')
    ax.set_title(r'3D resultant speed; streamlines = in-plane $(U,V)$')
    ax.legend(handles=[ibm_patch], fontsize=7)

    if title:
        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
    else:
        plt.tight_layout()
    plt.savefig(savename, dpi=300, bbox_inches='tight')
    plt.show()


def plot2D_streamlines_vorticityZ(x, y, U, V, vorticity, title_prefix, xname, yname, savename, xfill, yfill, resolution):
    """Plots vorticity contours as the background and streamlines in black on top."""

    # Create a uniform grid in the y-direction while keeping x unchanged
    y_uniform = np.linspace(y.min(), y.max(), resolution)  # Uniform y-grid
    X, Y = np.meshgrid(x, y_uniform, indexing='xy')  # Meshgrid

    # Interpolate U, V, and vorticity onto the uniform grid
    U_interp = RectBivariateSpline(y, x, U)
    V_interp = RectBivariateSpline(y, x, V)
    vort_interp = RectBivariateSpline(y, x, vorticity)

    U_uniform = U_interp(y_uniform, x)
    V_uniform = V_interp(y_uniform, x)
    vort_uniform = vort_interp(y_uniform, x)

    # **Enhance contrast using normalization**
    # Clamp so vcenter=0 is always strictly between vmin and vmax (required by TwoSlopeNorm)
    vmin = min(np.min(vorticity), -1e-10)
    vmax = max(np.max(vorticity),  1e-10)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    levels = np.linspace(vmin, vmax, 101)
    
    # **Plotting**
    fig, ax = plt.subplots(figsize=(8, 6))

    # **Plot vorticity contours as the background**
    contour = ax.contourf(X, Y, vort_uniform, levels=levels, cmap='seismic', norm=norm, extend='both')  # Adjust alpha for visibility
    #contour = ax.contourf(X, Y, vort_uniform, levels=100, cmap='coolwarm', norm=norm, alpha=0.8)  # Adjust alpha for visibility

    # **Plot streamlines in black**
    ax.streamplot(X, Y, U_uniform, V_uniform, color='black', linewidth=0.1, density=3)

    # **Add colorbar**
    cbar = fig.colorbar(contour, ax=ax, label="Vorticity")
    # cbar.set_ticks([-500, -250, 0, 250, 500])  # Adjust tick spacing for better clarityy
    cbar.locator = MaxNLocator(nbins=7)
    cbar.update_ticks()

    # **Overlay black IBM region**
    ax.fill(xfill, yfill, facecolor='black', zorder=3)  # Ensures it's on top

    # **Set axes limits starting from 0**
    ax.set_xlim(0, x.max())
    ax.set_ylim(0, y.max())

    # **Titles and Labels**
    ax.set_title(title_prefix)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)

    # **Save and Show Plot**
    plt.savefig(savename, dpi=300)
    plt.show()


# ── PhAvg_rotated per-run plots (moved out of the main file for readability) ──
def plot_wavenumber_field(mfld, savename, title, x_in, y_in, x_oro_in, y_oro_in,
                          mask0, l_in, sponge_j, hill_hgt, fig_dir):
    """Signed vertical-wavenumber field  m(x,z)·l_in  (inner units, fluid only)
    with the IBM solid, crest and sponge markers.  `savename` is a bare filename
    joined onto `fig_dir`."""
    _lim = int(min(sponge_j, np.size(y_in) - 1))
    _z   = y_in[:_lim]
    _w   = (mfld[:_lim, :] * l_in) * mask0[:_lim, :]      # inner-unit, fluid only
    _vmax = float(np.nanpercentile(np.abs(_w), 98))       # robust to node spikes
    _vmax = _vmax if _vmax > 0 else 1.0
    _lv  = np.linspace(-_vmax, _vmax, 200)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    _cf = ax.contourf(x_in, _z, np.clip(_w, -_vmax, _vmax),
                      levels=_lv, cmap='RdBu_r', extend='both')
    ax.fill(x_oro_in, y_oro_in, facecolor='black')        # IBM solid
    ax.axhline(y_in[int(hill_hgt)], color='g', ls='--', lw=0.8, label='crest $h$')
    ax.axhline(y_in[int(sponge_j)], color='m', ls=':',  lw=1.0, label='sponge')
    plt.colorbar(_cf, ax=ax, label=r'$m\,\ell_{in}$  (sign: phase-line tilt)')
    ax.set_xlabel(r'$x^+$'); ax.set_ylabel(r'$z^+$'); ax.set_title(title)
    ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, savename), dpi=300)
    plt.show()


def plot_fig4_budget(zin, zout, C_zx, V_zx, R_zx, T_zx, C_zy, V_zy, R_zy, T_zy,
                     ustar, G, veer_deg, label, fig_dir):
    """Kostelecky & Ansorge (2024) figure-4 panels — PURE plotting.

    Every quantity is COMPUTED BY THE CALLER (PhAvg[_rotated].py, PLOT 32r) and
    passed in, so this function stays reusable for any budget profile set:
      zin, zout            inner (z⁺) / outer (z⁻ = y/u*) wall-normal axes
      (C,V,R,T)_zx, _zy    Coriolis / viscous / Reynolds / total budget terms
      ustar, G             normalisation scales (inner /u*², outer ·10⁻³/G²)
      veer_deg             geostrophic veer (deg), title annotation only
    Panels mirror fig. 4: (a) τ_zx and (b) τ_zy inner; (c,d) the same in outer
    units.  Saves Fig4_momentum_budget_{label}.png in fig_dir.
    """
    u2 = ustar**2; G2 = G**2

    def _panel(ax, x, C, V, R, T, sc, xlim, title, ylab, xlab):
        ax.plot(x, C/sc, color='blue',   label='Coriolis C')
        ax.plot(x, V/sc, color='red',    label='Viscous V')
        ax.plot(x, R/sc, color='orange', label='Reynolds R')
        ax.plot(x, T/sc, color='black',  lw=2, label='Total ⟨τ⟩')
        ax.axhline(0, color='grey', lw=0.5); ax.set_xlim(*xlim)
        ax.set_title(title, fontsize=9); ax.set_ylabel(ylab); ax.set_xlabel(xlab)
        ax.grid(alpha=0.3)

    fig, axs = plt.subplots(2, 2, figsize=(12, 9), dpi=200)
    _panel(axs[0, 0], zin, C_zx, V_zx, R_zx, T_zx, u2, (0, 100),
           f'(a) $\\tau_{{zx}}$ inner — {label}', r'$\langle\tau\rangle^{+}_{zx}$', r'$z^{+}$')
    _panel(axs[0, 1], zin, C_zy, V_zy, R_zy, T_zy, u2, (0, 100),
           '(b) $\\tau_{zy}$ inner', r'$\langle\tau\rangle^{+}_{zy}$', r'$z^{+}$')
    _panel(axs[1, 0], zout, C_zx, V_zx, R_zx, T_zx, G2*1e-3, (0, 1.2),
           '(c) $\\tau_{zx}$ outer', r'$\langle\tau\rangle^{-}_{zx}\cdot10^{-3}$', r'$z^{-}$')
    _panel(axs[1, 1], zout, C_zy, V_zy, R_zy, T_zy, G2*1e-3, (0, 1.2),
           '(d) $\\tau_{zy}$ outer', r'$\langle\tau\rangle^{-}_{zy}\cdot10^{-3}$', r'$z^{-}$')
    axs[0, 0].legend(fontsize=7, loc='upper right')
    fig.suptitle(f'Integrated momentum budget (eq. 4.2) — {label}   '
                 f'[Method-2 $u_*$ = {ustar:.4f}, geostrophic veer = {veer_deg:.1f}°]')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'Fig4_momentum_budget_{label}.png'), dpi=200)
    plt.show()


def plot_eps_slope(x_in, h_in, h_gen_in, s_raw, s_gen, gen_labels, s_opt,
                   scan_w_in, scan_ES, scan_rms, scan_max, w_opt_in,
                   y_in, slope2d, x_oro_in, y_oro_in, savename,
                   fit=None, analytic=None, zlim=None, title=None,
                   plateau_in=None, es_note=None, hk=None):
    """Generalized slope of the IBM indicator eps — PURE plotting (6 panels).

    Every quantity is COMPUTED BY THE CALLER (PhAvg_rotated.py, PLOT 00) from the
    functions.py helpers (eps_surface_height / generalized_slope / eps_slope_scan /
    fit_cosine_surface / eps_generalized_gradient) and passed in, so this routine
    stays reusable for any staircase surface.  Panels:

      (a) surface elevation h⁺(x⁺) — raw staircase, coarse-grained, cosine FIT
      (b) fit residual h − h_fit — what the single cosine does not capture
      (c) Δ-sensitivity: ES, s_rms, max|s| vs Δ⁺ — the small-Δ blow-up, the
          plateau, and the large-Δ roll-off; the chosen Δ* is marked
      (d) slope dh/dx — the raw two-point derivative (spiky, deliberately shown
          off-scale) against the generalized slope at several widths Δ and the
          exact derivative of the fitted cosine
      (e) surface inclination θ = arctan(dh/dx) in RADIANS (right axis: degrees)
      (f) the 2-D generalized slope angle θ = arctan(−∂ₓε̄/∂_zε̄) in RADIANS, in
          the mollified interface band, with the orography outline

    Parameters
    ----------
    x_in, y_in           inner-unit axes (x⁺, z⁺).
    h_in                 (nx,) raw staircase surface elevation, z⁺.
    h_gen_in             (nx,) coarse-grained surface elevation at Δ*, z⁺.
    s_raw                (nx,) two-point cell-resolution slope (dimensionless).
    s_gen, gen_labels    list of (nx,) generalized slopes + their legend labels.
    s_opt                (nx,) the CHOSEN generalized slope (at Δ*) — the result.
    scan_w_in            (nW,) scanned widths Δ in INNER units.
    scan_ES/rms/max      (nW,) the three reductions from functions.eps_slope_scan.
    w_opt_in             chosen Δ* in inner units.
    slope2d              (ny, nx) 2-D generalized slope, NaN outside the band.
    x_oro_in, y_oro_in   orography outline (inner units).
    savename             full output path of the PNG.
    fit                  optional dict from functions.fit_cosine_surface with 'h'
                         and 'resid' already in z⁺ plus 's' and a 'label' string;
                         panel (b) is blanked when None.
    analytic             optional dict of the nominal-geometry reference with keys
                         'h' (nx, z⁺), 's' (nx), 'ES', 's_rms', 's_max'; skipped
                         entirely when None.
    zlim                 z⁺ cap for panel (f) (default: 1.35× the crest height).
    title                figure suptitle.
    plateau_in           optional (Δ⁺_lo, Δ⁺_hi) bracket shaded in panel (c).
    es_note              optional text box for panel (c) (the ES classification).
    hk                   optional steepness 2πh/λ, annotated on panel (e) — where
                         it belongs, since hk = tan θ_max is that panel's maximum.
    """
    fig, axs = plt.subplots(2, 3, figsize=(19, 9), dpi=200)

    # ── (a) surface elevation + cosine fit ───────────────────────────────────
    ax = axs[0, 0]
    ax.plot(x_in, h_in, color='0.55', lw=0.8, drawstyle='steps-mid',
            label=r'raw staircase  $h^+$ (eps)')
    ax.plot(x_in, h_gen_in, color='crimson', lw=1.8,
            label=r'coarse-grained  $\Delta^+=%.0f$' % w_opt_in)
    if fit is not None and fit.get('h') is not None:
        ax.plot(x_in, fit['h'], color='tab:blue', lw=1.4,
                label=fit.get('label', 'cosine fit'))
    if analytic is not None and analytic.get('h') is not None:
        ax.plot(x_in, analytic['h'], color='black', ls=':', lw=1.2,
                label='nominal sinusoid')
    ax.set_xlabel(r'$x^+$'); ax.set_ylabel(r'$z^+$')
    ax.set_title('(a) IBM surface elevation + cosine fit', fontsize=9)
    # 'upper center': a valley puts its crests at BOTH ends, so the mid-top is the
    # only region the curves never enter.
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='upper center', framealpha=0.92)

    # ── (b) cosine-fit residual ──────────────────────────────────────────────
    ax = axs[0, 1]
    if fit is not None and fit.get('resid') is not None:
        _rs = np.asarray(fit['resid'])
        ax.plot(x_in, _rs, color='0.45', lw=0.7, drawstyle='steps-mid',
                label=r'$h^+ - h^+_{\rm fit}$')
        _rr = float(np.sqrt(np.mean(_rs**2)))
        for _s, _l in ((_rr, r'$\pm$rms = %.3f' % _rr), (-_rr, None)):
            ax.axhline(_s, color='crimson', ls='--', lw=1.0, label=_l)
        if fit.get('dz_cell') is not None:      # one grid cell = the quantisation
            ax.axhline(0.5*fit['dz_cell'], color='tab:blue', ls=':', lw=1.0,
                       label=r'$\pm\frac{1}{2}\,\Delta z^+_{\rm cell}$')
            ax.axhline(-0.5*fit['dz_cell'], color='tab:blue', ls=':', lw=1.0)
        ax.axhline(0, color='grey', lw=0.5)
        ax.legend(fontsize=7, loc='upper right', ncol=2)
        if fit.get('note'):
            ax.text(0.02, 0.03, fit['note'], transform=ax.transAxes,
                    ha='left', va='bottom', fontsize=7.5,
                    bbox=dict(fc='white', ec='0.7', alpha=0.85, pad=2.5))
    else:
        ax.text(0.5, 0.5, 'no cosine fit supplied', transform=ax.transAxes,
                ha='center', va='center', fontsize=9, color='0.5')
    ax.set_xlabel(r'$x^+$'); ax.set_ylabel(r'residual  $z^+$')
    ax.set_title('(b) cosine-fit residual — the staircase quantisation', fontsize=9)
    ax.grid(alpha=0.3)

    # ── (d) slope: raw (spiky) vs generalized ────────────────────────────────
    ax = axs[1, 0]
    ax.plot(x_in, s_raw, color='0.72', lw=0.5, alpha=0.55, zorder=1,
            label='raw 2-point (off-scale spikes)')
    _cols = ['tab:blue', 'crimson', 'tab:green', 'tab:purple', 'tab:orange']
    _ref = []
    for _i, (_s, _lb) in enumerate(zip(s_gen, gen_labels)):
        # the under-resolved curve stays thin/faint: it is evidence, not a result
        _small = 'small' in str(_lb)
        ax.plot(x_in, _s, color=_cols[_i % len(_cols)],
                lw=0.9 if _small else 1.8, alpha=0.7 if _small else 1.0,
                zorder=2 if _small else 3, label=_lb)
        _ref.append(_s)
    if fit is not None and fit.get('s') is not None:
        ax.plot(x_in, fit['s'], color='black', ls='--', lw=1.3, zorder=4,
                label=fit.get('label', 'cosine fit') + ' (exact)')
        _ref.append(fit['s'])
    elif analytic is not None and analytic.get('s') is not None:
        ax.plot(x_in, analytic['s'], color='black', ls='--', lw=1.3, zorder=4,
                label='nominal sinusoid')
        _ref.append(analytic['s'])
    # y-range from the MEDIAN of the per-curve maxima, so neither the raw spikes
    # nor a deliberately under-resolved Δ can crush the converged curves flat —
    # they simply run off the top, which is what the panel is showing.
    _m = float(np.nanmedian([np.nanmax(np.abs(np.ravel(_r))) for _r in _ref])) \
         if _ref else 1.0
    _m = _m if _m > 0 else 1.0
    ax.set_ylim(-1.8*_m, 1.8*_m)
    ax.axhline(0, color='grey', lw=0.5)
    ax.set_xlabel(r'$x^+$'); ax.set_ylabel(r'$\mathrm{d}h/\mathrm{d}x$')
    ax.set_title('(d) generalized slope vs the cell-resolution derivative', fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='upper left', framealpha=0.92)

    # ── (e) surface inclination in RADIANS ───────────────────────────────────
    # θ = arctan(dh/dx).  Plotted from the CONVERGED generalized slope (Δ*), so it
    # is spike-free by construction; the cosine fit gives the analytic comparison.
    # Right-hand axis carries the same curve in degrees for readability.
    ax = axs[1, 1]
    _th_opt = np.arctan(np.asarray(s_opt))
    ax.plot(x_in, _th_opt, color='crimson', lw=1.8,
            label=r'generalized  $\Delta^+=%.0f$' % w_opt_in)
    if fit is not None and fit.get('s') is not None:
        ax.plot(x_in, np.arctan(np.asarray(fit['s'])), color='black', ls='--',
                lw=1.3, label=fit.get('label', 'cosine fit') + ' (exact)')
    _thm = float(np.nanmax(np.abs(_th_opt)))
    _thm = _thm if _thm > 0 else 1.0
    ax.axhline(0, color='grey', lw=0.5)
    ax.set_ylim(-1.35*_thm, 1.35*_thm)
    ax.set_xlabel(r'$x^+$')
    ax.set_ylabel(r'$\theta=\arctan(\mathrm{d}h/\mathrm{d}x)$   [rad]')
    ax.set_title(r'(e) surface inclination $\theta$ in radians', fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='upper left', framealpha=0.92)
    _axd = ax.twinx()                                  # same data, degree ticks
    _axd.set_ylim(np.degrees(-1.35*_thm), np.degrees(1.35*_thm))
    _axd.set_ylabel(r'$\theta$  [deg]', fontsize=8)
    _axd.tick_params(labelsize=7)
    # hk = tan θ_max is the third route to the steepness, so this panel is where
    # it belongs: the annotation and the curve maximum are the same statement.
    _tnote = r'$\max|\theta| = %.4f$ rad $= %.2f^\circ$' % (_thm, np.degrees(_thm))
    if hk is not None:
        _tnote += '\n' + r'$hk = 2\pi h/\lambda = \tan\theta_{\max} = %.4f$' % hk
    ax.text(0.98, 0.04, _tnote, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=7.5,
            bbox=dict(fc='white', ec='0.7', alpha=0.85, pad=2.5))

    # ── (c) width sensitivity + plateau ──────────────────────────────────────
    ax = axs[0, 2]
    if plateau_in is not None:                 # the sub-tolerance run = the plateau
        ax.axvspan(plateau_in[0], plateau_in[1], color='tab:green', alpha=0.10,
                   label='plateau')
    ax.loglog(scan_w_in, scan_max, color='tab:purple', marker='.', ms=3,
              label=r'$\max|s_\Delta|$')
    ax.loglog(scan_w_in, scan_rms, color='tab:blue', marker='.', ms=3, lw=2,
              label=r'$s_{\rm rms}$  (selects $\Delta_*$)')
    ax.loglog(scan_w_in, scan_ES, color='crimson', marker='.', ms=3,
              label=r'ES $=\langle|s_\Delta|\rangle$  ($\Delta$-invariant)')
    _hl = fit if (fit is not None and fit.get('ES') is not None) else analytic
    if _hl is not None:
        for _k, _c in (('s_max', 'tab:purple'), ('s_rms', 'tab:blue'), ('ES', 'crimson')):
            if _hl.get(_k) is not None:
                ax.axhline(_hl[_k], color=_c, ls=':', lw=1.0)
    ax.axvline(w_opt_in, color='black', ls='--', lw=1.0,
               label=r'chosen $\Delta^{+}_{*}=%.0f$' % w_opt_in)
    ax.set_xlabel(r'coarse-graining width  $\Delta^+$')
    ax.set_ylabel('slope statistic')
    ax.set_title('(c) resolution dependence — staircase spikes (small $\\Delta$), '
                 'plateau, over-smoothing (large $\\Delta$)', fontsize=9)
    ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=7, loc='lower left')
    if es_note:
        ax.text(0.98, 0.95, es_note, transform=ax.transAxes, ha='right', va='top',
                fontsize=7.5, bbox=dict(fc='white', ec='0.7', alpha=0.85, pad=2.5))

    # ── (f) 2-D generalized slope angle in the interface band, in RADIANS ────
    ax = axs[1, 2]
    # ∇ε̄ is concentrated ON the interface, so the band is only a few cells thick —
    # crop just above the crest or the ribbon shrinks to a line.
    _zc  = float(np.nanmax(h_in)) if np.isfinite(np.nanmax(h_in)) else float(y_in[-1])
    _zl  = zlim if zlim is not None else 1.35*max(_zc, y_in[1])
    _lim = int(np.searchsorted(y_in, _zl)) + 1
    _lim = int(min(max(_lim, 2), np.size(y_in)))
    _ang = np.arctan(slope2d[:_lim, :])                       # radians
    _v   = float(np.nanpercentile(np.abs(_ang), 99)) if np.any(np.isfinite(_ang)) else 1.0
    _v   = _v if _v > 0 else 1.0
    _cf  = ax.contourf(x_in, y_in[:_lim], np.clip(_ang, -_v, _v),
                       levels=np.linspace(-_v, _v, 101), cmap='RdBu_r', extend='both')
    ax.plot(x_oro_in, y_oro_in, color='black', lw=0.8)
    _cb = plt.colorbar(_cf, ax=ax,
                       label=r'$\theta=\arctan(\mathrm{d}h/\mathrm{d}x)$  [rad]')
    _cb.ax.tick_params(labelsize=7)
    ax.set_ylim(0, y_in[_lim - 1])
    ax.set_xlabel(r'$x^+$'); ax.set_ylabel(r'$z^+$')
    ax.set_title(r'(f) 2-D generalized slope from $\nabla\bar\varepsilon$ '
                 '[rad]', fontsize=9)

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(savename, dpi=200)
    plt.show()
