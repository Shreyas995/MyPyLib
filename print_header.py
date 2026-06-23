#!/usr/bin/env python3
"""
print_header.py -- read the raw header of a tlab flow field file and print it.

Standalone: the header reader is copied in below (no imports from functions.py).

The tlab header is stream binary (no Fortran record markers):
    offset (int32)  = total header size in bytes = 5*4 + n_params*8
    nx     (int32)
    ny     (int32)
    nz     (int32)
    nt     (int32)
    params (float64 * n_params)

Usage:
    python3 print_header.py 220000          # -> flow.220000.1
    python3 print_header.py flow.220000.1   # full file name
    python3 print_header.py 220000 --base flow --comp 2   # -> flow.220000.2
"""
import os
import sys
import argparse
import numpy as np

HEADER_BYTES = 52  # 5 int32 (20) + 4 float64 params (32)


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
                raise ValueError("Offset value is too small, it must be greater than the size of an integer.")

            # Read the grid dimensions and nt
            nx = read_fortran_record(f, np.dtype('<i4'))
            ny = read_fortran_record(f, np.dtype('<i4'))
            nz = read_fortran_record(f, np.dtype('<i4'))
            nt = read_fortran_record(f, np.dtype('<i4'))
            # Calculate the size of params
            remaining_header_size = offset - 5 * sizeofint
            params_size = int(remaining_header_size / sizeofreal)

            # Read params if there are any
            params = []
            if params_size > 0:
                for i in range(params_size):
                    params_record = read_fortran_record(f, np.dtype('<f8'))  # 'f8' for double precision float
                    params.append(params_record)

            return offset, nx, ny, nz, nt, params

    except Exception as e:
        # Print the error message and return a default value
        # print(f'Error reading header: {e}')
        return None, None, None, None, None, None


def main():
    ap = argparse.ArgumentParser(description="Print the raw header of a tlab flow field file.")
    ap.add_argument('target',
                    help="iteration number (e.g. 220000 -> flow.220000.1) "
                         "OR a full file name (e.g. flow.220000.1)")
    ap.add_argument('--base', default='flow',
                    help="filename prefix when target is an iteration number (default: flow)")
    ap.add_argument('--comp', default='1',
                    help="component index when target is an iteration number (default: 1)")
    ap.add_argument('-n', '--nbytes', type=int, default=HEADER_BYTES,
                    help="number of raw header bytes to print (default: 52)")
    args = ap.parse_args()

    # If the target is a plain integer (iteration number) build flow.<iter>.<comp>;
    # otherwise treat it as a full file name / path.
    if args.target.isdigit():
        path = '{}.{}.{}'.format(args.base, args.target, args.comp)
    else:
        path = args.target

    if not os.path.isfile(path):
        sys.exit("error: file not found: {}".format(path))

    # --- raw bytes ---
    with open(path, 'rb') as f:
        raw = f.read(args.nbytes)

    print("file        : {}".format(path))
    print("raw bytes   : {} (requested {})".format(len(raw), args.nbytes))
    print("hex         : {}".format(raw.hex()))

    # --- parsed header ---
    offset, nx, ny, nz, nt, params = read_header(path)
    if offset is None:
        sys.exit("error: read_header failed (not a valid tlab header?)")

    print("offset (B)  : {}".format(offset))
    print("nx          : {}".format(nx))
    print("ny          : {}".format(ny))
    print("nz          : {}".format(nz))
    print("nt          : {}".format(nt))
    print("params      : {}".format(params))


if __name__ == '__main__':
    main()
