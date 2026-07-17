#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
saveresults.py — backward-compatibility shim.

The variable list (``var_names``) and all array/pickle I/O were consolidated into
IO.py.  This module now simply re-exports IO so that legacy callers doing
``from saveresults import *`` (PhAvg.py, PhAvg_v2.py, test_interpolation.py, …)
keep working unchanged.  New code should ``import IO`` directly.

The realpath-based sys.path insert makes ``from IO import *`` resolve even when
this file is reached through a symlink (root-level or per-simulation data dir):
realpath() follows the symlink back to the MyPyLib master where IO.py lives.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))  # MyPyLib (real dir of IO.py)

from IO import *          # noqa: F401,F403  (re-export var_names + write/read/pickle fns)
