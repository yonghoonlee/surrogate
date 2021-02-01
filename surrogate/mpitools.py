#!/usr/bin/env python
# coding: utf-8
# Copyright (c) 2021 Yong Hoon Lee

import os
import sys

def under_mpirun():
    """Return True if executed under mpirun."""
    for name in os.environ.keys():
        if name == "OMPI_COMM_WORLD_RANK" or \
           name == "MPIEXEC_HOSTNAME" or \
           name.startswith("MPIR_") or \
           name.startswith("MPICH_"):
            return True
    return False

if under_mpirun():
    from mpi4py import MPI
    def debug(*msg):
        newmsg = ["%d: " % MPI.COMM_WORLD.rank] + list(msg)
        for m in newmsg:
            sys.stdout.write("%s " % m)
        sys.stdout.write("\n")
        sys.stdout.flush()
else:
    MPI = None