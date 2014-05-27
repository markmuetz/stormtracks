#!/bin/bash
gfortran -I/usr/local/include cyclone.f90 -lnetcdff -o cyclone
#gfortran -O0 -Og -I/usr/local/include cyclone_modified.f95 -lnetcdff -o cyclone_modified

# Use to check array bounds.
#gfortran -fcheck=all -I/usr/local/include cyclone_modified.f95 -lnetcdff -o cyclone_modified
#gfortran -I/usr/local/include cyclone_modified.f95 -lnetcdff -o cyclone_modified
gfortran -Wall -I/usr/local/include cyclone_modified.f95 -lnetcdff -o cyclone_modified
mv cyclone_modified ../bin/cyclone_modified
mv cyclone ../bin/cyclone
