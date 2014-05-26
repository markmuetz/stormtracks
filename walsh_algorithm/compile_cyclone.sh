#!/bin/bash
gfortran -I/usr/local/include cyclone.f90 -lnetcdff -o cyclone
gfortran -I/usr/local/include cyclone_modified.f90 -lnetcdff -o cyclone_modified
mv cyclone ../bin/cyclone
mv cyclone_modified ../bin/cyclone_modified
