#!/bin/bash
gfortran -I/usr/local/include cyclone.f90 -lnetcdff -o cyclone
mv cyclone ../bin/cyclone
