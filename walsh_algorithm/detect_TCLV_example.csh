#!/bin/csh

# This script reads in the files and creates detection files.

# Requires the compiled fortran routine cyclone.f90 (cyclone.o) and the 
# namelist file nml.default

# Directories, file names and location of compiled cyclone.f90 code will
# need to be altered 
set WALSHDIR = walsh_algorithm
set BINDIR = bin
set INDIR = data/TCMIP_algorithm/example_tclv
set OUTDIR = output
set fname  = example
set outfile = tclv_out
set y = 1999

#  Only remove following files for analysis from beginning of simulation
#  Comment out for an analysis that is restarted.
#rm ${outfile}*
#rm tclv_out.relaxfile

if ! $?CYCDET then
    echo "Need to set CYCDET"
    exit 1
endif

cd ${CYCDET}

while ($y <= 1999)
      echo 'Retrieving files for ' ${y}
      ln -fs ${CYCDET}/${INDIR}/${fname}_${y}*.nc ${BINDIR}/
      echo 'Starting detections for ' ${y}
      foreach m (01 02)
         sed  "s/prefix/'${outfile}'/" ${WALSHDIR}/nml.default > ${BINDIR}/tmp.nml
         sed  "s/fname/'${fname}_${y}${m}.nc'/" ${BINDIR}/tmp.nml > ${BINDIR}/nml.nml
	 cd ${BINDIR}
         ./cyclone > ../${OUTDIR}/junk.log
         tail -1 ../${OUTDIR}/junk.log >> ../${OUTDIR}/$outfile.log
         echo `tail -1 ../${OUTDIR}/junk.log `
         cp tclv_out.relaxfile ../${OUTDIR}/tclv_out.relaxfile_${y}${m}
	 cd ..
      end
      #rm ${fname}_${y}*.nc .
      @ y++
end
#rm junk.log tmp.nml
       
