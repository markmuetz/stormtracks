!
      program cyclone
!
!     Cyclone detection program initially developed by Kevin Walsh with
!     modifications by Kim Nguyen, Tony Rafter and Debbie Abbs

!     The program reads in 4-D data from a climate model, currently using
!     monthly netCDF files, with data written tw   ice daily on a lat-lon grid. 
!     The data required are u,v and T at 850, 700, 500 and 300 hPa, 
!     the 10m windspeed and MSLP.  This version also uses the topography 
!     field from the model to define a landmask and uses surface skin 
!     temperature to check lows are only formed >26C.

!     This is currently run from detect_TCLV.csh (where the input files 
!     are defined) and requires the file nml.default where the detection 
!     criteria are specified. 

!     Output files are monthly detections, a file of all detections a listing 
!     of the criteria used and the 'relaxflag' file identifying if there has 
!     been a detection at each location in the previous timestep. Each file 
!     is prefixed with the value defined by outfile in the namelist. The
!     'relaxflag' file should be deleted when analysing the first dataset of 
!     a simulation and a new file is written at the completion of the analysis
!     of a months output. 

!     Variable names may need to be altered for different models.  
!     The date stamps will also need to be altered and time increments 
!     need to be changed if the data is more than twice daily 

      use netcdf

      implicit none 
      include 'netcdf.inc'

      integer, parameter :: nvmax=1000 ! Maximum no. of vortices per archive

      integer il,jl,kl,tl,ijk,ij,iq,m,n,i,j,k,iarch,prev,       &
              k850,k700,k500,k300,jv
      integer lonid, latid,levid,timeid,nlon,nlat,nlevs,ntimes,  &
              status,ierr,ier,mode
      integer start(2),count(2),ihr,iyear,iyear2,irecmnth,irecday,  &
              kdate,march,ticker,cyc(nvmax),ix,iy,iz
      integer farch, narch, id,jd
      integer iv,isumt,jmax,jmin,imax,imin,jaround,iaround,      &
              ipoint,jpoint,isum,iwmax,jwmax,jtestmin,jtestmax,  &
              itestmin,itestmax,itest,jtest,ipoint2,jpoint2,     &
              ntest,ips,jps,ilm2,jlm2,itan,jtan,nv
      integer idvt,ncid,pslid
      integer timest(2),timeco(2),timerco(2)

      integer last_ips(nvmax),last_jps(nvmax)

      real pi, rearth, r
      real tcrit,vortcrit,wspcrit,ocscrit,wchkcrit,t300crit,      &
           pmslcrit,radius,wspthresh
      real wsp(nvmax),dlat,dlon,tsum850,tsum700,tsum500,tsum300,  &
           pmslsum,tave850,tave700,tave500,tave300,pmslav,        &
           tanom700,tanom500,wsum850,wsum300,wave850,wave300,     &
           vorttest,psmin,ttest,dist,costhet,sinthet,    &
           u850mag,ratio,xlt,xgt,ylt,ygt,utan,wind_top,wspeed10

      character tdim*10,timorg*100
      character*95 ifile
      character*80 outfile,critfile,allfile,relaxfile
      character cdate*10,ctime*8,cyr*4,cyr2*4,chdate*8,vortex(nvmax)*100
      character *2 cmth,cday,chr,cmin,cv
      character string(nvmax)*100

      logical t300flag,debug, file_there
      logical psflag,wspflag,vorflag,rotate, location

      integer, allocatable :: nxwidth(:,:),nywidth(:,:),     &
                              nxtwidth(:,:),nytwidth(:,:)
      integer, allocatable :: nxv(:,:),nyv(:,:),nxw(:,:),nyw(:,:)
      integer, allocatable :: nvortex(:)

      real, allocatable :: pmsl(:,:),u10(:,:),zs(:,:),tsu(:,:),       &
                           dx(:,:),dy(:,:),xw(:,:),yw(:,:)
      real, allocatable :: uin(:,:,:),vin(:,:,:),tin(:,:,:)
      real, allocatable :: rlat(:),rlon(:),level(:),times(:)
      real, allocatable :: ut(:),vt(:)
      real, allocatable :: tanomsum(:,:),tanom850(:,:),tanom300(:,:),  &
                           pmslanom(:,:),tanomdiff(:,:),vort(:,:),     &
                           wspdchek(:,:),wspeedmx(:,:),utantot(:,:)

      character(len=10), allocatable :: flag(:,:)
      logical, allocatable :: relaxflag(:,:)

      data pi/3.1415926536/,rearth/6371.22e3/,r/287./  
      data debug/.false./
      data start/1,1/
!
!
! weights for tangential velocities for OCS calculation
!
      real weightan(5,5)                ! weights for OCS calculation
      data weightan/0.000,0.031,0.055,0.031,0.000,                        &
                   0.030,0.078,0.056,0.078,0.030,                        &
                   0.053,0.057,0.000,0.057,0.053,                        &
                   0.030,0.078,0.056,0.078,0.030,                        &
                   0.000,0.031,0.055,0.031,0.000/

! ------------- DEFAULT CRITERIA -------------

! default values for logical flags
      data psflag,wspflag,t300flag/.false.,.false.,.true./
! temperature 
      data tcrit/0./
!
! vorticity 
      data vortcrit/3.5e-5/
!
! wind speed 
      data wspcrit/15./
!
! level wind speed 
      data wchkcrit/5./
!
! 300 hPa temp 
      data t300crit/0.5/
!
! --------------- NAMELIST --------------------
!
      namelist/nml/farch,narch,ifile,outfile,tcrit,vortcrit,wspcrit,        &
        ocscrit,wchkcrit,t300crit,pmslcrit,t300flag,radius,debug,id,jd

!  ------------- Start of code -----------------
!
! read in namelist control information
!
      open (unit=5,file='nml.nml')
      read(5,nml)
        close(5)
      write(6,*) ' Output files are '
      write(6,nml)
!
! open input netCDF outfile
!
      status=nf_open(ifile, mode, ncid)
      if(status.ne.0) then
        print *,' cannot open netCDF file; error code ',status
        stop
      end if

!  determine if relaxflag.dat exists.  File should not exist for the first
!  time of simulation but will be there when later files from the simulation
!  are processed

       relaxfile = outfile(1:len_trim(outfile))//'.relaxfile'
       file_there = .false.
       inquire (file = relaxfile, exist=file_there)
       if (.not. file_there) then
            print *,'relaxflag.dat does not exist'
       endif  


! turn OFF fatal netcdf errors
      call ncpopt(0)

!        --------------------------------------
!     Get dimensions (co-ordinate variables)
!        --------------------------------------

!        Pressure levels

      status = nf_inq_dimid(ncid, 'lev', levid)
      status = nf_inq_dimlen(ncid, levid, nlevs)
      status = nf_inq_varid(ncid, 'lev', levid)
      allocate (level(nlevs))
      status = nf_get_vara_real(ncid, levid, 1, nlevs, level)

!        Longitudes

      status = nf_inq_dimid(ncid, 'lon', lonid)
      status = nf_inq_dimlen(ncid, lonid, nlon)
      status = nf_inq_varid(ncid, 'lon', lonid)
      allocate (rlon(nlon))
      status = nf_get_vara_real(ncid, lonid, 1, nlon, rlon)

!        Latitudes

      status = nf_inq_dimid(ncid, 'lat', latid)
      status = nf_inq_dimlen(ncid, latid, nlat)
      status = nf_inq_varid(ncid, 'lat', latid)
      allocate (rlat(nlat))
      status = nf_get_vara_real(ncid, latid, 1, nlat, rlat)



      il = nlon
      jl = nlat
      kl = nlevs

      ij=il*jl
      ijk=il*jl*kl

      count(1) = nlon
      count(2) = nlat

!     Allocate space arrays

      allocate (ut(ijk),vt(ijk))

      allocate (pmsl(il,jl),u10(il,jl),uin(il,jl,kl),vin(il,jl,kl), &
                tin(il,jl,kl),dx(il,jl),dy(il,jl),                  &
                xw(il,jl),yw(il,jl),zs(il,jl),tsu(il,jl),           &
                nxwidth(il,jl),nywidth(il,jl),                      &
                nxtwidth(il,jl),nytwidth(il,jl),                    &
                tanomsum(il,jl), tanom850(il,jl), tanom300(il,jl),  &
                pmslanom(il,jl), tanomdiff(il,jl), vort(il,jl),     &
                wspdchek(il,jl),wspeedmx(il,jl),utantot(il,jl),     &
                flag(il,jl),relaxflag(il,jl))

      ut = 0.
      vt = 0.

!     Date and Time

      status = nf_inq_dimid(ncid, 'time', timeid)
      status = nf_inq_dimlen(ncid, timeid, ntimes)
      allocate (times(ntimes))
      status = nf_get_vara_real(ncid, latid, 1, ntimes, times)

       march = ntimes
       tl = ntimes

       allocate (nvortex(tl), nxv(nvmax,tl),nyv(nvmax,tl),        &
                 nxw(nvmax,tl),nyw(nvmax,tl))

! get time units, and origin (contained in units text)

      call ncagtc(ncid,timeid,'units',timorg,100,ier)

! The following date stamps may need to be changed for other data sets
        cdate = timorg(15:25)
        ctime = timorg(26:34)

        cyr = cdate(1:4)
        cmth = cdate(6:7)
        cday = cdate(9:10)
        chr = ctime(1:2)
        cmin = ctime(4:5)

        print *, cdate, ' ', ctime, ' ', cyr, ' ', cmth, ' ',    &
                 cday, ' ', chr, ' ', cmin

        read(cyr,'(i4.4)')iyear
        write(cyr,'(i4.4)')iyear
        read(cmth,'(i2.2)')irecmnth
        read(cday,'(i2.2)')irecday
        read(chr,'(i2.2)')ihr
        read(cmin,'(i2.2)')imin
        chdate = cyr//cmth//cday
        read(chdate,'(i8.8)')kdate
        iyear2 = iyear + 1
        write(cyr2,'(i4.4)')iyear2

!       Open the output file for TCLV hits.  Save criteria for this
!       set to critfile, based on outfile prefix
        critfile = outfile(1:len_trim(outfile))//'.criteria'
        allfile = outfile(1:len_trim(outfile))//'.all'
        outfile = outfile(1:len_trim(outfile))//'_'//cyr//cmth//'.dat'
        print *,outfile(1:len_trim(outfile))
        print *,critfile(1:len_trim(critfile))
        print *,allfile(1:len_trim(allfile))
 
      open(unit=11,file=outfile,status='unknown',recl=120)
!
!----Comment following lines out if using daily model data----
! Set start time and day of actual cyclone analysis
         if(farch.ne.1)then
          irecday = irecday + int((farch-1)/2)

          if(mod(float(farch-1),2.).eq.0)then
            ihr = 00
          else
            ihr = 12
          endif
         endif
!-------------------------------------------------------------

! turn on fatal netcdf errors
      call ncpopt(NCVERBOS+NCFATAL)

      narch=min(narch,march)

      timest(1) = 1

      timeco(1) = 3
      timeco(2) = 1
      timerco(1) = 2
      timerco(2) = 1
      ix = nlon
      iy = nlat
      iz = nlevs


! Determine levels for 850, 700, 500, and 300 hPa

      do k = 1,nlevs
         if (level(k) .eq. 850)k850 = k
         if (level(k) .eq. 700)k700 = k
         if (level(k) .eq. 500)k500 = k
         if (level(k) .eq. 300)k300 = k
      enddo
      print *,'850, 700, 500, 300 hPa levels ', k850,k700,k500,k300
        

! set up the search area for wind and temperature
! radius has to be in metre 

      do m=2,nlat-1

         dlat=(0.5*(rlat(m+1) - rlat(m-1))/180.)*pi
         do n=2,nlon-1
            dlon=(0.5*(rlon(n+1) - rlon(n-1))/180. )*pi
            dx(n,m) = rearth*cos(rlat(m)/180.*pi) * dlon
            dy(n,m) = rearth*dlat
            xw(n,m)=(radius/dx(n,m))
            yw(n,m)=(radius/dy(n,m))
            nxwidth(n,m)=nint(xw(n,m))
            nywidth(n,m)=nint(yw(n,m))
            if(mod(nxwidth(n,m),2).ne.0)nxwidth(n,m)=nxwidth(n,m)+1
            if(mod(nxwidth(n,m),2).ne.0)nywidth(n,m)=nywidth(n,m)+1
            nytwidth(n,m)=nxwidth(n,m)
            nxtwidth(n,m)=2*nxwidth(n,m)
        enddo
      enddo

      do n=1,nlat
          nxwidth(1,n)=nxwidth(2,n)
          nxtwidth(1,n)=nxtwidth(2,n)
          nywidth(1,n)=nywidth(2,n)
          nytwidth(1,n)=nytwidth(2,n)
          dx(1,n)=dx(2,n)
          dy(1,n)=dy(2,n)
        enddo
        do n=1,nlat
          nxwidth(nlon,n)=nxwidth(nlon-1,n)
          nxtwidth(nlon,n)=nxtwidth(nlon-1,n)
          nywidth(nlon,n)=nywidth(nlon-1,n)
          nytwidth(nlon,n)=nytwidth(nlon-1,n)
          dx(nlon,n)=dx(nlon-1,n)
          dy(nlon,n)=dy(nlon-1,n)
        enddo

        do n=1,nlon
          nxwidth(n,1)=nxwidth(n,2)
          nxtwidth(n,1)=nxtwidth(n,2)
          nywidth(n,1)=nywidth(n,2)
          nytwidth(n,1)=nytwidth(n,2)
          dx(n,1)=dx(n,2)
          dy(n,1)=dy(n,2)
        enddo
        do n=1,nlon
          nxwidth(n,nlat)=nxwidth(n,nlat-1)
          nxtwidth(n,nlat)=nxtwidth(n,nlat-1)
          nywidth(n,nlat)=nywidth(n,nlat-1)
          nytwidth(n,nlat)=nytwidth(n,nlat-1)
          dx(n,nlat)=dx(n,nlat-1)
          dy(n,nlat)=dy(n,nlat-1)
        enddo

!        write the criteria used to the head of each month's output files
!
         write(11,*) 'Cyclone detections for year ',cyr,', month ',cmth
         write(11,*) ' '
         write(11,*) 'CRITERIA USED: '
         write(11,1200) ' Temperature anomaly criterion  ',tcrit
         write(11,1201) vortcrit
         write(11,1200) ' Wind speed criterion           ',wspcrit
         write(11,1200) ' Wind check criterion           ',wchkcrit
         write(11,1200) ' OCS criterion                  ',ocscrit
         write(11,1200) ' T300 criterion                 ',t300crit
         write(11,*)    'T300 flag                      ',t300flag
         write(11,*)    'Radius                         ',radius
         write(11,*)    'MSLP anomaly criterion         ',pmslcrit
         write(11,*) ' '
         write(11,'(A34,A47,A20)')' YYYY MM DD HHHH   LON    LAT     ',&
                     'PMIN     VORTICITY   WIND   SUM   DIFF  OCS    ',&
                     'WMAX_LON    WMAX_LAT'
         write(11,*) ' '

1200     format(A32,F5.1)
1201     format(' Vorticity criterion       ',e10.3)


          if (.not. file_there) then
             open(unit=12,file=critfile,status='replace')
             write(12,*) 'CRITERIA USED: '
             write(12,1200) ' Temperature anomaly criterion  ',tcrit
             write(12,1201) vortcrit
             write(12,1200) ' Wind speed criterion           ',wspcrit
             write(12,1200) ' Wind check criterion           ',wchkcrit
             write(12,1200) ' OCS criterion                  ',ocscrit
             write(12,1200) ' T300 criterion                 ',t300crit
             write(12,*)    'T300 flag                      ',t300flag
             write(12,*)    'Radius                         ',radius
             write(12,*)    'MSLP anomaly criterion         ',pmslcrit
             close (12)
             open(unit=13,file=allfile,status='replace',recl=120)
          else
             open(unit=13,file=allfile,status='old',position='append',recl=120)
          endif

      if (debug) then
        print *,'Debug at ',rlon(id),rlat(jd)
        print *
      endif
!
! set values of points where vortices were detected in the previous timestep
!
          if(file_there)then
             open(unit=7,file=relaxfile)
             read(7,*)relaxflag
             close(7)
          else
           do i=1,il
             do j=1,jl
               relaxflag(i,j) = .false.
             enddo
           enddo
          endif

!     read in surface height

      call histrd1(ncid,iarch,il,jl,'zs',ix,iy,zs)
      wspthresh = wspcrit


! ###########################################################################
! ####################  loop over WHOLE month data  #########################
! ###########################################################################

       do 9000 iarch=farch,narch

        print *, 'date: ',iyear,irecmnth,irecday,ihr
!
! reset arrays for checking for multiple detections of the same
! vortex in the same archive interval
!
         do i=1,nvmax
           nyv(i,iarch)=0
         enddo
         do i=1,nvmax
           nxv(i,iarch)=0
         enddo

          if(iarch.gt.farch.and.nvortex(iarch-1).gt.0)then
             relaxflag = .false.
             do prev=1,nvortex(iarch-1)
                 last_ips(prev) = nxv(prev,iarch-1)
                 last_jps(prev) = nyv(prev,iarch-1)
                 do i=1,il
                  do j=1,jl
                   if(i.ge.(last_ips(prev)-nxwidth(i,j)).and.   &
                      i.le.(last_ips(prev)+nxwidth(i,j)).and.  &
                      j.ge.(last_jps(prev)-nywidth(i,j)).and.  &
                      j.le.(last_jps(prev)+nywidth(i,j))) &
                        relaxflag(i,j) = .true.
                        
! where N is the number of the cyclone in the last timestep(s) around which
! this area is being calculated
                  enddo
                 enddo
             enddo
          endif
!
! set variable arrays for this timestep
!
         call histrd4(ncid,iarch,il,jl,kl,'temp',ix,iy,tin)
         call histrd4(ncid,iarch,il,jl,kl,'u',ix,iy,uin)
         call histrd4(ncid,iarch,il,jl,kl,'v',ix,iy,vin)

         call histrd1(ncid,iarch,il,jl,'u10',ix,iy,u10)
         call histrd1(ncid,iarch,il,jl,'psl',ix,iy,pmsl)
         call histrd1(ncid,iarch,il,jl,'tsu',ix,iy,tsu)

!        If psl is NOT already in pascals then convert to pascal 
!        otherwise comment out
!
         do j=1,jl
            do i=1,il
               pmsl(i,j) = 100.*pmsl(i,j)
            enddo         
         enddo         
!
!        calculate relative vorticity; here use fourth-order accurate
!        approximation

         call vort5(il,jl,uin,vin,dx,dy,vort)
!
!        calculate temperature anomalies for use later
!
         do j=1,jl
           do i=1,il
              tsum850 = 0.
              tsum700 = 0.
              tsum500 = 0.
              tsum300 = 0.
          
              isumt=0
              jmax = min(j+nytwidth(i,j),jl)
              jmax = max(jmax,2*nytwidth(i,j)+1)
              jmin = max(j-nytwidth(i,j),1)
              jmin = min(jmin,jl-2*nytwidth(i,j))
              imax = min(i+nxtwidth(i,j),il)
              imax = max(imax,2*nxtwidth(i,j)+1)
              imin = max(i-nxtwidth(i,j),1)
              imin = min(imin,il-2*nxtwidth(i,j))

              do jaround=jmin,jmax
                do iaround=imin,imax
                  isumt = isumt + 1
                  ipoint = iaround
                  jpoint = jaround
!
!                 calculate mean of temperatures at each level in this region
!
                  tsum850 = tsum850 + tin(ipoint,jpoint,k850)
                  tsum700 = tsum700 + tin(ipoint,jpoint,k700)
                  tsum500 = tsum500 + tin(ipoint,jpoint,k500)
                  tsum300 = tsum300 + tin(ipoint,jpoint,k300)
                end do !iaround=imin,imax
              end do !jaround=jmin,jmax

              tave850 = tsum850/(1.*isumt)
              tave700 = tsum700/(1.*isumt)
              tave500 = tsum500/(1.*isumt)
              tave300 = tsum300/(1.*isumt)

              tanom850(i,j) = tin(i,j,k850) - tave850
              tanom700 = tin(i,j,k700) - tave700
              tanom500 = tin(i,j,k500) - tave500
              tanom300(i,j) = tin(i,j,k300) - tave300
              tanomdiff(i,j) = tanom300(i,j) - tanom850(i,j)
              tanomsum(i,j) = tanom700 + tanom500 + tanom300(i,j)
              if (debug .and. i.eq.id .and. j.eq.jd) then
                     print *, 'anom diff, anom sum',     &
                              tanomdiff(i,j), tanomsum(i,j)
                     print *, 'anom 700, 500, 300',      &
                              tanom700, tanom500, tanom300(i,j)
              endif

           end do      !(i=1,il)
         end do        !(j=1,jl)

         do j=1,jl
           do i=1,il
             wsum850 = 0.
             wsum300 = 0.
             pmslsum = 0
!
!            also calculate mean wind speed at 850 and 300 hPa
!            and mslp anomaly
!
             isum = 0
             jmax = min(j+nywidth(i,j),jl)
             jmax = max(jmax,2*nywidth(i,j)+1)
             jmin = max(j-nywidth(i,j),1)
             jmin = min(jmin,jl-2*nywidth(i,j))
             imax = min(i+nxwidth(i,j),il)
             imax = max(imax,2*nxwidth(i,j)+1)
             imin = max(i-nxwidth(i,j),1)
             imin = min(imin,il-2*nxwidth(i,j))

             do jaround=jmin,jmax
               do iaround=imin,imax
                  isum = isum + 1
                  ipoint = iaround
                  jpoint = jaround
                  wsum850 = wsum850+sqrt(uin(ipoint,jpoint,k850)**2+ &
                           vin(ipoint,jpoint,k850)**2)
                  wsum300 = wsum300+sqrt(uin(ipoint,jpoint,k300)**2+ &
                           vin(ipoint,jpoint,k300)**2)
                  pmslsum = pmslsum + pmsl(ipoint,jpoint)
               end do
             end do

             wave850 = wsum850/(1.*isum)
             wave300 = wsum300/(1.*isum)
             pmslav = pmslsum/(1.*isum)
!
!            use averages instead of maximum wind speed location
!
             wspdchek(i,j) = wave850 - wave300
             pmslanom(i,j) = pmsl(i,j) - pmslav
             if (debug .and. i.eq.id .and. j.eq.jd) then
                print *,'wind av 850 & 300 ', wave850, wave300,    &
                        relaxflag(i,j)
                print *,'pmslanom ',pmslanom(i,j)
             endif

           end do    !i=1,il
         enddo       !j=1,jl
!
!        set number of vortices to zero
!
         nvortex(iarch) = 0

!        loop over all points - do not allow TC formation to occur 
!        poleward of 30deg or over land.

         do j=1,jl        
           do i=1,il        
!              if (relaxflag(i,j)) wspthresh = 0.8*wspcrit
              if (abs(rlat(j)).le.30. .or. relaxflag(i,j)) then

!                   Want to skip outside edges of domain...
!
                   if(j.le.nywidth(i,j).or.j.ge.jl-nywidth(i,j))then
                        goto 990
                   endif
                   if(i.le.nxwidth(i,j).or.i.ge.il-nxwidth(i,j))then
                        goto 990
                   endif
!
!              set logical flags to false
!
               psflag = .false.
               wspflag = .false.
               vorflag = .false.
               rotate = .false.
               location = .false.
!     
!              test vorticity criterion; note sign reversal in 
!              southern Hemisphere because cyclones rotate the other way!
!
               if(rlat(j).lt.0.) then
                     vorttest = -vort(i,j)
               elseif(rlat(j).gt.0) then
                     vorttest = vort(i,j)
               endif                          !(rlat(j).lt.0.)

               if(vorttest.gt.vortcrit) then
                 vorflag = .true.
                 if(debug .and. i.eq.id .and. j.eq.jd)    &
                     print *, vortcrit,vorttest,i,j,rlat(j),rlon(i)
!
!                Now test velocity and minimum pressure criteria
!
!                Check to see if there is a minimum pressure in 
!                this region i.e. that there is a point 
!                which has a lower pressure than 
!                any of the surrounding points
!
!                then search in area surrounding vort. min 
!                for the minimum pressure. Pmin <= 1005 hPa. 
!
                 ntest = 0
                 psmin = 100500.
            
                 jtestmin = j-nywidth(i,j)+1
                 jtestmax = j+nywidth(i,j)-1
                 itestmin = i-nxwidth(i,j)+1
                 itestmax = i+nxwidth(i,j)-1
            
                 do jtest=jtestmin,jtestmax
                    do itest=itestmin,itestmax
                       if(pmsl(itest,jtest) .lt.psmin) then
                         psmin = pmsl(itest,jtest)
                         ips = itest
                         jps = jtest
                         psflag = .true.
                       end if
                    enddo      !itest=itestmin,itestmax
                 enddo         !jtest=jtestmin,jtestmax

!                require the pressure minimum to be over the sea (zs .gt. 0.5) and in a region of
!                SST higher than 26C.
 
                 if (tsu(ips,jps).ge.299.15 .and. zs(ips,jps).le.0.5)   &
                         location = .true.
!
!                further confirm that this is an actual 
!                pressure minimum i.e. that
!                all points around it are of higher pressure

                 do jtest=jps-1,jps+1
                   do itest=ips-1,ips+1
                       if(pmsl(itest,jtest).lt.psmin) then
                         psflag = .false.
                       end if
                    enddo       !itest=ipsmin-1,ipsmin+1
                 enddo          !jtest=jpsmin-1,jpsmin+1
!
!                if the pressure is a minimum at ipoint, jpoint
!                set psflag to true and set analysis 
!                points ips and jps for 
!                further criteria i.e. minimum pressure point taken to be 
!                centre of storm as in Bengtsson
!                
                 if (psflag) then
!                  Set flag if there is rotation
!                  modified to +/- 2 TR 22-12-05
                   if (uin(ips,jps-2,k850)/uin(ips,jps+2,k850) .lt. 0 .and.  &
                       vin(ips-2,jps,k850)/vin(ips+2,jps,k850) .lt. 0) then
                       rotate = .true.
                   endif 
                 endif

!                Require that pmsl(ips,jps) be  pmslcrit hPa lower than 
!                surrounding av.  If not, then reset psflag. 
                 if (pmslanom(ips,jps) .gt. pmslcrit*100.) psflag = .false.
!
!                double-check that ips and jps have not been set outside the
!                permitted detection bounds; in other words, the OCS calculation
!                below needs at least two points on either side
!
                 ilm2 = il - 2
                 jlm2 = jl - 2
                 if(ips.lt.3 .or. ips.gt.ilm2 .or. jps.lt.3 .or.       &
                   jps.gt.jlm2) then
                   psflag=.false.
                 end if
!
!                Find maximum wind speed and location in this region
!                Put all wind speeds in this region in an array 
!                corresponding to the maximum w/speed at 10m in region 
!                surrounding the vortex 
!
                 ntest = 0
                 wspeedmx(i,j) = 0.
                 do jaround=jps-nywidth(i,j),jps+nywidth(i,j)
                    do iaround=ips-nxwidth(i,j),ips+nxwidth(i,j)
                       if(u10(iaround,jaround).gt.wspeedmx(i,j)) then
                         wspeedmx(i,j) = u10(iaround,jaround)
                         iwmax = iaround
                         jwmax = jaround
                         if(debug .and. i.eq.id .and. j.eq.jd) then
                            print *,u10(iaround,jaround),wspeedmx(i,j),  &
                                    rlon(iaround),rlat(jaround)
                         endif
                       end if
                    enddo         !iaround=i-nxwidth,i+nxwidth
                 enddo            !jaround=j-nywidth,j+nywidth

                 wind_top=wspeedmx(i,j)

                 if(wspeedmx(i,j).ge.wspthresh) then
                    wspflag = .true.
                    if(debug .and. i.eq.id .and. j.eq.jd)            &
                        print *,'windspeed criterion true ',         &
                        wspeedmx(i,j),iwmax,jwmax
                 end if

                 if (debug .and. ips.eq.id .and. jps.eq.jd) then
                        print *,i,j,psflag,wspflag,vorflag,rotate
                        print *
                        print *,vorttest,i,j,rlon(i),rlat(j)
                        print *,wspeedmx(i,j),iwmax,jwmax,rlon(iwmax),  &
                                rlat(jwmax)
                        print *,psmin,ips,jps,rlon(ips),rlat(jps)
                        print *,tanomsum(ips,jps),tcrit,relaxflag(i,j)
                        print *,pmslanom(ips,jps)
                 endif
!
!                temperature criterion calculation            
!            
                 if(psflag .and. wspflag .and. vorflag .and. rotate) then

                   if(location .or. relaxflag(i,j)) then
                     if(tanomsum(ips,jps).gt.tcrit.or. relaxflag(i,j)) then

!                       if t300flag is true, compare 300 hPa temperature 
!                       anomaly and 850 hpa temp. anom. If not true, 
!                       compare to specified t300 anomaly criterion
!
                        if(t300flag) then
                           ttest = tanom850(ips,jps)
                        else
                           ttest = t300crit
                        end if    

                        if(debug .and. i.eq.id .and. j.eq.jd) then
                          print*,'ttest,tanom300= ',    &
                                  ttest,tanom300(ips,jps)
                        endif
                        if(tanom300(ips,jps).ge.ttest.or.     &
                           relaxflag(i,j)) then
!
!                        relative windspeed criterion; test wind 
!                        mean windspeed at 850 versus 300 hPa
!                        here use wspdchek at point of maximum wind speed
!
                        if(debug .and. i.eq.id .and. j.eq.jd) then
                          print *,'Entering windspeed criterion check '
                          print*,'wspdchek= ',wspdchek(ips,jps)
                        endif
                        if(wspdchek(ips,jps).ge.wchkcrit.or.     &
                           relaxflag(i,j)) then
                           if (debug .and. i.eq.id .and. j.eq.jd) then
                               print *,'Enter OCS calculation '
                           endif
!
!                            now calculate OCS
!
!                            first calculate tangential wind speed 
!                            at the required points
!
                             utantot(ips,jps)=0.
                             do itan=ips-2,ips+2
                                do jtan=jps-2,jps+2
!
!                                  calculate angle of this point 
!                                  relative to centre of storm
!
                                  if(itan.eq.ips .and. jtan.eq.jps) then
!
!                                     centre of storm; no tangential velocity
!   
                                      utan = 0.
          
                                  else
                                                  
                                      dist = sqrt((1.*(itan-ips))**2 +  &
                                           (1.*(jtan-jps))**2)
                        
                                      costhet = (itan-ips)/dist
                                      sinthet = (jtan-jps)/dist
!
!                                     tangential velocity at this point; 
!                                     here, wind directions are taken from the 
!                                     850 hPa winds, while speeds are taken 
!                                     from 10m wind speeds; include weights
!                                     to calculate mean OCS strength
!
                                      u850mag = sqrt(uin(itan,jtan,k850)**2   &
                                             + vin(itan,jtan,k850)**2)

                                      ratio = u10(itan,jtan)/u850mag

                                      utan = ratio*(uin(itan,jtan,k850)*      &
                                     sinthet-vin(itan,jtan,k850)*costhet)     &
                                     *weightan(itan-ips+3,jtan-jps+3)
                                  end if
!
!                                 sign convention for hemispheres
!                        
                                  if(rlat(jtan).ge.0) then
                                     utantot(ips,jps) = utantot(ips,jps) - utan
                                  else
                                     utantot(ips,jps) = utantot(ips,jps) + utan
                                  end if  

                                enddo      !jtan=jps-2,jps+2
                             enddo         !itan=ips-2,ips+2
!
!
!                            test for OCS strength criterion
!
                             if(utantot(ips,jps).ge.ocscrit.or. &
                                relaxflag(i,j))then
!
!                              recalculate the maximum wind speed 
!                              to reflect the maximum wind within 
!                              (nxwidth,nywidth) grid point of 
!                              the centre of the storm, not just the first
!                              point which verifies the criteria
!     
                               ntest = 0
                               wspeedmx(ips,jps) = 0.
                               do jaround=jps-(nywidth(i,j)+1),jps+(nywidth(i,j)-1)
                                 do iaround=ips-(nxwidth(i,j)+1),ips+(nxwidth(i,j)-1)
                                   ntest = ntest + 1

                                   ipoint = iaround
                                   jpoint = jaround 
                                   if(iaround.gt.il)ipoint=iaround-il   
                                   if(iaround.lt.1)ipoint=iaround+il

                                   wspeed10 = u10(ipoint,jpoint)

                                   if(wspeed10.gt.wspeedmx(ips,jps)) then
                                     wspeedmx(ips,jps) = wspeed10
                                     iwmax = ipoint
                                     jwmax = jpoint
                                   end if
                                 enddo  !iaround=ips-nxwidth,ips+nxwidth
                               enddo    !jaround=jps-nywidth,jps+nywidth
                               wspeedmx(ips,jps)=max(wind_top,wspeedmx(ips,jps))

!
!                                count the vortices
!
!                                check to be sure that this vortex has 
!                                NOT already been recorded.  Ignore vortices 
!                                that are identical or
!                                within one grid point of an existing vortex
                                 do nv=1,nvortex(iarch)
                                   if(ips.ge.nxv(nv,iarch)-1 .and.   &
                                      ips.le.nxv(nv,iarch)+1 .and.   &
                                      jps.ge.nyv(nv,iarch)-1 .and.   &
                                      jps.le.nyv(nv,iarch)+1 ) &
                                   goto 990
                                 enddo

                                 nvortex(iarch) = nvortex(iarch) + 1

                        write(string(nvortex(iarch)),100)cyr,irecmnth, &
                                 irecday,ihr,cmin,rlon(ips),rlat(jps), &
                                 pmsl(ips,jps)/100.,vort(i,j), &
                                 wspeedmx(ips,jps),tanomsum(ips,jps),  &
                                 tanomdiff(ips,jps),utantot(ips,jps),  &
                                 rlon(iwmax),rlat(jwmax)
        
                                 wsp(nvortex(iarch))=wspeedmx(ips,jps)
                                 if(nvortex(iarch).gt.nvmax) then
                                   write(11,*) ' Too many vortices '
                                   print *, ' Too many vortices '
                                   write(11,*) ' Increase nvmax '
                                   stop
                                 end if

                                 nxv(nvortex(iarch),iarch) = ips
                                 nyv(nvortex(iarch),iarch) = jps
                                 nxw(nvortex(iarch),iarch) = iwmax
                                 nyw(nvortex(iarch),iarch) = jwmax
!
                             endif    !(utantot(ips,jps).ge.ocscrit) then
                         endif        !(wspdchek(iwmax,jwmax).ge.wchkcrit) 
                        endif         !(tanom300(ips,jps).ge.ttest) then
                     endif            !(tanomsum(ips,jps).gt.tcrit) then
                   endif              !(location) then
                 endif                !(psflag .and. wspflag) then
               endif                  !(vorttest.gt. vortcrit) then
990            continue
            endif                      !(abs(rlat(j)).le.70.) then
           enddo                     !i=nxwidth+1,il-nxwidth
         enddo                        !j=nywidth+1,jl-nywidth
100      format(A4,I3.2,2I3.2,A2,2F7.1,F9.1,E13.3,F7.1,3F6.1,2F11.1)
101      format(a30,2(f8.3,1x),2(1x,i3),a6,e10.3)

102      format(' 10m wind speed ',f4.1,' temp anomaly ', f4.1,        &
        ' 300 hPa tanom ',f4.1,' OCS ',f4.1)

!        write out vortices for these locations.
!
            cyc=1
            if(nvortex(iarch).gt.1)then
             ticker=0
             do j=1,nvortex(iarch)-1
! looking at nwwidth gridpoints n/s/e/w of (ips,jps)
! don't want 2 vortices within radius of each other - choose the strongest
! and the characteristics associated with that vortex
                if(cyc(j).ne.0)then
                xlt=nxv(j,iarch)-nxwidth(nxv(j,iarch),nyv(j,iarch))
                xgt=nxv(j,iarch)+nxwidth(nxv(j,iarch),nyv(j,iarch))
                ylt=nyv(j,iarch)-nywidth(nxv(j,iarch),nyv(j,iarch))
                ygt=nyv(j,iarch)+nywidth(nxv(j,iarch),nyv(j,iarch))
                nv=nvortex(iarch)
! for each of the other vortices at this timestep...
                do i=j+1,nv
                 do k=xlt,xgt
                  do m=ylt,ygt
                   if(k.eq.nxv(i,iarch).and.m.eq.nyv(i,iarch).and. &
                      cyc(i).ne.0)then
                     if (vort(nxv(i,iarch),nyv(i,iarch)) .lt.     &
                         vort(nxv(j,iarch),nyv(j,iarch)))   then
                         iv = nxv(i,iarch)
                         jv = nyv(i,iarch)
                         rlon(nxv(j,iarch))=rlon(iv)
                         rlat(nxv(j,iarch))=rlat(iv)
                         pmsl(nxv(j,iarch),nyv(j,iarch))=pmsl(iv,jv)
                         vort(nxv(j,iarch),nyv(j,iarch))=vort(iv,jv)
                         wspeedmx(nxv(j,iarch),nyv(j,iarch))=wspeedmx(iv,jv)
                         tanomsum(nxv(j,iarch),nyv(j,iarch))=tanomsum(iv,jv)
                         tanomdiff(nxv(j,iarch),nyv(j,iarch))=tanomdiff(iv,jv)
                         utantot(nxv(j,iarch),nyv(j,iarch))=utantot(iv,jv)
                         nxw(j,iarch) = nxw(i,iarch)
                         nyw(j,iarch) = nyw(i,iarch)
                     endif

                     write(string(j),100)cyr,irecmnth,irecday,ihr,cmin,&
                          rlon(nxv(j,iarch)),rlat(nyv(j,iarch)),       &
                          pmsl(nxv(j,iarch),nyv(j,iarch))/100.,        &
                          vort(nxv(j,iarch),nyv(j,iarch)),             &
                          wspeedmx(nxv(j,iarch),nyv(j,iarch)),         &
                          tanomsum(nxv(j,iarch),nyv(j,iarch)),         &
                          tanomdiff(nxv(j,iarch),nyv(j,iarch)),        &
                          utantot(nxv(j,iarch),nyv(j,iarch)),          &
                          rlon(nxw(j,iarch)),rlat(nyw(j,iarch))
                     write(string(i),*)"0"
                     cyc(i)=0
                   endif
                  enddo ! m=ylt,ygt
                 enddo ! k=xlt,xgt
                enddo ! i=j+1,nvortex(iarch)
                endif ! cyc(j).ne.0
             enddo ! j=1,nvortex(iarch)-1
            endif ! (nvortex(iarch).gt.1)

            do i=1,nvortex(iarch)
              if(cyc(i).ne.0)then
                  write(11,*)string(i)(1:len_trim(string(i)))
                  write(13,*)string(i)(1:len_trim(string(i)))
                endif
            enddo

           if (iarch .lt. narch) then
! increment date and time for this timestep
!----Comment following lines out if using daily model data----
! If there are more than 2 times daily this will also need to be altered
! e. 4xdaily if(ihr.eq.0)then ihr=06 elseif(ihr.eq.6)then ihr=12 etc
           if(ihr.eq.0)then
             ihr=12
           else
             ihr=00
!-------------------------------------------------------------
             irecday=irecday+1
             if((irecmnth.eq.1.or.irecmnth.eq.3.or.irecmnth.eq.5.or.   &
                irecmnth.eq.7.or.irecmnth.eq.8.or.irecmnth.eq.10.or.   &
                irecmnth.eq.12).and.irecday.eq.32)then
!               Changing month
                 irecday=1
                 irecmnth=irecmnth+1
                 if(irecmnth.eq.13)then
!                 Changing year
                   iyear=iyear+1
                   write(cyr,'(i4.4)')iyear
                   irecmnth=1
                 endif
!                print *, 'date: ',iyear,irecmnth,irecday,ihr
             elseif((irecmnth.eq.4.or.irecmnth.eq.6.or.irecmnth.eq.11) &
                    .and.irecday.eq.31)then
!                Changing month
                 irecday=1
                 irecmnth=irecmnth+1
             elseif(irecmnth.eq.2.and.irecday.eq.29)then
!                Changing month
                 irecday=1
                 irecmnth=irecmnth+1                 
             endif
           endif

           endif    ! if (iarch .lt. narch) then
 

9000  continue      !iarch=1,narch

!        Set relaxflag ready for next file.
         relaxflag = .false.
          if(nvortex(narch).gt.0)then
             
             do prev=1,nvortex(narch)
                 last_ips(prev) = nxv(prev,narch)
                 last_jps(prev) = nyv(prev,narch)
                 do i=1,il
                  do j=1,jl
                   if(i.ge.(last_ips(prev)-nxwidth(i,j)).and.   &
                      i.le.(last_ips(prev)+nxwidth(i,j)).and.  &
                      j.ge.(last_jps(prev)-nywidth(i,j)).and.  &
                      j.le.(last_jps(prev)+nywidth(i,j))) &
                        relaxflag(i,j) = .true.
                  enddo
                 enddo
             enddo
          endif
!

        open(unit=13,file=relaxfile,status='unknown')
        write(13,*)relaxflag
        close(13)
!           enddo

      call ncclos(ncid,ierr)
      print *,'Detections completed for ', cyr//cmth

      end

!*********************************************************************       

      subroutine vort5(nx,ny,u,v,dx,dy,vort)
       
      real u(nx,ny),v(nx,ny),dx(nx,ny),dy(nx,ny),vort(nx,ny)
!
! calculate vorticities according to fourth-order accurate method
! from P.582 of "Mesoscale Meteorology and Forecasting", ed Ray.
!
222   format(a10,10(10(1x,f6.2),/))

      do j=3,ny-2
        do i=3,nx-2

          deltax = dx(i,j)
          deltay = dy(i,j)

          dudy1 = 2.*(u(i,j+1) - u(i,j-1))/(3.*deltay)
          dudy2 = (u(i,j+2) - u(i,j-2))/(12.*deltay)
          dudy = dudy1 - dudy2
          dvdx1 = 2.*(v(i+1,j) - v(i-1,j))/(3.*deltax)
          dvdx2 = (v(i+2,j) - v(i-2,j))/(12.*deltax)
          dvdx = dvdx1 - dvdx2
          vort(i,j) = dvdx - dudy
        end do
      end do  
!     print 223,'vort5:vort= ',((vort(i,j),i=30,40),j=25,35)
223   format(a10,15(6(1x,E12.5),/))
      return
      end 

!*********************************************************************       
 
      subroutine histrd1(histid,iarch,il,jl,name,ix,iy,var)


      character name*(*)

      integer start(3),count(3)

      real var(il,jl)

      ix=il
      iy=jl
      start(1) = 1
      start(2) = 1
      start(3) = iarch

      count(1) = ix
      count(2) = iy
      count(3) = 1

! read data
      id = ncvid(histid,name,ierr)
      call ncvgt(histid,id,start,count,var,ierr)


      return ! histrd1
      end
!***************************************************************************
      subroutine histrd4(histid,iarch,il,jl,kl,name,ix,iy,var)

      character name*(*)

      integer start(4),count(4),n
      real subvar(il,jl)

      real var(il,jl,kl)

      start(1) = 1
      start(2) = 1
      start(3) = 1
      start(4) = iarch

      count(1) = il
      count(2) = jl
      count(3) = 1
      count(4) = 1

! read data
      id = ncvid(histid,name,ierr)
      do n=1,kl
          start(3)=n
          k=n

        call ncvgt(histid,id,start,count,subvar,ierr)

        do j=1,jl
          do i=1,il
             var(i,j,k)=subvar(i,j)
          end do
        end do

      end do

      return ! histrd4
      end


      
