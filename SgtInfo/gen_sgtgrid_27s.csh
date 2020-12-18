#!/bin/csh

#
# gen_sgtgrid.csh
#
# This script determines the target source locations to be used in the SGT computation.
# These locations will be written into a SGT coordinate file (in the subdirectory SgtCords).
# The variable "STATS" specifies the stations for which the SGTs will be computed. One SGT
# coordinate file will be determined for each station lised in "STATS".
#
# SGT coordinate locations are input as (lon,lat,dep), which are then converted to grid
# cooridnate indices for the specified FD model grid. This allows for rapid lookup when
# later accessing the SGTs for computing seismograms. SGT locations can be point source
# EQ locations, finite-fault subfault locations, or just random locations throughout the
# 3D model volume. Pretty much anywhere you think you might have a source point, you should
# generate an SGT location for that point.
#

set OUTDIR = SgtCords
mkdir -p $OUTDIR

#
# The following variables specify the stations for which SGT coordinates will be determined.
#

set STATS = ( NNZ )
set SLON = ( 173.3745 )
set SLAT = ( -41.2102 )

#
# The following variables specify the FD model coordinates
#

set NX = 176
set NY = 176
set NZ = 80
set HH = 2.000
set SUFX = -h${HH}

set MODEL_LAT = -42.09995
set MODEL_LON = 173.09991
set MODEL_ROT = 0.0
set XAZIM = `echo $MODEL_ROT | gawk '{print $1+90.0;}'`

#
# The following variables generate target EQ locations and/or finite-fault subfault locations
# for which SGTs will be computed. All of these target locations are given by (lon,lat,dep)
# with dep in km.
#
# For this example, the first set of variables specifies 3 point EQ locations, which go into the
# file "EQLOC_FILE".
#
# The second set of variables lists the (predetermined from somewhere else) finite-fault files
# for which SGTs will be determined. The format of the finite-fault files is simply (lon,lat,dep)
# for each subfault location.
#
# All of the above files are listed into the "FAULTLIST" file. The code will open each of these
# files sequentially, read the coordinates and determine the location mapping for the SGT
# location in the FD model grid.
#
# Comment lines in these files are specified by a leading "#".  These lines will be skipped.
#

#set FAULTLIST = fault_list.file
set FAULTLIST = NONE

#
# First, generate point EQ locations
#

set EQLOC_FILE = eqloc.txt
set ELON = ( 171.98 174.29 173.99 174.42 171.28 173.04 173.21 173.59 173.01 172.72 172.82 171.69 173.09 174.17 174.26 174.28 174.04 173.65 173.82 173.76 174.28 173.96 174.31 173.62 173.73 174.18 173.98 )
set ELAT = (   -43.18 -41.45 -41.77 -41.50 -43.06 -42.26 -42.39 -42.15 -42.95 -42.67 -42.75 -43.13 -42.96 -41.69 -41.66 -41.67 -41.76 -42.26 -42.27 -42.36 -41.73 -42.31 -41.75 -42.26 -42.24 -41.68 -41.85 )
set EDEP = (    7.00 10.00 6.00 18.00 8.00 11.00 11.00 5.00 4.00 8.00 11.00 8.00 6.00 16.00 18.00 14.00 13.00 6.00 13.00 10.00 16.00 15.00 17.00 17.00 16.00 18.00 11.00 )

echo "# 3 EQ point locations" > $EQLOC_FILE

set l = 0
foreach lon ( $ELON )
@ l ++

echo $lon $ELAT[$l] $EDEP[$l] >> $EQLOC_FILE

end

echo $EQLOC_FILE > $FAULTLIST

#
# Now, add (predetermined) finite-fault files to "FAULTLIST"
#

set FINITE_FAULT_FILES = ( sj1.gsf sj2.gsf )

foreach file ( $FINITE_FAULT_FILES )

echo $file >> $FAULTLIST

end

#
# The following variables specify (optional) input that will generate SGT locations
# at a series of depths and distances from the station location. The depths and distances
# are in groups of increasing values, i.e., denser near the station and coarser further
# away. To turn of this opton either set RADIUS_FILE = NONE, or do not specify a parameter
# entry for "radiusfile" in the parameter list for "gen_sgtgrid"
#

set RADIUS_FILE = NONE

#
# This will turn off the depth & distance option
#

set RADIUS_FILE = NONE

#
# Or using the following will specify SGT locations at depth samplings of ZINC changing at
# depth levels of ZLEV, along with distance samplings of RINC changing at disance levels of RLEV
#

set RADIUS_FILE = sgt.radiusfile

set ZLEV = (  5.0 24.0 60.0 )
set ZINC = (    5   10   25 )

set RLEV = ( 10.0 50.0 100.0 1000.0 )
set RINC = (   10   15    25     50 )

set IX_MIN = 40
set IX_MAX = `echo $NX $IX_MIN | gawk '{printf "%d\n",$1-$2;}'`

set IY_MIN = 40
set IY_MAX = `echo $NY $IY_MIN | gawk '{printf "%d\n",$1-$2;}'`

set IZ_START = 4
set IZ_MAX = 180

echo $#RLEV > $RADIUS_FILE
echo $RLEV >> $RADIUS_FILE
echo $RINC >> $RADIUS_FILE
echo $#ZLEV >> $RADIUS_FILE
echo $ZLEV >> $RADIUS_FILE
echo $ZINC >> $RADIUS_FILE

#
# The following first determines the X,Y grid location of the station, and then runs the code
#

set s = 0
foreach stat ( $STATS )
@ s ++

set SXY = `echo $SLON[$s] $SLAT[$s] | /home/rgraves/Bin/ll2xy mlon=$MODEL_LON mlat=$MODEL_LAT xazim=$XAZIM`
set XSRC = `echo $SXY[1] $HH $NX | gawk '{printf "%d\n",int(0.5*$3 + $1/$2);}'`
set YSRC = `echo $SXY[2] $HH $NY | gawk '{printf "%d\n",int(0.5*$3 + $1/$2);}'`

set OUTFILE = $OUTDIR/${stat}${SUFX}.cordfile

echo $NX $NY $NZ $HH $XSRC $YSRC
echo $IX_MIN $IX_MAX $IY_MIN $IY_MAX $IZ_START $IZ_MAX
echo $OUTFILE $FAULTLIST
echo $MODEL_LON $MODEL_LAT $MODEL_ROT

/home/rgraves/Bin/gen_sgtgrid nx=$NX ny=$NY nz=$NZ h=$HH xsrc=$XSRC ysrc=$YSRC \
	    modellon=$MODEL_LON modellat=$MODEL_LAT modelrot=$MODEL_ROT \
	    faultlist=$FAULTLIST \
	    radiusfile=$RADIUS_FILE \
            ixmin=$IX_MIN ixmax=$IX_MAX iymin=$IY_MIN iymax=$IY_MAX \
	    izstart=$IZ_START izmax=$IZ_MAX \
	    outfile=$OUTFILE

end
