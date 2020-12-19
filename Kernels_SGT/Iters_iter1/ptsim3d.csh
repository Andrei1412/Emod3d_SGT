#! /bin/csh

set BINDIR = ${HOME}/Bin

set STATS = ( AZ_TRO )
set SGTNAME = ( AZ_TRO )
set SLON = ( -116.4257 )
set SLAT = ( 33.5234 )

set RUNS = ( 20011031-m502 20050612-m519 20020102-m401 )
set ELON = (       -116.50       -116.57       -116.43 )
set ELAT = (         33.50         33.53         33.38 )
set EDEP = (          8.00          8.00         11.00 )
set EMOM = (    3.8032e+23    6.9263e+23    1.1546e+22 )
set ESTK = (          48.0          36.0          44.0 )
set EDIP = (          90.0          77.0          90.0 )
set ERAK = (          57.0          32.0          22.0 )

set NTOUT = -1
set NTOUT = 6000
set TSTART = -0.05     # to account for 1/2 width of cos pulse
set TSTART = 0.0

set r = 0
foreach run ( $RUNS )
@ r ++

set OUTDIR = ${run}/Vel
\mkdir -p $OUTDIR

set s = 0
foreach stat ( $STATS )
@ s ++

set SGT_MAIN_GFDIR = ../../SgtCalc/${stat}/SgtFiles

$BINDIR/ptsim3d stat=$stat slon=$SLON[$s] slat=$SLAT[$s] \
                   moment=$EMOM[$r] strike=$ESTK[$r] dip=$EDIP[$r] rake=$ERAK[$r] \
		   elon=$ELON[$r] elat=$ELAT[$r] edep=$EDEP[$r] \
                   sgt_xfile=${SGT_MAIN_GFDIR}/${SGTNAME[$s]}_fx.sgt \
                   sgt_yfile=${SGT_MAIN_GFDIR}/${SGTNAME[$s]}_fy.sgt \
                   sgt_zfile=${SGT_MAIN_GFDIR}/${SGTNAME[$s]}_fz.sgt \
		   ntout=$NTOUT outdir=$OUTDIR

foreach comp ( 000 090 ver )

wcc_header tst=$TSTART infile=$OUTDIR/${stat}.${comp} outfile=$OUTDIR/${stat}.${comp}

end

end

end
