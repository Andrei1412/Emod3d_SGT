#! /bin/csh

set BINDIR = ${HOME}/Bin

set STATS = ( AZ_TRO )
set SGTNAME = ( AZ_TRO )
set SLON = ( -116.4257 )
set SLAT = ( 33.5234 )

set RUNS = ( 20011031-m502 20050612-m519 20020102-m401 )

set SRFDIR = ./Srf

set NTOUT = 6000
set TSTART = -0.05     # to account for 1/2 width of cos pulse

set r = 0
foreach run ( $RUNS )
@ r ++

set OUTDIR = ${run}/Vel
\mkdir -p $OUTDIR

set s = 0
foreach stat ( $STATS )
@ s ++

set SGT_MAIN_GFDIR = ../../SgtCalc/${stat}/SgtFiles

$BINDIR/jbsim3d stat=$stat slon=$SLON[$s] slat=$SLAT[$s] \
                   rupmodfile=${SRFDIR}/${run}.srf \
                   sgt_xfile=${SGT_MAIN_GFDIR}/${SGTNAME[$s]}_fx.sgt \
                   sgt_yfile=${SGT_MAIN_GFDIR}/${SGTNAME[$s]}_fy.sgt \
                   sgt_zfile=${SGT_MAIN_GFDIR}/${SGTNAME[$s]}_fz.sgt \
		   ntout=$NTOUT outdir=$OUTDIR

foreach comp ( 000 090 ver )

wcc_header tst=$TSTART infile=$OUTDIR/${stat}.${comp} outfile=$OUTDIR/${stat}.${comp}

end

end

end
