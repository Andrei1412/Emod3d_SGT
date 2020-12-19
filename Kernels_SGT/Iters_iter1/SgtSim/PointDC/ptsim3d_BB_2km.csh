#! /bin/csh

#set BINDIR = ${HOME}/Bin
set BINDIR = /scale_wlg_persistent/filesets/home/rgraves/Bin/

set STATS = ( NNZ )
set SGTNAME = ( AAA )
set SLON = ( 173.37451 )
set SLAT = ( -41.21028 )

set RUNS = ( 2015p013973 2017p144774 3505099 )
set ELON = (       171.28       171.69       171.98 )
set ELAT = (       -43.06       -43.13       -43.18 )
set EDEP = (       8.00         8.00         7.00 )
set EMOM = (       8.95e+22     1.43e+23     2.15e+23 )
set ESTK = (       252.0        163.0        69.0 )
set EDIP = (       82.0         86.0         90.0 )
set ERAK = (       150.0        -4.0         151.0 )

set NTOUT = -1
#set NTOUT = 6000
set NTOUT = 2500
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

#set SGT_MAIN_GFDIR = ../../SgtCalc/${stat}/SgtFiles
set SGT_MAIN_GFDIR = ../../Dev_Strain/${stat}

$BINDIR/ptsim3d stat=$stat slon=$SLON[$s] slat=$SLAT[$s] \
                   moment=$EMOM[$r] strike=$ESTK[$r] dip=$EDIP[$r] rake=$ERAK[$r] \
		   elon=$ELON[$r] elat=$ELAT[$r] edep=$EDEP[$r] \
#                   sgt_xfile=${SGT_MAIN_GFDIR}/${SGTNAME[$s]}_fx.sgt \
#                   sgt_yfile=${SGT_MAIN_GFDIR}/${SGTNAME[$s]}_fy.sgt \
#                   sgt_zfile=${SGT_MAIN_GFDIR}/${SGTNAME[$s]}_fz.sgt \
                   sgt_xfile=${SGT_MAIN_GFDIR}/${SGTNAME}_fx.sgt \
                   sgt_yfile=${SGT_MAIN_GFDIR}/${SGTNAME}_fy.sgt \
                   sgt_zfile=${SGT_MAIN_GFDIR}/${SGTNAME}_fz.sgt \
		   ntout=$NTOUT outdir=$OUTDIR

foreach comp ( 000 090 ver )

$BINDIR/wcc_header tst=$TSTART infile=$OUTDIR/${stat}.${comp} outfile=$OUTDIR/${stat}.${comp}

end

end

end
