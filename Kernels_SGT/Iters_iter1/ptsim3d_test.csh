#! /bin/csh

set BINDIR = /scale_wlg_persistent/filesets/home/rgraves/Bin

set STATS = ( CACS )
set SGTNAME = ( AAA )
set SLON = ( 172.5300 )
set SLAT = ( -43.4832 )

set RUNS = ( 2012p149826 )
set ELON = (     172.8326 )
set ELAT = (     -43.4624 )
set EDEP = (      6.00 )
set EMOM = (    1.49e+22 )
set ESTK = (       49.0 )
set EDIP = (       47.0 )
set ERAK = (       103.0 )

set NTOUT = -1
set NTOUT = 2000
set TSTART = -0.5     # to account for 1/2 width of cos pulse
set TSTART = 0.0

set r = 0
foreach run ( $RUNS )
@ r ++

set OUTDIR = Dump_EQ/${run}/Vel
\mkdir -p $OUTDIR

set s = 0
foreach stat ( $STATS )
@ s ++

set SGT_MAIN_GFDIR = ../../Dev_Strain/${stat}

$BINDIR/ptsim3d stat=$stat slon=$SLON[$s] slat=$SLAT[$s] \
                   moment=$EMOM[$r] strike=$ESTK[$r] dip=$EDIP[$r] rake=$ERAK[$r] \
		   elon=$ELON[$r] elat=$ELAT[$r] edep=$EDEP[$r] \
                   sgt_xfile=${SGT_MAIN_GFDIR}/${SGTNAME[$s]}_fx.sgt \
                   sgt_yfile=${SGT_MAIN_GFDIR}/${SGTNAME[$s]}_fy.sgt \
                   sgt_zfile=${SGT_MAIN_GFDIR}/${SGTNAME[$s]}_fz.sgt \
		   ntout=$NTOUT outdir=$OUTDIR

foreach comp ( 000 090 ver )

$BINDIR/wcc_header tst=$TSTART infile=$OUTDIR/${stat}.${comp} outfile=$OUTDIR/${stat}.${comp}

end

end

end
