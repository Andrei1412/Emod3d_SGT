#!/bin/csh

set RUNS = ( AAA )

set INDIR = ./OutBin
set LOGDIR = SgtLog
set OUTDIR = SgtFiles

mkdir -p $OUTDIR

set FORCES  = ( fx fy fz )

set ZERO_OUTFILE = 1

#set s = 0
#foreach run ( $RUNS )
#@ s ++

foreach force ( $FORCES )

#set inroot = $INDIR/${run}-${force}_sgt-
set inroot = $INDIR/AAA-${force}_sgt-

set PROCS = `ls ${inroot}* | gawk -F"/" '{n=split($NF,a,".");m=split(a[n-1],b,"-");print b[m];}' `

set FILELIST = filelist.$force

\rm $FILELIST
foreach proc ( $PROCS )

echo "${inroot}${proc}.e3d" >> $FILELIST

end

set OUTFILE = $OUTDIR/AAA_${force}.sgt

/scale_wlg_persistent/filesets/home/rgraves/Bin/merge_sgt-nonmpi filelist=$FILELIST zero_outfile=$ZERO_OUTFILE outfile=$OUTFILE \
	logdir=$LOGDIR logname=AAA-${force} < /dev/null

#end
end
