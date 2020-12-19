#! /bin/csh

# START OF REQUIRED PARAMETERS #

set RUNS = ( 20011031-m502 20050612-m519 20020102-m401 )
set ELON = (       -116.50       -116.57       -116.43 )
set ELAT = (         33.50         33.53         33.38 )
set EDEP = (          8.00          8.00         11.00 )
set EMOM = (    3.8032e+23    6.9263e+23    1.1546e+22 )
set ESTK = (          48.0          36.0          44.0 )
set EDIP = (          90.0          77.0          90.0 )
set ERAK = (          57.0          32.0          22.0 )

# END OF REQUIRED PARAMETERS #

# FOLLOWING PARAMETERS MIGHT NEED TO BE ADJUSTED #

set SRFDIR = Srf
mkdir -p $SRFDIR

set STYPE     = cos
set DT        = 0.01
set RISE_TIME = 0.1

set VELMODEL = ../../Model/Mod-1D/mj-vs500.fd-h0.100

# END OF OPTIONAL PARAMETERS #

# FOLLOWING PARAMETERS DON"T NEED TO BE CHANGED #

set TARGET_AREA_KM = -1
set TARGET_SLIP_CM = -1
set NSTK    = 1
set NDIP    = 1

set r = 0
foreach run ( $RUNS )
@ r ++

set MPARS = `gawk -v zz=$EDEP[$r] 'BEGIN{zb=0.0;}{if(NR>1){if(zb<zz && $6>=zz){vs=$2;den=$3;}zb=$6;}}END{print vs,den;}' $VELMODEL `
echo $MPARS

set VS = $MPARS[1]
set DEN = $MPARS[2]

set DPARS = `echo $EMOM[$r] $TARGET_AREA_KM $TARGET_SLIP_CM $VS $DEN | gawk '{if($2>0){dd=sqrt($2);ss=($1*1.0e-20)/($2*$4*$4*$5);}else if($3>0){dd=sqrt(($1*1.0e-20)/($3*$4*$4*$5));ss=$3;}else{aa=exp(2.0*log($1)/3.0 - 14.7*log(10.0));dd=sqrt(aa);ss=($1*1.0e-20)/(aa*$4*$4*$5);}printf "%.5e %.3f\n",dd,ss;}'`
echo $EMOM[$r] $DPARS

set SRFFILE = $SRFDIR/${run}.srf

echo $ELON[$r] $ELAT[$r] $EDEP[$r] $DPARS[1] $DPARS[1] $ESTK[$r] $EDIP[$r] $ERAK[$r] $DPARS[2] | \
pointdc2srf stype=$STYPE dt=$DT risetime=$RISE_TIME version=2.0 vs=$VS den=$DEN > $SRFFILE

end
