#!/bin/csh

#Submit this script with: sbatch thefilename

#SBATCH --time=0:10:00   # walltime
###-#SBATCH --nodes=4    # number of nodes (40? cores per node)
#SBATCH --ntasks=320   # number of processor cores (i.e. tasks)
##SBATCH --mem-per-cpu=15G   # memory per CPU core
#SBATCH --error=test.e   # stderr file
#SBATCH --output=test.o   # stdout file
#SBATCH --exclusive
##SBATCH --hint=nomultithread

###***#SBATCH --job-name=nr02   # job name
###***#SBATCH --partition=   # queue to run in

echo "Starting" $SLURM_JOB_ID `date`
echo "Initiated on `hostname`"
echo ""
cd "$SLURM_SUBMIT_DIR"           # connect to working directory of sbatch

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

#set VERSION = 3.0.7-dev
set VERSION = 3.0.9
#set EMOD_DIR = /home/rgraves/Mpi/Emod3d/V3.0.7-dev
set EMOD_DIR = /home/rgraves/Mpi/Emod3d/V3.0.9
#set EMOD_DIR = ${HOME}/Mpi/Emod3d/V${VERSION}
#set MERGE_DIR = /nesi/nobackup/nesi00213/RunFolder/tdn27/rgraves/Mpi/Merge
set MERGE_DIR = /home/rgraves/Mpi/Merge

set Vel_es_i = ../../Vel_es/Vel_es_i
set KER = ../../Dev_Strain
set FWD = ../../../FwdSims/V3.0.7-xyz
set ADJ = ../../../AdjSims/V3.0.7-a2a_xyz

set MY_RUN_ID = `echo $user $SLURM_JOB_ID | gawk '{split($2,a,".");printf "%s-%s\n",$1,a[1];}'`
echo "MY_RUN_ID=" $MY_RUN_ID

#set NP = ` ADJ/set_run+merge_params_new_h.csh `  

rm ../../Dev_Strain/*.*
rm $FWD/OutBin/*.*
rm $Vel_es_i/*.*
rm $ADJ/Adj-InputAscii/*.*
rm $ADJ/Adj-SourceBin/*.*
rm $KER/*.*


srun $EMOD_DIR/emod3d-mpi par=FWD/e3d_mysource_i.par < /dev/null
echo "FW emod3d"
srun $MERGE_DIR/merge_P3-mpi par=FWD/merge_P3_xyz.par < /dev/null
echo "emod3d merged"
python -c "from qcore import timeseries as ts; ts.LFSeis('$FWD/OutBin').all2txt(prefix='$Vel_es_i/')"
echo "postprocessing  finished"
mv $FWD/OutBin/fwd01_xyzts.* $KER
echo "Move strains to kernels"


exit
