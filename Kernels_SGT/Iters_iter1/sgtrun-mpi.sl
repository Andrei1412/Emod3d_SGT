#!/bin/csh

#Submit this script with: sbatch thefilename

#SBATCH --time=1:00:00   # walltime
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

# The following lines specify the location of the site for which the SGT's will be
# computed. These lines need to be changed for each site.


set STAT = OXZ
set SLON = 172.05049
set SLAT = -43.32584
# The following lines specify the three orthogonal forces for the SGT computations.
# These lines do not need to be changed. Note that the force strength is 1.0e+20 dyne.
# This means the resulting SGTs corrensponds to a moment of 1.0e+20 dyne-cm.

set FORCES = ( fx fy fz )
set XMOM = ( 1.0e+20 0.0 0.0 )
set YMOM = ( 0.0 1.0e+20 0.0 )
set ZMOM = ( 0.0 0.0 1.0e+20 )

# The following sets up the MPI stuff and runs the code. This will be system dependent.

set VERSION = 3.0.8
set BINDIR = /scale_wlg_persistent/filesets/home/rgraves/Mpi/Emod3d/V${VERSION}

set f = 0
foreach force ( $FORCES )
@ f ++

#set NP = ` ./set_sgtparams.csh $STAT $SLON $SLAT $force $XMOM[$f] $YMOM[$f] $ZMOM[$f] `

#srun -n $NP ${BINDIR}/emod3d-mpi par=e3d-${force}.par < /dev/null
srun ${BINDIR}/emod3d-mpi par=e3d-${force}.par < /dev/null

end

./merge_sgt-nonmpi.csh

echo "Done" `date`
exit
