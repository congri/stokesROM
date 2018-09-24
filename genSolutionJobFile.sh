# Script that generates and submits a job file to solve Stokes flow given a mesh with random non-overlapping circular exclusions
CORES=4

NMESHESLO=0
NMESHESUP=2048

MARG_LO=0.008
MARG_R=0.008
MARG_U=0.008
MARG_LE=0.008

NEXCLUSIONPARAM1=9.0
NEXCLUSIONPARAM2=1.0

RPARAM1=-5.5
RPARAM2=0.5

#Set up file paths
DATESTR=`date +%m-%d-%H-%M-%N`	#datestring for jobfolder name
PROJECTDIR="/home/constantin/python/projects/stokesEquation"
# Set JOBNAME by hand for every job!
JOBNAME="solv_${DATESTR}"
JOBDIR="/home/constantin/python/jobs/$JOBNAME"

#Create job directory and copy source code
rm -rf $JOBDIR
mkdir $JOBDIR
cp -r $PROJECTDIR/* $JOBDIR
#Change directory to job directory; completely independent from project directory
cd $JOBDIR
CWD=$(printf "%q\n" "$(pwd)")
rm job_file.sh

#write job file
printf "#PBS -N $JOBNAME
#PBS -l nodes=1:ppn=$CORES,walltime=240:00:00
#PBS -o $CWD
#PBS -e $CWD
#PBS -m abe
#PBS -M mailscluster@gmail.com

#Switch to job directory
cd $JOBDIR
#Set parameters
sed -i \"24s/.*/meshes = np.arange(${NMESHESLO}, ${NMESHESUP})/\" ./genSolution_cluster.py
sed -i \"33s/.*/nExclusionParams = (${NEXCLUSIONPARAM1}, ${NEXCLUSIONPARAM2})/\" ./genSolution_cluster.py
sed -i \"40s/.*/r_params = (${RPARAM1}, ${RPARAM2})/\" ./genSolution_cluster.py
sed -i \"39s/.*/margins = (${MARG_LO}, ${MARG_R}, ${MARG_U}, ${MARG_LE})/\" ./genSolution_cluster.py




#Activate fenics environment and run python
source activate fenics3
/home/constantin/anaconda3/envs/fenics3/bin/python3.6 ./genSolution_cluster.py


" >> job_file.sh

chmod +x job_file.sh
#directly submit job file
qsub job_file.sh
#execute job_file.sh in shell
#./job_file.sh



