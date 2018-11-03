# Script that generates and submits a job file to solve Stokes flow given a mesh with random non-overlapping circular exclusions
NMESHESLO=0
NMESHESUP=2500

MARG_LO=0.003
MARG_R=0.003
MARG_U=0.003
MARG_LE=0.003

NEXCLUSIONPARAM1=8.1
NEXCLUSIONPARAM2=0.6

RPARAM1=-5.53
RPARAM2=0.3

#Set up file paths
DATESTR=`date +%m-%d-%H-%M-%N`	#datestring for jobfolder name
PROJECTDIR="/home/constantin/python/projects/stokesEquation"
# Set JOBNAME by hand for every job!
JOBNAME="solv_engin_${DATESTR}"
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
#PBS -l walltime=240:00:00
#PBS -o /home/constantin/OEfiles
#PBS -e /home/constantin/OEfiles
#PBS -m abe
#PBS -M mailscluster@gmail.com

#Switch to job directory
cd $JOBDIR
#Set parameters
sed -i \"24s/.*/meshes = np.arange(${NMESHESLO}, ${NMESHESUP})/\" ./genSolution_cluster.py
sed -i \"33s/.*/nExclusionParams = (${NEXCLUSIONPARAM1}, ${NEXCLUSIONPARAM2})/\" ./genSolution_cluster.py
sed -i \"48s/.*/r_params = (${RPARAM1}, ${RPARAM2})/\" ./genSolution_cluster.py
sed -i \"47s/.*/margins = (${MARG_LO}, ${MARG_R}, ${MARG_U}, ${MARG_LE})/\" ./genSolution_cluster.py




#Activate fenics environment and run python
source activate fenics3
/home/constantin/anaconda3/envs/fenics3/bin/python3.6 ./genSolution_cluster.py


" >> job_file.sh

chmod +x job_file.sh
#directly submit job file
#qsub job_file.sh
#execute job_file.sh in shell
./job_file.sh



