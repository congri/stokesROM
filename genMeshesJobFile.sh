NMESHES=2048
NELEMENTS=256
NEX1=8.35    #number of exclusion parameters
NEX2=0.6
RPARAMSLO=-5.53
RPARAMSHI=0.3
MARGIN_LO=0.003
MARGIN_R=0.003
MARGIN_U=0.003
MARGIN_LE=0.003


CORES=1

DATESTR=`date +%m-%d-%H-%M-%N`	#datestring for jobfolder name
#Set up file paths
PROJECTDIR="/home/constantin/python/projects/stokesEquation"
JOBNAME="genMesh_nElements=${NELEMENTS}nParams1=${NEX1}nParams2=${NEX2}margins=${MARGIN_LO}_${MARGIN_R}_${MARGIN_U}_${MARGIN_LE}r=${RPARAMSLO}_${RPARAMSHI}"
JOBDIR="/home/constantin/python/jobs/${JOBNAME}_${DATESTR}"

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
#PBS -l walltime=1000:00:00
#PBS -o $CWD
#PBS -e $CWD
#PBS -m abe
#PBS -M mailscluster@gmail.com

#Switch to job directory
cd $JOBDIR
#Set parameters
sed -i \"15s/.*/nMeshes = $NMESHES/\" ./generateMeshes.py
sed -i \"16s/.*/nElements = $NELEMENTS  # PDE discretization/\" ./generateMeshes.py
sed -i \"22s/.*/nExclusionParams = ($NEX1, $NEX2)/\" ./generateMeshes.py
sed -i \"26s/.*/margins = ($MARGIN_LO, $MARGIN_R, $MARGIN_U, $MARGIN_LE)/\" ./generateMeshes.py
sed -i \"30s/.*/r_params = ($RPARAMSLO, $RPARAMSHI)/\" ./generateMeshes.py



#Activate fenics environment and run python
source activate fenics
/home/constantin/anaconda3/envs/fenics/bin/python3.6 ./generateMeshes.py


" >> job_file.sh

chmod +x job_file.sh
#directly submit job file
qsub job_file.sh
#execute job_file.sh in shell
#./job_file.sh



