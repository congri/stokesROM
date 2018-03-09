NMESHES=1024
NELEMENTS=128
NEX1=7.5    #number of exclusion parameters
NEX2=1.0
RPARAMSLO=-4.5
RPARAMSHI=0.5
MARGIN_LO=-1
MARGIN_R=-1
MARGIN_U=-1
MARGIN_LE=-1


CORES=1


#Set up file paths
PROJECTDIR="/home/constantin/python/projects/stokesEquation"
JOBNAME="genMesh_nElements=${NELEMENTS}nParams1=${NEX1}nParams2=${NEX2}margins=${MARGIN_LO}_${MARGIN_R}_${MARGIN_U}_${MARGIN_LE}r=${RPARAMSLO}_${RPARAMSHI}"
JOBDIR="/home/constantin/python/jobs/$JOBNAME"

#Create job directory and copy source code
rm -r $JOBDIR
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
sed -i \"15s/.*/nMeshes = $NMESHES/\" ./generateMeshes.py
sed -i \"16s/.*/nElements = $NELEMENTS  # PDE discretization/\" ./generateMeshes.py
sed -i \"22s/.*/nExclusionParams = ($NEX1, $NEX2)/\" ./generateMeshes.py
sed -i \"26s/.*/margins = ($MARGIN_LO, $MARGIN_R, $MARGIN_U, $MARGIN_LE)/\" ./generateMeshes.py
sed -i \"29s/.*/r_params = ($RPARAMSLO, $RPARAMSHI)/\" ./generateMeshes.py



#Activate fenics environment and run python
source activate fenics
/home/constantin/anaconda3/envs/fenics/bin/python3.6 ./generateMeshes.py


" >> job_file.sh

chmod +x job_file.sh
#directly submit job file
qsub job_file.sh
#execute job_file.sh in shell
#./job_file.sh



