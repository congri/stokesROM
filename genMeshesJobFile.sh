NMESHES=1024
NELEMENTS=128
NEXMIN=128
NEXMAX=129
RPARAMSLO=0.003
RPARAMSHI=0.015
MARGIN_LO=0
MARGIN_R=0.03
MARGIN_U=0
MARGIN_LE=0.03


CORES=1


#Set up file paths
PROJECTDIR="/home/constantin/python/projects/stokesEquation"
JOBNAME="genMesh_nElements=${NELEMENTS}nExMin=${NEXMIN}nExMax=${NEXMAX}margins=${MARGIN_LO}_${MARGIN_R}_${MARGIN_U}_${MARGIN_LE}r=${RPARAMSLO}_${RPARAMSHI}"
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
sed -i \"21s/.*/nExclusionsMin = $NEXMIN/\" ./generateMeshes.py
sed -i \"22s/.*/nExclusionsMax = $NEXMAX/\" ./generateMeshes.py
sed -i \"25s/.*/margins = ($MARGIN_LO, $MARGIN_R, $MARGIN_U, $MARGIN_LE)/\" ./generateMeshes.py
sed -i \"27s/.*/r_params = ($RPARAMSLO, $RPARAMSHI)/\" ./generateMeshes.py



#Activate fenics environment and run python
source activate fenics
/home/constantin/anaconda3/envs/fenics/bin/python3.6 ./generateMeshes.py


" >> job_file.sh

chmod +x job_file.sh
#directly submit job file
qsub job_file.sh
#execute job_file.sh in shell
#./job_file.sh



