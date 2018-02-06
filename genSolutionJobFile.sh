NMESHESLO=0
NMESHESUP=1023
NELEMENTS=128
NEXMIN=2048
NEXMAX=2040
RPARAMSLO=-4.6
RPARAMSHI=0.15
MARGIN_LO=-1
MARGIN_R=0.02
MARGIN_U=-1
MARGIN_LE=0.02


CORES=4


#Set up file paths
PROJECTDIR="/home/constantin/python/projects/stokesEquation"
JOBNAME="genSolution_nElements=${NELEMENTS}nExMin=${NEXMIN}nExMax=${NEXMAX}margins=${MARGIN_LO}_${MARGIN_R}_${MARGIN_U}_${MARGIN_LE}r=${RPARAMSLO}_${RPARAMSHI}"
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
sed -i \"29s/.*/meshes = np.arange($NMESHESLO, $NMESHESUP)  # vector of random meshes to load/\" ./genStokesData.py
sed -i \"31s/.*/nElements = $NELEMENTS  # PDE discretization/\" ./genStokesData.py
sed -i \"44s/.*/nExclusionsMin = $NEXMIN/\" ./genStokesData.py
sed -i \"45s/.*/nExclusionsMax = $NEXMAX/\" ./genStokesData.py
sed -i \"49s/.*/margins = ($MARGIN_LO, $MARGIN_R, $MARGIN_U, $MARGIN_LE)/\" ./genStokesData.py
sed -i \"50s/.*/r_params = ($RPARAMSLO, $RPARAMSHI)/\" ./genStokesData.py



#Activate fenics environment and run python
source activate fenics
/home/constantin/anaconda3/envs/fenics/bin/python3.6 ./genStokesData.py


" >> job_file.sh

chmod +x job_file.sh
#directly submit job file
qsub job_file.sh
#execute job_file.sh in shell
#./job_file.sh



