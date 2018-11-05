NMESHES=2500
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


#construct job file
echo "#!/bin/bash" >> ./job_file.sh
echo "#SBATCH --job-name=${JOBNAME}" >> ./job_file.sh
echo "#SBATCH --partition batch_SNB,batch_SKL" >> ./job_file.sh
echo "#SBATCH --output=/home/constantin/OEfiles/${JOBNAME}.%j.out" >> ./job_file.sh
echo "#SBATCH --error=/home/constantin/OEfiles/${JOBNAME}.%j.err" >> ./job_file.sh
echo "#SBATCH --mail-type=ALL" >> ./job_file.sh
echo "#SBATCH --mail-user=mailscluster@gmail.com " >> ./job_file.sh
echo "#SBATCH --time=1000:00:00" >> ./job_file.sh

echo "sed -i \"17s/.*/nMeshes = $NMESHES/\" ./generateMeshes.py" >> ./job_file.sh
echo "sed -i \"18s/.*/nElements = $NELEMENTS  # PDE discretization/\" ./generateMeshes.py" >> ./job_file.sh
echo "sed -i \"24s/.*/nExclusionParams = ($NEX1, $NEX2)/\" ./generateMeshes.py" >> ./job_file.sh
echo "sed -i \"28s/.*/margins = ($MARGIN_LO, $MARGIN_R, $MARGIN_U, $MARGIN_LE)/\" ./generateMeshes.py" >> ./job_file.sh
echo "sed -i \"32s/.*/r_params = ($RPARAMSLO, $RPARAMSHI)/\" ./generateMeshes.py" >> ./job_file.sh


#Activate fenics environment and run python
echo "source activate fenics" >> ./job_file.sh 
echo "/home/constantin/anaconda3/envs/fenics/bin/python3.6 ./generateMeshes.py" >> ./job_file.sh


chmod +x job_file.sh
#directly submit job file
#qsub job_file.sh
#execute job_file.sh in shell
./job_file.sh



