# Script that generates and submits a job file to solve Stokes flow given a mesh with random non-overlapping circular exclusions
NMESHESLO=0
NMESHESUP=2500

MARG_LO=0.003
MARG_R=0.003
MARG_U=0.003
MARG_LE=0.003

NEXCLUSIONPARAM1=8.35
NEXCLUSIONPARAM2=0.6

RPARAM1=-5.53
RPARAM2=0.3

SIGMAGPR=0.4
LENGTHSCALE=0.1
LENGTHSCALER=0.05
SIGMOID=1.5

#Set up file paths
DATESTR=`date +%m-%d-%H-%M-%N`	#datestring for jobfolder name
PROJECTDIR="/home/constantin/python/projects/stokesEquation"
# Set JOBNAME by hand for every job!
JOBNAME="solv_GPR_N1=${NEXCLUSIONPARAM1}_N2=${NEXCLUSIONPARAM2}_l=${LENGTHSCALE}_s=${SIGMOID}_R1=${RPARAM1}_R2=${RPARAM2}_SGPR=${SIGMAGPR}_lr=${LENGTHSCALER}_${DATESTR}"
JOBDIR="/home/constantin/python/jobs/$JOBNAME"
JOBSCRIPT="${JOBDIR}/genSolution_cluster.py"

#Create job directory and copy source code
rm -rf $JOBDIR
mkdir $JOBDIR
SOLUTIONSCRIPT="${PROJECTDIR}/genSolution_cluster.py"
cp  $SOLUTIONSCRIPT $JOBSCRIPT
#Change directory to job directory; completely independent from project directory
cd $JOBDIR

#construct job file
echo "#!/bin/bash" >> ./job_file.sh
echo "#SBATCH --job-name=${JOBNAME}" >> ./job_file.sh
echo "#SBATCH --partition batch_SNB,batch_SKL" >> ./job_file.sh
echo "#SBATCH --output=/home/constantin/OEfiles/${JOBNAME}.%j.out" >> ./job_file.sh
echo "#SBATCH --mail-type=ALL" >> ./job_file.sh
echo "#SBATCH --mail-user=mailscluster@gmail.com " >> ./job_file.sh
echo "#SBATCH --time=1000:00:00" >> ./job_file.sh

#Set parameters
echo "sed -i \"24s/.*/meshes = np.arange(${NMESHESLO}, ${NMESHESUP})/\" ./genSolution_cluster.py" >> ./job_file.sh
echo "sed -i \"33s/.*/nExclusionParams = (${NEXCLUSIONPARAM1}, ${NEXCLUSIONPARAM2})/\" ./genSolution_cluster.py" >> ./job_file.sh
echo "sed -i \"43s/.*/sig_scale = ${SIGMOID}/\" ./genSolution_cluster.py" >> ./job_file.sh
echo "sed -i \"42s/.*/cov_l = ${LENGTHSCALE}/\" ./genSolution_cluster.py" >> ./job_file.sh
echo "sed -i \"44s/.*/sigmaGP_r = ${SIGMAGPR}/\" ./genSolution_cluster.py" >> ./job_file.sh
echo "sed -i \"45s/.*/lengthScale_r = ${LENGTHSCALER}/\" ./genSolution_cluster.py" >> ./job_file.sh
echo "sed -i \"50s/.*/r_params = (${RPARAM1}, ${RPARAM2})/\" ./genSolution_cluster.py" >> ./job_file.sh
echo "sed -i \"49s/.*/margins = (${MARG_LO}, ${MARG_R}, ${MARG_U}, ${MARG_LE})/\" ./genSolution_cluster.py" >> ./job_file.sh

#Activate fenics environment and run python
echo "source ~/.bashrc" >> ./job_file.sh
echo "conda activate fenics_new" >> job_file.sh
echo "python -u ./genSolution_cluster.py" >> job_file.sh
echo "" >> job_file.sh


chmod +x job_file.sh
#directly submit job file
sbatch job_file.sh
#execute job_file.sh in shell
#./job_file.sh



