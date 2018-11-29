# Generates microstructural data, i.e. circles and radii of nonoverlap. disks
NDATA=2500


#margins to the domain boundary where no exclusions should be
MARG1=0.003		
MARG2=0.003
MARG3=0.003
MARG4=0.003

#parameters of spatial distribution of exclusion centers
X1="squaredExponential"

#Set up file paths
PROJECTDIR="/home/constantin/python/projects/stokesEquation"

LENGTHSCALE=0.08
LENGTHSCALER=0.05
SIGMAGPR=0.4
SIGMOID=1.0
N1=7.8			#usually lognormal mu for number of exclusions
N2=0.2
R1=-5.23
R2=0.3

JOBNAME="genMicrostruct_GP_GPR_muN=${N1}_sigN=${N2}_l=${LENGTHSCALE}_lr=${LENGTHSCALER}_sigma_mu_r=${SIGMAGPR}_sig_scale=${SIGMOID}_sigma_r=${R2}"
DATESTR=`date +%m-%d-%H-%M-%N`	#datestring for jobfolder name

JOBDIR="/home/constantin/python/jobs/${JOBNAME}_${DATESTR}"
JOBSCRIPT="${JOBDIR}/genCorrMicrostruct.m"

#Create job directory and copy source code
rm -rf $JOBDIR
mkdir $JOBDIR
MICROSTRUCTGENSCRIPT="${PROJECTDIR}/rom/genMicrostruct/genCorrMicrostruct.m"
cp $MICROSTRUCTGENSCRIPT $JOBSCRIPT
#Change directory to job directory; completely independent from project directory
cd $JOBDIR
CWD=$(printf "%q\n" "$(pwd)")

#construct job file
echo "#!/bin/bash" >> ./job_file.sh
echo "#SBATCH --job-name=${JOBNAME}" >> ./job_file.sh
echo "#SBATCH --partition batch_SNB,batch_SKL" >> ./job_file.sh
echo "#SBATCH --output=/home/constantin/OEfiles/${JOBNAME}.%j.out" >> ./job_file.sh
echo "#SBATCH --mail-type=ALL" >> ./job_file.sh
echo "#SBATCH --mail-user=mailscluster@gmail.com " >> ./job_file.sh
echo "#SBATCH --time=1000:00:00" >> ./job_file.sh

echo " " >> ./job_file.sh

echo "sed -i \"8s/.*/lengthScale = ${LENGTHSCALE};/\" ./genCorrMicrostruct.m" >> ./job_file.sh
echo "sed -i \"9s/.*/lengthScale_r = ${LENGTHSCALER};/\" ./genCorrMicrostruct.m" >> ./job_file.sh
echo "sed -i \"10s/.*/sigmaGP_r = ${SIGMAGPR};/\" ./genCorrMicrostruct.m" >> ./job_file.sh
echo "sed -i \"12s/.*/sigmoid_scale = ${SIGMOID};/\" ./genCorrMicrostruct.m" >> ./job_file.sh
echo "sed -i \"14s/.*/nExclusionParams = [${N1}, ${N2}];/\" ./genCorrMicrostruct.m" >> ./job_file.sh
echo "sed -i \"15s/.*/margins = [${MARG1}, ${MARG2}, ${MARG3}, ${MARG4}];/\" ./genCorrMicrostruct.m" >> ./job_file.sh
echo "sed -i \"16s/.*/rParams = [${R1}, ${R2}];/\" ./genCorrMicrostruct.m" >> ./job_file.sh
echo "sed -i \"18s/.*/nMeshes = 0:${NDATA};/\" ./genCorrMicrostruct.m" >> ./job_file.sh



#run matlab script
echo "/home/programs/matlab/bin/matlab -nodesktop -nodisplay -nosplash -r \"genCorrMicrostruct ; quit;\"" >> ./job_file.sh


chmod +x job_file.sh
#directly submit job file
sbatch job_file.sh
#execute job_file.sh in shell
#./job_file.sh


