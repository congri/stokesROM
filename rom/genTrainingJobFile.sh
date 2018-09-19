BCX="u_x=0.0-2.0*x[1]"
BCY="u_y=1.0-2.0*x[0]"

N1=5.5
N2=1.0
R1=-4.5
R2=0.7
MARG1=0.01
MARG2=0.01
MARG3=0.01
MARG4=0.01
XMU="0.7_0.3"
XCOV="0.2_0.0_0.3"

GRADIENTSAMPLESSTART=1
GRADIENTSAMPLESEND=1
STOCHOPTTIME=30    

NTRAIN=128
NTESTSTART=0
NTESTEND=1023

MAXEMITER=30

NCORES=16
if [ $NTRAIN -lt $NCORES ]; then
NCORES=$NTRAIN
fi
echo N_cores=
echo $NCORES

NAMEBASE="full_elbo_score_start=2x2"
DATESTR=`date +%m-%d-%H-%M-%N`	#datestring for jobfolder name
PROJECTDIR="/home/constantin/python/projects/stokesEquation/rom"
JOBNAME="${NAMEBASE}/${DATESTR}_nTrain=${NTRAIN}"
JOBDIR="/home/constantin/python/data/stokesEquation/meshSize=128/nonOverlappingDisks/margins=${MARG1}_${MARG2}_${MARG3}_${MARG4}/N~logn/mu=${N1}/sigma=${N2}/x~gauss/mu=${XMU}/cov=${XCOV}/r~logn/mu=${R1}/sigma=${R2}/p_bc=0.0/split/autosplit/${JOBNAME}"

#Create job directory and copy source code
mkdir -p "${JOBDIR}"
cp -r $PROJECTDIR/* "$JOBDIR"
#Remove existing data folder
rm -r $JOBDIR/data

#Change directory to job directory; completely independent from project directory
cd "$JOBDIR"
echo $PWD
CWD=$(printf "%q\n" "$(pwd)")
rm job_file.sh

#write job file
printf "#PBS -N $JOBNAME
#PBS -l nodes=1:ppn=$NCORES,walltime=240:00:00
#PBS -o /home/constantin/OEfiles
#PBS -e /home/constantin/OEfiles
#PBS -m abe
#PBS -M mailscluster@gmail.com

#Switch to job directory
cd \"$JOBDIR\"

#Set parameters
sed -i \"19s/.*/nTrain = ${NTRAIN};/\" ./trainModel.m
sed -i \"35s/.*/u_bc = {'${BCX}', '${BCY}'}/\" ./StokesData.m
sed -i \"7s/.*/        numberParams = [${N1}, ${N2}]/\" ./StokesData.m
sed -i \"10s/.*/        r_params = [${R1}, ${R2}]/\" ./StokesData.m
sed -i \"9s/.*/        margins = [${MARG1}, ${MARG2}, ${MARG3}, ${MARG4}]/\" ./StokesData.m
sed -i \"12s/.*/        coordDist_mu = '${XMU}'/\" ./StokesData.m
sed -i \"13s/.*/        coordDist_cov = '${XCOV}'/\" ./StokesData.m
sed -i \"17s/.*/maxCompTime = ${STOCHOPTTIME};/\" ./VI/efficientStochOpt.m
sed -i \"18s/.*/nSamplesStart = ${GRADIENTSAMPLESSTART};/\" ./VI/efficientStochOpt.m
sed -i \"19s/.*/nSamplesEnd = ${GRADIENTSAMPLESEND};/\" ./VI/efficientStochOpt.m
sed -i \"85s/.*/        max_EM_epochs = ${MAXEMITER}/\" ./ModelParams.m
sed -i \"11s/.*/testSamples = ${NTESTSTART}:${NTESTEND};/\" ./predictionScript.m




#Run Matlab
/home/matlab/R2017a/bin/matlab -nodesktop -nodisplay -nosplash -r \"trainModel ; quit;\"" >> job_file.sh

chmod +x job_file.sh
#directly submit job file
qsub job_file.sh
#./job_file.sh	#to test in shell









