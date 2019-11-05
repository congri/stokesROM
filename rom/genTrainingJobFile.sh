BCX="u_x=1.0-0.0x[1]"
BCY="u_y=1.0-0.0x[0]"

N1=7.8			#usually lognormal mu for number of exclusions
N2=0.2		    #usually lognormal sigma for number of exclusions
R1=-5.23		#usually lognormal mu for radii distribution
R2=0.3			#usually lognormal sigma for radii distribution

#margins to the domain boundary where no exclusions should be
MARG1=0.003		
MARG2=0.003
MARG3=0.003
MARG4=0.003

SIGMAGPR=0.4
LENGTHSCALE=0.08
LENGTHSCALER=0.05
SIGMOID=1.2
ORIGINREJECTION=0

#parameters of spatial distribution of exclusion centers
COORDDIST="GP"
RADIIDIST="lognGP"
X1="squaredExponential"

GRADIENTSAMPLESSTART=1
GRADIENTSAMPLESEND=1
#define that in matlab file!
#STOCHOPTTIME=120    

NSTART=0
NTRAIN=32
NTESTSTART=512
NTESTEND=1660

MAXEMEPOCHS="1000"

NAMEBASE="error_vs_data8x8"
DATESTR=`date +%m-%d-%H-%M-%N`	#datestring for jobfolder name
PROJECTDIR="/home/constantin/python/projects/stokesEquation/rom"
JOBNAME="${DATESTR}_nTrain=${NTRAIN}_nStart=${NSTART}_bc=${BCX}_${BCY}_${NAMEBASE}_epochs=${MAXEMEPOCHS}"
JOBDIR_BASE="constantin/python/data/stokesEquation/meshSize=256/nonOverlappingDisks/margins=${MARG1}_${MARG2}_${MARG3}_${MARG4}/N~logn/mu=${N1}/sigma=${N2}/x~${COORDDIST}/"

#location
if [ "$COORDDIST" = "GP" ]; then
    JOBDIR_BASE="${JOBDIR_BASE}cov=${X1}/l=${LENGTHSCALE}/sig_scale=${SIGMOID}"
elif [ "$COORDDIST" = "tiles" ]; then
    JOBDIR_BASE="${JOBDIR_BASE}"
elif [ "$COORDDIST" = "gauss" ]; then
    JOBDIR_BASE="${JOBDIR_BASE}mu=${X1}/cov=${LENGTHSCALE}"
elif [ "$COORDDIST" = "engineered" ]; then
    JOBDIR_BASE="${JOBDIR_BASE}"
fi

#radii
if [ "$RADIIDIST" = "lognGP" ]; then
    JOBDIR_BASE="${JOBDIR_BASE}/r~lognGP/mu=${R1}/sigma=${R2}/sigmaGP_r=${SIGMAGPR}/l=${LENGTHSCALER}"
else
    JOBDIR_BASE="${JOBDIR_BASE}/r~logn/mu=${R1}/sigma=${R2}"
fi

if [ "${ORIGINREJECTION}" -gt "0" ]; then
    JOBDIR_BASE="${JOBDIR_BASE}/origin_rejection=${ORIGINREJECTION}"
fi

#boundary conditions
JOBDIR_BASE="${JOBDIR_BASE}/p_bc=0.0/${BCX}_${BCY}"
JOBDIR_BASE="${JOBDIR_BASE}/$JOBNAME"
JOBDIR_MASTER="/home/${JOBDIR_BASE}"
JOBDIR_ETH="/home_eth/${JOBDIR_BASE}"

#Create job directory and copy source code
mkdir -p "${JOBDIR_MASTER}"
cp -r "$PROJECTDIR/featureFunctions" $JOBDIR_MASTER
cp -r "$PROJECTDIR/mesh" $JOBDIR_MASTER
cp -r "$PROJECTDIR/aux" $JOBDIR_MASTER
cp -r "$PROJECTDIR/comp" $JOBDIR_MASTER
cp -r "$PROJECTDIR/FEM" $JOBDIR_MASTER
cp -r "$PROJECTDIR/rom" $JOBDIR_MASTER
cp -r "$PROJECTDIR/VI" $JOBDIR_MASTER
cp "$PROJECTDIR/ModelParams.m" $JOBDIR_MASTER
cp "$PROJECTDIR/predictionScript.m" $JOBDIR_MASTER
cp "$PROJECTDIR/StokesData.m" $JOBDIR_MASTER
cp "$PROJECTDIR/StokesROM.m" $JOBDIR_MASTER
cp "$PROJECTDIR/trainModel.m" $JOBDIR_MASTER


#Change directory to job directory; completely independent from project directory
cd "$JOBDIR_MASTER"
echo "Current directory:"
echo $PWD

NCORES=8
if [ $NTRAIN -lt $NCORES ]; then
    NCORES=$NTRAIN
fi

#construct job file string
echo "#!/bin/bash" >> ./job_file.sh
echo "#SBATCH --job-name=${JOBNAME}" >> ./job_file.sh
echo "#SBATCH --partition batch_SKL,batch_SNB" >> ./job_file.sh
echo "#SBATCH --nodes 1-1" >> ./job_file.sh
echo "#SBATCH --mincpus=${NCORES}" >> ./job_file.sh     #node is not shared with other jobs
echo "#SBATCH --output=/home_eth/constantin/OEfiles/${JOBNAME}.%j.out" >> ./job_file.sh
echo "#SBATCH --error=/home_eth/constantin/OEfiles/${JOBNAME}.%j.err" >> ./job_file.sh
echo "#SBATCH --mail-type=ALL" >> ./job_file.sh
echo "#SBATCH --mail-user=mailscluster@gmail.com " >> ./job_file.sh
echo "#SBATCH --time=240:00:00" >> ./job_file.sh

echo "" >> ./job_file.sh
echo "#Switch to job directory" >> ./job_file.sh
echo "cd \"$JOBDIR_ETH\"" >> ./job_file.sh
echo "" >> ./job_file.sh
echo "#Set parameters" >> ./job_file.sh
echo "sed -i \"19s/.*/nTrain = ${NTRAIN};/\" ./trainModel.m" >> ./job_file.sh
echo "sed -i \"21s/.*/nStart = ${NSTART};/\" ./trainModel.m" >> ./job_file.sh
echo "sed -i \"7s/.*/        numberParams = [${N1}, ${N2}]/\" ./StokesData.m" >> ./job_file.sh
echo "sed -i \"9s/.*/        margins = [${MARG1}, ${MARG2}, ${MARG3}, ${MARG4}]/\" ./StokesData.m" >> ./job_file.sh
echo "sed -i \"10s/.*/        r_params = [${R1}, ${R2}]/\" ./StokesData.m" >> ./job_file.sh
echo "sed -i \"11s/.*/        coordDist = '${COORDDIST}'/\" ./StokesData.m" >> ./job_file.sh
echo "sed -i \"14s/.*/        radiiDist = '${RADIIDIST}'/\" ./StokesData.m" >> ./job_file.sh
echo "sed -i \"20s/.*/        sigmaGP_r = ${SIGMAGPR}/\" ./StokesData.m" >> ./job_file.sh
echo "sed -i \"21s/.*/        l_r = ${LENGTHSCALER}/\" ./StokesData.m" >> ./job_file.sh
echo "sed -i \"22s/.*/        origin_rejection = ${ORIGINREJECTION}/\" ./StokesData.m" >> ./job_file.sh
if [ "$COORDDIST" = "GP" ]; then
echo "sed -i \"17s/.*/        densityLengthScale = '${LENGTHSCALE}'/\" ./StokesData.m" >> ./job_file.sh
echo "sed -i \"18s/.*/        sigmoidScale = '${SIGMOID}'/\" ./StokesData.m" >> ./job_file.sh
elif [ "$COORDDIST" = "gauss" ]; then
echo "sed -i \"12s/.*/        coordDist_mu = '${X1}'/\" ./StokesData.m" >> ./job_file.sh
echo "sed -i \"13s/.*/        coordDist_cov = '${LENGTHSCALE}'/\" ./StokesData.m" >> ./job_file.sh
fi
echo "sed -i \"44s/.*/        u_bc = {'${BCX}', '${BCY}'}/\" ./StokesData.m" >> ./job_file.sh
echo "sed -i \"20s/.*/nSamplesStart = ${GRADIENTSAMPLESSTART};/\" ./VI/efficientStochOpt.m" >> ./job_file.sh
echo "sed -i \"21s/.*/nSamplesEnd = ${GRADIENTSAMPLESEND};/\" ./VI/efficientStochOpt.m" >> ./job_file.sh
echo "sed -i \"99s/.*/        max_EM_epochs = ${MAXEMEPOCHS}/\" ./ModelParams.m" >> ./job_file.sh
echo "sed -i \"12s/.*/testSamples = ${NTESTSTART}:${NTESTEND};/\" ./predictionScript.m" >> ./job_file.sh
echo "#Run Matlab" >> ./job_file.sh
echo "/home_eth/programs/matlab/bin/matlab -nodesktop -nodisplay -nosplash -r \" trainModel ; quit ; \"" >> ./job_file.sh


chmod +x job_file.sh
#directly submit job file
sbatch job_file.sh
#./job_file.sh	#to test in shell









