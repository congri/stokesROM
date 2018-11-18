BCX="u_x=0.0-2.0x[1]"
BCY="u_y=1.0-2.0x[0]"

N1=8.35			#usually lognormal mu for number of exclusions
N2=0.6			#usually lognormal sigma for number of exclusions
R1=-5.53		#usually lognormal mu for radii distribution
R2=0.3			#usually lognormal sigma for radii distribution

#margins to the domain boundary where no exclusions should be
MARG1=0.003		
MARG2=0.003
MARG3=0.003
MARG4=0.003

#parameters of spatial distribution of exclusion centers
COORDDIST="GP"
X1="squaredExponential"
X2=0.1      #length scale in GP
X3=1.5      #sigmoid scale in GP

GRADIENTSAMPLESSTART=1
GRADIENTSAMPLESEND=1
STOCHOPTTIME=120    

NSTART=96
NTRAIN=32
NTESTSTART=256
NTESTEND=1110

MAXEMEPOCHS=100

NAMEBASE="GP_4x4"
DATESTR=`date +%m-%d-%H-%M-%N`	#datestring for jobfolder name
PROJECTDIR="/home/constantin/python/projects/stokesEquation/rom"
JOBNAME="${DATESTR}_nTrain=${NTRAIN}_${NAMEBASE}"
JOBDIR="/home/constantin/python/data/stokesEquation/meshSize=256/nonOverlappingDisks/margins=${MARG1}_${MARG2}_${MARG3}_${MARG4}/N~logn/mu=${N1}/sigma=${N2}/x~${COORDDIST}/"
if [ "$COORDDIST" = "GP" ]; then
    JOBDIR="${JOBDIR}cov=${X1}/l=${X2}/sig_scale=${X3}/r~logn/mu=${R1}/sigma=${R2}/p_bc=0.0/${BCX}_${BCY}"
elif [ "$COORDDIST" = "tiles" ]; then
    JOBDIR="${JOBDIR}r~logn/mu=${R1}/sigma=${R2}/p_bc=0.0/${BCX}_${BCY}"
elif [ "$COORDDIST" = "gauss" ]; then
    JOBDIR="${JOBDIR}mu=${X1}/cov=${X2}/r~logn/mu=${R1}/sigma=${R2}/p_bc=0.0/${BCX}_${BCY}"
elif [ "$COORDDIST" = "engineered" ]; then
    JOBDIR="${JOBDIR}r~logn/mu=${R1}/sigma=${R2}/p_bc=0.0/${BCX}_${BCY}"
fi
JOBDIR="${JOBDIR}/$JOBNAME"

#Create job directory and copy source code
mkdir -p "${JOBDIR}"
cp -r "$PROJECTDIR/featureFunctions" $JOBDIR
cp -r "$PROJECTDIR/mesh" $JOBDIR
cp -r "$PROJECTDIR/aux" $JOBDIR
cp -r "$PROJECTDIR/comp" $JOBDIR
cp -r "$PROJECTDIR/FEM" $JOBDIR
cp -r "$PROJECTDIR/rom" $JOBDIR
cp -r "$PROJECTDIR/VI" $JOBDIR
cp "$PROJECTDIR/ModelParams.m" $JOBDIR
cp "$PROJECTDIR/predictionScript.m" $JOBDIR
cp "$PROJECTDIR/StokesData.m" $JOBDIR
cp "$PROJECTDIR/StokesROM.m" $JOBDIR
cp "$PROJECTDIR/trainModel.m" $JOBDIR


#Change directory to job directory; completely independent from project directory
cd "$JOBDIR"
echo "Current directory:"
echo $PWD

NCORES=16
if [ $NTRAIN -lt $NCORES ]; then
    NCORES=$NTRAIN
fi

#construct job file string
echo "#!/bin/bash" >> ./job_file.sh
echo "#SBATCH --job-name=${JOBNAME}" >> ./job_file.sh
echo "#SBATCH --partition batch_SNB,batch_SKL" >> ./job_file.sh
echo "#SBATCH --nodes 1-1" >> ./job_file.sh
echo "#SBATCH --mincpus=${NCORES}" >> ./job_file.sh     #node is not shared with other jobs
echo "#SBATCH --output=/home/constantin/OEfiles/${JOBNAME}.%j.out" >> ./job_file.sh
echo "#SBATCH --error=/home/constantin/OEfiles/${JOBNAME}.%j.err" >> ./job_file.sh
echo "#SBATCH --mail-type=ALL" >> ./job_file.sh
echo "#SBATCH --mail-user=mailscluster@gmail.com " >> ./job_file.sh
echo "#SBATCH --time=240:00:00" >> ./job_file.sh

echo "" >> ./job_file.sh
echo "#Switch to job directory" >> ./job_file.sh
echo "cd \"$JOBDIR\"" >> ./job_file.sh
echo "" >> ./job_file.sh
echo "#Set parameters" >> ./job_file.sh
echo "sed -i \"19s/.*/nTrain = ${NTRAIN};/\" ./trainModel.m" >> ./job_file.sh
echo "sed -i \"21s/.*/nStart = ${NSTART};/\" ./trainModel.m" >> ./job_file.sh
echo "sed -i \"7s/.*/        numberParams = [${N1}, ${N2}]/\" ./StokesData.m" >> ./job_file.sh
echo "sed -i \"9s/.*/        margins = [${MARG1}, ${MARG2}, ${MARG3}, ${MARG4}]/\" ./StokesData.m" >> ./job_file.sh
echo "sed -i \"10s/.*/        r_params = [${R1}, ${R2}]/\" ./StokesData.m" >> ./job_file.sh
echo "sed -i \"11s/.*/        coordDist = '${COORDDIST}'/\" ./StokesData.m" >> ./job_file.sh
if [ "$COORDDIST" = "GP" ]; then
echo "sed -i \"17s/.*/        densityLengthScale = '${X2}'/\" ./StokesData.m" >> ./job_file.sh
echo "sed -i \"18s/.*/        sigmoidScale = '${X3}'/\" ./StokesData.m" >> ./job_file.sh
elif [ "$COORDDIST" = "gauss" ]; then
echo "sed -i \"12s/.*/        coordDist_mu = '${X1}'/\" ./StokesData.m" >> ./job_file.sh
echo "sed -i \"13s/.*/        coordDist_cov = '${X2}'/\" ./StokesData.m" >> ./job_file.sh
fi
echo "sed -i \"39s/.*/        u_bc = {'${BCX}', '${BCY}'}/\" ./StokesData.m" >> ./job_file.sh
echo "sed -i \"17s/.*/maxCompTime = ${STOCHOPTTIME};/\" ./VI/efficientStochOpt.m" >> ./job_file.sh
echo "sed -i \"18s/.*/nSamplesStart = ${GRADIENTSAMPLESSTART};/\" ./VI/efficientStochOpt.m" >> ./job_file.sh
echo "sed -i \"19s/.*/nSamplesEnd = ${GRADIENTSAMPLESEND};/\" ./VI/efficientStochOpt.m" >> ./job_file.sh
echo "sed -i \"88s/.*/        max_EM_epochs = ${MAXEMEPOCHS}/\" ./ModelParams.m" >> ./job_file.sh
echo "sed -i \"12s/.*/testSamples = ${NTESTSTART}:${NTESTEND};/\" ./predictionScript.m" >> ./job_file.sh
echo "#Run Matlab" >> ./job_file.sh
echo "/home/programs/matlab/bin/matlab -nodesktop -nodisplay -nosplash -r \" trainModel ; quit ; \"" >> ./job_file.sh


chmod +x job_file.sh
#directly submit job file
sbatch job_file.sh
#./job_file.sh	#to test in shell









