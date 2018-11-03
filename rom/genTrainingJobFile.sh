BCX="u_x=1.0"
BCY="u_y=0.0"

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
COORDDIST="engineered"
X1="squaredExponential"
X2=0.1
X3=1.5

GRADIENTSAMPLESSTART=1
GRADIENTSAMPLESEND=1
STOCHOPTTIME=30    

NTRAIN=64
NTESTSTART=0
NTESTEND=1023

MAXEMITER=400

NCORES=16
if [ $NTRAIN -lt $NCORES ]; then
NCORES=$NTRAIN
fi
echo N_cores=
echo $NCORES

NAMEBASE="engineered_split_random"
DATESTR=`date +%m-%d-%H-%M-%N`	#datestring for jobfolder name
PROJECTDIR="/home/constantin/python/projects/stokesEquation/rom"
JOBNAME="${NAMEBASE}/${DATESTR}_nTrain=${NTRAIN}"
JOBDIR="/home/constantin/python/data/stokesEquation/meshSize=256/nonOverlappingDisks/margins=${MARG1}_${MARG2}_${MARG3}_${MARG4}/N~logn/mu=${N1}/sigma=${N2}/x~${COORDDIST}/"
if [ "$COORDDIST" = "GP" ]; then
JOBDIR="${JOBDIR}cov=${X1}/l=${X2}/sig_scale=${X3}/r~logn/mu=${R1}/sigma=${R2}/p_bc=0.0/${BCX}_${BCY}/${JOBNAME}"
elif [ "$COORDDIST" = "tiles" ]; then
JOBDIR="${JOBDIR}r~logn/mu=${R1}/sigma=${R2}/p_bc=0.0/${BCX}_${BCY}/${JOBNAME}"
elif [ "$COORDDIST" = "gauss" ]; then
JOBDIR="${JOBDIR}mu=${X1}/cov=${X2}/r~logn/mu=${R1}/sigma=${R2}/p_bc=0.0/${BCX}_${BCY}/${JOBNAME}"
elif [ "$COORDDIST" = "engineered" ]; then
JOBDIR="${JOBDIR}r~logn/mu=${R1}/sigma=${R2}/p_bc=0.0/${BCX}_${BCY}/${JOBNAME}"
fi

#Create job directory and copy source code
mkdir -p "${JOBDIR}"
cp -r $PROJECTDIR/* "$JOBDIR"
#Remove existing data folder
rm -r $JOBDIR/data

#Change directory to job directory; completely independent from project directory
cd "$JOBDIR"
echo $PWD
CWD=$(printf "%q\n" "$(pwd)")
rm ./job_file.sh

#construct job file string
echo "#PBS -N $JOBNAME" >> ./job_file.sh
echo "#PBS -l nodes=1:ppn=$NCORES,walltime=240:00:00" >> ./job_file.sh
echo "#PBS -o /home/constantin/OEfiles" >> ./job_file.sh
echo "#PBS -e /home/constantin/OEfiles" >> ./job_file.sh
echo "#PBS -m abe" >> ./job_file.sh
echo "#PBS -M mailscluster@gmail.com" >> ./job_file.sh
echo "" >> ./job_file.sh
echo "#Switch to job directory" >> ./job_file.sh
echo "cd \"$JOBDIR\"" >> ./job_file.sh
echo "" >> ./job_file.sh
echo "#Set parameters" >> ./job_file.sh
echo "sed -i \"19s/.*/nTrain = ${NTRAIN};/\" ./trainModel.m" >> ./job_file.sh
echo "sed -i \"7s/.*/        numberParams = [${N1}, ${N2}]/\" ./StokesData.m" >> ./job_file.sh
echo "sed -i \"9s/.*/        margins = [${MARG1}, ${MARG2}, ${MARG3}, ${MARG4}]/\" ./StokesData.m" >> ./job_file.sh
echo "sed -i \"10s/.*/        r_params = [${R1}, ${R2}]/\" ./StokesData.m" >> ./job_file.sh
echo "sed -i \"11s/.*/        coordDist = '${COORDDIST}'/\" ./StokesData.m" >> ./job_file.sh
if [ "$COORDDIST" = "GP" ]; then
echo "sed -i \"17s/.*/        densityLengthScale = '0.1'/\" ./StokesData.m" >> ./job_file.sh
echo "sed -i \"18s/.*/        sigmoidScale = '1.5'/\" ./StokesData.m" >> ./job_file.sh
elif [ "$COORDDIST" = "gauss" ]; then
echo "sed -i \"12s/.*/        coordDist_mu = '${X1}'/\" ./StokesData.m" >> ./job_file.sh
echo "sed -i \"13s/.*/        coordDist_cov = '${X2}'/\" ./StokesData.m" >> ./job_file.sh
fi
echo "sed -i \"39s/.*/        u_bc = {'${BCX}', '${BCY}'}/\" ./StokesData.m" >> ./job_file.sh
echo "sed -i \"17s/.*/maxCompTime = ${STOCHOPTTIME};/\" ./VI/efficientStochOpt.m" >> ./job_file.sh
echo "sed -i \"18s/.*/nSamplesStart = ${GRADIENTSAMPLESSTART};/\" ./VI/efficientStochOpt.m" >> ./job_file.sh
echo "sed -i \"19s/.*/nSamplesEnd = ${GRADIENTSAMPLESEND};/\" ./VI/efficientStochOpt.m" >> ./job_file.sh
echo "sed -i \"87s/.*/        max_EM_epochs = ${MAXEMITER}/\" ./ModelParams.m" >> ./job_file.sh
echo "sed -i \"11s/.*/testSamples = ${NTESTSTART}:${NTESTEND};/\" ./predictionScript.m" >> ./job_file.sh
echo "#Run Matlab" >> ./job_file.sh
echo "/home/matlab/R2017a/bin/matlab -nodesktop -nodisplay -nosplash -r \"trainModel ; quit;\"" >> ./job_file.sh
echo "sed -i \"9s/.*/        margins = [${MARG1}, ${MARG2}, ${MARG3}, ${MARG4}]/\" ./StokesData.m" >> ./job_file.sh
echo "sed -i \"9s/.*/        margins = [${MARG1}, ${MARG2}, ${MARG3}, ${MARG4}]/\" ./StokesData.m" >> ./job_file.sh
echo "sed -i \"9s/.*/        margins = [${MARG1}, ${MARG2}, ${MARG3}, ${MARG4}]/\" ./StokesData.m" >> ./job_file.sh
echo "sed -i \"9s/.*/        margins = [${MARG1}, ${MARG2}, ${MARG3}, ${MARG4}]/\" ./StokesData.m" >> ./job_file.sh


chmod +x job_file.sh
#directly submit job file
qsub job_file.sh
#./job_file.sh	#to test in shell









