BCX="u_x=-0.8 + 2.0*x[1]"
BCY="u_y=-1.2 + 2.0*x[0]"

N1=6.5
N2=1.0
R1=-4.5
R2=0.5
MARG1=-1
MARG2=-1
MARG3=-1
MARG4=-1
MU1=0.5
MU2=0.5
COV1=0.5
COV2=0.5

NTRAIN=64

NCORES=8
if [ $NTRAIN -lt $NCORES ]; then
NCORES=$NTRAIN
fi
echo N_cores=
echo $NCORES

NAMEBASE="stokesTrain"
DATESTR=`date +%m-%d-%H-%M-%N`	#datestring for jobfolder name
PROJECTDIR="/home/constantin/python/projects/stokesEquation/rom"
JOBNAME="${NAMEBASE}_nTrain=${NTRAIN}"
JOBDIR="/home/constantin/python/data/stokesEquation/meshes/meshSize=128/nNonOverlapCircExcl=logn${N1}-${N2}/coordDist=gauss_mu=[${MU1}, ${MU2}]cov=[[${COV1}, 0.0], [0.0, ${COV2}]]_margins=(${MARG1}, ${MARG2}, ${MARG3}, ${MARG4})/radiiDist=logn_r_params=(${R1}, ${R2})/${JOBNAME}"
SPOOL_FILE=/home/constantin/spooledOutput/${DATESTR}_${JOBNAME}

echo $JOBDIR

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
sed -i \"9s/.*/nTrain = ${NTRAIN};/\" ./trainModel.m
sed -i \"34s/.*/u_bc{1} = \'${BCX}\';/\" ./trainModel.m
sed -i \"35s/.*/u_bc{2} = \'${BCY}\';/\" ./trainModel.m
sed -i \"7s/.*/        numberParams = [${N1}, ${N2}]/\" ./StokesData.m
sed -i \"10s/.*/        r_params = [${R1}, ${R2}]/\" ./StokesData.m
sed -i \"9s/.*/        margins = [${MARG1}, ${MARG2}, ${MARG3}, ${MARG4}]/\" ./StokesData.m



#Run Matlab
/home/matlab/R2017a/bin/matlab -nodesktop -nodisplay -nosplash -r \"trainModel ; quit;\" | tee ${SPOOL_FILE}" >> job_file.sh

chmod +x job_file.sh
#directly submit job file
qsub job_file.sh
#./job_file.sh	#to test in shell

