NTRAIN=128

NCORES=16
if [ $NTRAIN -lt $NCORES ]; then
NCORES=$NTRAIN
fi
echo N_cores=
echo $NCORES

NAMEBASE="error"
DATESTR=`date +%m-%d-%H-%M-%N`	#datestring for jobfolder name
PROJECTDIR="/home/constantin/python/projects/stokesEquation"
JOBNAME="${NAMEBASE}_nTrain=${NTRAIN}"
SPOOL_FILE=/home/constantin/spooledOutput/${DATESTR}_${JOBNAME}

JOBDIR="/home/constantin/python/data/${JOBNAME}_${DATESTR}"


#Create job directory and copy source code
mkdir -p "${JOBDIR}"
cp -r $PROJECTDIR/* "$JOBDIR"
#Remove existing data folder
rm -r $PROJECTDIR/rom/data
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
cd \"$JOBDIR/rom\"

#Set parameters
sed -i \"9s/.*/nTrain = ${NTRAIN};/\" ./trainModel.m

#Run Matlab
/home/matlab/R2017a/bin/matlab -nodesktop -nodisplay -nosplash -r \"trainModel ; quit;\" | tee ${SPOOL_FILE}" >> job_file.sh

chmod +x job_file.sh
#directly submit job file
qsub job_file.sh
#./job_file.sh	#to test in shell

