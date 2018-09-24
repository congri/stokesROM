NMESHESLO=9
NMESHESHI=2048
NELEMENTS=256
NPARAMS1=8.35
NPARAMS2=0.6
RPARAMSLO=-5.53
RPARAMSHI=0.3
MARGIN_LO=0.003
MARGIN_R=0.003
MARGIN_U=0.003
MARGIN_LE=0.003


CORES=16


#Set up file paths
PROJECTDIR="/home/constantin/python/projects/stokesEquation"
JOBNAME="genSolution_nElements=${NELEMENTS}nExMin=${NEXMIN}nExMax=${NEXMAX}margins=${MARGIN_LO}_${MARGIN_R}_${MARGIN_U}_${MARGIN_LE}r=${RPARAMSLO}_${RPARAMSHI}"
JOBDIR="/home/constantin/python/jobs/$JOBNAME"

#Create job directory and copy source code
rm -rf $JOBDIR
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
sed -i \"27s/.*/    samples = np.arange(${NMESHESLO}, ${NMESHESHI})/\" ./stokesdata.py
sed -i \"28s/.*/    nElements = ${NELEMENTS}/\" ./stokesdata.py
sed -i \"32s/.*/    nExclusionParams = (${NPARAMS1}, ${NPARAMS2})/\" ./stokesdata.py
sed -i \"37s/.*/    rParams = (${RPARAMSLO}, ${RPARAMSHI})/\" ./stokesdata.py
sed -i \"38s/.*/    margins = (${MARGIN_LO}, ${MARGIN_R}, ${MARGIN_U}, ${MARGIN_LE})/\" ./stokesdata.py




#Activate fenics environment and run python
source activate fenics2
/home/constantin/anaconda3/envs/fenics2/bin/python3.6 ./genSolutionScript.py


" >> job_file.sh

chmod +x job_file.sh
#directly submit job file
qsub job_file.sh
#execute job_file.sh in shell
#./job_file.sh



