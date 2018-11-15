BCX='u_x=0.0-2.0x[1]'
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

NTRAIN=4
NTESTSTART=128
NTESTEND=1012

MAXEMEPOCHS=100

NAMEBASE="GP_2x2"
DATESTR=`date +%m-%d-%H-%M-%N`	#datestring for jobfolder name
PROJECTDIR="/home/constantin/python/projects/stokesEquation/rom"
JOBNAME="TEST_${DATESTR}_nTrain=${NTRAIN}_${NAMEBASE}"
JOBDIR="/home/constantin/python/data/stokesEquation/meshSize=256/nonOverlappingDisks/margins=${MARG1}_${MARG2}_${MARG3}_${MARG4}/N~logn/mu=${N1}/sigma=${N2}/x~${COORDDIST}/"
JOBDIR="${JOBDIR}cov=${X1}/l=${X2}/sig_scale=${X3}/r~logn/mu=${R1}/sigma=${R2}/p_bc=0.0/${BCX}_${BCY}/${JOBNAME}"

#Create job directory and copy source code
mkdir -p "${JOBDIR}"
cp $PROJECTDIR/test_script.m $JOBDIR

#Change directory to job directory; completely independent from project directory
cd "$JOBDIR"
echo $PWD

/home/programs/matlab/bin/matlab -nodesktop -nodisplay -nosplash -r "test_script; quit;"


