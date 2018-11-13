# Generates microstructural data, i.e. circles and radii of nonoverlap. disks
NDATA=200


#margins to the domain boundary where no exclusions should be
MARG1=0.003		
MARG2=0.003
MARG3=0.003
MARG4=0.003

#parameters of spatial distribution of exclusion centers
COORDDIST="GP"
X1="squaredExponential"

#Set up file paths
PROJECTDIR="/home/constantin/python/projects/stokesEquation"

N1=8.7			#usually lognormal mu for number of exclusions
for N2 in $(seq 0.1 0.1 0.3);	#usually lognormal sigma for number of exclusions
do
	echo "N2=${N2}"
	for R1 in $(seq -5.3 0.2 -4.9);			#usually lognormal mu for radii distribution
	do
		echo "R1=${R1}"
		for R2 in $(seq 0.3 0.2 0.7);			#usually lognormal sigma for radii distribution
		do
			echo "R2=${R2}"
			for X2 in $(seq 0.05 0.05 0.2);			#GP: length scale
			do
				echo "X2=${X2}"
				for X3 in $(seq 1.0 0.5 2.0);			#GP: sigmoid transform
				do
					echo "X3=${X3}"
					JOBNAME="genMesh_muN=${N1}_sigN=${N2}_muR=${R1}_sigR=${R2}_l=${X2}_sigm=${X3}"
					DATESTR=`date +%m-%d-%H-%M-%N`	#datestring for jobfolder name

					JOBDIR="/home/constantin/python/jobs/${JOBNAME}_${DATESTR}"
					JOBSCRIPT="${JOBDIR}/genMesh_cluster.py"

					#Create job directory and copy source code
					rm -rf $JOBDIR
					mkdir $JOBDIR
					MICROSTRUCTGENSCRIPT="${PROJECTDIR}/genMesh_cluster.py"
					cp $MICROSTRUCTGENSCRIPT $JOBSCRIPT
					#Change directory to job directory; completely independent from project directory
					cd $JOBDIR
					CWD=$(printf "%q\n" "$(pwd)")

					#construct job file
					echo "#!/bin/bash" >> ./job_file.sh
					echo "#SBATCH --job-name=${JOBNAME}" >> ./job_file.sh
					echo "#SBATCH --partition batch_SNB,batch_SKL" >> ./job_file.sh
					echo "#SBATCH --output=/home/constantin/OEfiles/${JOBNAME}.%j.out" >> ./job_file.sh
					echo "#SBATCH --error=/home/constantin/OEfiles/${JOBNAME}.%j.err" >> ./job_file.sh
					echo "#SBATCH --mail-type=ALL" >> ./job_file.sh
					echo "#SBATCH --mail-user=mailscluster@gmail.com " >> ./job_file.sh
					echo "#SBATCH --time=1000:00:00" >> ./job_file.sh

					echo " " >> ./job_file.sh
					echo "sed -i \"12s/.*/nMeshes = $NDATA/\" ./genMesh_cluster.py" >> ./job_file.sh
					echo "sed -i \"19s/.*/nExclusionParams = [$N1, $N2]/\" ./genMesh_cluster.py" >> ./job_file.sh
					echo "sed -i \"20s/.*/coordinateDist = '${COORDDIST}'/\" ./genMesh_cluster.py" >> ./job_file.sh
					echo "sed -i \"23s/.*/margins = [$MARG1, $MARG2, $MARG3, $MARG4]/\" ./genMesh_cluster.py" >> ./job_file.sh
					echo "sed -i \"27s/.*/r_params = [$R1, $R2]/\" ./genMesh_cluster.py" >> ./job_file.sh
					echo "sed -i \"32s/.*/covFun = '${X1}'/\" ./genMesh_cluster.py" >> ./job_file.sh
					echo "sed -i \"33s/.*/cov_l = $X2/\" ./genMesh_cluster.py" >> ./job_file.sh
					echo "sed -i \"34s/.*/sig_scale = $X3/\" ./genMesh_cluster.py" >> ./job_file.sh


					#Activate fenics environment and run python
					echo "source ~/.bashrc" >> ./job_file.sh
					echo "conda activate mshr_new" >> ./job_file.sh
					echo "ulimit -s 32000" >> ./job_file.sh 
					echo "python -u ./genMesh_cluster.py" >> ./job_file.sh
					echo "" >> ./job_file.sh

					chmod +x job_file.sh
					#directly submit job file
					sbatch job_file.sh
					sleep 5
					#execute job_file.sh in shell
					#./job_file.sh
				done
			done
		done
	done
done


