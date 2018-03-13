#PBS -N stokesTrain_nTrain=
#PBS -l nodes=1:ppn=8,walltime=240:00:00
#PBS -o /home/constantin/OEfiles
#PBS -e /home/constantin/OEfiles
#PBS -m abe
#PBS -M mailscluster@gmail.com

#Switch to job directory
cd ""

#Set parameters
sed -i "34s/.*/u_bc{1} = 'u_x=-0.8 + 2.0*x[1]';/" ./trainModel.m
sed -i "35s/.*/u_bc{2} = 'u_y=-1.2 + 2.0*x[0]';/" ./trainModel.m
sed -i "7s/.*/        numberParams = [6.5, 1.0]/" ./StokesData.m
sed -i "10s/.*/        r_params = [-4.5, .5]/" ./StokesData.m
sed -i "9s/.*/        margins = [-1, -1, -1, -1]/" ./StokesData.m



#Run Matlab
/home/matlab/R2017a/bin/matlab -nodesktop -nodisplay -nosplash -r "trainModel ; quit;" | tee /home/constantin/spooledOutput/03-09-19-17-096932727_stokesTrain_nTrain=