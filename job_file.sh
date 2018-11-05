#!/bin/bash
#SBATCH --job-name=genMesh_nElements=256nParams1=8.35nParams2=0.6margins=0.003_0.003_0.003_0.003r=-5.53_0.3
#SBATCH --partition batch_SNB,batch_SKL
#SBATCH --output=/home/constantin/OEfiles/genMesh_nElements=256nParams1=8.35nParams2=0.6margins=0.003_0.003_0.003_0.003r=-5.53_0.3.%j.out
#SBATCH --error=/home/constantin/OEfiles/genMesh_nElements=256nParams1=8.35nParams2=0.6margins=0.003_0.003_0.003_0.003r=-5.53_0.3.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mailscluster@gmail.com 
#SBATCH --time=1000:00:00
sed -i "17s/.*/nMeshes = 2500/" ./generateMeshes.py
sed -i "18s/.*/nElements = 256  # PDE discretization/" ./generateMeshes.py
sed -i "24s/.*/nExclusionParams = (8.35, 0.6)/" ./generateMeshes.py
sed -i "28s/.*/margins = (0.003, 0.003, 0.003, 0.003)/" ./generateMeshes.py
sed -i "32s/.*/r_params = (-5.53, 0.3)/" ./generateMeshes.py
source activate fenics3
/home/constantin/anaconda3/envs/fenics3/bin/python3.6 ./generateMeshes.py
