#!/bin/bash
NITER=128

for i in `seq 1 $NITER`;
do
    ./genMeshesJobFile.sh
    #./genSolutionJobFile.sh
    sleep 60    #needed s.t. no two jobs are generating the same mesh
    echo $i
done
