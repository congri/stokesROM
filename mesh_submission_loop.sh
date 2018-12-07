#!/bin/bash
NITER=50

for i in `seq 1 $NITER`;
do
    sleep 1
    ./genMeshesJobFile.sh
    #./genSolutionJobFile.sh
    sleep 300    #needed s.t. no two jobs are generating the same mesh
    echo $i
done
