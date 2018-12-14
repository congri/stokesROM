#!/bin/bash
NITER=4

for i in `seq 1 $NITER`;
do
    sleep 1
    #./genMeshesJobFile.sh
    ./genSolutionJobFile.sh
    sleep 30    #needed s.t. no two jobs are generating the same mesh
    echo $i
done
