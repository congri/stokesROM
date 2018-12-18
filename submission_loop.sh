#!/bin/bash
NITER=16

for i in `seq 1 $NITER`;
do
    sleep 0
    #./genMeshesJobFile.sh
    ./genSolutionJobFile.sh
    sleep 10    #needed s.t. no two jobs are generating the same mesh
    echo $i
done
