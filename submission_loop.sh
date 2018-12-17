#!/bin/bash
NITER=18

for i in `seq 1 $NITER`;
do
    sleep 0
    #./genMeshesJobFile.sh
    ./genSolutionJobFile.sh
    sleep 30    #needed s.t. no two jobs are generating the same mesh
    echo $i
done
