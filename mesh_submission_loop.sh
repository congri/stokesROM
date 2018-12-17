#!/bin/bash
NITER=16

for i in `seq 1 $NITER`;
do
    sleep 1
    ./genMeshesJobFile.sh
    #./genSolutionJobFile.sh
    sleep 60    #needed s.t. no two jobs are generating the same mesh
    echo $i
done
