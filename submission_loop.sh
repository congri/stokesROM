#!/bin/bash
NITER=60

for i in `seq 1 $NITER`;
do
    ./genMeshesJobFile.sh
    #./genSolutionJobFile.sh
    sleep 120    #needed s.t. no two jobs are generating the same mesh
    echo $i
done
