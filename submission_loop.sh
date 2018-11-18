#!/bin/bash
NITER=300

for i in `seq 1 $NITER`;
do
    ./genMeshesJobFile.sh
    sleep 30    #needed s.t. no two jobs are generating the same mesh
    echo $i
done
