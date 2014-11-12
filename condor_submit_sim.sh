#!/bin/bash

GAME=$1
NETWORK=$2
RESULT=$3
SKIPFRAMES=$4
MAXFRAMES=$5
MAXNOREW=$6
DROPRATE=$7
JOBDIR=$(dirname $RESULT)
JOBFILE=$(mktemp -p $JOBDIR)

if [[ $# -lt 3 ]]; then
    echo "usage: $(basename $0) <game> <network-file> <results-file>"
fi

cat > $JOBFILE <<EOL
######################################### 
# 
# Example 4: Show off some fancy features 
# and local predicates. 
# 
#########################################

+Group = "GRAD" 
+Project = "AI_ROBOTICS"

+ProjectDescription = "Transfer learning in ALE"

Universe = vanilla
Executable = /usr/local/bin/python
Arguments = $PWD/simulator_job.py $GAME $NETWORK $RESULT $SKIPFRAMES $MAXFRAMES $MAXNOREW $DROPRATE $NUM_EVALS $DISPLAY_SCREEN
Requirements = InMastodon && Arch == "x86_64"

Error = $RESULT.err
Output = $RESULT.out
Log = $RESULT.log

Queue
EOL

condor_submit $JOBFILE

sleep 5

rm $JOBFILE

