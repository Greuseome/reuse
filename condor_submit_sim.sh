#!/bin/bash

JOBFILE=$(mktemp)
GAME=$1
NETWORK=$2
RESULT=$3

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

+ProjectDescription = "ALE Simulator Job"

Universe = vanilla
Executable = /usr/local/bin/python
Arguments = $PWD/simulator_job.py $GAME $NETWORK $RESULT
Requirements = InMastodon && Arch == "x86_64"

Error = $RESULT.err
Output = $RESULT.out
Log = $RESULT.log

Queue
EOL

condor_submit $JOBFILE
