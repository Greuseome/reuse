
JOBFILE=$(mktemp)
GAME=$1
NETWORK=$2
RESULT=$3

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
Arguments = /u/mhollen/sift/reuse/simulator_job.py $GAME $NETWORK $RESULT
Requirements = InMastodon && Arch == "x86_64"

Error = condor_test.err.\$(Process) 
Output = condor_test.out.\$(Process) 
Log = condor_test.log

Queue
EOL

condor_submit $JOBFILE
