# Generate random net from size distribution of input nets
# Usage: this out_file [input nets]

import sys
sys.path.append('/v/filer4b/v20q001/ekm/Research/dev/reuse')
import cPickle
import ReuseNetwork as NN
import numpy as np

# Set size of input and output layers based on domain and transfer protocol
NUM_INPUT = 1 # since only transferring hidden
NUM_OUTPUT = 10 # using atari compressed output

# load argmuments
out_file = sys.argv[1]
input_nets = [cPickle.load(open(s,'r')) for s in sys.argv[2:]]
print "num input nets: " + str(len(input_nets))

# compute size of random net hidden layer
avg_hidden = sum([s.numHidden for s in input_nets])/float(len(input_nets))
print "avg hidden nodes: " + str(avg_hidden)

# construct random net
random_hidden = int(round(avg_hidden))
print "random hidden: " + str(random_hidden)
random_net = NN.ReuseNetwork(NUM_INPUT,NUM_OUTPUT, None)
for h in range(random_hidden): random_net.addHidden()
num_nodes = random_net.numNodes
random_net.edgeWeights = np.random.randn(num_nodes,num_nodes)
random_net.nodeBias = list(np.random.randn(num_nodes))

# write to file
cPickle.dump(random_net,open(out_file,'wb'),2)

