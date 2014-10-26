# Test G-ESP on parity problem

# extended to include Atari parity test

import GSP
import pickle
import ReuseNetwork as NN
import numpy as np
import copy
import NetViz
import time
import sys
import ConfigParser

# load settings file
config = ConfigParser.ConfigParser()
config.read(sys.argv[1])

# switch to test with or without reuse
TEST_REUSE = True 

INPUT_SIZE = 4

# use Atari substrates or just single input
ATARI = config.getboolean('task','atari')
SUBSTRATE_WIDTH = 8
SUBSTRATE_HEIGHT = 10

if TEST_REUSE: k = 2
else: k = INPUT_SIZE
   
reusables = []
while k <= INPUT_SIZE:
    print str(k) + ' bit parity'
    # solve k-bit parity using smaller parities
    
    # generate data
    data = [[[],0]]
    n = k
    while n > 0:
        temp = []
        for i,o in data:
            temp.append([i+[0],o])
            temp.append([i+[1],(o+1)%2])
        data = temp
        n -= 1

    ne = GSP.GSP(k,1,reusables,config)

    best_fitness = -1
    best_net = None
    generation = 0

    while best_fitness < 0.9*(2**k):
        
        curr_best_fitness = -1
        
        for i in range(ne.populationSize):
            currnet = ne.testNet(i)
            fitness = 0
            for d in data:
                inputs = []
                if ATARI:
                    for i in d[0]:
                        inputs.extend([i]*(SUBSTRATE_WIDTH*SUBSTRATE_HEIGHT))
                else: inputs = d[0]
                currnet.setInputs(np.array(inputs))
                currnet.activate()
                o = currnet.readOutputs()
                fitness += 1 - abs(d[1] - o[0])
            ne.evaluate(fitness,i)

            if fitness > curr_best_fitness:
                curr_best_fitness = fitness
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_net = copy.deepcopy(currnet)
                    print fitness
                    NetViz.visualize(best_net)

        if generation % 100 == 0:
            print "Gen " + str(generation)
        
        ne.nextGen() 
        generation += 1
    
    print "Generation "+str(generation)+", task complete."
    NetViz.visualize(best_net)
    best_net.clearCharges()
    reusables.append(best_net)
    k += 1

time.sleep(100)
