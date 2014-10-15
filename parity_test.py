# Test G-ESP on XOR problem

# extended to include Atari parity test

import GSP as NE
import copy
import ReuseNetwork as NN



TEST_REUSE = False # switch to test with or without reuse
INPUT_SIZE = 2
ATARI = True # use Atari substrates or just single input

SUBSTRATE_WIDTH = 8
SUBSTRATE_HEIGHT = 10

if TEST_REUSE:
    # build xor to reuse
    if ATARI:
        x = 5 # TO DO    
    else:
        xor = NN.ReuseNetwork(2,1)
        h = xor.addHidden(1.5)
        xor.addConnection(0,2,1)
        xor.addConnection(1,2,1)
        xor.addConnection(0,h,1)
        xor.addConnection(1,h,1)
        xor.addConnection(h,2,-2)
    
    ne = NE.NeuroEvolution(INPUT_SIZE,1,[xor],ATARI)

else:
    ne = NE.NeuroEvolution(INPUT_SIZE,1,[],ATARI)

# generate data
data = [[[],0]]
n = INPUT_SIZE
while n > 0:
    temp = []
    for i,o in data:
        temp.append([i+[0],o])
        temp.append([i+[1],(o+1)%2])
    data = temp
    n -= 1

best_fitness = -1
best_net = None
generation = 0

while best_fitness < 0.9*(2**INPUT_SIZE):
    
    curr_best_fitness = -1

    for i in range(ne.populationSize):
        currnet = ne.testNet(i)
        fitness = 0
        for d in data:
            currnet.clearCharges()
            inputs = []
            if ATARI:
                for i in d[0]:
                    inputs.extend([i]*(SUBSTRATE_WIDTH*SUBSTRATE_HEIGHT))
            else: inputs = d[0]
            currnet.setInputs(inputs)
            currnet.activate(currnet.inputs)
            o = currnet.readOutputs()
            fitness += 1 - abs(d[1] - o[0])
        ne.evaluate(fitness,i)

        if fitness > curr_best_fitness:
            curr_best_fitness = fitness
            #numIns = 0
            #for g in currnet.reuseGenomes[3]:
            #    if g[3] == 'in':
            #        numIns += 1
            #currnet.visualize()

        if fitness > best_fitness:
            best_fitness = fitness
            best_net = copy.deepcopy(currnet)
            best_net.visualize()
            print "New best: " + str(fitness)
    
    if generation % 10 == 0:
        print "Gen " + str(generation) + ", Best: " + str(curr_best_fitness) 
    ne.nextGen()
    generation += 1
print "Generation "+str(generation)+", task complete."
'''
for d in data:
    inputs = []
    if ATARI:
        for i in d[0]:
            inputs.extend([i]*(SUBSTRATE_WIDTH*SUBSTRATE_HEIGHT))
    else: inputs = d[0]
    best_net.clearCharges()
    best_net.setInputs(inputs)
    best_net.activate(best_net.inputs)
    best_net.visualize()
    x = raw_input("")
'''
ne = NE.NeuroEvolution(INPUT_SIZE,1,[copy.deepcopy(best_net)],ATARI)

best_fitness = -1
best_net = None
generation = 0

while best_fitness < 0.9*(2**INPUT_SIZE):
    
    curr_best_fitness = -1

    for i in range(ne.populationSize):
        currnet = ne.testNet(i)
        fitness = 0
        for d in data:
            currnet.clearCharges()
            inputs = []
            if ATARI:
                for i in d[0]:
                    inputs.extend([i]*(SUBSTRATE_WIDTH*SUBSTRATE_HEIGHT))
            else: inputs = d[0]
            currnet.setInputs(inputs)
            currnet.activate(currnet.inputs)
            o = currnet.readOutputs()
            fitness += 1 - abs(d[1] - o[0])
        ne.evaluate(fitness,i)

        if fitness > curr_best_fitness:
            curr_best_fitness = fitness
            #numIns = 0
            #for g in currnet.reuseGenomes[3]:
            #    if g[3] == 'in':
            #        numIns += 1
            #currnet.visualize()

        if fitness > best_fitness:
            best_fitness = fitness
            best_net = copy.deepcopy(currnet)
            best_net.visualize()
            print "New best: " + str(fitness)
    
    if generation % 10 == 0:
        print "Gen " + str(generation) + ", Best: " + str(curr_best_fitness) 
    ne.nextGen()
    generation += 1
print "Generation "+str(generation)+", task complete."

for d in data:
    inputs = []
    if ATARI:
        for i in d[0]:
            inputs.extend([i]*(SUBSTRATE_WIDTH*SUBSTRATE_HEIGHT))
    else: inputs = d[0]
    best_net.clearCharges()
    best_net.setInputs(inputs)
    best_net.activate(best_net.inputs)
    best_net.visualize()
    x = raw_input("")




