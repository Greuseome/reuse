import sys
import GSP as NE
import numpy as np
import copy
import pickle

from simulator import Simulator


def evolve_atari_network(game, input_size):

    ne = NE.GSP(input_size,18,2,[],True)

    NUM_GENERATIONS = 1000

    best_fitness = -10000000
    best_net = None
    generation = 0

    while generation < NUM_GENERATIONS:
        curr_best_fitness = -10000000

        # test each subpopulation
        for i in range(ne.populationSize):
            sim = Simulator(game)
            currnet = ne.testNet(i)
            fitness = 0

            while sim.running():
                sim.read()
                fitness += sim.reward

                #print "{}, ".format(fitness), 

                currnet.clearCharges()
                currnet.setInputs(sim.objects.reshape(input_size*80,1))
                currnet.activate()
                output = currnet.readOutputs()

                sim.write('{},18\n'.format(np.argmax(output)))

            ne.evaluate(fitness, i)

            if fitness > curr_best_fitness:
                curr_best_fitness = fitness
                #currnet.visualize()

            if fitness > best_fitness:
                best_fitness = fitness
                pickle.dump(currnet, open('nets/{}.net'.format(game),'wb'))
                #best_net = copy.deepcopy(currnet)
                #best_net.visualize()

            print "gen: {}\ti: {}\trew: {}\tbest: {}\t end: {}" \
                   .format(generation, i, fitness,
                           best_fitness, sim.terminated)
            sim.kill()

        with open('nets/{}.curve'.format(game),'a') as curve:
            curve.write(str(curr_best_fitness)+',') 

        print "Gen " + str(generation) + ", Best: " + str(curr_best_fitness) 
        ne.nextGen()
        generation += 1
    print "Generation "+str(generation)+", task complete."



if __name__ == '__main__':
    if len(sys.argv) > 1:
        game = sys.argv[1]
        input_size = int(sys.argv[2])
    else: 
        game = 'breakout'
        input_size = 2
    evolve_atari_network(game,input_size)


"""
for d in data:
    best_net.clearCharges()
    best_net.setInputs(inputs)
    best_net.activate(best_net.inputs)
    best_net.visualize()
    x = raw_input("")
"""




