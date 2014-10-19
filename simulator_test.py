import GSP as NE
import numpy as np
import copy

from simulator import Simulator


def evolve_atari_network(game, input_size):

    ne = NE.GSP(input_size*80,18,8,[],False)

    best_fitness = -1
    best_net = None
    generation = 0

    while best_fitness < 0.9*(2**input_size):
        curr_best_fitness = -1

        # test each subpopulation
        for i in range(ne.populationSize):
            sim = Simulator(game)
            currnet = ne.testNet(i)
            fitness = 0

            while sim.running():
                sim.read()
                fitness += sim.reward

                print "{}, ".format(fitness), 

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
                #best_net = copy.deepcopy(currnet)
                #best_net.visualize()

            print "gen: {}\ti: {}\trew: {}\tbest: {}\t end: {}" \
                   .format(generation, i, fitness,
                           best_fitness, sim.terminated)


        print "Gen " + str(generation) + ", Best: " + str(curr_best_fitness) 
        ne.nextGen()
        generation += 1
    print "Generation "+str(generation)+", task complete."

evolve_atari_network('freeway',3)

"""
for d in data:
    best_net.clearCharges()
    best_net.setInputs(inputs)
    best_net.activate(best_net.inputs)
    best_net.visualize()
    x = raw_input("")
"""




