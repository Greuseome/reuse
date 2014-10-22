import sys, os, time
import GSP as NE
import numpy as np
import copy
import cPickle
from uuid import uuid1

from simulator import SimulatorJob

def evolve_atari_network(game, input_size):

    ne = NE.GSP(input_size,18,2,[],True)

    NUM_GENERATIONS = 1000

    game_dir = os.path.join(os.getcwdu(), 'nets', game, str(uuid1()))
    if not os.path.exists(game_dir):
        os.makedirs(game_dir)

    print "all files saved to:"
    print game_dir

    best_fitness = -10000000
    best_net = None
    generation = 0

    while generation < NUM_GENERATIONS:
        curr_best_fitness = -10000000

        generation_dir = os.path.join(game_dir, 'gen-%04d' % generation)
        if not os.path.exists(generation_dir):
            os.makedirs(generation_dir)

        # test each subpopulation (run each on condor)
        all_sims = []
        for i in range(ne.populationSize):
            currnet_base = os.path.join(generation_dir, 'net-%04d' % i)
            currnet_result = '{}.result'.format(currnet_base)
            currnet_netfile= '{}.net'.format(currnet_base)

            # save network config to file
            currnet = ne.testNet(i)
            cPickle.dump(currnet, open(currnet_netfile,'wb'))

            # run simulator on network
            sim = SimulatorJob(i, game, currnet_netfile, currnet_result)
            all_sims.append(sim)

        # wait for all sims to finish
        finished = False
        while not finished:
            finished = np.all([s.done() for s in all_sims])
            time.sleep(1)

        for sim in all_sims:
            fitness = sim.reward()
            print "sim-{}: {}".format(sim.index, sim.reward())

            ne.evaluate(fitness, sim.index)

            if fitness > curr_best_fitness:
                curr_best_fitness = fitness

            if fitness > best_fitness:
                best_fitness = fitness

        with open('nets/{}.curve'.format(game),'a') as curve:
            curve.write(str(curr_best_fitness)+',') 

        print "Gen " + str(generation) + ", Best: " + str(curr_best_fitness) 
        ne.nextGen()
        generation += 1
    print "Generation "+str(generation)+", task complete."



if __name__ == '__main__':
    if len(sys.argv) > 1:
        game = sys.argv[1]
    else: game = 'breakout'
    evolve_atari_network(game,2)


"""
for d in data:
    best_net.clearCharges()
    best_net.setInputs(inputs)
    best_net.activate(best_net.inputs)
    best_net.visualize()
    x = raw_input("")
"""




