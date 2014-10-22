import sys, os, time
import shutil
import GSP as NE
import numpy as np
from uuid import uuid1

from simulator import SimulatorJob

def find_num_objects(game):
    images_dir = os.path.join('/u/mhollen/sift/ale-assets/images', game)
    return len(os.listdir(images_dir))


def evolve_atari_network(game, input_size):

    ne = NE.GSP(input_size, 18, 2, [], True)

    NUM_GENERATIONS = 1000

    evolve_id = str(uuid1())
    game_dir = os.path.join(os.getcwdu(), 'results', game, evolve_id)
    if not os.path.exists(game_dir):
        os.makedirs(game_dir)

    print '*'*40
    print "All files will be saved to:"
    print game_dir
    print '*'*40

    best_fitness = -10000000
    generation = 0

    for generation in range(NUM_GENERATIONS):
        curr_best_fitness = -10000000

        generation_dir = os.path.join(game_dir, 'gen-%04d' % generation)
        if not os.path.exists(generation_dir):
            os.makedirs(generation_dir)

        # test each subpopulation (run each on condor)
        all_sims = []
        for i in range(ne.populationSize):
            currnet_base = os.path.join(generation_dir, '.net-%04d' % i)
            currnet_result = '{}.result'.format(currnet_base)
            currnet_netfile = '{}.net'.format(currnet_base)

            # save network config to file
            currnet = ne.testNet(i)
            currnet.save(currnet_netfile)

            # run simulator on network
            sim = SimulatorJob(i, game, currnet_netfile, currnet_result)
            all_sims.append(sim)

        # wait for all sims to finish
        finished = False
        while not finished:
            finished = np.all([s.done() for s in all_sims])
            time.sleep(.2)

        for sim in all_sims:
            fitness = sim.reward

            ne.evaluate(fitness, sim.index)

            if fitness > curr_best_fitness:
                curr_best_fitness = fitness

            if fitness > best_fitness:
                # save the best
                best_fitness = fitness
                os.rename(sim.netfile,
                          os.path.join(game_dir,'best_network.pkl'))
            else:
                # delete temp network file
                os.remove(sim.netfile)

        # delete generation info
        shutil.rmtree(generation_dir)

        # write current best fitness to file
        with open(os.path.join(game_dir, 'fitness.history'),'a') as curve:
            curve.write(str(curr_best_fitness)+',') 

        print "Gen " + str(generation) + ", Best: " + str(curr_best_fitness) 

        # evolve
        ne.nextGen()

    print "Generation "+str(generation)+", task complete."


if __name__ == '__main__':
    if len(sys.argv) > 1:
        game = sys.argv[1]
    else:
        game = 'breakout'
    evolve_atari_network(game, find_num_objects(game))


