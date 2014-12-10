import sys
import os
import time
import signal
import shutil
import datetime
import ConfigParser
import GSP as NE
import cPickle
import argparse

from simulator import SimulatorJob

def find_num_objects(game):
    images_dir = os.path.join('images', game)
    if not os.path.exists(images_dir):
        raise Exception("Error: cannot find Atari Game images in {}" \
                        .format(images_dir))
    return len(os.listdir(images_dir))

# keep this global for easy cleanup
generation_dir = None
all_sims = []

def evolve_atari_network(settings_file, label=None):
    # read settings file
    config = ConfigParser.ConfigParser()
    config.read(settings_file)

    # load settings
    game = config.get('task','game')
    input_size = find_num_objects(game)

    num_generations = config.getint('evolution','num_generations')
    if len(config.get('evolution','start_gen')) > 0:
        start_gen = config.getint('evolution','start_gen')
    else: start_gen = 0

    compressed_output = config.getboolean('topology','compressed_output')
    if compressed_output:
        output_size = 10
    else: output_size = 18

    # load reuse nets
    reusables = []
    reuseFiles = config.get('task','source_networks')
    if len(reuseFiles) > 0:
        for r in reuseFiles.split(','):
            reusables.append(cPickle.load(open(r,'r')))

    # init evolution or restart from checkpoint
    checkpoint = config.get('task','checkpoint')
    if len(checkpoint) > 0:
        ne = cPickle.load(open(checkpoint,'r'))
    else:
        ne = NE.GSP(input_size, output_size, reusables, config)

    # define experiment folder
    evolve_id = datetime.datetime.now().strftime('%Y%M%d-%H%M-%f')
    if not label:
        game_dir = os.path.join(os.getcwdu(), 'results', game, evolve_id)
    else:
        game_dir = os.path.join(os.getcwdu(), 'results', game, label, evolve_id)

    if not os.path.exists(game_dir):
        os.makedirs(game_dir)

    fitness_file = os.path.join(game_dir, 'fitness.history')
    bestnet_file = os.path.join(game_dir, 'best_network.pkl')
    ne_file      = os.path.join(game_dir, 'ne.pkl')

    # copy settings into experiment folder
    shutil.copy(settings_file, os.path.join(game_dir,'settings.ini'))

    # make'em readable by everyone -- for science!
    for f in [settings_file, fitness_file, bestnet_file, ne_file, game_dir]:
        os.chmod(f, 777)

    print '*'*40
    print "All files will be saved to:"
    print game_dir
    print '*'*40

    best_fitness = -10000000

    for generation in range(start_gen, num_generations):
        curr_best_fitness = -10000000

        generation_dir = os.path.join(game_dir, 'gen-%04d' % generation)
        if not os.path.exists(generation_dir):
            os.makedirs(generation_dir)

        # test each subpopulation (run each on condor)
        all_sims = []
        print "submitting jobs..."
        for i in range(ne.populationSize):
            currnet_base = os.path.join(generation_dir, '.net-%04d' % i)
            currnet_result = '{}.result'.format(currnet_base)
            currnet_netfile = '{}.net'.format(currnet_base)

            # save network config to file
            currnet = ne.testNet(i)
            currnet.save(currnet_netfile)

            # run simulator on network
            sim = SimulatorJob(i, game, currnet_netfile, currnet_result, config)
            all_sims.append(sim)

        print "waiting to finish..."
        # wait for all sims to finish
        sims_finished = [False]*ne.populationSize 
        finished = False
        while not finished:
            for i in range(len(sims_finished)):
                if not sims_finished[i]:
                    sims_finished[i] = all_sims[i].done()
            finished = all(sims_finished)
            time.sleep(.2)

        print "analyzing..."
        for sim in all_sims:
            fitness = sim.reward

            ne.evaluate(fitness, sim.index)

            if fitness > curr_best_fitness:
                curr_best_fitness = fitness

            if fitness > best_fitness:
                # save the best
                best_fitness = fitness
                os.rename(sim.netfile, bestnet_file)
            else:
                # delete temp network file
                os.remove(sim.netfile)

        # delete generation info
        shutil.rmtree(generation_dir, ignore_errors=True)

        # write current best fitness to file
        with open(fitness_file,'a') as curve:
            curve.write(str(curr_best_fitness)+',') 

        # save neuroevolution progress       
        with open(ne_file, 'wb') as world_state:
            cPickle.dump(ne, world_state, 2)

        print "+ gen: {}, best: {}, Best: {}".format(generation, 
                                                     curr_best_fitness,
                                                     best_fitness)

        # evolve
        print "evolving..."
        ne.nextGen()

    # write newline for easy formatting later
    with open(os.path.join(game_dir, 'fitness.history'),'a') as curve:
        curve.write('\n') 

    print "Generation "+str(generation)+", task complete."


def exit_gracefully(signal, frame):
    print "exiting prematurely!..."
    # delete generation info
    shutil.rmtree(generation_dir, ignore_errors=True)
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, exit_gracefully)

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label', help="label for experiment")
    parser.add_argument('-o', '--outdir', help="root output results directory")
    parser.add_argument("settings_file", help="settings file for experiment")
    args = parser.parse_args()

    evolve_atari_network(args.settings_file, label=args.label)


