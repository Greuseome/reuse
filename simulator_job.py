from simulator import Simulator
import cPickle
import sys
import numpy as np

def run_game(game, net, result_file):
    currnet = cPickle.load(open(net,'r'))
    sim = Simulator(game)
    fitness = 0

    while sim.running():
        sim.read()
        fitness += sim.reward

        currnet.clearCharges()
        currnet.setInputs(sim.objects)
        currnet.activate()
        output = currnet.readOutputs()

        sim.write('{},18\n'.format(np.argmax(output)))

    f = open(result_file, 'w')
    f.write('{}'.format(fitness))
    f.close()

if __name__ == '__main__':
    if len(sys.argv) > 2:
        game = sys.argv[1]
        net  = sys.argv[2]
        result_file = sys.argv[3]
        run_game(game, net, result_file)
    else:
        raise Exception("usage: simulator_job.py game /path/to/pickled/net /path/to/result/file")
