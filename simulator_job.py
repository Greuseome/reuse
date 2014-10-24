from simulator import Simulator
import cPickle
import sys, os
import numpy as np
import tempfile
import shutil

def run_game(game, net, result_file):
    currnet = cPickle.load(open(net,'r'))
    sim = Simulator(game)
    fitness = 0
    i = 0

    while sim.running():
        success = sim.read()
        if not success:
            break

        fitness += sim.reward

        currnet.clearCharges()
        currnet.setInputs(sim.objects)
        currnet.activate()
        output = currnet.readOutputs()

        sim.write('{},18\n'.format(np.argmax(output)))
        i += 1

    tmp = tempfile.mktemp()
    f = open(tmp, 'w')
    f.write('{}'.format(fitness))
    f.close()

    shutil.move(tmp, result_file)

if __name__ == '__main__':
    if len(sys.argv) > 2:
        game = sys.argv[1]
        net  = sys.argv[2]
        result_file = sys.argv[3]
        run_game(game, net, result_file)
    else:
        raise Exception("usage: simulator_job.py game /path/to/pickled/net /path/to/result/file")
