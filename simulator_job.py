from simulator import Simulator
import pickle
import sys
import numpy as np

def run_game(game, net, result_file):
    sim = Simulator(game)
    currnet = pickle.load(open(net,'r'))
    fitness = 0

    while sim.running():
        sim.read()
        fitness += sim.reward

        currnet.clearCharges()
        currnet.setInputs(sim.objects)
        currnet.activate()
        output = currnet.readOutputs()

        f = open(result_file, 'a')
        f.write('reward: {}\n'.format(sim.reward))
        f.close()

        sim.write('{},18\n'.format(np.argmax(output)))

    sim.kill()

if __name__ == '__main__':
    if len(sys.argv) > 2:
        game = sys.argv[1]
        net  = sys.argv[2]
        result_file = sys.argv[3]
        run_game(game, net, result_file)
    else:
        raise Exception("usage: simulator_job.py game /path/to/pickled/net /path/to/result/file")
