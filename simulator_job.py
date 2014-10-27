from simulator import Simulator
import cPickle
import sys, os
import numpy as np
import tempfile
import shutil

def run_game(game,
             net,
             result_file, 
             skip_num_frames,
             max_num_frames,
             max_secs_without_reward,
             drop_rate=0):

    currnet = cPickle.load(open(net,'r'))
    currnet.clearCharges()
    sim = Simulator(game,
                    currnet.numInput,
                    skip_num_frames,
                    max_num_frames,
                    max_secs_without_reward)
    fitness = 0

    while sim.running():
        success = sim.read()
        if not success:
            break

        fitness += sim.reward

        if np.random.random >= drop_rate: # activate if signal not dropped
            currnet.clearCharges()
            currnet.setInputs(sim.objects)
            currnet.activate()
            output = currnet.readOutputs()
            if len(output) == 18:
                action = np.argmax(output)
            else:
                # compressed repr: output0 is fire; output1-9 are dirs
                # relies on output functions with range centered around 0.5, e.g., [0,1]
                action = np.argmax(output[1:]) + 1
                if action != 1:
                    # add 8 if (fire) and (dir not noop)
                    if output[0] >= 0.5: 
                        action += 8                 
                else: 
                    # substract 1 if (not fire) and (dir noop)
                    if output[0] < 0.5: action -= 1

        else: action = 0 #noop if signal dropped

        sim.write('{},18\n'.format(action))
        i += 1

    tmp = tempfile.mktemp()
    f = open(tmp, 'w')
    f.write('{}'.format(fitness))
    f.close()

    shutil.move(tmp, result_file)

if __name__ == '__main__':
    if len(sys.argv) > 6:
        game = sys.argv[1]
        net  = sys.argv[2]
        result_file = sys.argv[3]
        skip_num_frames = int(sys.argv[4])
        max_num_frames = int(sys.argv[5])
        max_secs_without_reward = int(sys.argv[6])
        drop_rate = float(sys.argv[7])
        run_game(game, net, result_file, skip_num_frames, max_num_frames, max_secs_without_reward,drop_rate)
    else:
        raise Exception("usage: simulator_job.py game /path/to/pickled/net /path/to/result/file skip_num_frames max_num_frames max_secs_without_reward drop_rate")
