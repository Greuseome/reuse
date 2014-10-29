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
             drop_rate):

    currnet = cPickle.load(open(net,'r'))
    currnet.clearCharges()
    sim = Simulator(game,
                    currnet.numInput,
                    skip_num_frames,
                    max_num_frames,
                    max_secs_without_reward)
    fitness = 0
    action = 0
    curr_skip = 0
    skipped = 0
    while sim.running():
        success = sim.read()
        if not success:
            break

        fitness += sim.reward

        # activate if no intentional skip and signal not dropped
        if np.random.random() >= drop_rate and skipped <= curr_skip:
            currnet.clearCharges()
            currnet.setInputs(sim.objects)
            currnet.activate()
            output = currnet.readOutputs()
            if len(output) == 18 or len(output) == 19:
                action = np.argmax(output[:18])
            else:
                # compressed repr: output0 is fire; output1-9 are dirs
                # relies on output functions with range centered around 0.5, e.g., [0,1]
                action = np.argmax(output[1:10]) + 1
                if action != 1:
                    # add 8 if (fire) and (dir not noop)
                    if output[0] >= 0.5: 
                        action += 8                 
                else: 
                    # substract 1 if (not fire) and (dir noop)
                    if output[0] < 0.5: action -= 1
            
            # get the num repeats for the current action
            if len(output) % 2 == 1:
                skip_sum = np.arctanh(output[-1]*2+1)
                if skip_sum < 0: curr_skip = 0
                elif skip_sum > 3: curr_skip = 300
                else: curr_skip = np.floor(skip_sum*100)
        
        else: skipped += 1

        sim.write('{},18\n'.format(action))



    tmp = tempfile.mktemp()
    f = open(tmp, 'w')
    f.write('{}'.format(fitness))
    f.close()

    shutil.move(tmp, result_file)

if __name__ == '__main__':
    if len(sys.argv) > 7:
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
