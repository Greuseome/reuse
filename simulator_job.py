from simulator import Simulator
import cPickle
import sys, os
import numpy as np
import tempfile
import shutil

def run_game(game,
             net,
             result_file, 
             skip_num_frames=0,
             max_num_frames=50000,
             max_secs_without_reward=50000,
             drop_rate=0,
             num_evals=1,
             display_screen=False):

    currnet = cPickle.load(open(net,'r'))
    
    total_fitness = 0
    for e in range(num_evals):
        currnet.clearCharges()
        sim = Simulator(game,
                        currnet.numInput,
                        skip_num_frames,
                        max_num_frames,
                        max_secs_without_reward,
                        display)
        fitness = 0
        i = 0
        action = 0
        while sim.running():
            success = sim.read()
            if not success:
                break
            if sim.reward != 0: print sim.reward
            fitness += sim.reward

            if np.random.random() >= drop_rate: # activate if signal not dropped
                #currnet.clearCharges()
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
            sim.write('{},18\n'.format(action))
            i += 1
        
        total_fitness += fitness
        print "TOTAL FITNESS: "+str(total_fitness)

    avg_fitness = float(total_fitness)/num_evals
    tmp = tempfile.mktemp()
    f = open(tmp, 'w')
    f.write('{}'.format(avg_fitness))
    f.close()

    shutil.move(tmp, result_file)

if __name__ == '__main__':
    l = len(sys.argv)
    if l > 3:
        game = sys.argv[1]
        net  = sys.argv[2]
        result_file = sys.argv[3]
    if l > 4: skip_num_frames = int(sys.argv[4])
    else: skip_num_frames = 0
    if l > 5: max_num_frames = int(sys.argv[5])
    else: max_num_frames = 50000
    if l > 6: max_secs_without_reward = int(sys.argv[6])
    else: max_secs_without_reward = max_num_frames
    if l > 7: drop_rate = float(sys.argv[7])
    else: drop_rate = 0.25
    if l > 8: num_evals = int(sys.argv[8])
    else: num_evals = 1
    if l > 9: display = bool(sys.argv[9])
    else: display = False
    if l > 3:
        run_game(game, net, result_file, skip_num_frames, max_num_frames, max_secs_without_reward,drop_rate,num_evals,display)
    else:
        raise Exception("usage: simulator_job.py game /path/to/pickled/net /path/to/result/file skip_num_frames max_num_frames max_secs_without_reward drop_rate num_evals display_screen")
