import time, os
import subprocess
import numpy as np

class Simulator(object):
    delimiter = ':'
    max_num_frames = 50000 # ~14 mins game play time

    def __init__(self, game):
        self.game = game
        game_str = './run_ale.sh \
                    -game_controller fifo \
                    -max_num_frames {} \
                    /u/mhollen/sift/ale/roms/{}.bin' \
                    .format(self.max_num_frames, self.game)

        self.proc = subprocess.Popen(game_str.split(),
                                stdout = subprocess.PIPE,
                                stdin  = subprocess.PIPE)
        time.sleep(1)
        self.handshake()
        self.objects = np.empty(0, dtype=bool)
        self.terminated = False
        self.reward = 0

    def handshake(self):
        """     screen: no
                   ram: no
            frame-skip: 4
                reward: yes
               objects: yes
        """
        self.write('0,0,20,1,1\n')

    def write(self, data):
        """ send ALE data """
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def read(self):
        """ read from ALE fifo (stdout) and update values
        """
        line = self.proc.stdout.readline()
        # just keep reading until we get something worthwhile
        wait_count = 0
        while self.delimiter not in line:
            line = self.proc.stdout.readline()
            wait_count += 1
            if wait_count > 1000:
                raise Exception("Exceeded maximum wait for proper ALE output")

        # we should be getting objects and reward from ALE
        objects, episode = line.split(self.delimiter)[:2]
        # convert reward to array
        self.terminated =  bool(int(episode.split(',')[0]))
        self.reward     =  int(episode.split(',')[1])
        # convert objects to array
        self.objects = np.array([int(n) for n in objects], \
                                dtype=np.bool)

        if self.terminated:
            self.proc.kill()


    def running(self):
        """ use this for polling stdin """
        return self.proc.poll() is None and not self.terminated


class SimulatorJob(object):

    def __init__(self, index, game, netfile, resultfile):
        self.index = index
        self.game = game
        self.netfile = netfile
        self.resultfile = resultfile
        self.reward = None

        self.game_str = './condor_submit_sim.sh {} {} {}' \
                         .format(game, netfile, resultfile)

        print self.game_str

        self.proc = subprocess.Popen(self.game_str.split())

        self.proc.wait()
        if bool(self.proc.returncode):
            raise Exception('Error submitting job. {}'.format(self.game_str))

    def done(self):
        if os.path.exists(self.resultfile):
            self.update_reward()
            return True
        return False

    def update_reward(self):
        f = open(self.resultfile, 'r')
        self.reward = int(f.readline())
        f.close()

