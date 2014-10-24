import time, os
import subprocess
import numpy as np

class Simulator(object):
    delimiter = ':'
    max_num_frames = 50000 # ~14 mins game play time
    fps = 60

    def __init__(self,
                 game,
                 skip_num_frames=20,
                 noreward_stop=60):

        self.game = game
        self.objects = np.empty(0, dtype=bool)
        self.terminated = False
        self.reward = 0
        self.noreward_i = 0
        self.skip_num_frames = skip_num_frames

        self.noreward_stop = noreward_stop

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

    def handshake(self):
        """     screen: no
                   ram: no
            frame-skip: #
                reward: yes
               objects: yes
        """
        self.write('0,0,{},1,1\n'.format(self.skip_num_frames))

    def write(self, data):
        """ send ALE data """
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def end(self):
        if self.proc.returncode is None:
            self.proc.kill()
        self.terminated = True
        self.reward = 0
        self.objects = None

    def read(self):
        """ read from ALE fifo (stdout) and update values
        """

        if self.proc.returncode is not None:
            self.end()
            return False
        else:
            self.terminated = False
            self.reward = 0
            self.objects = None

        # just keep reading until we get something worthwhile
        wait_count = 0
        while True:
            line = self.proc.stdout.readline()
            if self.delimiter in line:
                break
            if 'DIE' in line:
                self.end()
                return False

            wait_count += 1
            if wait_count > 1000:
                return False

        # we should be getting objects and reward from ALE
        objects, episode = line.split(self.delimiter)[:2]
        # convert reward to array
        t, r = episode.split(',')
        if r != '' and t != '':
            self.terminated =  bool(int(t))
            self.reward     =  int(r)
            # convert objects to array
            self.objects = np.array([float(n) for n in objects], \
                                    dtype=np.float)
            if self.reward == 0:
                self.noreward_i += self.skip_num_frames
            else:
                self.noreward_i = 0

        if self.terminated:
            self.proc.kill()
            return False

        return True


    def running(self):
        # end the game if performing very badly
        if (self.noreward_i/self.fps) > self.noreward_stop:
            self.end()
            return False

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

        DEVNULL = open(os.devnull, 'wb')
        self.proc = subprocess.Popen(self.game_str.split(),
                                     stdin = DEVNULL,
                                     stdout= DEVNULL)

        self.proc.wait()
        if bool(self.proc.returncode):
            raise Exception('Error submitting job. {}'.format(self.game_str))

    def done(self):
        # check that path exists and that one line has been written
        if os.path.exists(self.resultfile) and os.stat(self.resultfile).st_size > 0:
            self.update_reward()
            return True
        return False

    def update_reward(self):
        f = open(self.resultfile, 'r')
        self.reward = int(f.readline())
        f.close()

