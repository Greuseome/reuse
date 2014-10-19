import time
import subprocess
import numpy as np

class Simulator(object):
    object_size = 80
    delimiter = ':'

    def __init__(self, game):
        self.game = game
        game_str = './run_ale.sh \
                    -game_controller fifo \
                    ../ale-assets/roms/{}.bin' \
                    .format(game)
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
        while self.delimiter not in line:
            print '*'*40
            print line
            print '*'*40
            line = self.proc.stdout.readline()

        # we should be getting objects and reward from ALE
        objects, episode = line.split(self.delimiter)[:2]

        # convert reward to array
        self.terminated =  bool(int(episode.split(',')[0]))
        self.reward     =  int(episode.split(',')[1])
        # convert objects to array
        self.objects = np.array([int(n) for n in objects], \
                                dtype=np.bool)


    def running(self):
        """ use this for polling stdin """
        return self.proc.poll() is None and not self.terminated

"""
sim = Simulator('freeway')

while sim.running():
    l = sim.read()
    print l
    sim.write('2,18\n')

sim.read()
sim.wait()
"""

