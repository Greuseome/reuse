import time, os
import subprocess
import numpy as np

class Simulator(object):
    delimiter = ':'
    fps = 60

    def __init__(self,
                 game,
                 numInput,
                 skip_num_frames,
                 max_num_frames,
                 noreward_stop,
                 display):

        self.game = game
        self.objects = np.empty(numInput,dtype=float)
        self.skip_num_frames = skip_num_frames
        self.max_num_frames = max_num_frames
        self.noreward_stop = noreward_stop

        self.terminated = False
        self.reward = 0
        self.noreward_i = 0
        if display: display_str = 'true'
        else: display_str = 'false'
        game_str = './run_ale.sh \
                    -game_controller fifo \
                    -max_num_frames {} \
                    -display_screen {} \
                    /u/mhollen/sift/ale/roms/{}.bin' \
                    .format(self.max_num_frames,display_str,self.game)

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
        
        if self.terminated:
            return False
        
        if self.proc.returncode is not None:
            self.end()
            return False
        else:
            self.terminated = False
            self.reward = 0
            #self.objects = None

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
            self.objects[:] = [float(x) for x in objects]
            if self.reward <= 0:
                self.noreward_i += self.skip_num_frames + 1
            else:
                self.noreward_i = 0
        
        if self.terminated:
            self.proc.kill()

        return True


    def running(self):
        # end the game if performing very badly
        if (self.noreward_i/self.fps) > self.noreward_stop:
            self.end()
            return False

        """ use this for polling stdin """
        return self.proc.poll() is None and not self.terminated


class SimulatorJob(object):

    def __init__(self, index, game, netfile, resultfile, config):
        self.index = index
        self.game = game
        self.netfile = netfile
        self.resultfile = resultfile
        self.reward = None

        # load ale params
        skip_num_frames = config.getint('ale','skip_num_frames')
        noreward_stop = config.getint('ale','max_secs_without_reward')
        max_num_frames = config.getint('ale','max_num_frames')
        drop_rate = config.getfloat('ale','drop_rate')
        num_evals = config.getint('evolution','num_evals_per_net')
        display = config.getboolean('ale','display_screen')       
        job_successful = False
        while not job_successful:
            self.game_str = './condor_submit_sim.sh {} {} {} {} {} {} {} {}' \
                         .format(game, 
                                 netfile, 
                                 resultfile,
                                 skip_num_frames,
                                 max_num_frames,
                                 noreward_stop,
                                 drop_rate,
                                 num_evals,
                                 display)

            DEVNULL = open(os.devnull, 'wb')
            self.proc = subprocess.Popen(self.game_str.split(),
                                     stdin = DEVNULL,
                                     stdout= DEVNULL)

            self.proc.wait()
                
            if bool(self.proc.returncode):
                print 'Error submitting job: retrying...'
            else: job_successful = True
            

    def done(self):
        # check that path exists and that one line has been written
        if os.path.exists(self.resultfile) and os.stat(self.resultfile).st_size > 0:
            self.update_reward()
            return True
        return False

    def update_reward(self):
        f = open(self.resultfile, 'r')
        self.reward = float(f.readline())
        f.close()
