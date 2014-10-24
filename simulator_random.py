from simulator import Simulator
import random
import sys


if len(sys.argv) > 1:
    game = sys.argv[1]
else:
    game = 'freeway'

print "Running: {}".format(game)

sim = Simulator(game)

reward = 0
i = 0

while sim.running():
    sim.read()
    action = random.randint(0, 17)
    reward += sim.reward
    print (
          "iter: {}; ".format(i),
          "reward: {}; ".format(sim.reward),
          "total: {}; ".format(reward),
          "done: {}".format(sim.terminated)
          )
    sys.stdout.flush()

    sim.write('{},18\n'.format(action))
    i += 1

