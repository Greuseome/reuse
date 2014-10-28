# Class for subpopulations for GSP

# probably can integrate easily into GSP.py

import numpy as np

class Subpopulation:
    
    def __init__(self, genomeSize, populationSize):
        self.genomeSize = genomeSize
        self.populationSize = populationSize
        self.individuals = np.random.randn(self.populationSize,self.genomeSize)
        self.fitness = np.empty(self.populationSize)
        self.fitness.fill(-1)
