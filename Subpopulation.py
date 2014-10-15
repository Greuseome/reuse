# Class for maintaining subpopulations for GReuseOME-ESP

import ReuseNetwork as nn
import numpy as np
import random

class SubPopulation:

    # switch for incremental v. fully connected
    fullyConnected = True

    # switches for atari experiments
    keep_substrate_geom = True
    tied_weights = True

    def __init__(self, currnet, r, numConnections,populationSize):
        self.n = 0
        self.numConnections = numConnections
        self.r = r
        self.currnet = currnet
        if r in currnet.hidden:
            self.kind = 'hidden'
        elif r in currnet.reuse:
            self.kind = 'reuse'
            self.recruit = currnet.reuse[r]
        else:
            raise Exception ('Bad r')
        self.individuals = [] # names of genomes
        self.genomes = {}
        self.fitness = {} # fitness of genomes
        for i in range(populationSize): self.addRandomGenome()
    
    def addRandomGenome(self):
        genome = []
        if self.kind == 'reuse':
            if self.fullyConnected:
                if self.keep_substrate_geom:
                    # connect substrates to substrates maintaining geometry
                    # note: substrate inputs have name (substrate,x,y)
                    if self.tied_weights:
                        for s0 in range(self.currnet.numSubstrates):
                            for s1 in range(self.recruit.numSubstrates):
                                genome.append([s0,(s1,self.r),random.gauss(0,1),'in'])
                    else:
                        for i in self.currnet.inputs:
                            for s in range(self.recruit.numSubstrates):
                                genome.append([i,((s,i[1],i[2]),self.r),
                                                random.gauss(0,1),'in'])
                else:
                    # connect all input to all input
                    for i0 in self.currnet.inputs:
                        for i1 in self.recruit.inputs:
                            genome.append([i0,(i1,self.r),random.gauss(0,1),'in'])
                # connect all output to all output
                for o1 in self.recruit.outputs:
                    for o0 in self.currnet.outputs:
                        genome.append([(o1,self.r),o0,random.gauss(0,1),'out'])
            else:
                for c in range(self.numConnections):
                    self.addConnection(genome)

        elif self.kind == 'hidden':
            genome.append([-1,-1,random.gauss(0,1)])
            for i in self.currnet.inputs:
                genome.append([i,self.r,random.gauss(0,1)])
            for o in self.currnet.outputs:
                genome.append([self.r,o,random.gauss(0,1)])
        self.addGenome(genome)

    def addGenome(self,genome):
        self.individuals.append(self.n)
        self.genomes[self.n] = genome
        self.fitness[self.n] = -1
        self.n += 1

    def removeIndividual(self,i):
        self.individuals.remove(i)
        del self.genomes[i]
        del self.fitness[i]

    def addConnection(self,genome):
        kind = random.choice(['in','out'])
        if kind == 'in':
            s = random.choice(self.currnet.inputs)
            t = (random.choice(self.recruit.inputs),self.r)
            #t = (random.choice(self.recruit.inputs+self.recruit.hidden),self.r)
        elif kind == 'out':
            s = (random.choice(self.recruit.outputs),self.r)
            #s = (random.choice(self.recruit.hidden+self.recruit.outputs),self.r)
            t = random.choice(self.currnet.outputs)
        weight = random.gauss(0,1)
        genome.append([s,t,weight,kind])

    def incrNumConnections(self):
        print "NEW CONNECTION"
        self.numConnections += 1
        for i in self.individuals:
            self.addConnection(self.genomes[i])

        
    
    

