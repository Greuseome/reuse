# Class for maintaining subpopulations for GReuseOME-ESP

import ReuseNetwork as nn
import numpy as np
import random

class SubPopulation:

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
            t = (random.choice(self.recruit.inputs+self.recruit.hidden),self.r)
        elif kind == 'out':
            s = (self.r,random.choice(self.recruit.hidden+self.recruit.outputs))
            t = random.choice(self.currnet.outputs)
        weight = random.gauss(0,1)
        genome.append([s,t,weight,kind])

    def incrNumConnections(self):
        print "NEW CONNECTION"
        self.numConnections += 1
        for i in self.individuals:
            self.addConnection(self.genomes[i])

        
    
    

