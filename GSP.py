# Manage and execute evolution in GReuseOME-ESP
# Fitness should be set by the simulator

import ReuseNetwork as NN
import numpy as np
import Subpopulation as SP
import random

class NeuroEvolution:

    # GA parameters
    numGenerations = 1000
    numInitialConnections = 3
    numInitialRecruits = 1
    populationSize = 100
    weightMutationRate = 0.1
    weightMutationStdev = 2
    terminalMutationRate = 0.05
    replacementRate = 0.5
    initialWeightStdev = 1.0
    subPopStagThreshold = 1000 # num gens w/o improvement -> incr connection
    netStagThreshold = 10 # num gens w/o improvement -> recruit new
    
    def __init__(self,numInput,numOutput,reusables=[]):
        self.numInput = numInput
        self.numOutput = numOutput
        self.currnet = NN.ReuseNetwork(numInput,numOutput)
        self.reusables = reusables # list of nets to potentially recruit
        self.bestNetSoFar = None
        self.bestFitnessSoFar = -1
        self.bestCurrFitness = -1
        self.gensWithoutImprovement = 0
        self.subPops = {}
        self.subPopCurrFitness = {}
        self.subPopBestFitness = {}
        self.subPopStag = {} # keep track of subPop stagnation
        self.testNets = []
        for i in range(self.numInitialRecruits):
            self.newRecruit()
    
    def newRecruit(self):
        print "NEW SUBPOP"
        if len(self.subPops) < len(self.reusables):
            # recruit net if there are ones you haven't used yet
            node = self.currnet.addReuse(self.reusables[len(self.subPops)])
        else: 
            # add hidden node fully-connected to input and output layers
            node = self.currnet.addHidden()
            for i in self.currnet.inputs:
                self.currnet.addConnection(i,node)
            for o in self.currnet.outputs:
                self.currnet.addConnection(node,o)
        sp = SP.SubPopulation(self.currnet,node,
                        self.numInitialConnections,self.populationSize)
        self.subPops[node] = sp
        self.subPopCurrFitness[node] = -1
        self.subPopBestFitness[node] = -1
        self.subPopStag[node] = 0

    def testNet(self,n):
        # set up and return the nth current complete net 
        for sp in self.subPops:
            subPop = self.subPops[sp]
            genome = subPop.genomes[subPop.individuals[n]]
            if subPop.kind == 'reuse':
                self.currnet.reuseGenomes[sp] = genome
            elif subPop.kind == 'hidden':
                self.currnet.nodeBias[sp] = genome[0][2]
                for gene in genome[1:]:
                    self.currnet.edgeWeights[gene[0],gene[1]] = gene[2]
            else: raise Exception ('bad of a kind')
        self.currnet.clearCharges()
        return self.currnet

    def evaluate(self,fitness,n):
        # set fitness of individuals in nth net
        for sp in self.subPops:
            self.subPops[sp].fitness[self.subPops[sp].individuals[n]] = fitness
            if fitness > self.subPopCurrFitness[sp]:
                self.subPopCurrFitness[sp] = fitness
        if fitness > self.bestCurrFitness:
            self.bestCurrFitness = fitness
        
    def mutate(self,sp):
        subPop = self.subPops[sp]
        for i in subPop.individuals:
            for gene in subPop.genomes[i]:
                if random.random() < self.weightMutationRate:
                    gene[2] += random.gauss(0,self.weightMutationStdev)
                if subPop.kind == 'reuse':
                    if random.random() < self.terminalMutationRate:
                        if isinstance(gene[0],int): kind = 'in'
                        else: kind = 'out'
                        if random.random() < 0.5:
                            # mutate source node
                            if kind == 'in':
                                gene[0] = random.choice(self.currnet.inputs)
                            else:
                                gene[0] = (sp,random.choice(
                                      subPop.recruit.hidden+subPop.recruit.outputs))
                        else:
                            # mutate target node
                            if kind == 'in':
                                gene[1] = (random.choice(
                                    subPop.recruit.inputs+subPop.recruit.hidden),sp)
                            else:
                                gene[1] = random.choice(self.currnet.outputs)


    def crossover(self,g1,g2):
        # one point crossover of genomes
        if len(g1) != len(g2): raise Exception ("Bad genome lengths")
        point = random.randrange(0,len(g1),1)
        return g1[:point] + g2[point:]

    def replace(self,sp):
        # replace via crossover with two parents, one from top quartile
        for sp in self.subPops.values():
            rank = sorted(sp.fitness, key=sp.fitness.get)
            topQuartile = rank[int(0.75*self.populationSize):]
            offspring = []
            for i in range(int(self.populationSize*self.replacementRate)):
                # breed new offspring
                g1 = sp.genomes[random.choice(topQuartile)]
                g2 = sp.genomes[random.choice(rank)]
                if sp.kind == 'reuse':
                    random.shuffle(g1)
                    random.shuffle(g2)
                offspring.append(self.crossover(g1,g2))

            for i in range(int(self.populationSize*self.replacementRate)):
                # least fit of last gen die
                sp.removeIndividual(rank[i])
            for i in offspring:
                # offspring replace them
                sp.addGenome(i)

    def nextGen(self):
        for sp in self.subPops:
            self.replace(sp)
            self.mutate(sp)
            # Decide whether to add new connection
            if self.subPops[sp].kind == 'reuse':
                if self.subPopCurrFitness[sp] <= self.subPopBestFitness[sp]:
                    self.subPopStag[sp] += 1
                    if self.subPopStag[sp] == self.subPopStagThreshold:
                        self.subPops[sp].incrNumConnections()
                        self.subPopStag[sp] = 0
                else:
                    self.subPopBestFitness[sp] = self.subPopCurrFitness[sp]
                    self.subPopStag[sp] = 0

        # Decide whether to add new recruit
        if self.bestCurrFitness <= self.bestFitnessSoFar:
            self.gensWithoutImprovement += 1
            if self.gensWithoutImprovement == self.netStagThreshold:
                self.newRecruit()
                self.gensWithoutImprovement = 0
        else:
            self.bestFitnessSoFar = self.bestCurrFitness
            self.gensWithoutImprovement = 0
        
        for sp in self.subPops:
            random.shuffle(self.subPops[sp].individuals)
            self.subPopCurrFitness[sp] = -1

    def evolve(self):
        for g in range(numGenerations):
            self.nextGen()



        
