# Manage and execute evolution in GReuseOME-ESP
# Fitness should be set by the simulator

# Extended for atari experiments

import ReuseNetwork as NN
import numpy as np
import Subpopulation as SP
import random
import copy

class NeuroEvolution:

    # GA parameters
    numInitialRecruits = 10
    numGenerations = 1000
    numInitialConnections = 11 # not currently used
    populationSize = 100 # make sure divisible by 4!!!!!
    weightMutationRate = 0.8
    weightMutationStdev = 0.5
    terminalMutationRate = 0.05 # not currently used
    connectionMutationRate = 0.1 # not currently used 
    replacementRate = 0.5 # make sure divides population size w/o remainder!!
    initialWeightStdev = 0
    netStagThreshold = 20 # num gens w/o improvement -> burst mutate or recruit new
    
    
    def __init__(self,numInput,numOutput,reusables=[],atari=False):
        self.numInput = numInput
        self.numOutput = numOutput
        self.currnet = NN.ReuseNetwork(numInput,numOutput,atari)
        self.reusables = reusables # list of nets to potentially recruit
        self.bestNetSoFar = None
        self.bestFitnessSoFar = -1
        self.bestCurrFitness = -1
        self.gensWithoutImprovement = 0
        self.subPops = {}
        self.subPopBestIndiv = {}
        self.subPopBestFitness = {}
        self.testNets = []
        for i in range(self.numInitialRecruits):
            self.newRecruit()
        self.triedBurst = False
    
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
        self.subPopBestFitness[node] = -1

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
            prevFitness = self.subPops[sp].fitness[self.subPops[sp].individuals[n]] 
            self.subPops[sp].fitness[self.subPops[sp].individuals[n]] = fitness
            # Switch to promote elitism:
            #self.subPops[sp].fitness[self.subPops[sp].individuals[n]] = max(
            #                                            fitness, prevFitness)
            if fitness > self.subPopBestFitness[sp]:
                self.subPopBestFitness[sp] = fitness
                subPop = self.subPops[sp]
                self.subPopBestIndiv[sp] = copy.deepcopy(subPop.genomes[subPop.individuals[n]])
                if fitness > self.bestCurrFitness:
                    self.bestCurrFitness = fitness
        
    def mutate(self):
        for sp in self.subPops:
            subPop = self.subPops[sp]
            for i in subPop.individuals:
                for gene in subPop.genomes[i]:
                    if random.random() < self.weightMutationRate:
                        gene[2] += random.gauss(0,self.weightMutationStdev)
                    if subPop.kind == 'reuse' and not subPop.fullyConnected:
                        # mutate terminals
                        subPop.genomes[i].remove(gene)
                        subPop.addConnection(subPop.genomes[i])
                        '''
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
                         '''


    def crossover(self,g1,g2):
        # one point crossover of genomes
        if len(g1) != len(g2): raise Exception ("Bad genome lengths")
        point = random.randrange(0,len(g1),1)
        return [g1[:point] + g2[point:], g2[:point] + g1[point:]]

    def burstMutate(self):
        for sp in self.subPops:
            subPop = self.subPops[sp]
            for i in subPop.individuals:
                subPop.genomes[i] = copy.deepcopy(self.subPopBestIndiv[sp])
                for gene in subPop.genomes[i]:
                    gene[2] += random.gauss(0,1)
                    
            
    def adaptNetworkSize(self):
        # decide whether to lesion or recruit new
        return

    def replace(self):
        # replace via crossover with two parents, one from top quartile
        for sp in self.subPops.values():
            rank = sorted(sp.fitness, key=sp.fitness.get)
            topQuartile = rank[int(0.75*self.populationSize):]
            offspring = []
            #for i in range(int(self.populationSize*self.replacementRate)):
                # breed new offspring
            #   g1 = sp.genomes[random.choice(topQuartile)]
            #   g2 = sp.genomes[random.choice(rank)]

            # recombination as specified in gomez thesis:
            # each indiv in top quartile bred with a high-ranking indiv to
            # generate two offspring, which replace the bottom half.
            for i in range(len(topQuartile)):
                g1 = sp.genomes[topQuartile[i]]
                g2 = sp.genomes[random.choice(topQuartile[i:])]
                if sp.kind == 'reuse' and not sp.fullyConnected:
                    random.shuffle(g1)
                    random.shuffle(g2)
                offspring.extend(self.crossover(g1,g2))

            for i in range(int(self.populationSize*self.replacementRate)):
                # least fit of last gen die
                sp.removeIndividual(rank[i])
            for i in offspring:
                # offspring replace them
                sp.addGenome(i)

    def nextGen(self):
        self.replace()
        self.mutate()

            # Decide whether to add new connection (rephrase this code if needed)
            #if self.subPops[sp].kind == 'reuse' and not self.subPops[sp].fullyConnected:
            #    if self.subPopCurrFitness[sp] <= self.subPopBestFitness[sp]:
            #        self.subPopStag[sp] += 1
            #        if self.subPopStag[sp] == self.subPopStagThreshold:
            #            self.subPops[sp].incrNumConnections()
            #            self.subPopStag[sp] = 0
            #    else:
            #        self.subPopBestFitness[sp] = self.subPopCurrFitness[sp]
            #        self.subPopStag[sp] = 0

        # Check stagnation 
        if self.bestCurrFitness <= self.bestFitnessSoFar:
            self.gensWithoutImprovement += 1
            if self.gensWithoutImprovement == self.netStagThreshold:
                if self.triedBurst:
                    self.newRecruit()
                else:
                    self.burstMutate()
                self.triedBurst = not self.triedBurst
                self.gensWithoutImprovement = 0
        else:
            self.bestFitnessSoFar = self.bestCurrFitness
            self.gensWithoutImprovement = 0
        
        for sp in self.subPops:
            random.shuffle(self.subPops[sp].individuals)

    def evolve(self):
        for g in range(numGenerations):
            self.nextGen()



        
