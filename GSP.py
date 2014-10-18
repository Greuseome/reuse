# GSP Neuroevolution class

import ReuseNetwork
import Subpopulation
import numpy as np
import copy

# Atari substrate dimensions
SUBSRATE_WIDTH = 8
SUBSTRATE_HEIGHT = 10
SUBSTRATE_SIZE = SUBSTRATE_WIDTH*SUBSTRATE_HEIGHT

class GSP:

    # GA parameters
    populationSize = 100
    mutationStdev = 0.5
    burstStagThreshold = 10
    burstStdev = 1
    burstsBeforeRecruit = 5

    def __init__(self,numInput,numOutput,numInitialRecruits=5,reusables=[],atari=True):
        self.atari = atari
        self.numInput = numInput
        if atari:
            self.numSubstrates = numInput
            self.numInput *= SUBSTRATE_SIZE
            self.reuseNumSubstrates = []
        self.numOutput = numOutput
        self.currnet = ReuseNetwork.ReuseNetwork(numInput,numOutput)
        self.reusables = copy.deepcopt(reusables)
        self.numReused = 0
        self.bestNetSoFar = None
        self.bestFitnessSoFar = -1
        self.bestCurrFitness = -1
        self.gensWithoutImprovement = 0
        self.burstsWithoutImrovement = 0
        self.subPops = [] # make this an np array instead of list of pntrs?
        self.subPopBestIndiv = []
        self.subPopBestFitness = []
        for i in range(numInitialRecruits): self.newRecruit()

    def newRecruit(self):
        if len(reusables) > 0:
            recruit = self.reusables.pop()
            self.currnet.addReuse(recruit)
            self.numReused += 1

            if atari:
                self.reuseNumSubstrates.append(recruit.numInput / SUBSTRATE_SIZE)
                numInputWeights = self.numSubstrates * self.reuseNumSubstrate[-1]
            else:
                numInputWeights = self.numInput*recruit.numInput

            numOutputWeights = self.numOutput*recruit.numOutput
            genomeSize = numInputWeights+numOutputWeights

        else:
            self.currnet.addHidden()
            genomeSize = 1 + self.numInput + self.numOutput # plus 1 for node bias

        self.subPops.append(Subpopulation.Subpopulation(genomeSize,self.populationSize)
        self.subPopBestIndiv.append(None)
        self.subPopBestFitness.append(-1)

    def testNet(self,i):
        # set up the ith net for evaluation
        for sp in range(self.numReused):
            genome = self.subPops[sp].individuals[i]
            currGene = 0
            # set input-to-input weights
            startIdx = self.currnet.reuseInfo[sp][0]
            if atari:
                for j in range(self.numSubstrates):
                    for k in range(self.reuseNumSubstrates[sp]):
                        self.currnet.edgeWeights[j*SUBSTRATE_SIZE:j*SUBSTRATE_SIZE+SUBSTRATE_SIZE,
                            startIdx+k*SUBSTRATE_SIZE:startIdx+k*SUBSTRATE_SIZE+SUBSTRATE_SIZE].fill(
                            genome[currGene])
                        currGene += 1
            else:
                numReuseInputs = self.currnet.reuseInfo[sp][2]
                numConnections = numReuseInputs * self.numInputs
                self.currnet.edgeWeights[:currnet.reuseStart,startIdx:startIdx+numReuseInputs] = (
                    genome[currGene:currGene+numConnections].shape(self.numInputs,numReuseInputs))
                currGene += numConnections

            # set output-to-output weights
            reuseOutputStart = startIdx + self.currnet.reuseInfo[sp][4]
            endIdx = startIdx + self.currnet.reuseInfo[sp][5]
            numReuseOutputs = endIdx - reuseOutputStart
            numConnections = numReuseOutput * self.numOutput
            self.currnet.edgeWeights[reuseOutputStart:endIdx,currnet.outputStart:] = (
                genome[currGene:currGene+numConnections].shape(numReuseOutputs,self.numOutputs))
            currGene += numConnections

        for sp in range(len(self.subPops)-len(self.reused)):
            genome = self.subPops[sp].individuals[i]
            idx = self.currnet.hiddenStart+sp
            # set input weights
            self.currnet.edgeWeights[:currnet.reuseStart,idx] = genome[currGene:currGene+self.numInputs]
            currGene += self.numInputs
            # set output weights
            self.currnet.edgeWeights[idx,currnet.outputStart:] = genome[currGene:currGene+self.numOutputs]
            currGene += self.numOutputs

        return self.currnet

    def evaluate(self,fittness,i):
        # set fitness of ith individual in each subpop
        for sp in range(len(self.subPops)):
            self.subPops[sp].fitness[i] = fitness
            if fitness > self.subPopBestFitness[sp]
                self.subPopBestFitness[sp] = fitness
                self.subPopBestIndiv[sp] = np.copy(self.subPops[sp].individual[i])
                if fitness > self.bestCurrFitness: self.bestCurrFitness = fitness
    
    def mutate(self):
        for subPop in self.subPops:
            subPop.individuals += mutationStdev*np.random.randn(subPop.individuals.shape)

    def crossover(g1,g2):
        # one point crossover of genomes
        point = np.random.randint(0,len(g1)-1)
        return g1[:point] + g2[point:], g2[:point] + g1[point:]

    def replace(self):
        for subPop in self.subPops:
            # sort individuals by fitness
            p = subPop.fitness.argsort()
            subPop.individuals = subPop.individuals[p]
            subPop.fitness = subPop.fitness[sp]
            
            j = 0 # current indiv to be replaced
            for i in range(0.75*self.populationSize,self.populationSize):
                # crossover
                g1 = subPop.individuals[i]
                g2 = np.random.choice[subPop.individuals[i:]]
                point = np.random.randint(0,len(g1)-1)
                subPop.individuals[j] = g1[:point] + g2[point:]
                j += 1
                subPop.individuals[j] = g2[:point] + g1[point:]
                j += 1                

    def burstMutate(self):
        for sp in self.subPops:
            self.subPops[sp].individuals = (self.burstStdev
                * np.random.randn(self.subPops[sp].individuals.shape)
                + self.subPopBestIndiv[sp])
    
    def adaptNetworkSize(self):
        # decide whether to lesion or recruit new


    def nextGen(self):

        self.replace()
        self.mutate()

        if self.bestCurrFitness <= self.bestFitnessSoFar:
            self.gensWithoutImprovement += 1
            if self.gensWithoutImprovement == self.burstStagThreshold:
                self.burstsWithoutImprovement += 1
                if self.burstsWithoutImprovement == self.burstsBeforeRecruit:
                    self.newRecruit()
                    self.burstsWithoutImprovement = 0
                else:
                    self.burstMutate()
                self.gensWithoutImprovement = 0
        else:
            self.bestFitnessSoFar = self.bestCurrFitness
            self.gensWithoutImprovement = 0

        for subPop in subPops: subPop.individuals.random.shuffle()



              

    

