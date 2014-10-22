# GSP Neuroevolution class

import ReuseNetwork
import Subpopulation
import numpy as np
import copy

# Atari substrate dimensions
SUBSTRATE_WIDTH = 8
SUBSTRATE_HEIGHT = 10
SUBSTRATE_SIZE = SUBSTRATE_WIDTH*SUBSTRATE_HEIGHT

# Self loops on hidden nodes?
SELF_LOOPS = True

class GSP:

    # GA parameters
    populationSize = 100
    mutationStdev = 0.5
    burstStagThreshold = 5
    burstStdev = 1
    burstsBeforeRecruit = 1

    def __init__(self,numInput,numOutput,numInitialRecruits=5,reusables=[],atari=True):
        self.atari = atari
        self.numInput = numInput
        if atari:
            self.numSubstrates = numInput
            self.numInput *= SUBSTRATE_SIZE
            self.reuseNumSubstrates = []
        self.numOutput = numOutput
        self.currnet = ReuseNetwork.ReuseNetwork(self.numInput,self.numOutput)
        self.reusables = copy.deepcopy(reusables)
        self.numReused = 0
        self.bestNetSoFar = None
        self.bestFitnessSoFar = -1
        self.bestCurrFitness = -1
        self.gensWithoutImprovement = 0
        self.burstsWithoutImprovement = 0
        self.subPops = [] # make this an np array instead of list of pntrs?
        self.subPopBestIndiv = []
        self.subPopBestFitness = []
        for i in range(numInitialRecruits): self.newRecruit()

    def newRecruit(self):
        if len(self.reusables) > 0:
            print "New Reuse"
            recruit = self.reusables.pop()
            self.currnet.addReuse(recruit)
            self.numReused += 1

            if self.atari:
                self.reuseNumSubstrates.append(recruit.numInput / SUBSTRATE_SIZE)
                numInputWeights = self.numSubstrates * self.reuseNumSubstrates[-1]
            else:
                numInputWeights = self.numInput*recruit.numInput

            numOutputWeights = self.numOutput*recruit.numOutput
            genomeSize = numInputWeights+numOutputWeights

        else:
            print "New Hidden"
            self.currnet.addHidden()
            genomeSize = 1 + self.numInput + self.numOutput # first gene node bias
            if SELF_LOOPS: genomeSize += 1 # second gene self loop weight
        self.subPops.append(Subpopulation.Subpopulation(genomeSize,self.populationSize))
        self.subPopBestIndiv.append(None)
        self.subPopBestFitness.append(-1)

    def testNet(self,i):
        # set up the ith net for evaluation

        # apply reuse genomes
        for sp in range(self.numReused):
            genome = self.subPops[sp].individuals[i]
            currGene = 0
            # set input-to-input weights
            startIdx = self.currnet.reuseInfo[sp][0]
            if self.atari:
                for j in range(self.numSubstrates):
                    for k in range(self.reuseNumSubstrates[sp]):
                        self.currnet.edgeWeights[j*SUBSTRATE_SIZE:j*SUBSTRATE_SIZE+SUBSTRATE_SIZE,
                            startIdx+k*SUBSTRATE_SIZE:startIdx+k*SUBSTRATE_SIZE+SUBSTRATE_SIZE].fill(
                            genome[currGene])
                        currGene += 1
            else:
                numReuseInputs = self.currnet.reuseInfo[sp][2]
                numConnections = numReuseInputs * self.numInput
                self.currnet.edgeWeights[:self.currnet.reuseStart,
                    startIdx:startIdx+numReuseInputs] = (
                    genome[currGene:currGene+numConnections].reshape(self.numInput,numReuseInputs))
                currGene += numConnections

            # set output-to-output weights
            reuseOutputStart = startIdx + self.currnet.reuseInfo[sp][4]
            endIdx = startIdx + self.currnet.reuseInfo[sp][5]
            numReuseOutputs = self.currnet.reuseInfo[sp][1] - self.currnet.reuseInfo[sp][4]
            numConnections = numReuseOutputs * self.numOutput
            self.currnet.edgeWeights[reuseOutputStart:endIdx,self.currnet.outputStart:] = (
                genome[currGene:currGene+numConnections].reshape(numReuseOutputs,self.numOutput))
            currGene += numConnections

        # apply hidden genomes
        for sp in range(len(self.subPops)-self.numReused):
            genome = self.subPops[sp+self.numReused].individuals[i]
            idx = self.currnet.hiddenStart+sp
            self.currnet.nodeBias[idx] = genome[0] 
            currGene = 1 # skip node bias
            if SELF_LOOPS:
                self.currnet.edgeWeights[idx,idx] = genome[1]
                currGene += 1 # skip self loop weight
            # set input weights
            #print len(genome)
            #print len(self.currnet.edgeWeights[:self.currnet.reuseStart,idx])
            #print len(genome[currGene:currGene+self.numInput])
            self.currnet.edgeWeights[:self.currnet.reuseStart,idx] = genome[currGene:currGene+self.numInput]
            currGene += self.numInput
            # set output weights
            self.currnet.edgeWeights[idx,self.currnet.outputStart:] = genome[currGene:currGene+self.numOutput]
            currGene += self.numOutput
            if currGene != len(genome): raise Exception('Bad genome application')

        return self.currnet

    def evaluate(self,fitness,i):
        # set fitness of ith individual in each subpop
        for sp in range(len(self.subPops)):
            self.subPops[sp].fitness[i] = fitness
            if fitness > self.subPopBestFitness[sp]:
                self.subPopBestFitness[sp] = fitness
                self.subPopBestIndiv[sp] = np.copy(self.subPops[sp].individuals[i])
                if fitness > self.bestCurrFitness: self.bestCurrFitness = fitness
    
    def mutate(self):
        for subPop in self.subPops:
            subPop.individuals += self.mutationStdev*np.random.randn(self.populationSize,
                                                                        subPop.genomeSize)


    def replace(self):
        for subPop in self.subPops:
            # sort individuals by fitness
            p = subPop.fitness.argsort()
            subPop.individuals = subPop.individuals[p]
            subPop.fitness = subPop.fitness[p]
            
            j = 0 # current indiv to be replaced
            for i in range(int(0.75*self.populationSize),self.populationSize):
                # crossover
                g1 = subPop.individuals[i]
                idx = np.random.randint(i,self.populationSize)
                g2 = subPop.individuals[idx]
                point = np.random.randint(0,len(g1)-1)
                subPop.individuals[j][:point] = g1[:point] 
                subPop.individuals[j][point:] = g2[point:]
                j += 1
                subPop.individuals[j][:point] = g2[:point] 
                subPop.individuals[j][point:] = g1[point:]
                j += 1                

    def burstMutate(self):
        for sp in range(len(self.subPops)):
            self.subPops[sp].individuals = (self.burstStdev
                * np.random.randn(self.populationSize,self.subPops[sp].genomeSize)
                + self.subPopBestIndiv[sp])
    
    def adaptNetworkSize(self):
        # decide whether to lesion or recruit new
        decision = 0


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

        for subPop in self.subPops: np.random.shuffle(subPop.individuals)


              

    

