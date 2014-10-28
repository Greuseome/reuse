# GSP Neuroevolution class

import ReuseNetwork
import Subpopulation
import numpy as np
import copy

# Atari substrate dimensions
SUBSTRATE_WIDTH = 8
SUBSTRATE_HEIGHT = 10
SUBSTRATE_SIZE = SUBSTRATE_WIDTH*SUBSTRATE_HEIGHT

class GSP:

    def __init__(self,numInput,numOutput,reusables,config):

        # load GA params
        self.populationSize = config.getint('evolution','population_size')
        self.mutationStdev = config.getfloat('evolution','mutation_stdev')
        self.burstStagThreshold = config.getint('evolution','stag_threshold') 
        self.randomReplaceRate = config.getfloat('evolution','random_replace_rate')
        self.burstStdev = self.mutationStdev # keep these equal for now
        self.burstsBeforeRecruit = 1 # keep this constant for now

        numInitialRecruits = config.getint('evolution','num_initial_recruits')

        # load reuse topology settings
        self.input2input = config.getboolean('topology','reuse_input_to_input')
        self.input2hidden = config.getboolean('topology','reuse_input_to_hidden')
        
        # self loops on hidden nodes?
        self.loops = config.getboolean('topology','hidden_self_loops')

        # initialize attr
        self.atari = config.getboolean('task','atari')
        self.numInput = numInput
        if self.atari:
            self.numSubstrates = numInput
            self.numInput *= SUBSTRATE_SIZE
            self.reuseNumSubstrates = []
        self.numOutput = numOutput
        self.currnet = ReuseNetwork.ReuseNetwork(self.numInput,self.numOutput,config)
        self.reusables = copy.deepcopy(reusables)
        self.numReused = 0
        self.bestNetSoFar = None
        self.bestFitnessSoFar = -1000000
        self.bestCurrFitness = -1000000
        self.gensWithoutImprovement = 0
        self.burstsWithoutImprovement = 0
        self.subPops = [] # make this an np array instead of list of pntrs?
        self.subPopBestIndiv = []
        self.subPopBestFitness = []

        for i in range(numInitialRecruits): self.newRecruit()

    def newRecruit(self):
        # add reuse or hidden
        if len(self.reusables) > 0:
            print "New Reuse"
            recruit = self.reusables.pop()
            self.currnet.addReuse(recruit)
            self.numReused += 1

            genomeSize = 0
            
            # add input2input weights
            if self.input2input:
                if self.atari:
                    self.reuseNumSubstrates.append(recruit.numInput / SUBSTRATE_SIZE)
                    genomeSize += self.numSubstrates * self.reuseNumSubstrates[-1]
                else:
                    genomeSize += self.numInput*recruit.numInput
            
            # add input2hidden weights
            if self.input2hidden:
                genomeSize += self.numInput * recruit.numHidden
            
            # add output2output weights
            genomeSize += self.numOutput*recruit.numOutput

        else:
            print "New Hidden"
            self.currnet.addHidden()
            genomeSize = 1 + self.numInput + self.numOutput # first gene node bias
            if self.loops: genomeSize += 1 # second gene self loop weight
        
        # set up subpopulation for this recruit
        self.subPops.append(Subpopulation.Subpopulation(genomeSize,self.populationSize))
        self.subPopBestIndiv.append(None)
        self.subPopBestFitness.append(-1000000)

    def testNet(self,i):
        # set up the ith net for evaluation

        # apply reuse genomes
        for sp in range(self.numReused):
            genome = self.subPops[sp].individuals[i]
            currGene = 0
            startIdx = self.currnet.reuseInfo[sp][0]

            if self.input2input:
                # set input-to-input weights
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
            
            if self.input2hidden:
                # set input-to-hidden weights
                reuseHiddenStart = startIdx + self.currnet.reuseInfo[sp][3]
                reuseOutputStart = startIdx + self.currnet.reuseInfo[sp][4]
                reuseNumHidden = reuseOutputStart - reuseHiddenStart
                numConnections = self.numInput * reuseNumHidden
                self.currnet.edgeWeights[:self.currnet.reuseStart,reuseHiddenStart:reuseOutputStart] = (
                    genome[currGene:currGene+numConnections].reshape(self.numInput,reuseNumHidden))
                currGene += numConnections

            # set output-to-output weights
            reuseOutputStart = startIdx + self.currnet.reuseInfo[sp][4]
            endIdx = self.currnet.reuseInfo[sp][5]
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
            if self.loops:
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

        self.currnet.clearCharges()
        return self.currnet

    def evaluate(self,fitness,i):
        # set fitness of ith individual in each subpop
        for sp in range(len(self.subPops)):
            self.subPops[sp].fitness[i] = fitness
            if fitness >= self.subPopBestFitness[sp]:
                self.subPopBestFitness[sp] = fitness
                self.subPopBestIndiv[sp] = np.copy(self.subPops[sp].individuals[i])
            if fitness > self.bestCurrFitness: self.bestCurrFitness = fitness
    
    def mutate(self):
        for subPop in self.subPops:
            subPop.individuals[:self.populationSize/2] += self.mutationStdev*np.random.randn(self.populationSize/2,subPop.genomeSize)


    def replace(self):
        for subPop in self.subPops:
            # sort individuals by fitness
            p = subPop.fitness.argsort()
            subPop.individuals = subPop.individuals[p]
            subPop.fitness = subPop.fitness[p]
            
            j = 0 # current indiv to be replaced
            
            # crossover replacements
            for i in range(int(0.75*self.populationSize),self.populationSize):
                g1 = subPop.individuals[i]
                idx = np.random.randint(i,self.populationSize)
                g2 = subPop.individuals[idx]
                if len(g1) != len(g2): raise Exception ('BADDD!!!!')
                point = np.random.randint(0,len(g1)-1)
                subPop.individuals[j,:point] = g1[:point] 
                subPop.individuals[j,point:] = g2[point:]
                j += 1
                subPop.individuals[j,:point] = g2[:point] 
                subPop.individuals[j,point:] = g1[point:]
                j += 1

            # random replacements
            numRandom = int(self.randomReplaceRate*self.populationSize)
            subPop.individuals[j:j+numRandom] = np.random.randn(
                                      numRandom,subPop.genomeSize)

    def burstMutate(self):
        print 'BURSTING'
        for sp in range(len(self.subPops)):
            self.subPops[sp].individuals[:] = (self.burstStdev
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
                if self.burstsWithoutImprovement == self.burstsBeforeRecruit:
                    self.newRecruit()
                else:
                    self.burstMutate()
                    self.burstsWithoutImprovement += 1
                self.gensWithoutImprovement = 0
        else:
            self.bestFitnessSoFar = self.bestCurrFitness
            print 'New Best'
            self.gensWithoutImprovement = 0
            self.burstsWithoutImprovement = 0
        
        self.bestCurrFitness = -1000000
        for subPop in self.subPops: np.random.shuffle(subPop.individuals)


              

    

