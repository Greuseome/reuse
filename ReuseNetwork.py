# Class for nets that reuse other nets. So optimized.

import numpy as np
import cPickle

def sigmoid(x,b):
    return (np.tanh(2*(x-b))+1)/2

class ReuseNetwork:

    def __init__(self, numInput, numOutput, config):
        self.numInput = numInput
        self.numOutput = numOutput
        self.numReuse = 0
        # (startIdx,numNodes,reuseStart,hiddenStart,outputStart,endIdx)
        self.reuseInfo = [] 
        self.numHidden = 0
        self.numNodes = self.numInput + self.numOutput
        self.edgeWeights = np.zeros( (self.numNodes,self.numNodes) )
        self.edgeCharges = np.zeros( (self.numNodes,self.numNodes) )
        self.outCharges = np.zeros(numOutput) # network output layer values
        self.nodeBias = [0.5]*self.numNodes # make numpy array!!!!

        # at what index does each layer start
        self.inputStart = 0
        self.reuseStart = self.numInput
        self.hiddenStart = self.numInput
        self.outputStart = self.numInput
        
        # types of transfer connections
        self.input2input = config.getboolean('topology','reuse_input_to_input') if config is not None else False
        self.input2hidden = config.getboolean('topology','reuse_input_to_hidden') if config is not None else False

       

    def addHidden(self, bias=0.5):
        self.addNodes(1,self.outputStart)
        self.nodeBias.insert(self.outputStart,bias)
        self.numHidden += 1
        self.outputStart += 1

    def addReuse(self,reuseNet):
        n = reuseNet.numNodes
        x = self.hiddenStart
        self.addNodes(reuseNet.numNodes,x)
        
        # fill in reuse net internal weights
        self.edgeWeights[x:x+n,x:x+n] = reuseNet.edgeWeights

        self.nodeBias = self.nodeBias[:x] + reuseNet.nodeBias + self.nodeBias[x:]
        self.numReuse += 1
        self.reuseInfo.append((x,n,reuseNet.reuseStart,
                  reuseNet.hiddenStart,reuseNet.outputStart,x+n))                   
        self.hiddenStart += n
        self.outputStart += n
        
    def addNodes(self,numNodes,startIdx):
        n = numNodes # how many to add
        x = startIdx # where to insert them

        # add rows and columns to edgeWeights and edgeCharges
        temp = np.zeros( (self.numNodes+n,self.numNodes+n) )
        temp[:x,:x] = self.edgeWeights[:x,:x]
        temp[:x,x+n:] = self.edgeWeights[:x,x:]
        temp[x+n:,:x] = self.edgeWeights[x:,:x]
        temp[x+n:,x+n:] = self.edgeWeights[x:,x:]
        self.edgeWeights = temp

        temp = np.zeros( (self.numNodes+n,self.numNodes+n) )
        temp[:x,:x] = self.edgeCharges[:x,:x]
        temp[:x,x+n:] = self.edgeCharges[:x,x:]
        temp[x+n:,:x] = self.edgeCharges[x:,:x]
        temp[x+n:,x+n:] = self.edgeCharges[x:,x:]
        self.edgeCharges = temp   

        self.numNodes += n
    
    def setInputs(self,inputs):
        # how can we optimize further? 
        r = self.reuseStart
        o = self.outputStart
        inputs = inputs.reshape(r,1)
        self.edgeCharges[:r,r:o] = inputs*self.edgeWeights[:r,r:o]

    def activate(self):
        # how can we optimize further?
        r = 0 # number of reused nets we've activated so far
        outputInchargeStart = self.reuseStart
        if r < len(self.reuseInfo):
            startIdx = self.reuseInfo[r][0]
            startHidden = startIdx + self.reuseInfo[r][2]
            startOutput = startIdx + self.reuseInfo[r][4]
            endIdx = self.reuseInfo[r][5]
            outputInchargeStart = startOutput

        for x in range(self.reuseStart,self.numNodes):
            if r < len(self.reuseInfo): # activate reuse net
                if self.input2input and x < startHidden: # reuse input
                    # connections come from currnet input
                    inCharges = np.sum(self.edgeCharges[:self.reuseStart,x])
                elif x < startOutput: # reuse reuse or hidden
                    inCharges = 0 
                    if self.input2input: # connections come from reused input
                        inCharges += np.sum(self.edgeCharges[startIdx:startHidden,x])
                    if self.input2hidden: # connections come from currnet input
                        inCharges += np.sum(self.edgeCharges[:self.reuseStart,x])
                    inCharges += self.edgeCharges[x,x] # self loops
                elif x < endIdx: # reuse output
                    # connections come from reused hidden
                    inCharges = np.sum(self.edgeCharges[startHidden:startOutput,x])
                else: 
                    r += 1 # move to next reuse net
                    if r < len(self.reuseInfo):
                        startIdx = self.reuseInfo[r][0]
                        startHidden = startIdx + self.reuseInfo[r][2]
                        startOutput = startIdx + self.reuseInfo[r][4]
                        endIdx = self.reuseInfo[r][5]
            elif x < self.outputStart: # currnet hidden
                # connections come from currnet input
                inCharges = np.sum(self.edgeCharges[:self.reuseStart,x])
                inCharges += self.edgeCharges[x,x] # self loops
            else: # currnet output
                # connections come from reuse output and currnet hidden
                inCharges = np.sum(self.edgeCharges[outputInchargeStart:self.outputStart,x])
            
            nodeOutput = sigmoid(inCharges,self.nodeBias[x])
            
            if x < self.outputStart:
                self.edgeCharges[x,x:] = nodeOutput*self.edgeWeights[x,x:]
            else: self.outCharges[x-self.outputStart] = nodeOutput

    def readOutputs(self):
        return self.outCharges

    def clearCharges(self):
        self.edgeCharges.fill(0)

    def save(self, path):
        cPickle.dump(self, open(path,'wb'),2)
        

if __name__=='__main__':
    # Tests
    x = 5
