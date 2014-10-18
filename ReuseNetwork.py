# Class for nets that reuse other nets. So optimized.

import numpy as np

def sigmoid(x,b):
    return (np.tanh(2*(x-b))+1)/2.0

class ReuseNetwork:

    def __init__(self, numInput, numOutput):
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
        self.nodeBias = [0.5]*self.numNodes

        # at what index does each layer start
        self.inputStart = 0
        self.reuseStart = self.numInput
        self.hiddenStart = self.numInput
        self.outputStart = self.numInput
    
    def addHidden(self, bias=0.5):
        self.addNodes(1,self.outputStart)
        self.nodeBias.insert(self.outputStart,bias)
        self.numHidden += 1
        self.outputStart += 1

    def addReuse(self,reuseNet):
        n = reuseNet.numNodes
        print n
        x = self.hiddenStart
        self.addNodes(reuseNet.numNodes,x)
        
        # fill in reuse net internal weights
        self.edgeWeights[x:x+n,x:x+n] = reuseNet.edgeWeights

        np.insert(self.nodeBias, x, reuseNet.nodeBias)
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
        r = self.reuseStart
        o = self.outputStart
        self.edgeCharges[:r,r:o] = inputs*self.edgeWeights[:r,r:o]

    def activate(self):
        # how can we optimize further?
        for x in range(self.reuseStart,self.numNodes):
            nodeOutput = sigmoid(np.sum(self.edgeCharges[:x,x]),self.nodeBias[x])
            if x < self.outputStart:
                self.edgeCharges[x,x:] = nodeOutput*self.edgeWeights[x,x:]
            else: self.outCharges[x-self.outputStart] = nodeOutput

    def readOutputs(self):
        return self.outCharges

    def clearCharges(self):
        self.edgeCharges.fill(0)
        

if __name__=='__main__':
    # Tests
    x = 5
