# Class for nets that reuse other nets
# Terminology:
# For now, net currently being trained is called 'currnet',
# a net being reused is called a 'recruit'

import math
import copy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def sigmoid(x,b):
    # b is bias
    return 1 / (1 + math.exp(-2*(x-b)))


class ReuseNetwork:

    def __init__(self, numInput, numOutput,name=None):
        self.name = name
        self.numNodes = 0
        self.inputs = []
        self.hidden = []
        self.reuse = {} # maps recruit label to actual recruit net
        self.recruits = []
        self.outputs = []
        self.outConnections = {} # maps a node to its outgoing edges
        self.inConnections = {} # maps a node to its incoming edges
        self.reuseGenomes = {} # maps a recruit label to its connection genome
        self.edgeWeights = {}
        self.inCharges = {} # incoming charge to a node
        self.outCharges = {} # outgoing charge from a node
        self.edgeCharges = {} # weighted charge along an edge
        self.nodeBias = {}

        for i in range(numInput):
            node = self.numNodes
            self.inputs.append(node)
            self.inCharges[node] = 0
            self.outCharges[node] = 0
            self.outConnections[node] = []
            self.inConnections[node] = []
            self.numNodes += 1
        for o in range(numOutput):
            node = self.numNodes
            self.outputs.append(node)
            self.nodeBias[node] = 0.5
            self.inCharges[node] = 0
            self.outCharges[node] = 0
            self.inConnections[node] = []
            self.outConnections[node] = []
            self.numNodes += 1

    def numNodes(self):
        return self.numNodes        

    def addHidden(self, bias=0.5):
        node = self.numNodes
        self.hidden.append(node)
        self.recruits.append(node)
        self.inCharges[node] = 0
        self.outCharges[node] = 0
        self.inConnections[node] = []
        self.outConnections[node] = []
        self.nodeBias[node] = bias
        self.numNodes += 1
        return node

    def addReuse(self,reuseNet):
        node = self.numNodes
        self.recruits.append(node)
        self.reuse[node] = copy.deepcopy(reuseNet)
        self.reuseGenomes[node] = []
        self.inConnections[node] = []
        self.outConnections[node] = []
        # ensure recruits get marked for potential activation
        for i in self.inputs: self.addConnection(i,node)
        for o in self.outputs: self.addConnection(node,o)
        self.numNodes += 1
        return node

    def addConnection(self,u,v,weight=0):
        self.outConnections[u].append(v)
        self.inConnections[v].append(u)
        self.edgeWeights[u,v] = weight
        self.edgeCharges[u,v] = 0

    def addGenome(self,r,genome):
        self.reuseGenomes[r] = genome

    def removeConnection(self,u,v):
        self.outConnections[u].remove(v)
        self.inConnections[v].remove(u)
        del self.EW[u,v]
        del self.EC[u,v]
    
    def fire(self,v):
        if v in self.reuse:
            r = self.reuse[v]
            r.clearCharges()
            reuseInputs = set()
            reuseOutConnections = []
            # apply genome
            for s,t,weight,kind in self.reuseGenomes[v]:
                if s in self.inputs:
                    # set up incharges to recruit
                    r.inCharges[t[0]] += self.outCharges[s]*weight
                    reuseInputs.add(t[0])
                elif t in self.outputs:
                    reuseOutConnections.append([s,t,weight])
                else:
                    # currently genomes can only map from currnet inputs
                    # to recruit, or from recruit to currnet outputs.
                    # this can be easily changed.
                    raise Exception ("Bad genome map.")
            r.activate(list(reuseInputs))
            for s,t,weight in reuseOutConnections:
                # collect outcharges from recruit
                self.edgeCharges[v,t] += r.outCharges[s[0]]*weight
        else:
            if v in self.inputs:
                self.outCharges[v] = self.inCharges[v]
            else:
                for u in self.inConnections[v]:
                    self.inCharges[v] += self.edgeCharges[u,v]
                self.outCharges[v] = sigmoid(self.inCharges[v],self.nodeBias[v])
            for w in self.outConnections[v]:
                self.edgeCharges[v,w] = self.outCharges[v]*self.edgeWeights[v,w]

    def setInputs(self,inputs):
        if len(inputs) != len(self.inputs):
            raise Exception("Wrong input size.")
        for i in range(len(self.inputs)):
            self.inCharges[i] = inputs[i]

    def activate(self,inputs):
        active = list(inputs)
        while len(active) > 0:
            curr = active.pop(0)
            self.fire(curr)
            for v in self.outConnections[curr]:
                if v not in active:
                    active.append(v)
                    if v not in inputs: 
                        self.inCharges[v] = 0
               

    def readOutputs(self):
        outputs = []
        for o in self.outputs:
            outputs.append(self.outCharges[o])
        return outputs

    def clearCharges(self):
        for n in self.inputs + self.hidden + self.outputs:
            self.inCharges[n] = 0
            self.outCharges[n] = 0
        for e in self.edgeCharges:
            self.edgeCharges[e] = 0
        for r in self.reuse.values():
            r.clearCharges()

    def __str__(self):
        # Change this depending on what info is useful
        s = 'Nodes:\n'
        s += str(self.I) + '\n'
        s += str(self.H) + '\n'
        s += str(self.O) + '\n'
        s += str('Edges:') + '\n'
        for u in self.outConnections:
            for v in self.outConnections[u]: 
                s += str((u,v)) + ' ' + str(self.edgeWeights[u,v]) + '\n'
        return s

    def visualize(self):
        # This visualizer has some bugs; you'll see...
        plt.ion()
        plt.clf()
        G = nx.DiGraph()
        pos = {}
        colors = []
        nodes = []
        nodesize = []
        self.drawReuse(-1,G,pos,colors,nodes,nodesize,0.0,1.0,0.0,1.0)
        nx.draw(G,pos,node_color=colors,nodelist=nodes,
                with_labels=False,node_size=nodesize)
        plt.draw()

    def drawReuse(self,v,G,pos,colors,nodes,nodesize,x1,x2,y1,y2):
        a = v # v is an id for current net being drawn
        temp0 = a
        
        # the positions given here are not perfect, feel free to change.

        for n in range(len(self.inputs)):
            node = (self.inputs[n],v)
            G.add_node(node)
            nodes.append(node)
            nodesize.append(300*(y2-y1))
            x = x1 + (n+1)*(x2-x1)/(len(self.inputs)+1)
            y = y1 + (y2-y1)/4.0
            pos[node] = (x,y)           
            if self.outCharges[self.inputs[n]] > 0.5:
                colors.append('r')
            else: colors.append('b')
          
        k = 0
        for n in self.reuse:
            length = len(self.reuse) + len(self.hidden)
            x = x1 + (k+1)*((x2-x1)/(length+1))
            y = y1 + 2*(y2-y1)/4
            x3 = x - (x2-x1)/(length+1)/2
            x4 = x + (x2-x1)/(length+1)/2
            y3 = y1 + (y2-y1)/3.0
            y4 = y2 - (y2-y1)/3.0
            a += 1
            temp1 = a
            plt.gca().add_patch(Rectangle((x3,y3),x4-x3,y4-y3,
                                   facecolor='white',zorder=0))
            # recursively draw recruits
            a = self.reuse[n].drawReuse(a,G,pos,colors,nodes,
                                        nodesize,x3,x4,y3,y4)
            k = k+1
            # draw connections from and to recruit based on genome
            for s,t,weight,kind in self.reuseGenomes[n]:
                if s in self.inputs:
                    G.add_edge((s,temp0),(t[0],temp1))    
                elif t in self.outputs:
                    G.add_edge((s[0],temp1),(t,temp0))    
                else:
                    raise Exception ("BAD GENOME MAP")

        for n in range(len(self.hidden)):
            node = (self.hidden[n],v)
            G.add_node(node)
            nodes.append(node)
            nodesize.append(300*(y2-y1))
            x = x1 + (n+1+len(self.reuse))*(x2-x1)/(len(self.hidden)+len(self.reuse)+1) 
            y = y1 + 2*(y2-y1)/4.0
            pos[node] = (x,y)           
            if self.outCharges[self.hidden[n]] > 0.5:
                colors.append('r')
            else: colors.append('b')

        for n in range(len(self.outputs)):
            node = (self.outputs[n],v)
            G.add_node(node)
            nodes.append(node)
            nodesize.append(300*(y2-y1))
            x = x1 + (n+1)*(x2-x1)/(len(self.outputs)+1)
            y = y1 + 3*(y2-y1)/4.0
            pos[node] = (x,y)           
            if self.outCharges[self.outputs[n]] > 0.5:
                colors.append('r')
            else: colors.append('b')
        
        for i in self.outConnections:
            for j in self.outConnections[i]:
                if j in self.hidden+self.outputs and i in self.inputs+self.hidden:
                    G.add_edge((i,v),(j,v),weight=self.edgeWeights[i,j],
                                charge = self.edgeCharges[i,j])
        
        return a



if __name__=='__main__':
    # Tests

    '''
    # XOR test    
    xor = ReuseNetwork(2,1,'xor1')
    h = xor.addHidden(1.5)
    xor.addConnection(0,2,1)
    xor.addConnection(1,2,1)
    xor.addConnection(0,h,1)
    xor.addConnection(1,h,1)
    xor.addConnection(h,2,-2)

    newxor = ReuseNetwork(2,1,'xor2')
    r = newxor.addReuse(xor)
    genome = [  [0,(0,r),1],
                [1,(1,r),1],
                [(2,r),2,1]]
    newxor.addGenome(r,genome)

    finalxor = ReuseNetwork(2,1,'xor3')
    r = finalxor.addReuse(xor)
    genome = [  [0,(0,r),1],
                [1,(1,r),1],
                [(2,r),2,1]]
    finalxor.addGenome(r,genome)

    xor4 = ReuseNetwork(2,1,'xor4')
    h = xor4.addHidden(1.5)
    xor4.addConnection(0,2,1)
    xor4.addConnection(1,2,1)
    xor4.addConnection(0,h,1)
    xor4.addConnection(1,h,1)
    xor4.addConnection(h,2,-2)

    3bitParity = ReuseNetwork(3,1)
    r = 3bitParity.addReuse(copy.deepcopy(xor))

    

    r = finalxor.addReuse(xor)
    genome = []
    finalxor.addGenome(r,genome)
    '''
    # adj test
    adj2 = ReuseNetwork(2,1)
    adj2.addConnection(0,2,0.3)
    adj2.addConnection(1,2,0.3)

    adj3 = ReuseNetwork(3,1)
    r = adj3.addReuse(adj2)
    genome = [  [0,(0,r),1],
                [1,(1,r),1],
                [(2,r),3,1]]
    adj3.addGenome(r,genome)

    r = adj3.addReuse(adj2)
    genome = [  [1,(0,r),1],
                [2,(1,r),1],
                [(2,r),3,1]]
    adj3.addGenome(r,genome)


    
    adj5 = ReuseNetwork(5,1)
    
    r = adj5.addReuse(adj3)
    genome = [  [0,(0,r),1],
                [1,(1,r),1],
                [2,(2,r),1],
                [(3,r),5,1]]
    adj5.addGenome(r,genome)
    h = adj5.addHidden()
    adj5.addConnection(4,h,0.3)
    adj5.addConnection(3,h,0.3)
    adj5.addConnection(h,5,1)

    r = adj5.addReuse(adj2)
    genome = [  [2,(1,r),1],
                [3,(0,r),1],
                [(2,r),5,1]]
    adj5.addGenome(r,genome)

        





    for b1 in [0,1]:
        for b2 in [0,1]:
            for b3 in [0,1]:
                for b4 in [0,1]:
                    for b5 in [0,1]:
                        adj5.clearCharges()
                        adj5.setInputs([b1,b2,b3,b4,b5])
                        adj5.activate(adj5.inputs)
                        print adj5.readOutputs()
                        adj5.visualize()
                        x = raw_input("")
    '''
    for b1 in [0,1]:
        for b2 in [0,1]:
            newxor.clearCharges()
            newxor.setInputs([b1,b2])
            newxor.activate(newxor.inputs)
            newxor.visualize()
            x = raw_input("")
            print newxor.readOutputs()
    '''
