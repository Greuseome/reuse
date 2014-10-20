# Visualizer for Reuse Nets

import ReuseNetwork
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def visualize(net):

    CUTOFF = 0.2 # cutoff weight for displaying edges

    plt.ion()
    plt.clf()
    G = nx.Graph()
    pos = {} # positions of nodes
    labels = {} # node labels
    edge_labels = {}
    colors = [] # node colors
    edgecolors = [] #edge colors: the bluer posser, redder negger
    nodes = []
    nodesize = [] 
    layer = [] # the layer each node is in
    layer_size = [0] # the size of each layer
    deepest_layer = 0

    # set node positions
    for x in range(net.numNodes):
        for r in range(net.numReuse-1):
            if x >= net.reuseInfo[r][0] and x < net.reuseInfo[r+1][0]:
                colors.append(r+1)
        if net.numReuse > 0 and x >= net.reuseInfo[-1][0] and x < net.hiddenStart:
            colors.append(net.numReuse+1)
        elif x < net.reuseStart or x >= net.hiddenStart:
            colors.append(0)
        G.add_node(x)
        nodes.append(x)
        currLayer = 0
        for y in range(x):
            if abs(net.edgeWeights[y,x]) > 0:
                if layer[y] >= currLayer: currLayer = layer[y] + 1
        layer.append(currLayer)
        if currLayer > deepest_layer:
            deepest_layer = currLayer
            layer_size.append(1)
        else:
            layer_size[currLayer] += 1
    
    nodes_in_layer = [0]*len(layer_size)
    for x in range(net.numNodes):
        l = layer[x]
        nodes_in_layer[l] += 1
        #pos[x] = (nodes_in_layer[l]/(float(layer_size[l])+1),l)
        pos[x] = (nodes_in_layer[l]/float(max(layer_size)),l)
    
    # add edges
    for x in range(net.numNodes):
        for y in range(net.numNodes):
            if abs(net.edgeWeights[x,y]) > CUTOFF:
                G.add_edge(x,y)
                #edge_labels[x,y] = "{0:.1f}".format(net.edgeWeights[x,y])
                edgecolors.append(net.edgeWeights[x,y])
        
    # actually draw everything
    nx.draw(G,pos,node_color=colors,cmap=plt.cm.Purples,with_labels=False)  
    nx.draw_networkx_edges(G,pos,edge_color=edgecolors,
                        edge_cmap=plt.cm.RdBu,width=5)
    #nx.draw_networkx_edge_labels(G,pos,edge_labels)
 
    plt.draw()
