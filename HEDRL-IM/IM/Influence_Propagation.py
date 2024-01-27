import copy
import csv

import math
import numpy as np

import torch

torch.set_printoptions(threshold=np.inf)
seed = 123
np.random.seed(seed)


class Env:
    #  Restaurants-Rev Music-Rev NDC Bars_Rev
    def __init__(self, maxSeedNum):
        self.nameList = ['Restaurants-Rev']  # 网络数据集路径
        self.graphIndex = -1  # 数据集标记
        self.thresholdIndex = 5  # 阈值标记
        self.maxSeedsNum = maxSeedNum  # 设置种子规模
        self.nextGraph()

    # 切换网络数据集，并读取相应的邻接表和降维向量 #
    def nextGraph(self):
        self.graphIndex += 1
        self.networkName = self.nameList[self.graphIndex]  # 网络数据集名称
        self.embedInfo = self.getEmbedInfo()
        self.totalReward = 0
        self.edgeContainNodes, self.nodeNum = self.getEdgeContainNodes()
        self.nodeContainEdges, self.degreeList = self.getNodeContainEdges()
        self.hyperedgeNum = len(self.edgeContainNodes)
        self.nodeTreshold = self.allocateNodeThreshold()
        self.edgeTreshold = self.allocateEdgeThreshold()
        self.dim = self.embedInfo.shape[1] + 1  # 设置DQN输入维数
        self.initInput = self.seeds2input(set([]), set([]))
        self.activateNode = set([])
        self.activateEdge = set([])

    # Clear seed node information and reset dimensionality reduction vector #
    def reset(self):
        self.seeds = set([])
        self.activateNode = set([])
        self.activateEdge = set([])
        self.totalReward = 0
        return self.initInput

    # Based on the seed nodes selected by DQN, simulate the process of influence propagation, and use the influence score as a reward #
    def step(self, node):
        if node in self.seeds:
            print("choose repeated node!!!!!!!!!!!!!", self.seeds, node)
            state = self.seeds2input(self.seeds)
            return state, 0, False
        self.seeds.add(node)
        self.Influence(node)
        influence = len(self.activateNode)

        reward = influence - self.totalReward
        self.totalReward = influence

        isDone = False

        # End when a certain number of seed nodes are selected
        if len(self.seeds) == self.maxSeedsNum:
            isDone = True
            for i in range(self.nodeNum):
                self.nodeTreshold[i][0] = 0
            for i in range(self.hyperedgeNum):
                self.edgeTreshold[i][0] = 0
            self.activateNode = set([])
            self.activateEdge = set([])
        state = self.seeds2input(copy.deepcopy(self.activateNode), copy.deepcopy(self.activateEdge))
        return state, reward, isDone

    # Concatenate the information of the current seed node as a new vector to form a new dimensionality reduction vector for the network dataset #
    def seeds2input(self, activateNode, activateEdge):
        input = np.array(self.embedInfo.cpu().detach().numpy())
        for i in range(self.nodeNum):
            if i in activateNode:
                self.degreeList[i] = -1
            else:
                currentReward = 0
                currentNode = int(i)
                newIv = activateNode.union({currentNode})
                E = set(self.nodeContainEdges[currentNode])
                E = E - activateEdge
                for e in E:
                    nodes = set(self.edgeContainNodes[e])
                    linkTreshold = len((nodes & newIv))
                    if linkTreshold >= self.edgeTreshold[e][1]:
                        currentReward += 1
                self.degreeList[currentNode] = currentReward
        self.degreeList = np.array(self.degreeList)
        self.degreeList = self.degreeList.reshape((self.nodeNum, 1))
        input = np.hstack((self.degreeList, input))
        return input

    # Obtain dimensionality reduction vectors for network datasets #
    def getEmbedInfo(self):
        print("graph name == ", self.networkName)
        embedInfo = torch.load('../data/' + self.networkName + '_N.pt')
        print('../data/' + self.networkName + '_feature.pt')
        return embedInfo

    # Obtain the maximum influence set of the node set
    def Influence(self, node):

        self.nodeTreshold[node][0] = 1
        self.activateNode.add(node)
        while True:
            isStope = True  # Whether to stop propagating flag
            # Node to propagate hyperedges
            for i, nodes in enumerate(self.edgeContainNodes):
                if self.edgeTreshold[i][0] != 1:
                    linkTreshold = len(set(nodes) & self.activateNode)
                    if linkTreshold >= self.edgeTreshold[i][1]:
                        self.edgeTreshold[i][0] = 1
                        self.activateEdge.add(i)
                        isStope = False
            if isStope:
                break
            #  Hyperedges to propagate node
            for i, edges in enumerate(self.nodeContainEdges):
                if self.nodeTreshold[i][0] != 1:
                    linkTreshold = len(set(edges) & self.activateEdge)
                    if linkTreshold >= self.nodeTreshold[i][1]:
                        self.nodeTreshold[i][0] = 1
                        self.activateNode.add(i)
                        isStope = False
            if isStope:
                break

    def getNodeContainEdges(self):
        nodeContainEdges = [[] for x in range(self.nodeNum)]
        for i, elem in enumerate(self.edgeContainNodes):
            for node in elem:
                nodeContainEdges[node].append(i)
        degreeList = []
        for edges in nodeContainEdges:
            degreeList.append(len(edges))
        return nodeContainEdges, degreeList

    def getEdgeContainNodes(self):
        fileName = '../data/' + self.networkName + '_e2n.pt'
        edgeContainNodes = []
        allNodes = set([])
        with open(fileName) as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                newRow = []
                for x in row:
                    newRow.append(int(x))
                    allNodes.add(int(x))
                edgeContainNodes.append(newRow)
        return edgeContainNodes, len(allNodes)

    #  Node threshold allocation function
    def allocateNodeThreshold(self):
        nodeTreshold = np.zeros((self.nodeNum, 2))
        for index in range(int(self.nodeNum)):
            sum_treshold = len(self.nodeContainEdges[index])
            threshold = math.ceil(sum_treshold * 0.5)
            nodeTreshold[index][1] = threshold
        return nodeTreshold

    #  Hyperedge threshold allocation function
    def allocateEdgeThreshold(self):
        edgeTreshold = np.zeros((self.hyperedgeNum, 2))
        for index in range(int(self.hyperedgeNum)):
            sum_treshold = len(self.edgeContainNodes[index])
            threshold = int(sum_treshold / 2)
            edgeTreshold[index][1] = threshold
        return edgeTreshold
