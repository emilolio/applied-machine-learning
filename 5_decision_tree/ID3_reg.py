# -------------------------- No changes needed until line 56 ----------------------------
import numpy as np
from collections import Counter
from ordered_set import OrderedSet
from graphviz import Digraph
from sklearn import tree, metrics, datasets
from ordered_set import OrderedSet


class ID3RegressionTreePredictor :


    def __init__(self, minSamplesLeaf = 1, minSamplesSplit = 2, maxDepth = 100, stopMSE = 0.0) :

        self.__nodeCounter = -1

        self.__dot = Digraph()

        self.__minSamplesLeaf = minSamplesLeaf
        self.__minSamplesSplit = minSamplesSplit
        self.__maxDepth = maxDepth
        self.__stopMSE = stopMSE

        self.__numOfAttributes = 0
        self.__attributes = None
        self.__target = None
        self.__data = None

        self.__tree = None

    def newID3Node(self):
        self.__nodeCounter += 1
        return {'id': self.__nodeCounter, 'splitValue': None, 'nextSplitAttribute': None, 'mse': None, 'samples': None,
                         'avgValue': None, 'nodes': None}


    def addNodeToGraph(self, node, parentid):
        nodeString = ''
        for k in node:
            if ((node[k] != None) and (k != 'nodes')):
                nodeString += "\n" + str(k) + ": " + str(node[k])

        self.__dot.node(str(node['id']), label=nodeString)
        if (parentid != None):
            self.__dot.edge(str(parentid), str(node['id']))
            nodeString += "\n" + str(parentid) + " -> " + str(node['id'])

        #print(nodeString)

        return


    def makeDotData(self) :
        return self.__dot

# --------------------------- YOUR TASKS t1 and t2 BELOW THIS LINE ------------------------------------

    # stubb that can be extended to a full blown MSE calculation.
    def calcMSE(self, dataIDXs):
        mse = 0.0
        avg = 0.0

        if len(dataIDXs) == 0:
            return mse, avg

        y_values = [self.__target[i] for i in dataIDXs]
        avg = np.mean(y_values)

        for y in y_values:
            mse += (y - avg) ** 2
        mse = mse / len(dataIDXs)

        return mse, avg


    # find the best split attribute out of the still possible ones ('attributes')
    # over a subset of self.__data specified through a list of indices (dataIDXs)
    def findSplitAttr(self, attributes, dataIDXs):
        print("here")
        minMSE = float("inf")
        splitAttr = ''
        splitMSEs = {}
        splitDataIDXs = {}
        splitAverages = {}

        for attr in attributes:
            attrIndex = list(self.__attributes.keys()).index(attr)
            uniqueValues = self.__attributes[attr]

            attr_mse = {}
            attr_avg = {}
            attr_idx = {}
            for val in uniqueValues:
                subsetDataIDXs = {i for i in dataIDXs if self.__data[i][attrIndex] == val}
                mse, avg = self.calcMSE(subsetDataIDXs)
                attr_mse[val] = mse
                attr_avg[val] = avg
                attr_idx[val] = subsetDataIDXs

            minVal = sum(attr_mse.values())
            if minVal < minMSE:
                minMSE = minVal
                splitAttr = attr
                splitMSEs = attr_mse
                splitAverages = attr_avg
                splitDataIDXs = attr_idx

        print(minMSE, splitAttr, splitMSEs, splitAverages, splitDataIDXs)
        return minMSE, splitAttr, splitMSEs, splitAverages, splitDataIDXs

    # --------------------- NO MORE CHANGES NEEDED UNTIL LINE 188 ------------------------------------

    # the starting point for fitting the tree
    # you should not need to change anything in here
    def fit(self, data, target, attributes):

        self.__numOfAttributes = len(attributes)
        self.__attributes = attributes
        self.__data = data
        self.__target = target


        dataIDXs = {j for j in range(len(data))}

        mse, avg = self.calcMSE(dataIDXs)

        attributesToTest = list(self.__attributes.keys())

        self.__tree = self.fit_rek( 0, None, '-', attributesToTest, mse, avg, dataIDXs)

        return self.__tree


    # the recursive tree fitting method
    # you should not need to change anything in here
    def fit_rek(self, depth, parentID, splitVal, attributesToTest, mse, avg, dataIDXs) :

        root = self.newID3Node()

        root.update({'splitValue':splitVal, 'mse': mse, 'samples': len(dataIDXs)})
        currentDepth = depth

        if (currentDepth == self.__maxDepth or mse <= self.__stopMSE or len(attributesToTest) == 0 or len(dataIDXs) < self.__minSamplesSplit):
            root.update({'avgValue':avg})
            self.addNodeToGraph(root, parentID)
            return root

        minMSE, splitAttr, splitMSEs, splitAverages, splitDataIDXs = self.findSplitAttr(attributesToTest, dataIDXs)


        root.update({'nextSplitAttribute': splitAttr, 'nodes': {}})
        self.addNodeToGraph(root, parentID)

        attributesToTestCopy = OrderedSet(attributesToTest)
        attributesToTestCopy.discard(splitAttr)

        #print(splitAttr, splitDataIDXs)

        for val in self.__attributes[splitAttr] :
            #print("testing " + str(splitAttr) + " = " + str(val))
            if( len(splitDataIDXs[val]) < self.__minSamplesLeaf) :
                root['nodes'][val] = self.newID3Node()
                root['nodes'][val].update({'splitValue':val, 'samples': len(splitDataIDXs[val]), 'avgValue': splitAverages[val]})
                self.addNodeToGraph(root['nodes'][val], root['id'])
                print("leaf, not enough samples, setting node-value = " + str(splitAverages[val]))

            else :
                root['nodes'][val] = self.fit_rek( currentDepth+1, root['id'], val, attributesToTestCopy, splitMSEs[val], splitAverages[val], splitDataIDXs[val])

        return root

    # Doing a prediction for a data set 'data' (starting method for the recursive tree traversal)
    def predict(self, data) :
        predicted = list()

        for i in range(len(data)) :
            predicted.append(self.predict_rek(data[i], self.__tree))

        return predicted

    # Recursively traverse the tree to find the value for the sample 'sample'
    def predict_rek(self, sample, node) :

        if(node['avgValue'] != None) :
            return node['avgValue']

        attr = node['nextSplitAttribute']
        dataIDX = list(self.__attributes.keys()).index(attr)
        val = sample[dataIDX]
        next = node['nodes'][val]

        return self.predict_rek( sample, next)

# -------------------------- YOUR TASK t3 BELOW THIS LINE ------------------------

    def score(self, data, target):
        y_pred = self.predict(data)
        y_true = target
        y_true_mean = np.mean(y_true)
        u = np.sum([(y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))])
        v = np.sum([(y_true[i] - y_true_mean) ** 2 for i in range(len(y_true))])
        r_squared = 1 - u / v
        return r_squared




