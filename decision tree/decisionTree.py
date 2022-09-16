from Node import *
from collections import Counter
from Split import *


class DecisionTree:
    attribute_length = 0
    root = Node(-1, "", "",0)
    split = 0  # 0 for Entropy, 1 for ME, 2 for Gini
    PossibleAttribute = {}  # A dictionary, key is the attribute index, value is list of possible attribute
    max_depth = 100
    TrainAttributeData = np.empty(1)
    TrainLabelData= np.empty(1)
    TestAttributeData= np.empty(1)
    TestLabelData= np.empty(1)
    numericalData = []
    unknowAsAttribute = True # by default unknow will be treated as an attribute, if this is false, unknown will be the most common value of this attribute

    def __init__(self, trainData, trainLabel, numAttribute, testData, testLabel, split=0, depth=100, numeric=None,unknown = True):
        if numeric is None:
            numeric = []
        self.split = split
        self.max_depth = depth
        self.TrainAttributeData = trainData
        self.TrainLabelData = trainLabel
        self.attribute_length = numAttribute
        self.TestAttributeData = testData
        self.TestLabelData = testLabel
        self.root = Node(-1, "", "",0)
        self.numericalData = numeric
        self.unknowAsAttribute = unknown

    def InitializePossibleAttribute(self, data):
        for i in range(self.attribute_length):
            self.PossibleAttribute[i] = []

        # treat unknown attribute as the most common attribute
        if not self.unknowAsAttribute:
            for col in range(self.TrainAttributeData.shape[1]):
                unique, pos = np.unique(self.TrainAttributeData[:, col], return_inverse=True)  # Finds all unique elements and their positions
                counts = np.bincount(pos)  # Count the number of each unique element
                maxpos = counts.argmax()

                self.TrainAttributeData[:, col] = np.where(self.TrainAttributeData[:, col]=="unknown",maxpos,self.TrainAttributeData[:, col])

                unique, pos = np.unique(self.TestAttributeData[:, col],
                                        return_inverse=True)  # Finds all unique elements and their positions
                counts = np.bincount(pos)  # Count the number of each unique element
                maxpos = counts.argmax()
                self.TestAttributeData[:, col] = np.where(self.TestAttributeData[:, col] == "unknown",maxpos,self.TrainAttributeData[:, col])
        # reassign numerical values to discrete binary value
        for i in self.numericalData:
            numericalArray = np.array(self.TrainAttributeData[:,i], dtype=int)
            median = np.median(numericalArray)
            self.TrainAttributeData[:,i] = np.where(numericalArray>median,0,1)

            numericalArray = np.array(self.TestAttributeData[:, i], dtype=int)
            median = np.median(numericalArray)
            self.TestAttributeData[:, i] = np.where(numericalArray >= median, 0, 1)

        for line in data:
            for i in range(self.attribute_length):
                choices = self.PossibleAttribute[i]
                if line[i] not in choices:
                    choices.append(line[i])
                self.PossibleAttribute[i] = choices

    def checkLabel(self, labels):
        if len(labels) == 0:
            return True
        labelName = labels[0]
        for Label in labels:
            if Label != labelName:
                return False
        return True

    def mostCommonLabel(self, labels):
        occurence_count = Counter(labels)
        return occurence_count.most_common(1)[0][0]

    # return the attributes and label list, with the BestSplitAttribute index
    def GetSubAttributeData(self, attributesData, labels, BestSplitAttribute, value):
        newAttributeData = []
        newLabelData = []
        for A in range(len(attributesData)):
            if attributesData[A][BestSplitAttribute] == value:
                newAttributeData.append(attributesData[A])
                newLabelData.append(labels[A])
        return newAttributeData, newLabelData

    # currentNode: the Node current located
    # attributesData: List of list, training data
    # labels: list of labels corresponding to the attributeData
    # attributes: list of attribute still remaining
    def ConstructTree(self, currentNode, attributesData, labels, attributes):
        # reach the max depth, force marking labels here
        if currentNode.depth == self.max_depth:
            currentNode.label = self.mostCommonLabel(labels)
            return

        # All examples have the same label
        if self.checkLabel(labels):
            if len(attributes) == 0:
                currentNode.label = self.mostCommonLabel(labels)
            else:
                currentNode.label = labels[0]
            return
        split = Split(attributesData, labels, self.PossibleAttribute,attributes,self.split)
        BestSplitAttribute = split.split()  # attribute index
        currentNode.attribute = BestSplitAttribute

        for v in self.PossibleAttribute[BestSplitAttribute]:
            newNode = Node(-1, "", v,depth=currentNode.depth+1)
            currentNode.addNode(newNode)

            subAttributesData, subLabelsData = self.GetSubAttributeData(attributesData, labels, BestSplitAttribute, v)
            if len(subAttributesData) == 0:
                newNode.label = self.mostCommonLabel(labels)
            else:
                tempAttribute = attributes.copy()
                tempAttribute.remove(BestSplitAttribute)
                self.ConstructTree(newNode, subAttributesData, subLabelsData, tempAttribute)

        return

    # Take in rows of lists of attribute
    # return predict correct rate
    def GetAccuracy(self, AttributeList,LabelList):
        result = []
        for rows in AttributeList:
            TargetNode = self.root
            while not TargetNode.label:
                AttributeName = rows[TargetNode.attribute]
                findAttribute = False
                for subNode in TargetNode.subChildren:
                    # find attribute, do deeper into the tree until hit a leaf
                    if subNode.attributeClass == AttributeName:
                        TargetNode = subNode
                        findAttribute = True
                        break
                if not findAttribute:
                    # didn't find an attribute
                    print("No attribute in tree", AttributeName)
                    return ""
            result.append(TargetNode.label)
        return (np.count_nonzero(np.array(result)== LabelList))/len(LabelList)

    def Predict(self):
        train = self.GetAccuracy(self.TrainAttributeData, self.TrainLabelData)
        test = self.GetAccuracy(self.TestAttributeData, self.TestLabelData)
        print("Decision tree accuracy percent for training data: ",train* 100,"%")
        print("Decision tree accuracy percent for testing data: ", test * 100, "%")
    def RunTree(self):
        self.InitializePossibleAttribute(self.TrainAttributeData)
        attributes = list(range(0, self.attribute_length))
        self.ConstructTree(self.root, self.TrainAttributeData, self.TrainLabelData, attributes)
        self.Predict()