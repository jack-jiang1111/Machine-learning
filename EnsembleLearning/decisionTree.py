from collections import Counter

from Node import *
from Split import *


class DecisionTree:
    attribute_length = 0
    root = Node(-1, "", "", 0)
    split = 0  # 0 for Entropy, 1 for ME, 2 for Gini
    PossibleAttribute = {}  # A dictionary, key is the attribute index, value is list of possible attribute
    max_depth = 100
    TrainAttributeData = np.empty(1)
    TrainLabelData = np.empty(1)
    TestAttributeData = np.empty(1)
    TestLabelData = np.empty(1)
    numericalData = []
    weight = []
    unknowAsAttribute = True  # by default unknow will be treated as an attribute, if this is false, unknown will be the most common value of this attribute
    fullData = []  # used for sample bagging, missing atrribute values
    randomForest = 0  # how many attribute we used

    def __init__(self, trainData, trainLabel, numAttribute, testData, testLabel, weight=None, split=0, depth=100,
                 numeric=None, unknown=True, fullData=None, randomForest=0):

        if numeric is None:
            numeric = []
        if weight is None:
            weight = np.ones(len(trainLabel)) / len(trainLabel)
        if fullData is None:
            fullData = trainData.copy()
        if randomForest == 0:
            randomForest = numAttribute

        self.randomForest = randomForest
        self.fullData = fullData
        self.weight = weight
        self.split = split
        self.max_depth = depth
        self.TrainAttributeData = trainData
        self.TrainLabelData = trainLabel
        self.attribute_length = numAttribute
        self.TestAttributeData = testData
        self.TestLabelData = testLabel
        self.root = Node(-1, "", "", 0)
        self.numericalData = numeric
        self.unknowAsAttribute = unknown
    def fillUnknow(self,data):
        for col in range(data.shape[1]):
            unique, pos = np.unique(data[:, col],
                                    return_inverse=True)  # Finds all unique elements and their positions
            counts = np.bincount(pos)  # Count the number of each unique element
            maxpos = counts.argmax()

            data[:, col] = np.where(data[:, col] == "unknown", maxpos, data[:, col])
    def InitializePossibleAttribute(self):


        # treat unknown attribute as the most common attribute
        if not self.unknowAsAttribute:
            self.fillUnknow(self.fullData)
            self.fillUnknow(self.TrainAttributeData)
            self.fillUnknow(self.TestAttributeData)

        # reassign numerical values to discrete binary value
        for i in self.numericalData:
            numericalArray = np.array(self.TrainAttributeData[:, i], dtype=int)
            median = np.median(numericalArray)
            self.TrainAttributeData[:, i] = np.where(numericalArray > median, 0, 1)

            numericalArray = np.array(self.TestAttributeData[:, i], dtype=int)
            median = np.median(numericalArray)
            self.TestAttributeData[:, i] = np.where(numericalArray > median, 0, 1)

            numericalArray = np.array(self.fullData[:, i], dtype=int)
            median = np.median(numericalArray)
            self.fullData[:, i] = np.where(numericalArray > median, 0, 1)
        MaxAttribute = np.arange(0, self.attribute_length, 1)
        if self.randomForest == self.attribute_length:
            AttributeList = MaxAttribute
        else:
            # random choose n attribute from [0,max attribute]
            AttributeList = np.random.choice(MaxAttribute,self.randomForest,replace = False)

        for i in AttributeList:
            self.PossibleAttribute[i] = []

        for line in self.fullData:
            for i in self.PossibleAttribute.keys():
                choices = self.PossibleAttribute[i]
                if line[i] not in choices:
                    choices.append(line[i])
                self.PossibleAttribute[i] = choices

        return AttributeList

    def checkLabel(self, labels):
        if len(labels) == 0:
            return True
        labelName = labels[0]
        for Label in labels:
            if Label != labelName:
                return False
        return True

    def mostCommonLabel(self, labels,dataIndex):
        lookUp, index = np.unique(labels,
                                  return_inverse=True)
        lookUpResult = {}
        for i in range(len(labels)):
            if labels[i] not in lookUpResult:
                lookUpResult[labels[i]] = 0
            lookUpResult[labels[i]] += self.weight[dataIndex[i]]
        maxLabel = "yes"
        maxCount = 0
        for keys in lookUpResult:
            if lookUpResult[keys]>maxCount:
                maxCount = lookUpResult[keys]
                maxLabel = keys

        return maxLabel
        #occurence_count = Counter(labels)
        #return occurence_count.most_common(1)[0][0]

    # return the attributes and label list, with the BestSplitAttribute index
    def GetSubAttributeData(self, attributesData, labels, BestSplitAttribute, value,DataIndex):
        newAttributeData = []
        newLabelData = []
        subDataIndex = []
        for A in range(len(attributesData)):
            if attributesData[A][BestSplitAttribute] == value:
                newAttributeData.append(attributesData[A])
                newLabelData.append(labels[A])
                subDataIndex.append(DataIndex[A])
        return newAttributeData, newLabelData,subDataIndex

    # currentNode: the Node current located
    # attributesData: List of list, training data
    # labels: list of labels corresponding to the attributeData
    # attributes: list of attribute still remaining
    def ConstructTree(self, currentNode, attributesData, labels, attributes,dataIndex):
        # reach the max depth, force marking labels here
        if currentNode.depth == self.max_depth:
            currentNode.label = self.mostCommonLabel(labels,dataIndex)
            return

        # All examples have the same label
        if self.checkLabel(labels):
            if len(attributes) == 0:
                currentNode.label = self.mostCommonLabel(labels,dataIndex)
            else:
                currentNode.label = labels[0]
            return
        elif len(attributes) == 0: # use all the attributes
            currentNode.label = self.mostCommonLabel(labels,dataIndex)
            return
        split = Split(attributesData, labels, self.PossibleAttribute, attributes, self.weight, self.split,
                      self.randomForest)
        BestSplitAttribute = split.split()  # attribute index
        currentNode.attribute = BestSplitAttribute

        for v in self.PossibleAttribute[BestSplitAttribute]:
            newNode = Node(-1, "", v, depth=currentNode.depth + 1)
            currentNode.addNode(newNode)

            subAttributesData, subLabelsData,subDataIndex = self.GetSubAttributeData(attributesData, labels, BestSplitAttribute, v,dataIndex)
            if len(subAttributesData) == 0:
                newNode.label = self.mostCommonLabel(labels,dataIndex)
            else:
                tempAttribute = attributes.copy()
                tempAttribute.remove(BestSplitAttribute)
                self.ConstructTree(newNode, subAttributesData, subLabelsData, tempAttribute,subDataIndex)

        return

    # Take in rows of lists of attribute
    # return predict correct rate
    def GetAccuracy(self, Data, LabelList, rate=True):
        result = []
        for rows in Data:
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
                    print("No attribute in tree, Attribute Name:", AttributeName)
                    print(TargetNode.attribute)
                    return ""
            result.append(TargetNode.label)
        if rate:
            return (np.count_nonzero(np.array(result) == LabelList)) / len(LabelList)
        else:
            return np.array(result) == LabelList, np.array(result)

    def Predict(self):
        train = self.GetAccuracy(self.TrainAttributeData, self.TrainLabelData)
        test = self.GetAccuracy(self.TestAttributeData, self.TestLabelData)
        print("Decision tree accuracy percent for training data: ", train * 100, "%")
        print("Decision tree accuracy percent for testing data: ", test * 100, "%")

    def RunTree(self):
        attributes = list(self.InitializePossibleAttribute())
        dataIndex = np.arange(0,len(self.TrainAttributeData),1)
        self.ConstructTree(self.root, self.TrainAttributeData, self.TrainLabelData, attributes,dataIndex)
        self.Predict()

    def RunTreeWithAdaboost(self):
        attributes = list(self.InitializePossibleAttribute())
        dataIndex = np.arange(0, len(self.TrainAttributeData), 1)
        self.ConstructTree(self.root, self.TrainAttributeData, self.TrainLabelData, attributes,dataIndex)
        predictTrain, hTrain = self.GetAccuracy(self.TrainAttributeData, self.TrainLabelData, rate=False)
        predictTest, hTest = self.GetAccuracy(self.TestAttributeData, self.TestLabelData, rate=False)

        return predictTrain, hTrain, predictTest, hTest
