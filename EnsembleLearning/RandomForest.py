import os
import random
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'DecisionTree'))

import DecisionTree.decisionTree as Tree
import numpy as np


class RandomForest:
    attribute_length = 0
    TrainAttributeData = np.empty(1)
    TrainLabelData = np.empty(1)
    TestAttributeData = np.empty(1)
    TestLabelData = np.empty(1)
    numericalData = []
    AttributeSize = 4
    unknowAsAttribute = True  # by default unknow will be treated as an attribute, if this is false, unknown will be the most common value of this attribute
    T = 500  # numbers of iteration
    sampleSize = 1000

    def __init__(self, trainData, trainLabel, numAttribute, testData, testLabel, AttributeSize=4, split=0, numeric=None,
                 unknown=True, T=500,sampleSize = 1000):
        if numeric is None:
            numeric = []
        self.split = split
        self.TrainAttributeData = trainData
        self.TrainLabelData = trainLabel
        self.attribute_length = numAttribute
        self.TestAttributeData = testData
        self.TestLabelData = testLabel
        self.numericalData = numeric
        self.unknowAsAttribute = unknown
        self.T = T
        self.AttributeSize = AttributeSize
        self.sampleSize = sampleSize

    def runRandomForest(self):
        FinalTesting = []
        FinalTraining = []
        for i in range(self.T):
            sampleDataIndex = []
            IndexRange = np.arange(0, len(self.TrainAttributeData) - 1, 1)
            for j in range(self.sampleSize):
                sampleDataIndex.append(random.choice(IndexRange))

            # create a sample of data
            trainDataSample = np.take(self.TrainAttributeData, sampleDataIndex, 0)
            trainLabelSample = np.take(self.TrainLabelData, sampleDataIndex, 0)
            tree = Tree.DecisionTree(trainDataSample, trainLabelSample, self.attribute_length.copy(),
                                     self.TestAttributeData.copy(), self.TestLabelData.copy(), None, 0, 15,
                                     self.numericalData.copy(), fullData=self.TrainAttributeData.copy(),
                                     randomForest=self.AttributeSize)
            predictTrainBool, hTrain, predictTest, hTest = tree.RunTreeWithAdaboost()
            FinalTesting.append(hTest)
            FinalTraining.append(hTrain)
            del tree

            finalTestArray = np.array(FinalTesting)
            finalTestArray = np.average(np.where(finalTestArray == "yes", 1, -1), 0)
            PredictResult = np.where(finalTestArray > 0, "yes", "no")
            TestAccuracy = (np.count_nonzero(np.array(PredictResult) == self.TestLabelData)) / len(self.TestLabelData)

            finalTrainArray = np.array(FinalTraining)
            finalTrainArray = np.average(np.where(finalTrainArray == "yes", 1, -1), 0)
            PredictResult = np.where(finalTrainArray > 0, "yes", "no")
            TrainAccuracy = (np.count_nonzero(np.array(PredictResult) == trainLabelSample)) / len(trainLabelSample)

            print("Accuracy for random forest: ", i, " iteration: ", TrainAccuracy,TestAccuracy)
