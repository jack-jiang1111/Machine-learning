import os
import random
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'DecisionTree'))

import DecisionTree.decisionTree as Tree
import numpy as np


class Bagging:
    attribute_length = 0
    TrainAttributeData = np.empty(1)
    TrainLabelData = np.empty(1)
    TestAttributeData = np.empty(1)
    TestLabelData = np.empty(1)
    numericalData = []
    sampleSize = 1000
    unknowAsAttribute = True  # by default unknow will be treated as an attribute, if this is false, unknown will be the most common value of this attribute
    T = 500  # numbers of iteration
    print = True
    def __init__(self, trainData, trainLabel, numAttribute, testData, testLabel, sampleSize=1000, split=0, numeric=None,
                 unknown=True, T=500,print = True):
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
        self.sampleSize = sampleSize
        self.print = print

    def runBagging(self):
        FinalTesting = []
        FinalTrain = []
        for i in range(self.T):
            sampleDataIndex = []
            IndexRange = np.arange(0, len(self.TrainAttributeData) - 1, 1)
            for j in range(self.sampleSize):
                sampleDataIndex.append(random.choice(IndexRange))

            # create a sample of data
            trainDataSample = np.take(self.TrainAttributeData, sampleDataIndex, 0)
            trainLabelSample = np.take(self.TrainLabelData, sampleDataIndex, 0)
            tree = Tree.DecisionTree(trainDataSample, trainLabelSample, self.attribute_length.copy(),
                                     self.TestAttributeData.copy(), self.TestLabelData.copy(), None, 0, 100,
                                     self.numericalData.copy(), fullData=self.TrainAttributeData.copy())
            predictTrainBool, hTrain, predictTest, hTest = tree.RunTreeWithAdaboost()
            FinalTesting.append(hTest)
            FinalTrain.append(hTrain)
            del tree

            finalTestArray = np.array(FinalTesting)
            finalTrainArray = np.array(FinalTrain)

            finalTestArray = np.average(np.where(finalTestArray == "yes", 1, -1), 0)
            finalTrainArray = np.average(np.where(finalTrainArray == "yes", 1, -1), 0)

            PredictTestResult = np.where(finalTestArray > 0, "yes", "no")
            PredictTrainResult = np.where(finalTrainArray > 0, "yes", "no")

            Testaccuracy = (np.count_nonzero(np.array(PredictTestResult) == self.TestLabelData)) / len(self.TestLabelData)
            Trainaccuracy = (np.count_nonzero(np.array(PredictTrainResult) == trainLabelSample)) / len(
                trainLabelSample)
            if self.print:
                print("Accuracy for Bagging: ", i, " iteration: ", Trainaccuracy, Testaccuracy)
        return FinalTesting
