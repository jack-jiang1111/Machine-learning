import os
import random
import sys

from decisionTree import *

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
            IndexRange = np.arange(0, len(self.TrainAttributeData) - 1, 1)
            sampleDataIndex = np.random.choice(IndexRange,self.sampleSize,replace=True)

            # create a sample of data
            trainDataSample = np.take(self.TrainAttributeData, sampleDataIndex, 0)
            trainLabelSample = np.take(self.TrainLabelData, sampleDataIndex, 0)
            tree = DecisionTree(trainDataSample, trainLabelSample, self.attribute_length.copy(),
                                     self.TestAttributeData.copy(), self.TestLabelData.copy(), None, 0, 100,
                                     self.numericalData.copy(), fullData=self.TrainAttributeData.copy(),
                                     randomForest=self.AttributeSize)
            predictTrainBool, hTrain, predictTest, hTest = tree.RunTreeWithAdaboost()
            FinalTesting.append(hTest)
            FinalTraining.append(hTrain)
            del tree

            if self.attribute_length == 19:
                finalTestArray = np.array(FinalTesting)
                finalTestArray = np.average(np.where(finalTestArray == "1", 1, -1), 0)
                PredictResult = np.where(finalTestArray > 0, "1", "0")
                TestAccuracy = (np.count_nonzero(np.array(PredictResult) == self.TestLabelData)) / len(self.TestLabelData)

                finalTrainArray = np.array(FinalTraining)
                finalTrainArray = np.average(np.where(finalTrainArray == "1", 1, -1), 0)
                PredictResult = np.where(finalTrainArray > 0, "1", "0")
                TrainAccuracy = (np.count_nonzero(np.array(PredictResult) == trainLabelSample)) / len(trainLabelSample)
            else:
                finalTestArray = np.array(FinalTesting)
                finalTestArray = np.average(np.where(finalTestArray == "yes", 1, -1), 0)
                PredictResult = np.where(finalTestArray > 0, "yes", "no")
                TestAccuracy = (np.count_nonzero(np.array(PredictResult) == self.TestLabelData)) / len(
                    self.TestLabelData)

                finalTrainArray = np.array(FinalTraining)
                finalTrainArray = np.average(np.where(finalTrainArray == "yes", 1, -1), 0)
                PredictResult = np.where(finalTrainArray > 0, "yes", "no")
                TrainAccuracy = (np.count_nonzero(np.array(PredictResult) == trainLabelSample)) / len(trainLabelSample)

            TrainAccuracyIteration = np.count_nonzero(np.array(predictTrainBool)) / len(predictTrainBool)
            TestAccuracyIteration = np.count_nonzero(np.array(predictTest)) / len(predictTest)
            print("Accuracy for random forest: ", i, " iteration: ", TrainAccuracy,TestAccuracy)
        return np.array(FinalTesting)