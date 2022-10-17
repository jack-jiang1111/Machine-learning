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
    T = 500 # numbers of iteration

    def __init__(self, trainData, trainLabel, numAttribute, testData, testLabel, sampleSize = 1000,split=0, numeric=None,
                 unknown=True, T=500):
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

    def runBagging(self):
        FinalTesting = []
        for i in range(self.T):
            sampleDataIndex = []
            IndexRange = np.arange(0,len(self.TrainAttributeData)-1,1)
            for j in range(self.sampleSize):
                sampleDataIndex.append(random.choice(IndexRange))

            # create a sample of data
            trainDataSample = np.take(self.TrainAttributeData,sampleDataIndex,0)
            trainLabelSample = np.take(self.TrainLabelData,sampleDataIndex,0)
            tree = Tree.DecisionTree(trainDataSample, trainLabelSample, self.attribute_length.copy(),
                                     self.TestAttributeData.copy(), self.TestLabelData.copy(), None, 0, 100, self.numericalData.copy(),fullData=self.TrainAttributeData.copy())
            predictTrainBool, hTrain, predictTest, hTest = tree.RunTreeWithAdaboost()
            FinalTesting.append(hTest)
            del tree

        finalArray = np.array(FinalTesting)
        finalArray = np.average(np.where(finalArray == "yes",1,-1),0)
        PredictResult = np.where(finalArray>0,"yes","no")
        accuracy = (np.count_nonzero(np.array(PredictResult) == self.TestLabelData)) / len(self.TestLabelData)
        print("Accuracy for bagging: ",self.T," iteration: ",accuracy)