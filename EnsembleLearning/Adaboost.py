import math
import os
import sys
from collections import Counter

from decisionTree import *
import numpy as np


class Adaboost:
    attribute_length = 0
    TrainAttributeData = np.empty(1)
    TrainLabelData = np.empty(1)
    TrainWeights = np.empty(1)
    TestAttributeData = np.empty(1)
    TestLabelData = np.empty(1)
    numericalData = []
    unknowAsAttribute = True  # by default unknow will be treated as an attribute, if this is false, unknown will be the most common value of this attribute
    T = 500

    def __init__(self, trainData, trainLabel, numAttribute, testData, testLabel, split=0, numeric=None,
                 unknown=True, T=500):
        if numeric is None:
            numeric = []
        self.split = split
        self.TrainAttributeData = trainData
        self.TrainLabelData = trainLabel
        self.TrainWeights = np.ones(len(self.TrainLabelData))
        self.attribute_length = numAttribute
        self.TestAttributeData = testData
        self.TestLabelData = testLabel
        self.numericalData = numeric
        self.unknowAsAttribute = unknown

        self.T = T

    def runmain(self):
        weights = (np.ones(len(self.TrainLabelData))).astype('double')
        weights = (weights / np.sum(weights))

        TestWeight = (np.ones(len(self.TestLabelData))).astype('double')
        TestWeight = (TestWeight / np.sum(TestWeight))

        hTrainFinal = np.zeros(len(self.TrainLabelData))
        hTestFinal = np.zeros(len(self.TestLabelData))

        # print info
        trainAccAll = []
        testAccAll = []

        for i in range(self.T):
            tree = DecisionTree(self.TrainAttributeData.copy(), self.TrainLabelData.copy(), self.attribute_length.copy(),
                                     self.TestAttributeData.copy(), self.TestLabelData.copy(), weights.copy(), 0, 1, self.numericalData.copy())
            predictTrainBool, hTrain, predictTestBool, hTest = tree.RunTreeWithAdaboost()

            # predict Train/Test Boolean, if predict correct, contain True, else contain False
            # hTrain/Test, contains the predict values (as labels)
            PredictWrongIndex = np.where(predictTrainBool==False)
            PredictWrongIndexTest = np.where(predictTestBool == False)

            # Weighted error
            error = np.sum(weights[PredictWrongIndex[0]])
            errorTest = np.sum(TestWeight[PredictWrongIndexTest[0]])
            alpha = 0.5 * np.log((1 - error) / error)

            # If predict correct, The boolean will be True and we will assign yi*ht(xi) to 1
            # else assign it to -1
            predictTrain = np.where(predictTrainBool == True, 1, -1)

            #update weighted vector
            Dt = weights * np.exp(-alpha * predictTrain)
            Dt = Dt/np.sum(Dt)  # normalized
            weights = Dt

            # credit card case
            if self.attribute_length == 19:
                # Save a copy of final predict result
                hTrain = np.where(hTrain=='1',1,-1)
                hTest = np.where(hTest=='1',1,-1)
            else: #bank
                hTrain = np.where(hTrain == 'yes', 1, -1)
                hTest = np.where(hTest == 'yes', 1, -1)

            hTrainFinal += alpha * hTrain
            hTestFinal += alpha * hTest




            # credit card case
            if self.attribute_length == 19:
                # Save a copy of final predict result
                FinalHypothesisTrain = np.where(hTrainFinal > 0, '1', '0')
                FinalHypothesisTest = np.where(hTestFinal > 0, '1', '0')
            else:  # bank
                FinalHypothesisTrain = np.where(hTrainFinal > 0, 'yes', 'no')
                FinalHypothesisTest = np.where(hTestFinal > 0, 'yes', 'no')
            trainAccAll.append((np.count_nonzero(np.array(FinalHypothesisTrain) == self.TrainLabelData)) / len(self.TrainLabelData))
            testAccAll.append((np.count_nonzero(np.array(FinalHypothesisTest) == self.TestLabelData)) / len(self.TestLabelData))
            print("trainAcc: ",error," testAcc: ",errorTest," trainAccAll: ",trainAccAll[i]," testAccAll: ",testAccAll[i])
            del tree