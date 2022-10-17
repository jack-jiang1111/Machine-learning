import math
import os
import sys
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'DecisionTree'))

import DecisionTree.Node as Node
import DecisionTree.Split as Split
import DecisionTree.decisionTree as Tree
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

        hTrainFinal = np.zeros(len(self.TrainLabelData))
        hTestFinal = np.zeros(len(self.TestLabelData))

        # print info
        trainAcc = []
        testAcc = []
        trainAccAll = []
        testAccAll = []

        for i in range(self.T):
            tree = Tree.DecisionTree(self.TrainAttributeData.copy(), self.TrainLabelData.copy(), self.attribute_length.copy(),
                                     self.TestAttributeData.copy(), self.TestLabelData.copy(), weights, 0, 1, self.numericalData.copy())
            predictTrainBool, hTrain, predictTestBool, hTest = tree.RunTreeWithAdaboost()

            # predict Train/Test Boolean, if predict correct, contain True, else contain False
            # hTrain/Test, contains the predict values (as labels)
            PredictWrongIndex = np.where(predictTrainBool==False)

            # Weighted error
            error = np.sum(weights[PredictWrongIndex])
            alpha = 0.5 * np.log((1 - error) / error)

            # If predict correct, The boolean will be True and we will assign yi*ht(xi) to 1
            # else assign it to -1
            predictTrain = np.where(predictTrainBool == True, 1, -1)

            #update weighted vector
            Dt = weights * np.exp(-alpha * predictTrain)
            Dt = Dt/np.sum(Dt)  # normalized
            weights = Dt

            # Save a copy of final predict result
            hTrain = np.where(hTrain=='yes',1,-1)
            hTest = np.where(hTest=='yes',1,-1)

            hTrainFinal += alpha * hTrain
            hTestFinal += alpha * hTest

            trainAcc.append(np.count_nonzero(predictTrainBool)/len(self.TrainLabelData))
            testAcc.append(np.count_nonzero(predictTestBool)/len(self.TestLabelData))

            FinalHypothesisTrain = np.where(hTrainFinal > 0, 'yes','no')
            FinalHypothesisTest = np.where(hTestFinal > 0, 'yes', 'no')

            trainAccAll.append((np.count_nonzero(np.array(FinalHypothesisTrain) == self.TrainLabelData)) / len(self.TrainLabelData))
            testAccAll.append((np.count_nonzero(np.array(FinalHypothesisTest) == self.TestLabelData)) / len(self.TestLabelData))
            print("trainAcc: ",trainAcc[i]," testAcc: ",testAcc[i]," trainAccAll: ",trainAccAll[i]," testAccAll: ",testAccAll[i])
            #if tree.root.attribute!=11:
            #    print(tree.root.attribute)
            del tree