import random
import sys

import numpy as np


class Split:
    choice = 0
    data = []
    labels = []
    Attributes = []  # a list of attribute index
    PossibleAttribute = {}

    # a list of dictionary, key is the values in attribute, value is count distribution
    labelDistribution = []
    weights = []
    RandomForest = 0 # Random forest attributes

    def __init__(self, data, labels, attributeValues, attributes, weights, choice=0,RandomForest=0):
        self.choice = choice
        self.data = data
        self.labels = labels
        self.PossibleAttribute = attributeValues
        self.Attributes = attributes
        self.labelDistribution = []
        self.RandomForest = RandomForest
        if len(weights):
            self.weights = weights
        else:
            self.weights = np.ones(len(labels))

    def dataProcessing(self):
        data = np.array(self.data)
        label = np.expand_dims(np.array(self.labels), axis=1)
        dataArr = np.append(data, label, axis=1)

        for i in self.Attributes:
            label_distribution = {}
            for value in self.PossibleAttribute[i]:
                mask = (dataArr[:, i] == value)

                indexArray = np.where(dataArr[:, i] == value)
                rows = dataArr[mask, :][:, -1]
                lookUp, index = np.unique(rows,
                                          return_inverse=True)  # counts is a 1-d list return the counts of different labels
                counts = np.bincount(index, weights=self.weights[indexArray])
                counts = counts / np.sum(counts) * len(index)

                if len(counts) != 0:
                    label_distribution[value] = counts
            self.labelDistribution.append(label_distribution)

    def split(self):
        self.dataProcessing()
        if self.choice == 0:
            return self.SplitByEntropy()
        elif self.choice == 1:
            return self.SplitByME()
        else:
            return self.SplitByGini()

    def SplitByEntropy(self):
        # for each attribute
        # for each value in attribute list, calculate gain
        # choose the largest gain attribute, return attribute index
        maxGain = -sys.maxsize - 1
        bestAttribute = 0
        index = 0
        for labelDistribution in self.labelDistribution:
            attribute_gain = 0
            for key in labelDistribution:
                value = labelDistribution[key]
                attribute_gain += (np.sum(value / np.sum(value) * np.log2(value / np.sum(value)))) * np.sum(value)
            if attribute_gain > maxGain:
                maxGain = attribute_gain
                bestAttribute = index
            index += 1
        return self.Attributes[bestAttribute]

    def SplitByME(self):
        maxGain = -sys.maxsize - 1
        bestAttribute = 0
        index = 0
        for labelDistribution in self.labelDistribution:
            attribute_gain = 0
            for key in labelDistribution:
                value = labelDistribution[key]
                if len(value) != 1:
                    attribute_gain -= (np.sum(value) - np.max(value)) / np.sum(value)
            if attribute_gain > maxGain:
                maxGain = attribute_gain
                bestAttribute = index
            index += 1
        return self.Attributes[bestAttribute]

    def SplitByGini(self):
        maxGain = -sys.maxsize - 1
        bestAttribute = 0
        index = 0
        for labelDistribution in self.labelDistribution:
            attribute_gain = 0
            for key in labelDistribution:
                value = labelDistribution[key]
                attribute_gain -= (1 - np.sum(np.power(value / np.sum(value), 2))) * np.sum(value)
            if attribute_gain > maxGain:
                maxGain = attribute_gain
                bestAttribute = index
            index += 1
        return self.Attributes[bestAttribute]
