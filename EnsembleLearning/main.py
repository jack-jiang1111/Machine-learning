import numpy as np

from Adaboost import *
from Bagging import *
from RandomForest import *


# Take in file name return a dictionary
def readFiles(CSVfile):
    attributeData = []
    labelData = []
    attributeLength = 0
    with open(CSVfile, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            attributeData.append(terms[:-1])
            labelData.append(terms[-1])
            attributeLength = len(terms) - 1

    return attributeData, labelData, attributeLength


def TestAdaboost(dataset=1):
    if dataset:
        TrainAttributeData, TrainLabelData, TrainNumAttribute = readFiles("bank/train.csv")
        TestAttributeData, TestLabelData, TestNumAttribute = readFiles("bank/test.csv")
        tree1 = Adaboost(np.array(TrainAttributeData), np.array(TrainLabelData), np.array(TrainNumAttribute),
                         np.array(TestAttributeData), np.array(TestLabelData), split=0,
                         numeric=[0, 5, 9, 11, 12, 13, 14], unknown=False, T=500)
    else:
        TrainAttributeData, TrainLabelData, TrainNumAttribute = readFiles("creditCard/train.csv")
        TestAttributeData, TestLabelData, TestNumAttribute = readFiles("creditCard/test.csv")
        tree1 = Adaboost(np.array(TrainAttributeData), np.array(TrainLabelData), np.array(TrainNumAttribute),
                         np.array(TestAttributeData), np.array(TestLabelData), split=0,
                         numeric=[0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], unknown=False, T=500)

    tree1.runmain()


def TestBagging(dataset=1):
    if dataset:
        TrainAttributeData, TrainLabelData, TrainNumAttribute = readFiles("bank/train.csv")
        TestAttributeData, TestLabelData, TestNumAttribute = readFiles("bank/test.csv")
        tree2 = Bagging(np.array(TrainAttributeData), np.array(TrainLabelData), np.array(TrainNumAttribute),
                        np.array(TestAttributeData), np.array(TestLabelData), sampleSize=1000, split=0,
                        numeric=[0, 5, 9, 11, 12, 13, 14], unknown=False, T=500)
    else:
        TrainAttributeData, TrainLabelData, TrainNumAttribute = readFiles("creditCard/train.csv")
        TestAttributeData, TestLabelData, TestNumAttribute = readFiles("creditCard/test.csv")
        tree2 = Bagging(np.array(TrainAttributeData), np.array(TrainLabelData), np.array(TrainNumAttribute),
                        np.array(TestAttributeData), np.array(TestLabelData), sampleSize=1000, split=0,
                        numeric=[0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], unknown=False, T=500)
    tree2.runBagging()


def TestRandomForest(dataset=1):
    if dataset:
        TrainAttributeData, TrainLabelData, TrainNumAttribute = readFiles("bank/train.csv")
        TestAttributeData, TestLabelData, TestNumAttribute = readFiles("bank/test.csv")
        tree = RandomForest(np.array(TrainAttributeData), np.array(TrainLabelData), np.array(TrainNumAttribute),
                            np.array(TestAttributeData), np.array(TestLabelData), sampleSize=50, split=0,
                            numeric=[0, 5, 9, 11, 12, 13, 14], unknown=False, T=500, AttributeSize=6)
    else:
        TrainAttributeData, TrainLabelData, TrainNumAttribute = readFiles("creditCard/train.csv")
        TestAttributeData, TestLabelData, TestNumAttribute = readFiles("creditCard/test.csv")
        tree = RandomForest(np.array(TrainAttributeData), np.array(TrainLabelData), np.array(TrainNumAttribute),
                            np.array(TestAttributeData), np.array(TestLabelData), sampleSize=50, split=0,
                            numeric=[0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], unknown=False, T=500,
                            AttributeSize=6)
    tree.runRandomForest()


def TestBiasBagging():
    TrainAttributeData, TrainLabelData, TrainNumAttribute = readFiles("bank/train.csv")
    TestAttributeData, TestLabelData, TestNumAttribute = readFiles("bank/test.csv")

    hTestCollection = []
    for i in range(100):
        tree = Bagging(np.array(TrainAttributeData), np.array(TrainLabelData), np.array(TrainNumAttribute),
                       np.array(TestAttributeData), np.array(TestLabelData), sampleSize=1000, split=0,
                       numeric=[0, 5, 9, 11, 12, 13, 14], unknown=False, T=500, print=False)
        hTestArr = tree.runBagging()
        hTestCollection.append(hTestArr)

    # single tree bias and variance
    Single_bias = 0.0
    Single_var = 0.0
    for label in TestLabelData:
        avg = 0
        TestPredict = []
        for hTestArr in hTestCollection:
            singleTree = hTestArr[0]
            labelLine = np.where(singleTree == 'yes', 1, -1)
            avg += labelLine
            TestPredict.append(labelLine)
        avg = avg / len(TestPredict)
        y = np.where(label == 'yes', 1, -1)

        Single_bias += np.power(y - avg, 2)
        Single_var += np.var(TestPredict)

    AvgSingleBias = Single_bias / len(TestAttributeData)
    AvgSingleVar = Single_var / len(TestAttributeData)
    print("single bias: ", np.average(AvgSingleBias), " single var: ", AvgSingleVar)

    # bagging tree bias and variance
    SumBaggingBias = 0.0
    SumBaggingVar = 0.0
    for label in TestLabelData:
        avg = 0
        TestPredict = []
        for hTestArr in hTestCollection:
            Label = 0
            for hTest in hTestArr:
                labelLine = np.where(hTest == 'yes', 1, -1)
                Label /= len(hTestArr)
                Label += labelLine
                avg += Label

            TestPredict.append(Label)
        avg = avg / len(TestPredict)

        y = np.where(label == 'yes', 1, -1)
        SumBaggingBias += np.power(y - avg, 2)
        SumBaggingVar += np.var(TestPredict)

    AvgBaggingBias = SumBaggingBias / len(TestAttributeData)
    AvgBaggingVar = SumBaggingVar / len(TestAttributeData)
    print("All bias: ", AvgBaggingBias, " All var: ", AvgBaggingVar)


def TestBiasRandomForest():
    TrainAttributeData, TrainLabelData, TrainNumAttribute = readFiles("bank/train.csv")
    TestAttributeData, TestLabelData, TestNumAttribute = readFiles("bank/test.csv")

    hTestCollection = []
    for i in range(20):
        tree = RandomForest(np.array(TrainAttributeData), np.array(TrainLabelData), np.array(TrainNumAttribute),
                            np.array(TestAttributeData), np.array(TestLabelData), sampleSize=50, split=0,
                            numeric=[0, 5, 9, 11, 12, 13, 14], unknown=False, T=500, AttributeSize=6)
        hTestArr = tree.runRandomForest()
        hTestCollection.append(hTestArr)

    # single tree bias and variance
    Single_bias = 0.0
    Single_var = 0.0
    for label in TestLabelData:
        avg = 0
        TestPredict = []
        for hTestArr in hTestCollection:
            singleTree = hTestArr[0]
            labelLine = np.where(singleTree == 'yes', 1, -1)
            avg += labelLine
            TestPredict.append(labelLine)
        avg = avg / len(TestPredict)
        y = np.where(label == 'yes', 1, -1)

        Single_bias += np.power(y - avg, 2)
        Single_var += np.var(TestPredict)

    AvgSingleBias = Single_bias / len(TestAttributeData)
    AvgSingleVar = Single_var / len(TestAttributeData)
    print("single bias: ", np.average(AvgSingleBias), " single var: ", AvgSingleVar)

    # bagging tree bias and variance
    SumBaggingBias = 0.0
    SumBaggingVar = 0.0
    for labels in TestLabelData:
        avg = 0
        TestPredict = []
        for hTestArr in hTestCollection:
            Label = 0
            for hTest in hTestArr:
                labelLine = np.where(hTest == 'yes', 1, -1)
                Label /= len(hTestArr)
                Label += labelLine
                avg += Label

            TestPredict.append(Label)
        avg = avg / len(TestPredict)

        y = np.where(labels == 'yes', 1, -1)
        SumBaggingBias += np.power(y - avg, 2)
        SumBaggingVar += np.var(TestPredict)

    AvgBaggingBias = SumBaggingBias / len(TestAttributeData)
    AvgBaggingVar = SumBaggingVar / len(TestAttributeData)
    print("All bias: ", AvgBaggingBias, " All var: ", AvgBaggingVar)


if __name__ == "__main__":
    while True:
        dataset = input('Dataset? b for Bank, c for Credit Card, e for exit\n')
        while dataset != 'b' and dataset != 'c' and dataset != 'e':
            print("Sorry, unrecognized dataset")
            dataset = input('Dataset? b for Bank, c for Credit Card, e for exit\n')
        if dataset == 'e':
            exit(0)
        run = input('Algorithm? 0 for Adaboost, 1 for Bagging, 2 for randomForest. Please wait for half minute until the print coming out\n')
        while run != '0' and run != '1' and run != '2':
            print("Sorry, unrecognized Algorithm\n")
            run = input('Algorithm? 0 for Adaboost, 1 for Bagging, 2 for randomForest\n')
        if run == '0':
            if dataset =='b':
                TestAdaboost()
            else:
                TestAdaboost(0)
        elif run == '1':
            if dataset == 'b':
                TestBagging()
            else:
                TestBagging(0)
        else:
            if dataset == 'b':
                TestRandomForest()
            else:
                TestRandomForest(0)
        print('\n')
    # TestAdaboost(0)
    # TestBagging(0)
    # TestRandomForest()
    # TestBiasBagging()
    # TestBiasRandomForest()


