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


def TestAdaboost():
    TrainAttributeData, TrainLabelData, TrainNumAttribute = readFiles("bank/train.csv")
    TestAttributeData, TestLabelData, TestNumAttribute = readFiles("bank/test.csv")
    tree1 = Adaboost(np.array(TrainAttributeData), np.array(TrainLabelData), np.array(TrainNumAttribute),
                     np.array(TestAttributeData), np.array(TestLabelData), split=0,
                     numeric=[0, 5, 9, 11, 12, 13, 14], unknown=False, T=500)
    tree1.runmain()


def TestBagging():
    TrainAttributeData, TrainLabelData, TrainNumAttribute = readFiles("bank/train.csv")
    TestAttributeData, TestLabelData, TestNumAttribute = readFiles("bank/test.csv")
    tree2 = Bagging(np.array(TrainAttributeData), np.array(TrainLabelData), np.array(TrainNumAttribute),
                    np.array(TestAttributeData), np.array(TestLabelData), sampleSize=1000, split=0,
                    numeric=[0, 5, 9, 11, 12, 13, 14], unknown=False, T=500)
    tree2.runBagging()


def TestRandomForest():
    TrainAttributeData, TrainLabelData, TrainNumAttribute = readFiles("bank/train.csv")
    TestAttributeData, TestLabelData, TestNumAttribute = readFiles("bank/test.csv")
    tree = RandomForest(np.array(TrainAttributeData), np.array(TrainLabelData), np.array(TrainNumAttribute),
                        np.array(TestAttributeData), np.array(TestLabelData), sampleSize=50, split=0,
                        numeric=[0, 5, 9, 11, 12, 13, 14], unknown=False, T=500, AttributeSize=6)
    tree.runRandomForest()


def TestBiasBagging():
    TrainAttributeData, TrainLabelData, TrainNumAttribute = readFiles("bank/train.csv")
    TestAttributeData, TestLabelData, TestNumAttribute = readFiles("bank/test.csv")

    hTestCollection = []
    for i in range(3):
        tree = Bagging(np.array(TrainAttributeData), np.array(TrainLabelData), np.array(TrainNumAttribute),
                       np.array(TestAttributeData), np.array(TestLabelData), sampleSize=1000, split=0,
                       numeric=[0, 5, 9, 11, 12, 13, 14], unknown=False, T=5,print = False)
        hTestArr = tree.runBagging()
        hTestCollection.append(hTestArr)

    # single tree bias and variance
    Single_bias = 0.0
    Single_var = 0.0
    for label in TestAttributeData:
        avg = 0
        TestPredict = []
        for hTestArr in hTestCollection:
            singleTree = hTestArr[0]
            labelLine = np.where(singleTree=='yes',1,-1)
            avg += labelLine
            TestPredict.append(labelLine)
        avg = avg/len(TestPredict)
        y = np.where(label == 'yes', 1, -1)

        Single_bias += np.power(y - avg, 2)
        Single_var += np.var(TestPredict)

    AvgSingleBias = Single_bias / len(TestAttributeData)
    AvgSingleVar = Single_var / len(TestAttributeData)
    print("single bias: ",np.average(AvgSingleBias)," single var: ",AvgSingleVar)

    # bagging tree bias and variance
    SumBaggingBias = 0.0
    SumBaggingVar = 0.0
    for label in TestAttributeData:
        avg = 0
        TestPredict = []
        for hTestArr in hTestCollection:
            Label = 0
            for hTest in hTestCollection:
                labelLine = np.where(hTest == 'yes', 1, -1)
                Label /= len(hTestArr)
                Label += labelLine
                avg += Label

            TestPredict.append(Label)
        avg = avg/len(TestPredict)

        y = np.where(label == 'yes', 1, -1)
        SumBaggingBias += np.power(y - avg, 2)
        SumBaggingVar += np.var(TestPredict)

    AvgBaggingBias = SumBaggingBias / len(TestAttributeData)
    AvgBaggingVar = SumBaggingVar / len(TestAttributeData)
    print("All bias: ", AvgBaggingBias, " All var: ", AvgBaggingVar)



if __name__ == "__main__":
    # TestAdaboost()
    # TestBagging()
    #TestRandomForest()
    TestBiasBagging()

# ToDo:
# 4. All experiments
