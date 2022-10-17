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
                    numeric=[0, 5, 9, 11, 12, 13, 14], unknown=False, T=10)
    tree2.runBagging()


def TestRandomForest():
    TrainAttributeData, TrainLabelData, TrainNumAttribute = readFiles("bank/train.csv")
    TestAttributeData, TestLabelData, TestNumAttribute = readFiles("bank/test.csv")
    tree = RandomForest(np.array(TrainAttributeData), np.array(TrainLabelData), np.array(TrainNumAttribute),
                        np.array(TestAttributeData), np.array(TestLabelData), sampleSize=1000, split=0,
                        numeric=[0, 5, 9, 11, 12, 13, 14], unknown=False, T=500, AttributeSize=6)
    tree.runRandomForest()


if __name__ == "__main__":
    TestAdaboost()
    #TestBagging()
    #TestRandomForest()
    print('\n')

# ToDo:
# 1. debug Adaboost
# 2. fix Bagging fully expanding trees
# 3. random forest attribute list problem
# 4. All experiments
