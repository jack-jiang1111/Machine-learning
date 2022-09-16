from decisionTree import *

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



def TestCar():
    TrainAttributeData, TrainLabelData, TrainNumAttribute = readFiles("car/train.csv")
    TestAttributeData, TestLabelData, TestNumAttribute = readFiles("car/test.csv")
    for i in range(6):
        print("Max depth: ",i+1)
        tree1 = DecisionTree(np.array(TrainAttributeData), np.array(TrainLabelData), np.array(TrainNumAttribute),np.array(TestAttributeData), np.array(TestLabelData),split=2,depth = i+1)
        tree1.RunTree()
def TestBank():
    TrainAttributeData, TrainLabelData, TrainNumAttribute = readFiles("bank/train.csv")
    TestAttributeData, TestLabelData, TestNumAttribute = readFiles("bank/test.csv")
    for i in range(6):
        print("Max depth: ",i+1)
        tree1 = DecisionTree(np.array(TrainAttributeData), np.array(TrainLabelData), np.array(TrainNumAttribute),np.array(TestAttributeData), np.array(TestLabelData),split=2,depth=i+1,numeric=[0,5,9,11,12,13,14],unknown=False)
        tree1.RunTree()
TestBank()