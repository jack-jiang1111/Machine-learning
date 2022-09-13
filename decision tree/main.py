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

# TODO: max depth tree
# TODO: numerial values
TrainAttributeData, TrainLabelData, TrainNumAttribute = readFiles("car/train.csv")
TestAttributeData, TestLabelData, TestNumAttribute = readFiles("car/test.csv")
tree = DecisionTree()
tree.RunTree(TrainAttributeData, TrainLabelData, TrainNumAttribute, 0)
tree.Predict(TestAttributeData, TestLabelData)