from LinearRegression import *

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


if __name__ == "__main__":
    TrainAttributeData, TrainLabelData, TrainNumAttribute = readFiles("train.csv")
    TestAttributeData, TestLabelData, TestNumAttribute = readFiles("test.csv")
    LinearRegressionModel = LinearRegression(TrainAttributeData, TrainLabelData,TestAttributeData, TestLabelData,TrainNumAttribute)
    LinearRegressionModel.SGD()
    #LinearRegressionModel.Batch()
    #LinearRegressionModel.Analysis()

    print('\n')

# ToDo:
# 1. debug Adaboost
# 2. fix Bagging fully expanding trees
# 3. random forest attribute list problem
# 4. All experiments
