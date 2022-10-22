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


    while True:
        run = input('Algorithm? 0 for Batch, 1 for SGD, 2 for Analysis\n')
        while run != '0' and run != '1' and run != '2':
            print("Sorry, unrecognized Algorithm\n")
            run = input('Algorithm? 0 for Batch, 1 for SGD, 2 for Analysis\n')
        if run == '0':
            LinearRegressionModel = LinearRegression(TrainAttributeData, TrainLabelData, TestAttributeData,
                                                     TestLabelData, TrainNumAttribute)
            LinearRegressionModel.Batch()
        elif run == '1':
            LinearRegressionModel = LinearRegression(TrainAttributeData, TrainLabelData, TestAttributeData,
                                                     TestLabelData, TrainNumAttribute)
            LinearRegressionModel.SGD()
        else:
            LinearRegressionModel = LinearRegression(TrainAttributeData, TrainLabelData, TestAttributeData,
                                                     TestLabelData, TrainNumAttribute)
            LinearRegressionModel.Analysis()

    #LinearRegressionModel.SGD()
    #LinearRegressionModel.Batch()
    #LinearRegressionModel.Analysis()

# ToDo:
# 1. debug Adaboost
# 2. fix Bagging fully expanding trees
# 3. random forest attribute list problem
# 4. All experiments
