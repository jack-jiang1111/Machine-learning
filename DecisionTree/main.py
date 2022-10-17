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


def TestCar(maxDepth,split):
    TrainAttributeData, TrainLabelData, TrainNumAttribute = readFiles("car/train.csv")
    TestAttributeData, TestLabelData, TestNumAttribute = readFiles("car/test.csv")

    tree1 = DecisionTree(np.array(TrainAttributeData), np.array(TrainLabelData), np.array(TrainNumAttribute),
                         np.array(TestAttributeData), np.array(TestLabelData), split=split, depth=maxDepth)
    tree1.RunTree()


def TestBank(maxDepth,split):
    TrainAttributeData, TrainLabelData, TrainNumAttribute = readFiles("bank/train.csv")
    TestAttributeData, TestLabelData, TestNumAttribute = readFiles("bank/test.csv")
    tree1 = DecisionTree(np.array(TrainAttributeData), np.array(TrainLabelData), np.array(TrainNumAttribute),
                         np.array(TestAttributeData), np.array(TestLabelData), split=split, depth=maxDepth,
                         numeric=[0, 5, 9, 11, 12, 13, 14], unknown=False)
    tree1.RunTree()


if __name__ == "__main__":
    while True:
        dataset = input('Dataset? b for Bank, c for Car, e for exit\n')
        while dataset != 'b' and dataset != 'c' and dataset!='e':
            print("Sorry, unrecognized dataset")
            dataset = input('Dataset? b for Bank, c for Car\n')
        if dataset =='e':
            exit(0)
        maxDepth = int(input('Max depth of tree, input a number\n'))
        while maxDepth < 1:
            print("Sorry, max depth should greater than zero\n")
            maxDepth = int(input('Max depth of tree, input a number\n'))
        split = input('Split Algorithm? 0 for entropy, 1 for Majority error, 2 for gini index\n')
        while split != '0' and split != '1' and split != '2':
            print("Sorry, unrecognized split\n")
            split = input('Split Algorithm? 0 for entropy, 1 for Majority error, 2 for gini index\n')
        if dataset=='b':
            TestBank(maxDepth,int(split))
        else:
            TestCar(maxDepth,int(split))
        print('\n')