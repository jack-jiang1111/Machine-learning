import numpy as np


class svm:
    TrainData = np.empty(1)
    TestData = np.empty(1)
    numAttribute = -1
    w = np.zeros(1)
    r = 0.2
    b = 1
    maxEpoch = 10

    def __init__(self, trainData, testData, numAttr):


        self.TrainData = np.array(trainData, dtype='float64')
        num_col = self.TrainData.shape[1]
        label = np.expand_dims(np.where(self.TrainData[:, -1] == 1, 1, -1),1)  # convert to 1/-1 for convient
        self.TrainData[:, num_col - 1] = 1
        self.TrainData = np.append(self.TrainData,label,1)

        self.TestData = np.array(testData, dtype='float64')
        label = np.expand_dims(np.where(self.TestData[:, -1] == 1, 1, -1), 1)  # convert to 1/-1 for convient
        self.TestData[:, num_col - 1] = 1
        self.TestData = np.append(self.TestData, label, 1)

        self.numAttribute = numAttr+1 #(add one for b)
        self.w = np.zeros(self.numAttribute)

    def TestError(self, data):
        x = np.array(data[:, 0:-1])
        y = np.matmul(x, self.w)
        y = np.where(y > 0, 1, -1)
        acc = np.count_nonzero(y == data[:, -1]) / len(data)
        return acc

    def runStandard(self):

        for i in range(self.maxEpoch):
            # shuffule data
            np.random.shuffle(self.TrainData)
            for index in range(len(self.TrainData)):
                error = (np.dot(self.w, self.TrainData[index][:-1]) + self.b) * self.TrainData[index][-1]
                if error <= 0:
                    # update weights
                    self.w += self.r * self.TrainData[index][:-1] * self.TrainData[index][-1]
            trainError = self.TestError(self.TrainData)
            testError = self.TestError(self.TestData)
            print("Epoch ", i, " w: ", np.round(self.w,3), " Train Error: ", np.round(1-trainError,3), " Test Error: ", np.round(1-testError,3))
