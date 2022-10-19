import random

import numpy as np

class LinearRegression:
    TrainAttributeData = np.empty(1)
    TrainLabelData = np.empty(1)
    TestAttributeData = np.empty(1)
    TestLabelData = np.empty(1)
    learningRate = 0.01
    numAttribute = -1

    def __init__(self,trainData,trainLabel,testData,testLabel,attributeLength,learningRate=0.01):
        self.TestAttributeData = np.transpose(np.array(testData).astype(float))
        self.TrainAttributeData = np.transpose(np.array(trainData).astype(float))
        self.TrainLabelData = np.transpose(np.array(trainLabel).astype(float))
        self.TestLabelData = np.transpose(np.array(testLabel).astype(float))
        self.learningRate = learningRate
        self.numAttribute = attributeLength

    def SGD(self):
        weights = np.zeros(self.TrainAttributeData.shape[0])
        tolerate = 1e-6

        it = 1
        epoch = 10000
        while it<epoch:
            loss = 0
            orginalW = weights.copy()
            for i in range(self.TrainAttributeData.shape[1]):
                X1 = self.TrainAttributeData[:,i]
                y1 = self.TrainLabelData[i]

                error = y1 - np.dot(weights, X1)
                grad = -error * X1
                # Update weights for all sample data
                weights -= self.learningRate * grad

                loss+=np.square(error)
            loss *= 0.5
            print("it: ",it," ",loss)

            delta = orginalW-weights
            it+=1
            if it%5==0:
               self.learningRate/=2
            if np.sqrt(np.sum(np.square(delta))) < tolerate:
                break
    def Batch(self):
        weights = np.transpose(np.zeros((self.numAttribute,1)).astype(float))
        tolerate = 1e-6

        it=1
        epoch = 100000
        while(it<epoch):
            Error = self.TrainLabelData-np.matmul(weights,self.TrainAttributeData)
            delta = self.learningRate*np.sum(Error*self.TrainAttributeData,axis=1)

            weights +=delta
            loss = 0.5 * np.sum(np.square(Error))
            print("Epoch: ",it," ",loss)
            if np.sqrt(np.sum(np.square(delta)))<tolerate:
                break
            it+=1
            if it%5==0:
               self.learningRate/=2
        PredictError = self.TestLabelData - np.matmul(weights, self.TestAttributeData)
        print(PredictError)
    def Analysis(self):
        weight = np.matmul(np.linalg.inv(np.matmul(self.TrainAttributeData,np.transpose(self.TrainAttributeData))),np.matmul(self.TrainAttributeData,self.TrainLabelData))
        print(weight)
        return weight