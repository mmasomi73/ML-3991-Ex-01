import os
import numpy as np
from colorama import Fore


class DataCollector:
    path = '../Dataset/UCI HAR Dataset/'
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    def __init__(self, log=False):
        if log:
            print(Fore.GREEN + "\t+----------------------------------------------")
            print(Fore.GREEN + "\tRead HAR Dataset Begin.")

        self.train_x = self.dataSetReader(self.path + 'train/X_train.txt')
        if log:
            print(Fore.GREEN + "\tTrain Data Has Been Read")
            print(Fore.GREEN + "\tTrain Data Has {} Features and {} Instances".format(self.train_x.shape[1],
                                                                                      self.train_x.shape[0]))

        self.train_y = self.dataSetReader(self.path + 'train/y_train.txt')
        if log:
            print(Fore.GREEN + "\tTrain Labels Has Been Read")
            print(Fore.GREEN + "\tTrain Data Has {} Class".format(np.unique(self.train_y)))

        self.test_x = self.dataSetReader(self.path + 'test/X_test.txt')
        if log:
            print(Fore.GREEN + "\tTest Data Has Been Read")
            print(Fore.GREEN + "\tTrain Data Has {} Features and {} Instances".format(self.test_x.shape[1],
                                                                                      self.test_x.shape[0]))

        self.test_y = self.dataSetReader(self.path + 'test/y_test.txt')
        if log:
            print(Fore.GREEN + "\tTest Labels Has Been Read")
            print(Fore.GREEN + "\tTest Data Has {} Class".format(len(np.unique(self.train_y))))
            print(Fore.GREEN + "\t-----=[HAR Dataset Has Totally {} Records in {} Class]=-----".format(
                self.test_x.shape[0] + self.train_x.shape[0], len(np.unique(self.train_y))))
            print(Fore.GREEN + "\t+----------------------------------------------")

    def dataSetReader(self, path):

        if not os.path.isfile(path):
            print(Fore.RED + "\t+------------------------+")
            print(Fore.RED + "\t| File or Path Incorrect |")
            print(Fore.RED + "\t+------------------------+")
            return np.array([])
        return np.loadtxt(path)

    def getTrainData(self):
        return self.train_x

    def getTrainLabel(self):
        return self.train_y

    def getTestData(self):
        return self.test_x

    def getTestLabel(self):
        return self.test_y
