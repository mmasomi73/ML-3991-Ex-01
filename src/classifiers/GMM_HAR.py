from sklearn.mixture import GaussianMixture
from colorama import Fore, Back, Style
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os


def dataSetReader(path):
    if not os.path.isfile(path):
        print(Fore.RED + "\t+------------------------+")
        print(Fore.RED + "\t| File or Path Incorrect |")
        print(Fore.RED + "\t+------------------------+")
        return np.array([])
    return np.loadtxt(path)


# -----= Read Dataset
train_x = dataSetReader('../../Dataset/UCI HAR Dataset/train/X_train.txt')
train_y = dataSetReader('../../Dataset/UCI HAR Dataset/train/y_train.txt')
test_x = dataSetReader('../../Dataset/UCI HAR Dataset/test/X_test.txt')
test_y = dataSetReader('../../Dataset/UCI HAR Dataset/test/y_test.txt')

clf = GaussianMixture(covariance_type='tied', n_components=6)
clf.fit(train_x, train_y)
result = []
currect = 0

labels_predict = clf.predict(test_x)
for i in range(len(labels_predict)):
    if labels_predict[i] == test_y[i]:
        currect += 1
    result.append([i, test_y[i], labels_predict[i]])

print("\n Accuracy : {}".format((currect / len(test_y)) * 100))
print("\n MissClassification : {}".format(((len(test_y) - currect) / len(test_y)) * 100))
