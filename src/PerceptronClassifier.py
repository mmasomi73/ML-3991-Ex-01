import datetime
from sklearn.linear_model import Perceptron


class PerceptronClassifier:
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    clf = object

    st_tr_time = 0
    en_tr_time = 0
    st_te_time = 0
    en_te_time = 0

    currect = 0

    labels_predict = []

    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def train(self):
        self.st_tr_time = datetime.datetime.now().timestamp()
        self.clf = Perceptron(max_iter=100)
        self.clf.fit(self.train_x, self.train_y)
        self.en_tr_time = datetime.datetime.now().timestamp()

    def predic(self):
        self.st_te_time = datetime.datetime.now().timestamp()
        self.labels_predict = self.clf.predict(self.test_x)
        self.en_te_time = datetime.datetime.now().timestamp()
        return self.labels_predict

    def getAccuracy(self):
        if len(self.labels_predict) == 0:
            self.predic()
        for i in range(len(self.labels_predict)):
            if self.labels_predict[i] == self.test_y[i]:
                self.currect += 1
        return (self.currect / len(self.test_y)) * 100

    def trainTime(self):
        return self.en_tr_time - self.st_tr_time

    def testTime(self):
        return self.en_te_time - self.st_te_time

    def printResult(self):
        if len(self.labels_predict) == 0:
            self.train()
            self.predic()
        print("\t+-----=[ Perceptron Classifier ]=----- ")
        print("\t| Train Time         : {}".format(self.trainTime()))
        print("\t| Test  Time         : {}".format(self.testTime()))
        print("\t| Accuracy           : {}".format((self.currect / len(self.test_y)) * 100))
        print("\t| MissClassification : {}".format(((len(self.test_y) - self.currect) / len(self.test_y)) * 100))
        # print("\n Coefficients : {}".format(self.clf.coef_))
        print("\t------------------------------------------- ")
