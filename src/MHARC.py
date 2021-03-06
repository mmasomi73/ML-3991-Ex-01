import datetime

from DataCollector import DataCollector
from SVMClassifier import SVMClassifier
from PerceptronClassifier import PerceptronClassifier
from KNeighborsClassifier import KNeighborsClassifier
from NaiveBayesClassifier import NaiveBayesClassifier
from GradientDescentClassifier import GradientDescentClassifier
from GaussianMixtureClassifier import GaussianMixtureClassifier
from LogisticRegressionClassifier import LogisticRegressionClassifier


class MHARC:
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    acc = {}
    accuracies = {}

    def __init__(self, log=False):
        self.dc = dc = DataCollector(log)
        self.train_x = dc.getTrainData()
        self.train_y = dc.getTrainLabel()
        self.test_x = dc.getTestData()
        self.test_y = dc.getTestLabel()

    def executor(self):
        for itr in range(10):
            print("-----= Iteration:{} ".format(itr))
            self.acc = {}
            self.GradientDescentDriver()
            self.NaiveBayesDriver()
            self.PerceptronDriver()
            self.LogisticRegressionDriver()
            self.GaussianMixtureDriver()
            self.SVMDriver()
            self.KNeighborsDriver()
            self.accuracies[itr] = self.acc

    def GradientDescentDriver(self):
        gradientdescent = GradientDescentClassifier(self.train_x, self.train_y, self.test_x, self.test_y)
        # -----= gradientDescent
        gradientdescent.train()
        gradientdescent_labels_1 = gradientdescent.predic()
        gradientdescent_acc = gradientdescent.getAccuracy()
        gradientdescent.printResult()
        self.acc['gradientdescent'] = {'accuracy': gradientdescent_acc,
                                       'train-time': gradientdescent.trainTime(),
                                       'test-time': gradientdescent.testTime()
                                       }

    def NaiveBayesDriver(self):
        naivebayes = NaiveBayesClassifier(self.train_x, self.train_y, self.test_x, self.test_y)
        # -----= naiveBayes
        naivebayes.train_1()
        naivebayes_labels = naivebayes.predic()
        naivebayes_acc = naivebayes.getAccuracy()
        naivebayes.printResult()
        self.acc['naivebayes-GaussianNB'] = {'accuracy': naivebayes_acc,
                                             'train-time': naivebayes.trainTime(),
                                             'test-time': naivebayes.testTime(),
                                             }

        naivebayes.train_2()
        naivebayes_labels_2 = naivebayes.predic()
        naivebayes_acc = naivebayes.getAccuracy()
        naivebayes.printResult()
        self.acc['naivebayes-MultinomialNB'] = {'accuracy': naivebayes_acc,
                                                'train-time': naivebayes.trainTime(),
                                                'test-time': naivebayes.testTime(),
                                                }

        naivebayes.train_3()
        naivebayes_labels_3 = naivebayes.predic()
        naivebayes_acc = naivebayes.getAccuracy()
        naivebayes.printResult()
        self.acc['naivebayes-ComplementNB'] = {'accuracy': naivebayes_acc,
                                               'train-time': naivebayes.trainTime(),
                                               'test-time': naivebayes.testTime(),
                                               }

        # -----= naiveBayes
        naivebayes.train_1()
        naivebayes_labels = naivebayes.predic()
        naivebayes_acc = naivebayes.getAccuracy()
        naivebayes.printResult()
        self.acc['naivebayes-GaussianNB'] = {'accuracy': naivebayes_acc,
                                             'train-time': naivebayes.trainTime(),
                                             'test-time': naivebayes.testTime(),
                                             }

    def PerceptronDriver(self):
        perceptron = PerceptronClassifier(self.train_x, self.train_y, self.test_x, self.test_y)
        # -----= gradientDescent
        perceptron.train()
        perceptron_labels_1 = perceptron.predic()
        perceptron_acc = perceptron.getAccuracy()
        perceptron.printResult()
        self.acc['perceptron'] = {'accuracy': perceptron_acc,
                                  'train-time': perceptron.trainTime(),
                                  'test-time': perceptron.testTime()
                                  }

    def LogisticRegressionDriver(self):
        logisticregression = LogisticRegressionClassifier(self.train_x, self.train_y, self.test_x, self.test_y)
        # -----= gradientDescent
        logisticregression.train()
        logisticregression_labels_1 = logisticregression.predic()
        logisticregression_acc = logisticregression.getAccuracy()
        logisticregression.printResult()
        self.acc['logisticregression'] = {'accuracy': logisticregression_acc,
                                          'train-time': logisticregression.trainTime(),
                                          'test-time': logisticregression.testTime()
                                          }

    def GaussianMixtureDriver(self):
        gaussianmixture = GaussianMixtureClassifier(self.train_x, self.train_y, self.test_x, self.test_y)
        # -----= gradientDescent
        gaussianmixture.train_1()
        gaussianmixture_labels = gaussianmixture.predic()
        gaussianmixture_acc = gaussianmixture.getAccuracy()
        gaussianmixture.printResult()
        self.acc['gaussianmixture-GaussianMixture'] = {'accuracy': gaussianmixture_acc,
                                                       'train-time': gaussianmixture.trainTime(),
                                                       'test-time': gaussianmixture.testTime(),
                                                       }

        gaussianmixture.train_2()
        gaussianmixture_labels_2 = gaussianmixture.predic()
        gaussianmixture_acc = gaussianmixture.getAccuracy()
        gaussianmixture.printResult()
        self.acc['gaussianmixture-BayesianGaussianMixture'] = {'accuracy': gaussianmixture_acc,
                                                               'train-time': gaussianmixture.trainTime(),
                                                               'test-time': gaussianmixture.testTime(),
                                                               }

    def SVMDriver(self):
        svm = SVMClassifier(self.train_x, self.train_y, self.test_x, self.test_y)
        # -----= SVM
        svm.train_1()
        svm_labels = svm.predic()
        svm_acc = svm.getAccuracy()
        svm.printResult()
        self.acc['svm-NuSVC'] = {'accuracy': svm_acc,
                                 'train-time': svm.trainTime(),
                                 'test-time': svm.testTime(),
                                 }

        svm.train_2()
        svm_labels_2 = svm.predic()
        svm_acc = svm.getAccuracy()
        svm.printResult()
        self.acc['svm-LinearSVC'] = {'accuracy': svm_acc,
                                     'train-time': svm.trainTime(),
                                     'test-time': svm.testTime(),
                                     }
        svm.train_3()
        svm_labels_3 = svm.predic()
        svm_acc = svm.getAccuracy()
        svm.printResult()
        self.acc['svm-SVC'] = {'accuracy': svm_acc,
                               'train-time': svm.trainTime(),
                               'test-time': svm.testTime(),
                               }

    def KNeighborsDriver(self):
        kneighbors = KNeighborsClassifier(self.train_x, self.train_y, self.test_x, self.test_y)
        # -----= KNeighbors
        kneighbors.train()
        kneighbors_labels = kneighbors.predic()
        kneighbors_acc = kneighbors.getAccuracy()
        kneighbors.printResult()
        self.acc['kneighbors'] = {'accuracy': kneighbors_acc,
                                  'train-time': kneighbors.trainTime(),
                                  'test-time': kneighbors.testTime(),
                                  }

    def SaveResults(self):
        file_name = '../outs/res-' + datetime.datetime.now().strftime('%d-%H-%M') + '.csv'
        file = open(file_name, "w+")
        file.write('iteration,classifier,accuracy,train-time,test-time\n')
        for itr, acc in self.accuracies.items():
            for key, acc in acc.items():
                file.write(
                    str(itr) + ',' +
                    str(key) + ',' +
                    str(acc['accuracy']) + ',' +
                    str(acc['train-time']) + ',' +
                    str(acc['test-time']) + '\n'
                )

    def KNNOptimizer(self):
        self.acc = {}
        for knn in range(30):
            for itr in range(10):
                kneighbors = KNeighborsClassifier(self.train_x, self.train_y, self.test_x, self.test_y)
                # -----= KNeighbors
                kneighbors.train_withK(2 * knn + 1)
                kneighbors_labels = kneighbors.predic()
                kneighbors_acc = kneighbors.getAccuracy()
                kneighbors.printResult()
                self.acc[str(2 * knn + 1) + 'NN-' + str(itr)] = {'accuracy': kneighbors_acc,
                                                                 'train-time': kneighbors.trainTime(),
                                                                 'test-time': kneighbors.testTime(),
                                                                 }

        file_name = '../outs/knn-res-' + datetime.datetime.now().strftime('%d-%H-%M') + '.csv'
        file = open(file_name, "w+")
        file.write('Number Of K,iteration,accuracy,train-time,test-time\n')
        for itr, acc in self.acc.items():
            key = itr.split('-')[0].split("NN")[0]
            i = itr.split('-')[1]
            file.write(
                str(key) + ',' +
                str(i) + ',' +
                str(acc['accuracy']) + ',' +
                str(acc['train-time']) + ',' +
                str(acc['test-time']) + '\n'
            )


if __name__ == '__main__':
    mharc = MHARC(True)
    mharc.KNNOptimizer()
    # mharc.SaveResults()
