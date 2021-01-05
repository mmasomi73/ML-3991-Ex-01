from DataCollector import DataCollector
from GradientDescentClassifier import GradientDescentClassifier


class MHARC:
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    def __init__(self, log=False):
        self.dc = dc = DataCollector(log)
        self.train_x = dc.getTrainData()
        self.train_y = dc.getTrainLabel()
        self.test_x = dc.getTestData()
        self.test_y = dc.getTestLabel()

    def executor(self):
        GDC = GradientDescentClassifier(self.train_x, self.train_y, self.test_x, self.test_y)
        GDC.train()
        GDC_labels = GDC.predic()
        GDC_acc = GDC.getAccuracy()
        GDC.printResult()


if __name__ == '__main__':
    mharc = MHARC(True)
    mharc.executor()
