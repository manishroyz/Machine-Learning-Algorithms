import numpy as np
import data_extract as de
from random_seed import random_seed
import random

#To alter seed value, set the value of "random_seed" variable in  'random_seed.py'
# Comment/ Uncomment the below line to unseed/seed the random functions
random.seed(random_seed)

class Perceptron:
    def __init__(self, margin = 0):
        self.weights = []
        self.bias = []
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.predictions = []
        self.accuracy = 0
        self.init_weights_bias(19)
        self.updates = 0
        self.t = 0
        self.margin = margin
        self.avg_weights = self.weights
        self.avg_bias = self.bias
        
    def train(self, data, lr, dynamic_lr = False, aggressive = False):
        if aggressive:
            self.train_aggressive(data, lr)
        else:
            data_mod = data
            lr_0 = lr
            self.x_train = data_mod[:,:-1]
            self.y_train = data_mod[:,-1]
            
            for index, i in enumerate(self.x_train):
                h = np.inner(i, self.weights) + self.bias
                f = self.y_train[index]
                if dynamic_lr:
                        lr = lr_0/(1+self.t)
                        self.t += 1
                if (h*f <= self.margin):
                    self.weights = self.weights + lr*f*i
                    self.bias = self.bias + lr*f
                    self.updates += 1    
                self.avg_weights = self.avg_weights + self.weights
                self.avg_bias = self.avg_bias + self.bias
            
    def train_aggressive(self, data, margin):
        data_mod = np.c_[np.ones((data.shape[0],1)), data]
        self.init_weights_bias(20)
        self.x_train = data_mod[:,:-1]
        self.y_train = data_mod[:,-1]

        for index, i in enumerate(self.x_train):
            h = np.inner(i, self.weights)
            f = self.y_train[index]
            if (h*f <= margin):
                lr = (margin - f*np.inner(self.weights, i))/(np.inner(i, i) + 1)
                self.weights = self.weights + lr*f*i
                self.bias = self.bias + lr*f
                self.updates += 1
        
    def init_weights_bias(self, cols):
        ran_init = random.uniform(-0.01, 0.01)
        self.weights = np.array([ran_init]*cols)
        self.bias = ran_init
        
    def predict(self, data, average = False, aggressive = False):
        if aggressive:
            self.predict_aggressive(data)
        else:
            self.x_test = data[:,:-1]
            self.y_test = data[:, -1]
            w = self.weights
            b = self.bias
            if average:
                w = self.avg_weights
                b = self.avg_bias
                
            preds = []
            for x in self.x_test:
                preds.append(np.inner(x, w) + b)
            self.predictions = preds
            self.accuracy = self.calc_accuracy()
            
    def predict_aggressive(self, data):
        data = np.c_[np.ones((data.shape[0],1)), data]
        self.x_test = data[:,:-1]
        self.y_test = data[:, -1]
        w = self.weights
        b = self.bias
                    
        preds = []
        for x in self.x_test:
            preds.append(np.inner(x, w) + b)
        self.predictions = preds
        self.accuracy = self.calc_accuracy()
        
    def calc_accuracy(self):
        correct = 0
        for i,x in enumerate(self.predictions):
            if x*self.y_test[i] >= 0:
                correct += 1
        return correct/len(self.y_test)*100

