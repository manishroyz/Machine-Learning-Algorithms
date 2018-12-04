import copy
import matplotlib.pyplot as plt
import numpy as np
import data_extract as dex
from perceptron import Perceptron
import time
from random_seed import random_seed


# Initialize train and test lists
train1, train2, train3, train4, train5 = [], [], [], [], []
test1, test2, test3, test4, test5 = [], [], [], [], []

# ............Plot functionality initialization
plt.ioff()
plt.close("all")
plt.xlabel('Epochs')
plt.ylabel('Accuracy(%)')
plt.legend(loc='upper left')


# ********************************************************************************************************************
# Build the dataset used for Cross-validation for the various Perceptron Algorithms

def set_cross_validation():

    global train1, train2, train3, train4, train5
    global test1, test2, test3, test4, test5
    
    train1 = dex.extract(['dataset/CVSplits/training00.data', 'dataset/CVSplits/training01.data', 'dataset/CVSplits/training02.data', 'dataset/CVSplits/training03.data'])
    test1 = dex.extract(['dataset/CVSplits/training04.data'])
    train2 = dex.extract(['dataset/CVSplits/training01.data', 'dataset/CVSplits/training02.data', 'dataset/CVSplits/training03.data', 'dataset/CVSplits/training04.data'])
    test2 = dex.extract(['dataset/CVSplits/training00.data'])
    train3 = dex.extract(['dataset/CVSplits/training02.data', 'dataset/CVSplits/training03.data', 'dataset/CVSplits/training04.data', 'dataset/CVSplits/training00.data'])
    test3 = dex.extract(['dataset/CVSplits/training01.data'])
    train4 = dex.extract(['dataset/CVSplits/training03.data', 'dataset/CVSplits/training04.data', 'dataset/CVSplits/training00.data', 'dataset/CVSplits/training01.data'])
    test4 = dex.extract(['dataset/CVSplits/training02.data'])
    train5 = dex.extract(['dataset/CVSplits/training04.data', 'dataset/CVSplits/training00.data', 'dataset/CVSplits/training01.data', 'dataset/CVSplits/training02.data'])
    test5 = dex.extract(['dataset/CVSplits/training03.data'])

# ********************************************************************************************************************

# ********************************************************************************************************************
# Function definition for Basic Perceptron

def simple_perceptron():

    global train1, train2, train3, train4, train5
    global test1, test2, test3, test4, test5
    # Given Learning Rates
    learning_rate_list = [1, 0.1, 0.01]
    accuracy_dictionary = {}
    
    # Perform Cross Validation on varied Learning Rates
    for lr in learning_rate_list:
        epoch = 10
        p1 = Perceptron()
        p2 = Perceptron()
        p3 = Perceptron()
        p4 = Perceptron()
        p5 = Perceptron()
                
        for x in range(epoch):
            np.random.shuffle(train1)
            np.random.shuffle(train2)
            np.random.shuffle(train3)
            np.random.shuffle(train4)
            np.random.shuffle(train5)
            
            p1.train(train1, lr)
            p2.train(train2, lr)
            p3.train(train3, lr)
            p4.train(train4, lr)
            p5.train(train5, lr)
        
        p1.predict(test1)
        p2.predict(test2)
        p3.predict(test3)
        p4.predict(test4)
        p5.predict(test5)

        # Averaging the accuracies
        accuracy_dictionary[lr] = (p1.accuracy + p2.accuracy + p3.accuracy + p4.accuracy + p5.accuracy)/5
    
    best_hyperparameter = max(accuracy_dictionary, key=accuracy_dictionary.get)

    # Computation using best hyperparameter value found above
    # train, dev, test datasets
    train = dex.extract(["dataset/diabetes.train"])
    dev = dex.extract(["dataset/diabetes.dev"])
    test = dex.extract(["dataset/diabetes.test"])
    
    epoch = 20
    epoch_acc_dictionary = []
    p = Perceptron()
        
    for i in range(epoch):
        np.random.shuffle(train)
        p.train(train, best_hyperparameter)
        p.predict(dev)
        epoch_acc_dictionary.append((copy.deepcopy(p), p.accuracy))
    
    p_best_wrt_epocs = max(epoch_acc_dictionary, key = lambda x:x[1])
    P = p_best_wrt_epocs[0]
    P.predict(test)
    
    # Results......
    print("Best Hyper-parameter (Learning Rate) = " + str(best_hyperparameter))
    print("Cross validation accuracy for best Hyper-parameter (Learning Rate) = " + str(round(accuracy_dictionary[best_hyperparameter],4)))
    print("The total number of updates the learning algorithm performs on the training set = " + str(epoch_acc_dictionary[-1][0].updates))
    print("Development set accuracy = " + str(round(epoch_acc_dictionary[-1][1], 4)))
    print("Test set accuracy = " + str(round(P.accuracy, 4)))

    # Plot Epocs Accuracy
    x_axis = list(range(1, 21))
    y_axis = [x[1] for x in epoch_acc_dictionary]
    plt.plot(x_axis, y_axis, label="Simple Perceptron")
    plt.ylim([1,100])
    plt.show()

# ********************************************************************************************************************


# ********************************************************************************************************************
# Function definition for Decaying Learning Rate Perceptron

def decaying_perceptron():
    global train1, train2, train3, train4, train5
    global test1, test2, test3, test4, test5
    # Given Learning Rates
    learning_rate_list = [1,0.1,0.01]
    accuracy_dictionary = {}

    # Perform Cross Validation on varied Learning Rates
    for lr in learning_rate_list:
        epoch = 10
        p1 = Perceptron()
        p2 = Perceptron()
        p3 = Perceptron()
        p4 = Perceptron()
        p5 = Perceptron()
        
        for x in range(epoch):
            np.random.shuffle(train1)
            np.random.shuffle(train2)
            np.random.shuffle(train3)
            np.random.shuffle(train4)
            np.random.shuffle(train5)
                     
            p1.train(train1, lr, True)
            p2.train(train2, lr, True)
            p3.train(train3, lr, True)
            p4.train(train4, lr, True)
            p5.train(train5, lr, True)
        
        p1.predict(test1)
        p2.predict(test2)
        p3.predict(test3)
        p4.predict(test4)
        p5.predict(test5)

        # Averaging the accuracies
        accuracy_dictionary[lr] = (p1.accuracy + p2.accuracy + p3.accuracy + p4.accuracy + p5.accuracy)/5
    
    best_hyperparameter = max(accuracy_dictionary, key=accuracy_dictionary.get)

    # Computation using best hyperparameter value found above
    # train, dev, test datasets
    train = dex.extract(["dataset/diabetes.train"])
    dev = dex.extract(["dataset/diabetes.dev"])
    test = dex.extract(["dataset/diabetes.test"])
    
    epoch = 20
    
    epoch_acc_dictionary = []
    p = Perceptron()
        
    for i in range(epoch):
        np.random.shuffle(train)
        p.train(train, best_hyperparameter, True)
        p.predict(dev)
        epoch_acc_dictionary.append((copy.deepcopy(p), p.accuracy))
    
    p_best_wrt_epocs = max(epoch_acc_dictionary, key = lambda x:x[1])
    P = p_best_wrt_epocs[0]
    P.predict(test)

    # Results......
    print("Best Hyper-parameter (Learning Rate) = " + str(best_hyperparameter))
    print("Cross validation accuracy for best Hyper-parameter (Learning Rate) = " + str(round(accuracy_dictionary[best_hyperparameter],4)))
    print("The total number of updates the learning algorithm performs on the training set  = " + str(epoch_acc_dictionary[-1][0].updates))
    print("Developement set accuracy = " + str(round(epoch_acc_dictionary[-1][1],4)))
    print("Test set accuracy = " + str(round(P.accuracy,4)))

    # Plot Epocs Accuracy
    x_axis = list(range(1,21))
    y_axis = [x[1] for x in epoch_acc_dictionary]
    plt.plot(x_axis, y_axis, label = "Perceptron with dynamic learning")
    plt.ylim([1,100])
    plt.show()

# ********************************************************************************************************************


# ********************************************************************************************************************
# Function definition for Margin Perceptron

def margin_perceptron():
    global train1, train2, train3, train4, train5
    global test1, test2, test3, test4, test5

    # Given Hyper-parameters
    learning_rate_list = [1,0.1,0.01]
    margins = [1, 0.1, 0.01]
    # Generating combinations of both hyper-parameters : margin & learning rate
    combinations = [(x,y) for x in margins for y in learning_rate_list]

    accuracy_dictionary = []

    # Perform Cross Validation on varied combinations formed above
    for c in combinations:
        epoch = 10
        p1 = Perceptron(c[0])
        p2 = Perceptron(c[0])
        p3 = Perceptron(c[0])
        p4 = Perceptron(c[0])
        p5 = Perceptron(c[0])
        
        for x in range(epoch):
            np.random.shuffle(train1)
            np.random.shuffle(train2)
            np.random.shuffle(train3)
            np.random.shuffle(train4)
            np.random.shuffle(train5)
                     
            p1.train(train1, c[1], True)
            p2.train(train2, c[1], True)
            p3.train(train3, c[1], True)
            p4.train(train4, c[1], True)
            p5.train(train5, c[1], True)
        
        p1.predict(test1)
        p2.predict(test2)
        p3.predict(test3)
        p4.predict(test4)
        p5.predict(test5)

        # Averaging the accuracies
        accuracy_dictionary.append((c,(p1.accuracy + p2.accuracy + p3.accuracy + p4.accuracy + p5.accuracy)/5))

    # Computation using best hyperparameter value found above
    best_hyperparameter_set = max(accuracy_dictionary, key = lambda x:x[1])
    best_hyperparameter = best_hyperparameter_set[0]

    # train, dev, test datasets
    train = dex.extract(["dataset/diabetes.train"])
    dev = dex.extract(["dataset/diabetes.dev"])
    test = dex.extract(["dataset/diabetes.test"])
    
    epoch = 20
    
    epoch_acc_dictionary = []
    p = Perceptron(best_hyperparameter[0])
        
    for i in range(epoch):
        np.random.shuffle(train)
        p.train(train, best_hyperparameter[1], True)
        p.predict(dev)
        epoch_acc_dictionary.append((copy.deepcopy(p), p.accuracy))
    
    p_best_wrt_epocs = max(epoch_acc_dictionary, key = lambda x:x[1])
    P = p_best_wrt_epocs[0]
    P.predict(test)

    # Results......
    print("Best margin = " + str(best_hyperparameter[0]))
    print("Best learning rate = " + str(best_hyperparameter[1]))
    print("Cross validation accuracy for best learning rate = " + str(round(best_hyperparameter_set[1], 4)))
    print("The total number of updates the learning algorithm performs on the training set = " + str(epoch_acc_dictionary[-1][0].updates))
    print("Developement set accuracy = " + str(round(epoch_acc_dictionary[-1][1], 4)))
    print("Test set accuracy = " + str(round(P.accuracy,4)))

    # Plot Epocs Accuracy
    x_axis = list(range(1,21))
    y_axis = [x[1] for x in epoch_acc_dictionary]
    plt.plot(x_axis, y_axis, label = "Margin Perceptron")
    plt.ylim([1,100])
    plt.show()

# ********************************************************************************************************************


# ********************************************************************************************************************
# Function definition for Averaged Perceptron
    
def averaged_perceptron():
    global train1, train2, train3, train4, train5
    global test1, test2, test3, test4, test5

    # Given learning rates
    learning_rate_list = [1,0.1,0.01]
    accuracy_dictionary = {}

    # Perform Cross Validation on varied combinations formed above
    for lr in learning_rate_list:
        epoch = 10
        p1 = Perceptron()
        p2 = Perceptron()
        p3 = Perceptron()
        p4 = Perceptron()
        p5 = Perceptron()
                
        for x in range(epoch):
            np.random.shuffle(train1)
            np.random.shuffle(train2)
            np.random.shuffle(train3)
            np.random.shuffle(train4)
            np.random.shuffle(train5)
            
            p1.train(train1, lr)
            p2.train(train2, lr)
            p3.train(train3, lr)
            p4.train(train4, lr)
            p5.train(train5, lr)
        
        p1.predict(test1, True)
        p2.predict(test2, True)
        p3.predict(test3, True)
        p4.predict(test4, True)
        p5.predict(test5, True)
        
        accuracy_dictionary[lr] = (p1.accuracy + p2.accuracy + p3.accuracy + p4.accuracy + p5.accuracy)/5
    
    best_hyperparameter = max(accuracy_dictionary, key=accuracy_dictionary.get)

    # Computation using best hyperparameter value found above
    train = dex.extract(["dataset/diabetes.train"])
    dev = dex.extract(["dataset/diabetes.dev"])
    test = dex.extract(["dataset/diabetes.test"])
    
    epoch = 20
    epoch_acc_dictionary = []
    p = Perceptron()
        
    for i in range(epoch):
        np.random.shuffle(train)
        p.train(train, best_hyperparameter)
        p.predict(dev, True)
        epoch_acc_dictionary.append((copy.deepcopy(p), p.accuracy))
    
    p_best_wrt_epocs = max(epoch_acc_dictionary, key = lambda x:x[1])
    P = p_best_wrt_epocs[0]
    P.predict(test, True)

    # Results......
    print("Best Hyper-parameter (Learning Rate) = " + str(best_hyperparameter))
    print("Cross validation accuracy for best Hyper-parameter (Learning Rate) = " + str(round(accuracy_dictionary[best_hyperparameter],4)))
    print("The total number of updates the learning algorithm performs on the training set = " + str(epoch_acc_dictionary[-1][0].updates))
    print("Developement set accuracy = " + str(round(epoch_acc_dictionary[-1][1], 4)))
    print("Test set accuracy = " + str(round(P.accuracy,4)))

    # Plot Epocs Accuracy
    x_axis = list(range(1,21))
    y_axis = [x[1] for x in epoch_acc_dictionary]
    plt.plot(x_axis, y_axis, label = "Averaged Perceptron")
    plt.ylim([1,100])
    plt.show()
        
# ********************************************************************************************************************


# ********************************************************************************************************************
# Function definition for Aggresive Perceptron

def aggresive_perceptron():
    global train1, train2, train3, train4, train5
    global test1, test2, test3, test4, test5

    # Given margin values
    learning_rate_list = [1,0.1,0.01]
    accuracy_dictionary = {}

    # Perform Cross Validation on varied combinations formed above
    for lr in learning_rate_list:
        epoch = 10
        p1 = Perceptron()
        p2 = Perceptron()
        p3 = Perceptron()
        p4 = Perceptron()
        p5 = Perceptron()
                
        for x in range(epoch):
            np.random.shuffle(train1)
            np.random.shuffle(train2)
            np.random.shuffle(train3)
            np.random.shuffle(train4)
            np.random.shuffle(train5)
            
            p1.train(train1, lr, aggressive = True)
            p2.train(train2, lr, aggressive = True)
            p3.train(train3, lr, aggressive = True)
            p4.train(train4, lr, aggressive = True)
            p5.train(train5, lr, aggressive = True)
        
        p1.predict(test1, aggressive = True)
        p2.predict(test2, aggressive = True)
        p3.predict(test3, aggressive = True)
        p4.predict(test4, aggressive = True)
        p5.predict(test5, aggressive = True)
        
        accuracy_dictionary[lr] = (p1.accuracy + p2.accuracy + p3.accuracy + p4.accuracy + p5.accuracy)/5

    best_hyperparameter = max(accuracy_dictionary, key=accuracy_dictionary.get)

    # Computation using best hyperparameter value found above
    train = dex.extract(["dataset/diabetes.train"])
    dev = dex.extract(["dataset/diabetes.dev"])
    test = dex.extract(["dataset/diabetes.test"])
    
    epoch = 20
    epoch_acc_dictionary = []
    p = Perceptron()
        
    for i in range(epoch):
        np.random.shuffle(train)
        p.train(train, best_hyperparameter, aggressive = True)
        p.predict(dev, aggressive = True)
        epoch_acc_dictionary.append((copy.deepcopy(p), p.accuracy))
    
    p_best_wrt_epocs = max(epoch_acc_dictionary, key = lambda x:x[1])
    P = p_best_wrt_epocs[0]
    P.predict(test, aggressive = True)

    # Results......
    print("Best Hyper-parameter (Margin) = " + str(best_hyperparameter))
    print("Cross validation accuracy for best Hyper-parameter = " + str(round(accuracy_dictionary[best_hyperparameter],4)))
    print("The total number of updates the learning algorithm performs on the training set = " + str(epoch_acc_dictionary[-1][0].updates))
    print("Development set accuracy = " + str(round(epoch_acc_dictionary[-1][1],4)))
    print("Test set accuracy = " + str(round(P.accuracy,4)))

    # Plot Epocs Accuracy
    x_axis = list(range(1,21))
    y_axis = [x[1] for x in epoch_acc_dictionary]
    plt.plot(x_axis, y_axis, label = "Aggressive Perceptron")
    plt.ylim([1,100])
    plt.show()
    
# ********************************************************************************************************************


# ********************************************************************************************************************
# Function Calls ..............

#To alter seed value, set the value of "random_seed" variable in  'random_seed.py'
# Comment/ Uncomment the below line to unseed/seed the random functions
np.random.seed(random_seed)

set_cross_validation()
print("***********************************************************************************")
print("***********************************************************************************")
print("Calling Simple Perceptron......")
print("***********************************************************************************")
simple_perceptron()
print("***********************************************************************************")
print("***********************************************************************************")
print("Calling Perceptron with Decaying learning rate............")
print("***********************************************************************************")
decaying_perceptron()
print("***********************************************************************************")
print("***********************************************************************************")
print("Calling Margin Perceptron..........")
print("***********************************************************************************")
margin_perceptron()    
print("***********************************************************************************")
print("***********************************************************************************")
print("Calling Averaged Perceptron....................")
print("***********************************************************************************")
averaged_perceptron()
print("***********************************************************************************")
print("***********************************************************************************")
print("Calling Aggressive Perceptron with Margin............")
print("***********************************************************************************")
aggresive_perceptron()
print("***********************************************************************************")
print("***********************************************************************************")

 
