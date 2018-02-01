# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 20:23:44 2018

@author: Sparsh Gupta
"""

from mnist import MNIST
import numpy as np

mndata = MNIST('')


def lin(x):
    return 2*x

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_(x):
    return np.exp(-x)/(np.multiply((1+np.exp(-x)),(1+np.exp(-x))))

def softmax(x):
    x_ = np.sum(np.exp(x), axis=1)
    return np.exp(x)/np.resize(x_, [x_.shape[0],1])

def ReLu(x):
    ret = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i,j] > 0:
                ret[i,j] = x[i,j]
    return ret

def ReLu_(x):
    ret = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i,j] > 0:
                ret[i,j] = 1
    return ret

def tanhx(x, a=1.7159, b=2.0/3.0):
    return a*np.tanh(b*x)

def tanhx_(x, a=1.7159, b=2.0/3.0):
    return a*b*(1 - np.multiply((np.tanh(b*x)),(np.tanh(b*x))))
    
def FeedForward(X_train_, Y_train_, X_test, Y_test, hidden, activations, activations_, eta_=0.1, batch_size=128, epochs = 150, holdoutRatio = 0.2,momentum=0):
    ##################Split Data into Training and Validation##################
    indxs = np.random.permutation(X_train_.shape[0])
    idxs = indxs[:int((1-holdoutRatio)*X_train_.shape[0])]
    X_train = []
    Y_train = []
    X_valid = []
    Y_valid = []
    for i in range(X_train_.shape[0]):
        if i in idxs:
            X_train.append(np.resize(X_train_[i,:],(X_train_.shape[1])))
            Y_train.append(np.resize(Y_train_[i,:],(Y_train_.shape[1])))
        else:
            X_valid.append(np.resize(X_train_[i,:],(X_train_.shape[1])))
            Y_valid.append(np.resize(Y_train_[i,:],(Y_train_.shape[1])))
    X_train = np.matrix(X_train)
    Y_train = np.matrix(Y_train)
    X_valid = np.matrix(X_valid)
    Y_valid = np.matrix(Y_valid)
    print('Train-Valid split done.')

    Weights = []
    prev = X_train.shape[1]
    
    #####################Initialize Weight Matrices#################
    for i in range(len(hidden)):
        curr = hidden[i]
        tmp = np.random.normal(0,1.0/np.sqrt(prev+1),[prev+1,curr])
        Weights.append(tmp)
        prev = curr
    Weights.append(np.random.normal(0,1.0/np.sqrt(prev+1), [prev+1, Y_train.shape[1]]))
    
    def forwardPass(Xdata, A, Z):
        curr_ = Xdata
        for i in range(len(Weights)):
            curr_ = np.hstack((curr_, np.ones([curr_.shape[0],1])))
            curr_ = np.dot(curr_, Weights[i])
            A.append(np.hstack((curr_, np.ones([curr_.shape[0],1]))))
            curr_ = activations[i](curr_)
            Z.append(np.hstack((curr_, np.ones([curr_.shape[0],1]))))
        return curr_
    
    def forwardPass_(Xdata, A, Z):
        curr_ = Xdata
        for i in range(len(laWeights)):
            curr_ = np.hstack((curr_, np.ones([curr_.shape[0],1])))
            curr_ = np.dot(curr_, laWeights[i])
            A.append(np.hstack((curr_, np.ones([curr_.shape[0],1]))))
            curr_ = activations[i](curr_)
            Z.append(np.hstack((curr_, np.ones([curr_.shape[0],1]))))
        return curr_
    
    def calculateLoss(Xtrain, Ytrain, A, Z):
        E_ = 0.0
        curr_ = forwardPass(Xtrain, A, Z)
        for i in range(Ytrain.shape[0]):
            for j in range(Ytrain[0].shape[0]):
                if Ytrain[i,j] == 1.0:
                    E_ -= np.log(curr_[i,j])
        return E_/len(Xtrain)
        
    def calculateAccuracy(Xtest, Ytest, A, Z):
        curr_ = forwardPass(Xtest, A, Z)
        curr_ = np.argmax(curr_, axis=1)
        count = 0
        for i in range(len(Ytest)):
            if int(Ytest[i,curr_[i]]) == 1:
                count+=1
        return count*1.0/len(Xtest)
    
    def shuffle(X_tr, Y_tr):
        indxs = np.random.permutation(X_tr.shape[0])
        idxs = indxs[:X_tr.shape[0]]
        X_tr_ = []
        Y_tr_ = []
        for i in idxs:
            X_tr_.append(np.resize(X_tr[i,:],(X_train_.shape[1])))
            Y_tr_.append(np.resize(Y_tr[i,:],(Y_train_.shape[1])))
        X_tr_ = np.matrix(X_tr_)
        Y_tr_ = np.matrix(Y_tr_)
        return X_tr_, Y_tr_
    
    #################Epochs Begin!##################################
    prev_mom = {}
    laWeights = []
    for i in range(len(Weights)):
        prev_mom[i] = np.zeros((Weights[i].shape[0],Weights[i].shape[1]))
        laWeights.append(Weights[i])
    training_loss = []
    validation_loss = []
    #testing_loss = []
    training_accuracy = []
    #testing_accuracy = []
    validation_accuracy = []
    saved_weights = []
    print('Initialization Complete. Training Begins...\n')
    for e in range(epochs):
        eta=eta_
        tra_shape = X_train.shape[1]
        temp = np.hstack((X_train,Y_train))
        np.random.shuffle(temp)
        X_tra = temp[:batch_size,:tra_shape]
        Y_tra = temp[:batch_size,tra_shape:]        
        Z = []
        A = []
        delta = {}
        curr = forwardPass_(X_tra,A,Z)

        delta[len(laWeights)-1] = Y_tra - curr
        i = len(laWeights) - 1
        while(i >= 0):
            if i == len(laWeights) - 1:
                delta[i-1] = np.multiply(activations_[i-1](A[i-1]), np.dot(delta[i],laWeights[i].T))
            elif i>0:
                delta[i-1] = np.multiply(activations_[i-1](A[i-1]), np.dot(delta[i][:,:-1],laWeights[i].T))
            if i == 0:    
                upd = momentum*prev_mom[i] - eta*np.dot(np.hstack((X_tra, np.ones([X_tra.shape[0],1]))).T, delta[i][:,:-1])/X_tra.shape[0]
            elif i == len(laWeights)-1:
                upd = momentum*prev_mom[i] - eta*np.dot(Z[i-1].T, delta[i])/X_tra.shape[0]
            else:
                upd = momentum*prev_mom[i] - eta*np.dot(Z[i-1].T, delta[i][:,:-1])/X_tra.shape[0] 
            Weights[i] = Weights[i] - upd
            prev_mom[i] = upd
            laWeights[i] = Weights[i] - momentum*upd
            i-=1
        
        training_accuracy.append(calculateAccuracy(X_train, Y_train, A, Z))
        validation_accuracy.append(calculateAccuracy(X_valid, Y_valid, A, Z))
        training_loss.append(calculateLoss(X_train, Y_train, A, Z))
        validation_loss.append(calculateLoss(X_valid, Y_valid, A, Z))
        saved_weights.append(Weights)
        if e%10 == 0:
            print('Epoch = ',e,'Train Acc = ',training_accuracy[e],'Valid Acc = ',validation_accuracy[e],'Train Loss = ',training_loss[e],'Valid Loss = ',validation_loss[e])
    
    training_accuracy = np.array(training_accuracy)
    validation_accuracy = np.array(validation_accuracy)
    training_loss = np.array(training_loss)
    validation_loss = np.array(validation_loss)
    idx = np.argmax(validation_accuracy)
    final_weights = saved_weights[idx]
    
    A = []
    Z = []
    Weights = final_weights
    testing_accuracy = calculateAccuracy(X_test,Y_test,A,Z)
    return idx,training_accuracy[idx],validation_accuracy[idx],training_loss[idx],validation_loss[idx],testing_accuracy,final_weights
    
def main():
    images_train, labels_train = mndata.load_training()
    images_test, labels_test = mndata.load_testing()
    print('Data Loaded.')
    images_train = np.matrix(images_train)
    images_train = images_train/127.5 - 1
    images_test = np.matrix(images_test)
    images_test = images_test/127.5 - 1
    print('Data Normalized.')
    tmp = np.zeros([len(labels_train),10])
    for i in range(len(labels_train)):
        tmp[i, labels_train[i]]=1
    labels_train = np.matrix(tmp)
    tmp = np.zeros([len(labels_test),10])
    for i in range(len(labels_test)):
        tmp[i, labels_test[i]]=1
    labels_test = np.matrix(tmp)
    print('Labels One-Hot Encoded.')
    
    ep = 200
    idx,tr_a,va_a,tr_l,va_l,te_acc,final_weights = FeedForward(images_train,labels_train, images_test, labels_test, [100,50], [ReLu, ReLu, softmax],[ReLu_,ReLu_],eta_=0.01,momentum=0.9,batch_size=256, epochs=ep)
    print('Best Fit Obtained at Epoch : ',idx)
    print('Training Loss at best fit : ',tr_l)
    print('Validation Loss at best fit : ',va_l)
    print('Training Accuracy at best fit : ',tr_a)
    print('Validation Accuracy at best fit : ',va_a)
    print('Testing Accuracy at best fit : ',te_acc)
    
if __name__ == "__main__":
    main()