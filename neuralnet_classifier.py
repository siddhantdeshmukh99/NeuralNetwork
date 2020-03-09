import random
import math
import numpy as np

class Layer:
    def __init__(self,size,no_inp,input_layer=False):
        self.perseptrons=[]
        self.no_inp=no_inp
        if input_layer == False:
            self.bias=random.uniform(-1/math.sqrt(self.no_inp),1/math.sqrt(self.no_inp))
            for i in range(size):
                self.perseptrons.append([])
            for y in self.perseptrons:
                for _ in range(self.no_inp):
                    y.append(random.uniform(-1/math.sqrt(self.no_inp),1/math.sqrt(self.no_inp)))
        else:
            for i in range(size):
                self.perseptrons.append([1])
            self.bias=0
            
class NeuralNetwork:
    def __init__(self,x_train,y_train):
        self.layers=[]
        self.x_train=x_train
        self.y_train=y_train
        self.no_opt=len(y_train.unique())
        self.layers.append(Layer(len(self.x_train.columns),1,input_layer=True))
        self.layers.append(Layer(self.no_opt,len(self.x_train.columns)))
        
    def addLayer(self,size):
        self.layers.pop()
        if len(self.layers) == 1:
            self.layers.append(Layer(size,len(self.x_train.columns)))
        else:
            self.layers.append(Layer(size,len(self.layers[-1].perseptrons)))
        self.layers.append(Layer(self.no_opt,size))
    
    def sigmoid(self,x):
        return 1/(1+math.exp(-x))
    
    def forwardPropogation(self,data):
        ret=[]
        for k,x in enumerate(self.layers):
            opt=[]
            for j,y in enumerate(x.perseptrons):
                sum=0
                for i,z in enumerate(y):
                    sum=sum+z*data[i]
                sum=sum+x.bias
                if k!=0:
                    opt.append(self.sigmoid(sum))
                else:
                    opt.append(sum)
            data=opt
            ret.append(opt)
        return ret
                    
    def backwardPropogation(self,output,target,learning_rate):
        err=[]
        for i,x in enumerate(output[-1]):                                  
            err.append(x*(1-x)*(target[i]-x))
        print(output[-1],target,end="\n")
            
        for j,x in reversed(list(enumerate(self.layers[:-1]))):
            err1=[]
            if j>0:
                for i,y in enumerate(x.perseptrons):
                    sum=0
                    for k,z in enumerate(err):
                        sum=sum+z*self.layers[j+1].perseptrons[k][i]
                    err1.append(sum*output[j][i]*(1-output[j][i]))
            s=0       
            for k,_ in enumerate(self.layers[j+1].perseptrons):
                s=s+err[k]
                for i,_ in enumerate(self.layers[j+1].perseptrons[k]):
                    self.layers[j+1].perseptrons[k][i]=self.layers[j+1].perseptrons[k][i]+(err[k]*output[j][i]*(learning_rate))
            self.layers[j+1].bias=self.layers[j+1].bias+s*(learning_rate)
            err=err1
        
    def train(self,learning_rate):
        for i,x in enumerate(self.x_train.values):
            opt=self.forwardPropogation(x)
            target=[]
            for y in self.y_train.unique():
                if y == self.y_train.values[i]:
                    target.append(1)
                else:
                    target.append(0)
            self.backwardPropogation(opt,target,learning_rate)
    
    def test(self,data):
        ret=[]
        for i,x in enumerate(data.values):
            ret.append(self.forwardPropogation(x)[-1])
        return ret
        
