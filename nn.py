# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 18:54:55 2014

@author: skew
"""
"""
import nn
reload(nn)
from nn import *
data=pd.read_table("ok.txt",",",header=0)
nn=NNet()
nn.set_dataset(data)
nn.set_alpha(1)
nn.set_n_iter(500)
nn.set_lambd(0)
nn.set_batch_size(112)

nn.add_n_layers(5)

nn.get_input_layer().add_n_vertex(2)

nn.get_input_layer().get_vertex(0).set_num_nodes(24)
nn.get_input_layer().get_vertex(1).set_num_nodes(12)

nn.get_layer(1).add_n_vertex(2)

nn.get_layer(1).get_vertex(0).set_num_nodes(6)
nn.get_layer(1).get_vertex(0).add_in_edge(nn.get_input_layer().get_vertex(0))

nn.get_layer(1).get_vertex(1).set_num_nodes(6)
nn.get_layer(1).get_vertex(1).add_in_edge(nn.get_input_layer().get_vertex(1))

nn.get_layer(2).add_n_vertex(1)
nn.get_layer(2).get_vertex(0).set_num_nodes(36)
nn.get_layer(2).get_vertex(0).add_in_edge(nn.get_layer(1).get_vertex(0))
nn.get_layer(2).get_vertex(0).add_in_edge(nn.get_layer(1).get_vertex(1))

nn.get_layer(3).add_n_vertex(1)
nn.get_layer(3).get_vertex(0).set_num_nodes(6)
nn.get_layer(3).get_vertex(0).add_in_edge(nn.get_layer(2).get_vertex(0))

nn.get_output_layer().add_n_vertex(1)
nn.get_output_layer().get_vertex(0).set_num_nodes(24)
nn.get_output_layer().get_vertex(0).add_in_edge(nn.get_layer(3).get_vertex(0))

nn.initialize_data()
nn.initialize_weight_matrixes(0.01)
nn.initialize_deltas()

nn.grad_descent()
"""
import pandas as pd
import numpy as np
import random 

def sigmoid(mtx):
    return 1./(1.+np.exp(-(mtx)))
    
#derivative of the sigmoid function used in back_prop   
def sigmoid_grad(z):
    """(matrix/vector/number) -> (matrix/vector/number)
    
    Return the derivative of the sigmoid function applied to a number,vector
    or matrix
    
    >>> sigmoid_grad(0)
    0.25
    """
    return sigmoid(z)*(1-sigmoid(z))

class Vertex:
    def __init__(self):
        self.num_nodes=0
        self.pointed_from=[]
        self.data=np.shape((0,0))
        self.weight_matrix=np.shape((0,0))
        self.grad=np.shape((0,0))
        self.delta=np.shape((0,0))
        self.grad_check=np.shape((0,0))
    def set_data(self,d):
        n=d.shape[1]+1
        self.data=np.zeros((d.shape[0],n))
        self.data[:,1:]=d
        self.set_num_nodes(n)
        
    def get_data(self):
        return self.data

    def get_weight_matrix(self):
        return self.weight_matrix
    def set_weight_matrix(self,m):
        #self.weight_matrix=np.ones((m.shape[0]+1,m.shape[1])
        #self.weight_matrix[:,1:]=m
        self.weight_matrix=m
    def add_in_edge(self,node_key):
        self.pointed_from.append(node_key)
        #self.weight_matrix=np.zeros(())
    def get_pointed_from(self):
        return self.pointed_from
    def get_num_nodes(self):
        return self.num_nodes
    def set_num_nodes(self,n):
        self.num_nodes=n

    
class NNet:
    def __init__(self):
        self.layers=[]
        self.cost=0
        self.target_variable=np.shape((0,0))
        self.alpha=0
        self.n_iter=0
        self.lambd=0
        self.dataset=pd.DataFrame()
    def set_dataset(self,d):
        data=one_hot_encoding(d, ['relation','person1','person2'])
        data=data.as_matrix()
        self.dataset=data
    def set_n_iter(self,n):
        self.n_iter=n
    def set_alpha(self,a):
        self.alpha=a
    def set_lambd(self,l):
        self.lambd=l
    def set_target_variable(self,y):
        self.target_variable=y
        
    def add_n_layers(self,num_layers):
        if num_layers>2:
            self.layers.append(InputLayer())
            for i in range((num_layers-2)):
                self.layers.append(HiddenLayer())
            self.layers.append(OutputLayer())
            
    def num_layers(self):
        return len(self.layers)
        
    def get_layer(self,num):
        return self.layers[num]
        
    def get_input_layer(self):
        return self.layers[0]
        
    def get_output_layer(self):
        return self.layers[-1]
    
        
        
    def forward_prop(self):
        for l_idx in range(1,(self.num_layers()-1)):
            for vertex in self.get_layer(l_idx).get_vertexes():
                for in_vertex in vertex.get_pointed_from():
                    vertex.data[:,1:]+=np.dot(in_vertex.data,in_vertex.weight_matrix)
                vertex.data[:,1:]=sigmoid(vertex.data[:,1:])
        for vertex in self.get_output_layer().get_vertexes():
                for in_vertex in vertex.get_pointed_from():
                    vertex.data+=np.dot(in_vertex.data,in_vertex.weight_matrix)
                vertex.data=sigmoid(vertex.data)
                    
                
    def get_batch_size(self):
        return self.batch_size
                
    def set_batch_size(self,b):
        self.batch_size=b
        
    def initialize_data(self):
        for l_idx in range(self.num_layers()-1):
            for vertex in self.get_layer(l_idx).get_vertexes():
                vertex.set_data(np.zeros((self.batch_size,vertex.get_num_nodes())))
        for vertex in self.get_output_layer().get_vertexes():
            vertex.set_data(np.zeros((self.batch_size,(vertex.get_num_nodes()-1))))
        
    def initialize_weight_matrixes(self,epsilon_init):
        for l_idx in range(1,(self.num_layers()-1)):
            for vertex in self.get_layer(l_idx).get_vertexes():
                for in_vertex in vertex.get_pointed_from():
                    in_vertex.set_weight_matrix(\
                            np.random.rand(in_vertex.get_num_nodes(),vertex.get_num_nodes()-1)\
                            *2*epsilon_init-epsilon_init)
        for vertex in self.get_output_layer().get_vertexes():
            for in_vertex in vertex.get_pointed_from():
                in_vertex.set_weight_matrix(\
                            np.random.rand(in_vertex.get_num_nodes(),vertex.get_num_nodes())\
                            *2*epsilon_init-epsilon_init)
                            
    def initialize_deltas(self):
        for l_idx in range(self.num_layers()):
            for vertex in self.get_layer(l_idx).get_vertexes():
                vertex.delta=0*vertex.data
                    
    def initialize_grad(self):
        for l_idx in range(self.num_layers()-1):
            for vertex in self.get_layer(l_idx).get_vertexes():
                vertex.grad=0*vertex.weight_matrix
                
    def initialize_gradent_checking(self):
        for l_idx in range(self.num_layers()-1):
            for vertex in self.get_layer(l_idx).get_vertexes():
                vertex.grad_check=0*vertex.weight_matrix
    def get_estimate(self):
        return self.get_output_layer().get_vertex(0).data
    def compute_cost(self):
        print self.target_variable.shape
        print self.get_estimate().shape
        print '-'
        print self.target_variable.shape
        print self.get_estimate().shape
        cost_pre=-(np.dot((self.target_variable.T)*\
            np.log(self.get_estimate())))-np.dot((1-self.target_variable.T),np.log(1-self.get_estimate()))
        self.cost=(1./self.batch_size)*np.sum(np.diag(cost_pre))
        #LAMBDA L8R
        #if self.lambd>0:
            #(self.lambd/(2*self.batch_size))
        
    def get_cost(self):
        return self.cost
        
    def set_data(self):
        #m=112
        #idxs=random.sample(xrange(m),m-self.batch_size)
        idxs=np.random.randint(self.dataset.shape[0],size=self.batch_size)
        #dataset=pd.read_table("ok.txt",",",header=0,skiprows=idxs)
        #relation=pd.get_dummies(dataset['relation']).as_matrix()
        #person1=pd.get_dummies(dataset['person1']).as_matrix()
        #person2=pd.get_dummies(dataset['person2']).as_matrix()
        self.get_input_layer().get_vertex(0).set_data(self.dataset[idxs,0:24])
        self.get_input_layer().get_vertex(1).set_data(self.dataset[idxs,24:36])
        self.set_target_variable(self.dataset[idxs,36:60])
        
    def gradient_check(self,epsilon):
        self.initialize_grad_checking()
        for layer_idx in range(self.num_layers()-1):
            for vertex in self.get_layer(layer_idx).get_vertexes():
                for i in range(vertex.weight_matrix.shape[0]):
                    for j in range(vertex.weight_matrix.shape[1]):
                        vertex.weight_matrix[i][j]+=epsilon
                        cost_plus=self.compute_cost()
                        vertex.weight_matrix[i][j]-=(2*epsilon)
                        cost_minus=self.compute_cost()
                        vertex.weight_matrix[i][j]+=+epsilon
                        vertex.grad_check[i][j]=(cost_plus-cost_minus)/(2*epsilon)
        
    def back_prop(self):
        self.initialize_grad()
        for vertex in self.get_output_layer().get_vertexes():
            vertex.delta=vertex.data-self.target_variable
        for layer_idx in range(1,self.num_layers()-2):
            for vertex in self.get_layer(-layer_idx).get_vertexes():
                if layer_idx==1:
                    for in_vertex in vertex.get_pointed_from():
                        in_vertex.delta=np.dot(vertex.delta,in_vertex.weight_matrix.T)*\
                                    sigmoid_grad(in_vertex.data)
                        in_vertex.grad+=((1./self.batch_size)*np.dot(in_vertex.data.T,vertex.delta))
                else:
                    for in_vertex in vertex.get_pointed_from():
                        print vertex.delta.shape
                        print in_vertex.weight_matrix.T.shape
                        in_vertex.delta=np.dot(vertex.delta[:,1:],in_vertex.weight_matrix.T)*\
                                    sigmoid_grad(in_vertex.data)
                        in_vertex.grad+=((1./self.batch_size)*np.dot(in_vertex.data.T,vertex.delta[:,1:]))
                
        
#        for layer_idx in range(self.num_layers()-1):
#            if layer_idx<=self.num_layers()-1:
#                for vertex in self.get_layer(layer_idx).get_vertexes():
#                    print vertex.data.shape
#                    print vertex.delta[:,1:].shape
#                    vertex.grad+=((1./self.batch_size)*np.dot(vertex.data,vertex.delta[:,1:]))
#                    #vertex.grad[1:,:]+=(self.lambd/self.batch_size)
#            else:
#                for vertex in self.get_layer(layer_idx).get_vertexes():
#                    vertex.grad+=((1./self.batch_size)*np.dot(vertex.data,vertex.delta))
#                    #vertex.grad[1:,:]+=(self.lambd/self.batch_size)
    def grad_descent(self):
        for i in range(self.n_iter):
            print 'Iteration '+str(i)
            self.set_data()
            self.forward_prop()
            self.back_prop()
            for layer_idx in range(self.num_layers()-1):
                for vertex in self.get_layer(layer_idx).get_vertexes():
                    vertex.weight_matrix-=(self.alpha*vertex.grad)
            #self.compute_cost()
            #print self.cost
        
    
        
        

class Layer:
    def __init__(self):
        self.vertexes=[]
        
    def add_n_vertex(self,num_vertex):
        for i in range(num_vertex):        
            self.vertexes.append(Vertex())
    def get_vertexes(self):
        return self.vertexes
        
    def get_vertex(self,num):
        return self.vertexes[num]
        
    def __str__(self):
        print self.vertexes

        
        
class InputLayer(Layer):
    def __init__(self):
        self.vertexes=[]
        self.data=pd.DataFrame()
    
    def set_data(self,data1):
        self.data=data1
    
    
    
    def get_data(self):
        return self.data

        
class HiddenLayer(Layer):
    pass
    
class OutputLayer(InputLayer,Layer):
    pass
    
        
class ConvolutionalLayer(Layer):
    pass

def one_hot_encoding(data, cols):
    for col in cols:
        uniques=np.unique(data[col].values)
        num_columns=pd.get_dummies(data[col])
        num_columns.columns=col+'='+uniques
        data=data.join(num_columns)
        #to remove last column uncomment the line below
        #data=data.join(num_columns.drop(num_columns.columns.values[-1],1))
        data=data.drop(col,1)
    return data
    
        
    