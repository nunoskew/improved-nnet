# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 20:05:22 2014

@author: nunomarques
"""
"""
import nn
reload(nn)
import nn_test
reload(nn_test)
from nn_test import *
from nn import *
data = scipy.io.loadmat('ex4data1.mat')
X=data['X']
y=data['y']
data = scipy.io.loadmat('ex4weights.mat')
theta1=data['Theta1'].T
theta2=data['Theta2'].T
X=np.concatenate((X,y),axis=1)
nn=NNet_test()
nn.set_dataset(X)

nn.set_alpha(1)
nn.set_n_iter(500)
nn.set_lambd(0)
nn.set_batch_size(5)

nn.add_n_layers(3)

nn.get_input_layer().add_n_vertex(1)
nn.get_input_layer().get_vertex(0).set_num_nodes(400)

nn.get_layer(1).add_n_vertex(1)
nn.get_layer(1).get_vertex(0).set_num_nodes(25)
nn.get_layer(1).get_vertex(0).add_in_edge(nn.get_input_layer().get_vertex(0))

nn.get_output_layer().add_n_vertex(1)
nn.get_output_layer().get_vertex(0).set_num_nodes(10)
nn.get_output_layer().get_vertex(0).add_in_edge(nn.get_layer(1).get_vertex(0))

nn.initialize_data()
nn.initialize_weight_matrixes(0.12)
nn.initialize_deltas()

#nn.set_data()
#nn.get_layer(0).get_vertex(0).set_weight_matrix(theta1)
#nn.get_layer(1).get_vertex(0).set_weight_matrix(theta2)
#nn.forward_prop()
#nn.get_estimate()
#nn.compute_cost()
nn.set_data()
nn.forward_prop()
nn.back_prop()
nn.gradient_check(0.0001)
nn.grad_descent()
"""

from nn import *
import scipy.io
class NNet_test(NNet):
    def set_data(self):
        #m=112
        #idxs=random.sample(xrange(m),m-self.batch_size)
        idxs=np.random.randint(self.dataset.shape[0],size=self.batch_size)
        #dataset=pd.read_table("ok.txt",",",header=0,skiprows=idxs)
        #relation=pd.get_dummies(dataset['relation']).as_matrix()
        #person1=pd.get_dummies(dataset['person1']).as_matrix()
        #person2=pd.get_dummies(dataset['person2']).as_matrix()
        self.get_input_layer().get_vertex(0).set_data(self.dataset[idxs,0:400])
        self.set_target_variable(self.dataset[idxs,400:401],np.unique(self.dataset[:,400:401]))
        
    def set_dataset(self,d):
        #data=one_hot_encoding(d, ['relation','person1','person2'])
        #data=data.as_matrix()
        self.dataset=d
        
    def gradient_check(self,epsilon):
        self.initialize_gradient_checking()
        for layer_idx in range(self.num_layers()-1):
            for vertex in self.get_layer(layer_idx).get_vertexes():
                for i in range(vertex.weight_matrix.shape[0]):
                    for j in range(vertex.weight_matrix.shape[1]):
                        vertex.weight_matrix[i][j]+=epsilon
                        self.forward_prop()
                        self.compute_cost()
                        cost_plus=self.get_cost()
                        vertex.weight_matrix[i][j]-=(2*epsilon)
                        self.forward_prop()
                        self.compute_cost()
                        cost_minus=self.get_cost()
                        vertex.weight_matrix[i][j]+=epsilon
                        vertex.grad_check[i][j]=(cost_plus-cost_minus)/(2*epsilon)
                        
    def initialize_gradient_checking(self):
        for l_idx in range(self.num_layers()-1):
            for vertex in self.get_layer(l_idx).get_vertexes():
                vertex.grad_check=0*vertex.weight_matrix
    
    def set_target_variable(self,y,classes):
        self.target_variable=extend_target_variable(y,classes)
    
    