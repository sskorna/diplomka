#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANN_core.py

Purpose:
    define class object of NN 

Version:
    1       no hidden layer 
    2       one hidden layer
    3       added biases 
    4       added argument for number of nodes in hidden layer
    5       added LLik functions as cost function
    6       added possibility of having 2nd hidden layer
    7       simplifying feedforward and backprop and restructuring the initialization of class
    8       solving problem with dimensions by ditching matrix multiplication of node_fun derivative

Date:
    2019/05/16

@author: ssa299

Notes:
    x: matrix[i,j] with i observation of j variables
    y: vector[j] of j true observation
    obj_fun: either lLik or squared_error, give as function name not as a string
    node_fun: either linear or sigmoid, give as function name not as a string
    hidden_layer: number of hidden layers,option of 1 or 2
    step_rate: rate multyplying average partial derivative
    bias: True or False, whether biases should be used
    num_nodes:  list of two integers, each specifying number of nodes for each layer 
                put zeros if want to skip one of the layers
    
"""
###########################################################
### Imports
import numpy as np
#import pandas as pd
# import matplotlib.pyplot as plt
# import scipy.optimize as opt
# import seaborn as sns
# import statsmodels.api as sm

# =============================================================================
# define sigmoid function
# =============================================================================

def sigmoid(x):
    
    return 1/(1+ np.exp(-x))

# =============================================================================
# define sigmoid derivative function
# =============================================================================

def sigmoid_der(x):
    
#    return sigmoid(x) * (1 - sigmoid(x))
    return x * (1 - x)

# =============================================================================
# define log likelihood function
# =============================================================================

def lLik(obs_val, sigma_pred):

    LLik = -(1/2) * np.log(2 * np.pi) - (1/2) * np.log(sigma_pred) - (1/2) * ((obs_val.T ** 2) / sigma_pred)

    return LLik	

# =============================================================================
# define derivation of log likelihood
# =============================================================================

def lLik_der(obs_val, sigma_pred):
    
    LLik_der = - 1 / sigma_pred ** (1/2) + (1 / sigma_pred ** (3/2)) * (obs_val)
    
    return LLik_der

# =============================================================================
# define squared errors
# =============================================================================

def square_error(obs_val, obs_pred):
    
    return (obs_val - obs_pred) ** 2

# =============================================================================
# define first derivative of squared errors
# =============================================================================
    
#in this case should be multiplied by -1 since with respect to obs_pred, but then 
#change in weights and biases is not += but -=
def square_error_der(obs_val, obs_pred):
    
    return 2 * (obs_val - obs_pred)

# =============================================================================
# define linear function for case when only one layer
# =============================================================================
def linear(x):
    
    return x

# =============================================================================
# define derivative of linear 
# =============================================================================
def linear_der(x):
        
    return np.ones(x.shape)
# =============================================================================
# class object NeuralNetwork
# =============================================================================
    
class NeuralNetwork:

    def __init__(self, x, y, obj_fun = square_error, node_fun = linear,
                 hidden_layer = 1, step_rate = 0.0001, bias = False, num_nodes = [1,0]):
        
        self.nodes1     = num_nodes[0]
        
        self.nodes2     = num_nodes[1]
        
        if obj_fun == square_error :
            
            self.obj_fun = square_error
            
            self.obj_fun_der = square_error_der
        
        elif obj_fun == lLik :
            
            self.obj_fun = lLik
            
            self.obj_fun_der = lLik_der
            
        else :
            
            print('set objective function to either squared errors or log-likelihood')
        
        self.input      = x
        
        self.y          = y

        self.output     = np.zeros(self.y.shape)
        
        self.step_rate = step_rate
        
        self.bias = bias * 1
        
        if hidden_layer == 0 :
            
            self.weights_out = self.weights_hidden = np.ones([1,1])
            
            self.weights_input = np.random.rand(self.input.shape[1], 1)
            
            self.bias_out = self.bias_hidden = self.bias_input = np.zeros([1,1])
            
            self.node1_fun = linear
            
            self.node1_fun_der = linear_der
            
            self.node2_fun = linear
            
            self.node2_fun_der = linear_der
        
        elif hidden_layer == 1 :
            
            self.weights_out = np.ones([1,1])
            
            self.weights_hidden = np.random.rand(self.nodes1, 1)
            
            self.weights_input = np.random.rand(self.input.shape[1], self.nodes1)
            
            self.bias_out = self.bias_hidden = np.zeros([1,1])
            
            self.bias_input = np.zeros([1, self.nodes1])
            
            self.node1_fun = node_fun
            
            if node_fun == linear :
                
                self.node1_fun_der = linear_der
                
            elif node_fun == sigmoid :
                
                self.node1_fun_der = sigmoid_der
            
            self.node2_fun = linear
            
            self.node2_fun_der = linear_der
            
        elif hidden_layer == 2 :
            
            self.weights_out = np.random.rand(self.nodes2, 1)
            
            self.weights_hidden = np.random.rand(self.nodes1, self.nodes2)
            
            self.weights_input = np.random.rand(self.input.shape[1], self.nodes1)
            
            self.bias_out = np.zeros([1,1])
            
            self.bias_hidden = np.zeros([1, self.nodes2])
            
            self.bias_input = np.zeros([1, self.nodes1])
            
            self.node1_fun = self.node2_fun = node_fun
            
            if node_fun == linear :
                
                self.node1_fun_der = self.node2_fun_der = linear_der
                
            elif node_fun == sigmoid :
                
                self.node1_fun_der = self.node2_fun_der = sigmoid_der
                
            else:
                
                print('set the node function to either sigmoid or linear')
                
        else:
            
            print('set number of hidden layers to 0, 1 or 2')
            
    def feedforward(self):
        
        
        self.layer1 = self.node1_fun(np.dot(self.input, self.weights_input) + self.bias_input)
        
        self.layer2 = self.node2_fun(np.dot(self.layer1, self.weights_hidden ) + self.bias_hidden)
        
        self.output = np.dot(self.layer2, self.weights_out) + self.bias_out
        
    def backprop(self):
        
        #calculate errors and partial derivatives
        
        error_out = self.obj_fun_der(self.y, self.output)
        
        d_weights_out = np.dot(self.layer2.T, error_out) / self.output.shape[0]
        
        error_hidden = np.dot(error_out, self.weights_out.T) *  self.node2_fun_der(self.layer2) #this is derivative in terms of f(x) maybe should change to derivative in terms of x, then it need to be layer1*weights_hidden
        
        d_weights_hidden = np.dot(self.layer1.T, error_hidden) / self.layer2.shape[0]
        
        error_input = np.dot(error_hidden, self.weights_hidden.T) * self.node1_fun_der(self.layer1)
        
        d_weights_input = np.dot(self.input.T, error_input) / self.input.shape[0]
        
        d_bias_out = np.dot(np.ones(self.output.shape[0]), error_out) / self.output.shape[0]
        
        d_bias_hidden = np.dot(np.ones(self.layer2.shape[0]), error_hidden) / self.layer2.shape[0]
        
        d_bias_input = np.dot(np.ones(self.input.shape[0]), error_input) / self.input.shape[0]
        
        #update weights and biases
        
        self.weights_out += d_weights_out * self.step_rate * (self.nodes2 > 0)
        
        self.weights_hidden += d_weights_hidden * self.step_rate * (self.nodes1 > 0) 
        
        self.weights_input += d_weights_input * self.step_rate
        
        self.bias_out += d_bias_out * self.step_rate * self.bias * (self.nodes2 > 0)
        
        self.bias_hidden += d_bias_hidden * self.step_rate * self.bias * (self.nodes1 > 0)
        
        self.bias_input += d_bias_input * self.step_rate * self.bias
    
    def cost(self):
        return np.sum(self.obj_fun(self.y, self.output))
    
#    def change_data(self, new_x, new_y):
#        self.x = new_x
#        self.y = new_y
