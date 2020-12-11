#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANN_core.py

Purpose:
    define class object of NN 

Date:
    2020/11/01

@author: Simon Skorna
    
"""
###########################################################
### Imports
import numpy as np
from scipy.special import gamma


# =============================================================================
# define sigmoid function
# =============================================================================

def sigmoid(x):
    
    return 1/(1+ np.exp(-x))

# =============================================================================
# define sigmoid derivative function
# =============================================================================

def sigmoid_der(x):
    
#     return sigmoid(x) * (1 - sigmoid(x))
    return x * (1 - x)

# =============================================================================
# define log likelihood function
# =============================================================================

def lLik(obs_val, sigma_pred, dist):
    
    if dist == 'normal': 
        
        LLik = -(1/2) * np.log(2 * np.pi) - (1/2) * np.log(sigma_pred ** 2) - (1/2) \
            * ((obs_val** 2) / sigma_pred ** 2)
    
    if dist == 'studentst':
        
        print('TBD')
        
    return LLik

# =============================================================================
# define derivation of log likelihood
# =============================================================================

def lLik_der(obs_val, sigma_pred, dist):
    
    if dist == 'normal':
        
        LLik_der = - 1 / sigma_pred + (1 / sigma_pred ** 3) * (obs_val ** 2)
    
    if dist == 'studentst':
        
        print('TBD')
    return LLik_der

# =============================================================================
# define squared errors
# =============================================================================

def square_error(obs_val, obs_pred, *args):
    
    return (obs_val - obs_pred) ** 2

# =============================================================================
# define first derivative of squared errors
# =============================================================================
    
#in this case should be multiplied by -1 since with respect to obs_pred, but then 
#change in weights and biases is not += but -=
def square_error_der(obs_val, obs_pred, *args):
    
    return  (obs_val - obs_pred)

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

    def __init__(self, x, y, obj_fun = square_error, dist = None, node_fun = linear,
                 hidden_layer = 1, step_rate = 0.001, bias = False, num_nodes = [1,0],
                L2_reg = False, lambda_reg = 0):
        
        self.nodes1     = num_nodes[0]
        
        self.nodes2     = num_nodes[1]
        
        if obj_fun == square_error :
            
            self.obj_fun = square_error
            
            self.obj_fun_der = square_error_der
        
        elif obj_fun == lLik :
            
            self.obj_fun = lLik
            
            self.obj_fun_der = lLik_der
            
            self.dist = dist
            
        else :
            
            print('set objective function to either squared errors or log-likelihood')
        
        self.input      = x
        
        self.y          = y

        self.output     = np.ones(self.y.shape)
        
        self.step_rate = step_rate
        
        self.bias = bias * 1
        
        self.L2_reg = L2_reg * 1
        
        self.lambda_reg = lambda_reg
        
        if hidden_layer == 0 :
            
            self.weights_out = self.weights_hidden = np.ones([1,1])
            
            # random initialization based on tds article
            self.weights_input = np.random.randn(self.input.shape[1], 1) * \ 
                np.sqrt(2/self.input.shape[1])
            
            self.bias_out = self.bias_hidden = self.bias_input = np.zeros([1,1])
            
            self.node1_fun = linear
            
            self.node1_fun_der = linear_der
            
            self.node2_fun = linear
            
            self.node2_fun_der = linear_der
        
        elif hidden_layer == 1 :
            
            self.weights_out = np.ones([1,1])
            
            # random initialization based on tds article
            self.weights_hidden = np.random.randn(self.nodes1, 1) * \ 
                np.sqrt(2/self.input.shape[1])
            
            # random initialization based on tds article
            self.weights_input = np.random.randn(
                self.input.shape[1], self.nodes1
            ) * np.sqrt(2/self.input.shape[1])
            
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
            
            # random initialization based on tds article
            # in addition to ensure that initial output wil be positive square it
            self.weights_out = (np.random.randn(self.nodes2, 1) * \ 
                np.sqrt(2/self.input.shape[1])) ** 2
            
            # random initialization based on tds article
            self.weights_hidden = np.random.randn(self.nodes1, self.nodes2) * \ 
                np.sqrt(2/self.input.shape[1])
            
            # random initialization based on tds article
            self.weights_input = np.random.randn(self.input.shape[1], self.nodes1) * \ 
                np.sqrt(2/self.input.shape[1])
            
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

        self.layer1 = self.node1_fun(np.dot(self.input, self.weights_input) + \ 
            self.bias_input)
        
        self.layer2 = self.node2_fun(np.dot(self.layer1, self.weights_hidden) + \ 
            self.bias_hidden)
        
        self.output = np.dot(self.layer2, self.weights_out) + self.bias_out
        
    def backprop(self):
        
        #calculate errors and partial derivatives
        
        error_out = self.obj_fun_der(self.y, self.output, self.dist)
        
#         print(error_out.head())
        
        d_weights_out = np.dot(self.layer2.T, error_out) 
        
#         print(d_weights_out)
        
        error_hidden = np.dot(error_out, self.weights_out.T) * \ 
            self.node2_fun_der(self.layer2) 
        
#         print(error_hidden[1:5])
        
        d_weights_hidden = np.dot(self.layer1.T, error_hidden) 
        
#         print(d_weights_hidden)
        
        error_input = np.dot(error_hidden, self.weights_hidden.T) * \ 
            self.node1_fun_der(self.layer1)
        
#         print(error_input[1:5])
        
        d_weights_input = np.dot(self.input.T, error_input)
        
#         print(d_weights_input)
        
        d_bias_out = np.dot(np.ones(self.output.shape[0]), error_out) 
        
        d_bias_hidden = np.dot(np.ones(self.layer2.shape[0]), error_hidden)
        
        d_bias_input = np.dot(np.ones(self.input.shape[0]), error_input)
        
        #update weights and biases
        
        self.weights_out += d_weights_out * self.step_rate * \ 
            (self.nodes2 > 0) - self.lambda_reg * self.L2_reg * self.weights_out
        
        self.weights_hidden += d_weights_hidden * self.step_rate * \ 
            (self.nodes1 > 0) - self.lambda_reg * self.L2_reg * self.weights_hidden
        
        self.weights_input += d_weights_input * self.step_rate \
            - self.lambda_reg * self.L2_reg *self.weights_input
        
        self.bias_out += d_bias_out * self.step_rate * self.bias * \ 
            (self.nodes2 > 0) - self.lambda_reg * self.L2_reg * self.bias_out
        
        self.bias_hidden += d_bias_hidden * self.step_rate * self.bias * \
            (self.nodes1 > 0) - self.lambda_reg * self.L2_reg * self.bias_hidden
        
        self.bias_input += d_bias_input * self.step_rate * self.bias \
            - self.lambda_reg * self.L2_reg * self.bias_input
    
    def cost(self, new_input = None, new_y = None):
        # check if different input specified
        if new_input is None and new_y is None: 
            # if not - get the cost from training
            return np.sum(self.obj_fun(self.y, self.output, self.dist))
        else:
            # if yes - get new prediction 
            temp_layer1 = self.node1_fun(np.dot(new_input, self.weights_input) \ 
                + self.bias_input)
            temp_layer2 = self.node2_fun(np.dot(temp_layer1, self.weights_hidden) \ 
                + self.bias_hidden)
            temp_output = np.dot(temp_layer2, self.weights_out) + self.bias_out

            return np.sum(self.obj_fun(new_y, temp_output, self.dist))
    
    def predict(self, new_input = None):
        # check if different input specified
        if new_input is None: 
            # if not - get the output from training
            return self.output
        else:
            # if yes - get new prediction 
            temp_layer1 = self.node1_fun(np.dot(new_input, self.weights_input) + \
                self.bias_input)
            temp_layer2 = self.node2_fun(np.dot(temp_layer1, self.weights_hidden) + \ 
                self.bias_hidden)
            temp_output = np.dot(temp_layer2, self.weights_out) + self.bias_out

            return temp_output
        
    def add_data(self, new_x, new_y):
     
        self.input = pd.concat([self.input, new_x])
        
        self.y = pd.concat([self.y, new_y])       
    
    # this has been here for debugging
    def get_weights(self):
        
        return [self.weights_input, self.weights_hidden, self.weights_out]
