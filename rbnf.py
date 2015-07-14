### Neural Network with Radial Basis Funtion 
import numpy as np 
from numpy import *



'''
INTUITION :
	Certain Neurons respond to certain inputs.Receptive fields.Controlled by the gaussian function ..
	Entire space divided into regions of reception and goal is to find a triggered neuron .
	Middle layer does non linear classification ,output layer is just linear classifier .Perceptron algorithm .
	Weight is not multiplied with inputs ,here act as mean centers .Remember :) 
	Middle layer weight can be determined by either kmeans or randomly picked datapoints ,one for each neuron in hidden layer

DIMENSION ANALYSIS 


	Input 	Weighti 		RBFNeurons 	 	   Weight2  	 Output   

  	4*1 	4*1 			4*1                1*4           1*1


'''


class NN:
	def __init__(self,ni,nr,no):
		self.ni = ni
		self.nr = nr
		self.no = no

		self.ai = np.ones((self.ni,1))
		self.ar = np.ones((self.nr,1))
		self.ao = np.ones((self.no,1))


		self.wi = np.zeros((self.ni,1))
		self.wo = np.zeros((no,nr))




	def train(self,inputs,target):
		self.wi = self.ai 

		# calculate alpha,beta 

		# compute ah values 


		# determine weight2

		#TRAIN THE WEIGHTS  


	def predict(self,inputs):
		#PREDICT 


data = numpy.array([[0,0],[0,1],[1,0],[1,1]])
target = [[0],[1],[1],[0],[1],[0]]

nn = NN(4,4,1)
nn.train(data,target)

