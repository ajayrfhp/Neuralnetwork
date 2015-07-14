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


	Input 	WeightIn		RBFNeurons 	 	   WeightOut	 Output   

  	4*2 	4*2 			4*1                1*1           4*1


REFERENCE LINK

	https://chrisjmccormick.wordpress.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/

'''

def gaussianActivation(vector,beta,mu):
	
	vector = vector.reshape(vector.shape[0],1)
	mu = mu.reshape(mu.shape[0],1)
	diff = dot(transpose(vector - beta),(vector - beta))
	value = e**(-beta*diff)
	return value





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
		self.ai = inputs
		self.wi = self.ai 
		
		# Calculate Mu value
		
		average = np.zeros(self.wi[0].shape)
		for vector in self.wi:
			average += vector

		average /= self.wi.shape[0]



		# calculate sigma,beta value

		sigma = 0
		for vector in self.wi:
			sigma += sum(abs(self.wi - average))/self.wi.shape[0]

		beta = 1 / (2 * sigma * sigma)


		
		for i in range(self.ai.shape[0]):
			self.ar[i] = gaussianActivation(self.ai[i],beta,average)




		self.wo = dot(linalg.pinv(self.ar),target)	

		self.ao = dot(self.ar,self.wo)

		print self.wi
		print self.ar
		print self.wo
		print self.ao
			
		  

		# compute ah values 


		# determine weight2

		#TRAIN THE WEIGHTS  


	def predict(self,inputs):
		return None
		#PREDICT 


data = np.array([[0,0],[0,1],[1,0],[1,1]])
target = np.array([[0],[1],[1],[0]])

nn = NN(2,4,1)
nn.train(data,target)



