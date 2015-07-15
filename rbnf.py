### Neural Network with Radial Basis Funtion 
import numpy as np 
from numpy import *
import csv
import random



'''
INTUITION :
	Certain Neurons respond to certain inputs.Receptive fields.Controlled by the gaussian function ..
	Entire space divided into regions of reception and goal is to find a triggered neuron .
	Middle layer does non linear classification ,output layer is just linear classifier .Perceptron algorithm .
	Weight is not multiplied with inputs ,here act as mean centers .Remember :) 
	Middle layer weight can be determined by either kmeans or randomly picked datapoints ,one for each neuron in hidden layer




DIMENSION ANALYSIS 


	Input 	WeightIn		RBFNeurons 	 	   WeightOut	 Output   


  	150*4    10*4            150*10              10*3           150*3

  	4*2      4*2             4*4                 4*1            4*1


REFERENCE LINK

	https://chrisjmccormick.wordpress.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/

'''

def gaussianActivation(vector,beta,mu):
	
	vector = vector.reshape(vector.shape[0],1)
	mu = mu.reshape(mu.shape[0],1)
	diff = dot(transpose(vector - mu),(vector - mu))
	value = e**(-beta*diff)
	
	return value


def readFromCSV(filename):
	data = []
	with open(filename,"rb") as file:
		reader = csv.reader(file,delimiter=',')
		[data.append(row) for row in reader ]
		for i in range(len(data)):
			for j in range(len(data[i])):
				data[i][j] = float(data[i][j])
	return data


def intToVector(data):
	n = len(set(data))
	answer = []
	for i in range(len(data)):
		tempVector = [0]*n
		tempVector[int(data[i])] = 1
		answer.append(tempVector)
	return answer


class NN:
	def __init__(self,ni,nr,no):
		self.ni = ni
		self.nr = nr
		self.no = no

		self.sigma = 0

		self.beta = 0
		self.ai = None
		self.ar = np.zeros((self.ni,self.nr))
		self.ao = None



		self.wi = np.zeros((self.ni,1))
		self.wo = np.zeros((nr,no))





	def train(self,inputs,target):
		self.ai = inputs
		self.wi = np.array(random.sample(self.ai,self.nr))


		
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
		self.sigma = sigma
		self.beta = beta

		

		for i in range(self.ai.shape[0]):
			for j in range(self.wi.shape[0]):
				self.ar[i][j] = self.ar[i][j] + gaussianActivation(self.ai[i],beta,self.wi[j])
			


		self.wo = dot(linalg.pinv(self.ar),target)	

		self.ao = dot(self.ar,self.wo)

		

		#print self.wi
		#print self.ar
		#print self.wo
		#print self.ao
			
		  

		# compute ah values 


		# determine weight2

		#TRAIN THE WEIGHTS  


	def predict(self,inputs):
		self.ai = inputs
		self.ar = np.zeros((inputs.shape[0],self.nr))

		for i in range(self.ai.shape[0]):
			for j in range(self.wi.shape[0]):
				self.ar[i][j] = self.ar[i][j] + gaussianActivation(self.ai[i],self.beta,self.wi[j])	

		self.ao = dot(self.ar,self.wo)
		

		

		
		return self.ao
		#PREDICT 

	def validate(self,output,expected):
		cnt = 0
		i =0
		for row in self.ao:
			listRow = list(row)
			listExpected = list(expected[i])
			if(listRow.index(max(listRow)) != listExpected.index(max(listExpected))):
				cnt+=1



			i+=1
		print cnt		


		







data = readFromCSV("irisData.csv")
data = np.array(data)

np.random.shuffle(data)


target = data[:,-1]
target = np.array(intToVector(target))
data = np.delete(data,-1,1)


'''
data = np.array([[0,0],[0,1],[1,0],[1,1]])
target = np.array([[1],[2],[1],[0]])
'''
target = target.reshape(target.shape[0],target.shape[1])


nn = NN(data.shape[0],10,target.shape[1])
nn.train(data[0:100],target)

predictions = nn.predict(data[101:])

nn.validate(predictions,target[101:])


