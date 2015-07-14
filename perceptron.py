from numpy import *
import numpy

inpt=array([[0,0],[0,1],[1,0],[1,1]])
target=array([[0],[1],[1],[1]])
n=inpt.shape[0]

##ADD BIAS LAYER

columnOnes=numpy.ones(n)
inpt=numpy.column_stack([columnOnes,inpt])


weights=numpy.ones(shape=(3,1))
iterations=300


learningRate=0.25

for i in range(iterations): 


	activation=dot(inpt,weights)
	
	activation=where(activation>0.5,1,0)
	
	correction=target-activation
	
	weightDelta=dot(inpt.transpose(),correction)*learningRate
	
	
	weights=weights+weightDelta
	print weights
		
	print '____________________________'

activation=dot(inpt,weights)


activation=where(activation>0.5,1,0)
print activation

'''
input		weight		activation		target

data *m     m*n 		data*n 			data*n

4*3			3*1			4*1				4*1


'''
	