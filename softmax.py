from numpy import *
import numpy
import random

def sigmoid(x):
  print x
  totalsum=0
  for i in range(x.shape[0]):
    totalsum+=e**x[i]
  for i in range(x.shape[0]):
    x[i]=e**x[i]/totalsum
  return x



'''
DIMENSION ANALYSIS
    4*3         2*4
3*1       4*1         2*1

'''


class NN:
    def __init__(self,ni,nh,no):
      self.ni=ni+1
      self.nh=nh
      self.no=no

      self.ai=numpy.ones((self.ni,1))
      self.ah=numpy.ones((self.nh,1))
      self.ao=numpy.ones((self.no,1))

      self.wi=2*numpy.random.random((self.nh,self.ni))-1
      self.wo=2*numpy.random.random((self.no,self.nh))-1
      

    def predict(self,inputs):
      for i in range(len(inputs)):
        self.forwardProp(inputs[i])
        print self.ao

    def train(self,inputs,targets):
      for i in range(10):

        for i in range(len(inputs)):
          self.forwardProp(inputs[i])
          self.backProp(inputs[i],targets[i])

    def backProp(self,inpt,target):

      target=target.reshape(len(target),1)
      


      do=(target-self.ao)
      dh=self.ah*(1-self.ah)*dot(self.wo.transpose(),do.transpose())
      self.wi+=(0.75*dot(dh,self.ai.transpose()))
      self.wo+=0.75*dot(do,self.ah.transpose())
      

      '''

      MY OWN
      do=(target-self.ao)*(dot(self.ao.transpose(),(1-self.ao))) #2*1
      eo=dot(do,self.ah.transpose())  
                                    
      dummy=dot(self.wo.transpose(),do)
      di=dot(dot(self.ah,(1-self.ah.transpose())),dummy)
      ei=dot(di,self.ai.transpose())
      self.wo=self.wo+0.915*(eo)
      self.wi=self.wi+0.915*(ei)
      '''


      
        
    def forwardProp(self,inpt):
      self.ai=inpt
      self.ai=self.ai.reshape(len(self.ai),1) 
      self.ah=sigmoid(dot(self.wi,self.ai))
      self.ao=sigmoid(dot(self.wo,self.ah))
      #print self.ao
      


nn=NN(2,4,1)

X=numpy.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
Y=numpy.array([[1],[2],[2],[1]])
nn.train(X,Y)
nn.predict(X)

