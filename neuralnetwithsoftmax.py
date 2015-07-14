from numpy import *
import numpy
import random

#######GLOBAL CONSTANTS

X=numpy.array([[1,0,0],[1,1.6,1.8],[1,1.5,1.5],[1,0.4,0.6],[1,1.75,1.3],[0,0.1,0.1]])
Y=[[0],[1],[1],[0],[1],[0]]
nFeatures=X.shape[0]

test=numpy.array([[1,0.25,0.25],[1,0.6,0.8],[1,2.5,1.5],[1,2.4,2.6],[1,0.75,0],[0,0.1,0.1]])


def sigmoid(x):
  return (1/(1+e**(-x)))

def softmax(x):
  totalSum=0
  for i in range(x.shape[0]):
    totalSum+=e**(x[i])
   
  for i in range(x.shape[0]):
    x[i]=(e**(x[i]))/totalSum
  

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
      self.wo=2*numpy.random.random((self.no*(nFeatures),self.nh))-1
      

    def predict(self,inputs):
      ans=[[]]
      for i in range(len(inputs)):
        self.forwardProp(inputs[i])
        subAns=[]
        for j in range(self.ao.shape[0]):
          subAns.append(self.ao[j])
        ans.append(subAns)
      ans.remove(ans[0])
     
      ans=numpy.array(ans)
      return ans      
        

    def train(self,inputs,targets):
      for k in range(10000):
        for i in range(len(inputs)):
          self.forwardProp(inputs[i])
          self.backProp(inputs[i],targets[i])

    def backProp(self,inpt,target):

      target=target.reshape(len(target),1)

      do=target-self.ao  
      

      dh=self.ah*(1-self.ah)*dot(self.wo.transpose(),do)
      self.wi+=0.75*dot(dh,self.ai.transpose())
      self.wo+=0.75*dot(do,self.ah.transpose())



      
        
    def forwardProp(self,inpt):
      self.ai=inpt
      self.ai=self.ai.reshape(len(self.ai),1) 
      self.ah=softmax(dot(self.wi,self.ai))
      self.ao=softmax(dot(self.wo,self.ah))
      
    def getClass(self,answer):
      newAnswer=[]      
      nRows=answer.shape[0]
      nCols=answer.shape[1]
      for i in range(nRows):
        thisMax=0
        thisClass=-1
        for j in range(nCols):
          if(answer[i][j]>thisMax):
            thisClass=j
            thisMax=answer[i][j]  
        newAnswer.append(thisClass)


      return newAnswer      

    def convertClass(self,output):
      newAnswer=[[]]
      for i in range(len(output)):
        subAnswer=[]
        for j in range(len(output)):
          subAnswer.append(0)
        subAnswer[output[i][0]]=1
        newAnswer.append(subAnswer)
      newAnswer.remove(newAnswer[0])
      return newAnswer    





nn=NN(2,4,1)


nFeatures=X.shape[0]
Y=numpy.array(nn.convertClass(Y))
nn.train(X,Y)
answer=nn.predict(test)
answer=nn.getClass(answer)
print answer