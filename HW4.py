#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 19:38:51 2019

@author: salvadorcartagena
"""

import numpy as np
import matplotlib.pyplot as plt

"""
#Open the data file with 'r' = read, and 'rt' = read text
#Store the data files onto seperate data variables
#Each data file holds a different class with 4 feature input vectors
# features means 4 dimensions for each class
#First find out how the number of rows per class
#For this assignment we expect to have 50 input variables for x
#We also expect to have 4 features per class of data, so 4 columns per data set
"""

raw1 = open('./x1.dat','rt')
raw2 = open('./x2.dat','rt')
raw3 = open('./x1.dat','rt')
data1 = np.loadtxt(raw1,delimiter=" ")
data2 = np.loadtxt(raw2,delimiter=" ")
data3 = np.loadtxt(raw3,delimiter=" ")

m1 = data1.shape[0]
m2 = data2.shape[0]
m3 = data3.shape[0]

"""
#We will only use first half the data for trainining
#The second half of data will be used later to compare to the output of the predictive model
#The length of the data, rows, wil be saves as N1, N2, and N3
#It will be used to parse through the data and save the first half to seperate columns columns
#And the rows will be the seperate features
#In other words, x_vector

#First generate empty x1, x2, x3, matrices with
#NF row of features
#N1, N2, N3, column of data entries for x

#Starting with row 1 (feature 1), populate matrices x1, x2, x3 with
#jth data input from data1, data2, data3
#up to the 50th NF input
"""

N1 = m1//2
N2 = m2//2
N3 = m3//2
N = N1+N2+N3
NF=data1.shape[1]
N_Classes = 3
print ('N1,N2,N3,NF=',N1,N2,N3,NF)
print()

x1= np.zeros((NF,N1))
x2= np.zeros((NF,N2))
x3= np.zeros((NF,N3))
x1_test= np.zeros((NF,N1))
x2_test= np.zeros((NF,N2))
x3_test= np.zeros((NF,N3))
X = np.zeros((NF, N))
y = np.zeros((N_Classes, 1))
y1 = None
y2 = None
y3 = None

for j in range(0,N1,1):
  x1[0:NF,j] = data1[j,0:NF]
  x1_test[0:NF, j] = data1[j+25, 0:NF]
print("x1", x1)
print()

for j in range(0,N2,1):
  x2[0:NF,j] = data2[j, 0:NF]
  x2_test[0:NF, j] = data2[j+25, 0:NF]
print("x2", x2)
print()

for j in range(0,N3,1):
  x3[0:NF,j] = data3[j, 0:NF]
  x3_test[0:NF, j] = data3[j+25, 0:NF]
print("x3", x3)
print()

for j in range(0, N1, 1):
    X[0:NF, j] = data1[j, 0:NF]
    
for j in range(0, N1, 1):
    X[0:NF, j+25] = data2[j, 0:NF]
    
for j in range(0, N1, 1):
    X[0:NF, j+50] = data3[j, 0:NF]
#print("X", X[:,:25])

"""
#Construct a training vector for N data entries per NF features
#In this case the training matrix transposed is what will be saved
#The training matrix transposed has N number of columns and NF number of rows

#Set empty matrices for individual training vectors with N number of columns

#Setting the training matrices t1, t2, t3 for x1, x2, x3

#Setting one traning matrix in case we combine x1, x2, x3 in one X matrix
"""


T1_transpose = np.zeros((N_Classes, N1))
T2_transpose = np.zeros((N_Classes, N2))
T3_transpose = np.zeros((N_Classes, N3))
T_transpose = np.zeros((N_Classes, N))


t1 = np.array([[1],[0],[0]])
t2 = np.array([[0],[1],[0]])
t3 = np.array([[0],[0],[1]])

for i in range(0, N_Classes, 1):
    for j in range(0, N1, 1):
        T1_transpose[i, j] = t1[i, 0]
        T2_transpose[i, j] = t2[i, 0]
        T3_transpose[i, j] = t3[i, 0]
print("T1", T1_transpose)
print()
print("T2", T2_transpose)
print()
print("T3", T3_transpose)
print()


for i in range(0, N_Classes, 1):
    for j in range(0, N1, 1):
        T_transpose[i, j] = t1[i, 0]
        T_transpose[i, j+25] = t2[i, 0]
        T_transpose[i, j+50] = t3[i, 0]
print("T_transpose", T_transpose)

"""
#Solving for w1, w2, w3 via least square
#W was found by using T_Transpose and X
#w1, w2, w3 were found using x1, x2, x3 and T1, T2, T3

#Solving for combined W matrix

#Solving for individual W1, W2, W3 weight matrices

#After solving for weights W1, W2, W3 for class 1, 2, and 3
#We now must test our training model via decision matrix
"""

A = np.matmul(X, np.transpose(X))
B = np.matmul(X, np.transpose(T_transpose))
W = np.linalg.solve(A, B)
#print("W", W)
W_transpose = np.transpose(W)
print("W_transpose", W_transpose)

A1 = np.matmul(x1, np.transpose(x1))
B1 = np.matmul(x1, np.transpose(T1_transpose))
W1 = np.linalg.solve(A1, B1)
print("w1", W1)
print()
W1_transpose = np.transpose(W1)
#print("w1_transpose", W1_transpose)

A2 = np.matmul(x2, np.transpose(x2))
B2 = np.matmul(x2, np.transpose(T2_transpose))
W2 = np.linalg.solve(A2, B2)
print("w2", W2)
print()
W2_transpose = np.transpose(W2)
#print("w2_transpose", W2_transpose)

A3 = np.matmul(x3, np.transpose(x3))
B3 = np.matmul(x3, np.transpose(T3_transpose))
W3 = np.linalg.solve(A3, B3)
print("w3", W3)
print()
W3_transpose = np.transpose(W3)
#print("W3_transpose", W3_transpose)

"""
#In order to fill in the decision matrix, we must compare y's for class1, class2, and class3
#Whichever y is greatest, it is that class for which y belongs that the rows of predictions will iterate
#Meanwhile, while prediction rows are filled out, the actual column for the class that x came from is filled
#
"""

Confusion_Matrix = np.zeros((N_Classes, N_Classes))

y1 = np.dot(W1_transpose, x1_test)
#print("y1", y1)
#print("y1_transpose", np.transpose(y1))

y2 = np.dot(W2_transpose, x2_test)
#print("y2", y2)
#print("y2_transpose", np.transpose(y2))

y3 = np.dot(W3_transpose, x3_test)
#print("y3", y3)
#print("y3_transpose", np.transpose(y3))

for i in range(0, N1, 1):
    
    #actual class 1, guess class 1
    if (y1[0, i] >= y2[1, i]) and (y1[0, i] >= y3[2, i]):
        Confusion_Matrix[0, 0] = Confusion_Matrix[0, 0] + 1
    #actual class 1, guess class 3  
    elif (y3[2, i] >= y1[0, i]) and (y3[2, i] >= y2[1, i]):
        Confusion_Matrix[2, 0] = Confusion_Matrix[2, 0] + 1
    #actual class 1, guess class 2
    elif (y2[1, i] >= y1[0, i]) and (y2[1, i] >= y3[2, i]):
        Confusion_Matrix[1, 0] = Confusion_Matrix[1, 0] + 1
    
    #actual class 2, guess class 2
    if (y2[1, i] >= y1[0, i]) and (y2[1, i] >= y3[2, i]):
        Confusion_Matrix[1, 1] = Confusion_Matrix[1, 1] + 1
    #actual class 2, guess class 3
    elif (y3[2, i] >= y1[0, i]) and (y3[2, i] >= y2[1, i]):
        Confusion_Matrix[2, 1] = Confusion_Matrix[2, 1] + 1
    #actual class 2, guess class 1
    elif (y2[1, i] >= y1[0, i]) and (y2[1, i] >= y3[2, i]):
        Confusion_Matrix[0, 1] = Confusion_Matrix[0, 1] + 1
    
    #actual class 3, guess class 3
    if (y3[2, i] >= y2[1, i]) and (y3[2, i] >= y1[0, i]):
        Confusion_Matrix[2, 2] = Confusion_Matrix[2, 2] + 1
    #actual class 3, guess class 1
    elif (y1[0, i] >= y3[2, i]) and (y1[0, i] >= y2[1, i]):
        Confusion_Matrix[0, 2] = Confusion_Matrix[0, 2] + 1
    #actual class 3, guess class 2       
    else:#elif (y2[1, i] <= y3[2, i]) and (y2[1, i] >= y1[0, i]):
        Confusion_Matrix[1, 2] = Confusion_Matrix[1, 2] + 1

print()
print("Confusion Matrix")
print("Actual C1, C2, C3")
print("Guess")
print("C1")
print("C2")
print("C3")
print(Confusion_Matrix)
print()
"""
#using the Fisher Criterion with the LDA approach, we want to
#find the mean of each class' features
#mean1, mean2, and mean3 are matrices with 1 row and 4 columns
#Each column of mean1, mean2, mean3 holds means of the 25 data entries of the feature
#The MEAN_OF_TOTAL will be used later for the LDA approach, it will not appear in Least Square
"""
mean1 = np.mean(x1,axis=1)
mean2 = np.mean(x2,axis=1)
mean3 = np.mean(x3,axis=1)
MEAN_OF_TOTAL = (N1*mean1 + N2*mean2 + N3*mean3)/N
print('mean1',mean1)
print()
print('mean2',mean2)
print()
print('mean3',mean3)
print()

"""
#We now look for each class covariance Sk
#Each class covariance gets summated to the within class covariance matrix Sw
#Each class covariance is equal to the summation of x input minus class mean (m1, m2, or m3)
#times itself transposed
#When we find the within class covariance matrix, we can use it to solve for the weight

#First make 3 vaiables that will hold the summation of x - mean
#Populate the matrix first with NF 0_rows = 4
#and N1, N2, N3 0_columns = 25
#i will iterate through the feature rows
#an on each feature row
#j will iterate through each x input variable column
#For example
#x1_minus_mean1 will hold in each column space, a value for the feature's data input 
#minus the feature's mean
#x1[i,j] - mean1[i]
#and this will happen for each row which is holds a seperate feature

#We are not yet done with aquiring each class covariance
#Each class covariance Sk is obtained by the summation of xk_minus_meank times its transpose
#In this case, it is done by a matrix multiplication for example
#x1_minus_mean1 matrix times x1_minus_mean1_transpose
#These two operations (MxM^T & summation(M times M^T) are equivalent

#After solving for class variance S1, S2, S3 
#Solve for SB using MEAN_OF_TOTAL = (N1*mean1 + N2*mean2 + N3*mean3)/N
"""

x1_minus_mean1 = np.zeros((NF,N1));
x2_minus_mean2 = np.zeros((NF,N2));
x3_minus_mean3 = np.zeros((NF,N3));

for i in range(0,NF,1):
	for j in range (0,N1,1):
		x1_minus_mean1[i,j] = x1[i,j]-mean1[i]
	for j in range (0,N2,1):
		x2_minus_mean2[i,j] = x2[i,j]-mean2[i]
	for j in range (0,N3,1):
		x3_minus_mean3[i,j] = x3[i,j]-mean3[i]

S1 = np.matmul(x1_minus_mean1,np.transpose(x1_minus_mean1))
S2 = np.matmul(x2_minus_mean2,np.transpose(x2_minus_mean2))
S3 = np.matmul(x3_minus_mean3,np.transpose(x3_minus_mean3))

SW = S1+S2+S3
print("SW", SW)
print()
print ('S1')
for i in range(0,NF,1):
	print(i,S1[i,:])
print()
print ('S2')
for i in range(0,NF,1):
	print(i,S2[i,:])
print()
print('S3')
for i in range(0,NF,1):
	print(i,S3[i,:])
print()


SB= np.zeros( (NF,NF));

for i in range (0,NF,1):
	for j in range (0,NF,1):
		SB[i,j] = N1*(mean1[i]-MEAN_OF_TOTAL[i])*(mean1[j]-MEAN_OF_TOTAL[j]) \
		+ N2*(mean2[i]-MEAN_OF_TOTAL[i])*(mean2[j]-MEAN_OF_TOTAL[j]) \
		+ N3*(mean3[i]-MEAN_OF_TOTAL[i])*(mean3[j]-MEAN_OF_TOTAL[j])

print('SB')
for i in range(0,NF,1):
	print(i,SB[i,:])
print()

"""
#Once SB and SW are solved, we can solve for J

#By doing linalg.solve(SW, SB) we get (SW**-1)(SB)

# EIGENVALUE/VECTOR APPROACH
# [INVERSE(SW) *SB -J]W =0
#We want to use the biggest eigenvalue and its vector
"""

INV_SW_SB = np.linalg.solve(SW, SB);
value,vec = np.linalg.eig (INV_SW_SB);
print ()
print ('eigenvalues of INV_SW*SB:')
print (value)
print()
print ('eigenvectors of INV_SW*SB:')
print (vec)
print ()

J = np.max(value)
index = np.where(value==np.max(value))
q = index[0]
print ('maximum eigenvalue and index',J,q[0])

W = vec[:,q[0]];
# UNIT VECTOR SCALING
arg = W[0]*W[0] +W[1]*W[1]
arg = np.sqrt(arg)
W=W/arg
print ('J=',J,'W=',W)

plt.subplot(3,1,1)
plt.scatter(x1[0,:],x1[1,:],alpha=1.0,c='r');
plt.scatter(x2[0,:],x2[1,:],alpha=1.0,c='g');
plt.scatter(x3[0,:],x3[1,:],alpha=1.0,c='b');

#plot projection line collinear with W
qx=np.linspace(-12,12,20)
plt.plot(qx,W[1]/W[0]*qx,linestyle='-',c='r',label='vector w largest eigenvalue')
plt.plot(qx,W[1]/W[0]*qx,linestyle='-',c='r',label='vector w largest eigenvalue')

plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')


plt.subplot(3,1,2)

qq= (x1[0,:]*W[0]+x1[1,:]*W[1])
qy=np.zeros( (qq.shape))
plt.scatter(qq,qy,alpha=1.0,c='r');

qq= (x2[0,:]*W[0]+x2[1,:]*W[1])
plt.scatter(qq,qy,alpha=1.0,c='g');

#qq= (x3[0,:]*W[0]+x3[1,:]*W[1])
#plt.scatter(qq,qy,alpha=1.0,c='b');

#fo-dist
#variance

var1=0
var2=0
var3=0
m1= mean1[0]*W[0] +mean1[1]*W[1]
m2= mean2[0]*W[0] +mean2[1]*W[1]
m3= mean3[0]*W[0] +mean3[1]*W[1]


for i in range(0,NF,1):
	for j in range (0,NF,1):
		var1 = var1+W[i]*S1[i,j]*W[j]/N1
		var2 = var2+W[i]*S2[i,j]*W[j]/N2
		var3 = var3+W[i]*S3[i,j]*W[j]/N3

qxx=np.linspace(-20,20,200)
plt.plot(qxx,1/np.sqrt(2*var1*np.pi) *np.exp(-(qxx-m1)**2/(2*var1)),linewidth=2,c='r')
plt.plot(qxx,1/np.sqrt(2*var2*np.pi) *np.exp(-(qxx-m2)**2/(2*var2)),linewidth=2,c='g')
plt.plot(qxx,1/np.sqrt(2*var3*np.pi) *np.exp(-(qxx-m3)**2/(2*var3)),linewidth=2,c='b')
plt.show()

#projected mean var
print ('projected mean var')
print( m1,var1)
print( m2,var2)
print( m3,var3)
