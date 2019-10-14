# THREE-CLASS 1-K LS 
# Least square
import numpy as np
import matplotlib.pyplot as plt

raw1= open('./x1.dat','rt')
raw2= open('./x2.dat','rt')
raw3= open('./x3.dat','rt')
data1=np.loadtxt(raw1,delimiter=" ")
data2=np.loadtxt(raw2,delimiter=" ")
data3=np.loadtxt(raw3,delimiter=" ")

print ('data array shape', data1.shape)
print ('data array shape', data2.shape)
print ('data array shape', data3.shape)
M1=data1.shape[0]
M2=data2.shape[0]
M3=data3.shape[0]
NF=data1.shape[1]

print('M1,M2,M3,NF=',M1,M2,M3,NF)

#extract training data 50%
N1=M1//2
N2=M2//2
N3=M3//2
N= N1+N2+N3
# classes
K=3
########################################
x1= np.zeros( (NF,N1));
x2= np.zeros( (NF,N2));
x3= np.zeros( (NF,N3));
for j in range(0,N1,1):
	x1[0:NF,j]=data1[j,0:NF];
for j in range(0,N2,1):
  	x2[0:NF,j]=data2[j,0:NF];
for j in range(0,N3,1):
  	x3[0:NF,j]=data3[j,0:NF];
######################################
# XTILDE vector
#####################################
NFP1 =NF+1
x1t= np.zeros( (NFP1,N1));
x2t= np.zeros( (NFP1,N2));
x3t= np.zeros( (NFP1,N3));

for j in range(0,N1,1):
	x1t[0,j]=1
	x1t[1:NFP1,j]=x1[:,j]
for j in range(0,N2,1):
	x2t[0,j]=1
	x2t[1:NFP1,j]=x2[:,j]
for j in range(0,N3,1):
	x3t[0,j]=1
	x3t[1:NFP1,j]=x3[:,j]
#######################################
# XTILDE TRANSPOSE
#######################################
XT= np.zeros( (NFP1,N1+N2+N3));
for i in range (0,NFP1,1):
	XT[i,0:N1]      =x1t[i,0:N1]
	XT[i,N1:N1+N2]  =x2t[i,0:N2]
	XT[i,N1+N2:N1+N2+N3]=x3t[i,0:N3]

#######################################
# T TRANSPOSE
#######################################
TT= np.zeros( (K ,N1+N2+N3));
TT[0,0:N1]      =1
TT[1,N1:N1+N2]  =1
TT[2,N1+N2:N1+N2+N3]=1
#######################################

A= np.matmul(XT,np.transpose(XT))
B= np.matmul(XT,np.transpose(TT))
W=np.linalg.solve(A,B)

print( 'W=================')
print(W)
print( 'W=================')

##################################
R=np.matmul(np.transpose(W),XT)

#################################
print('###################')
confused= np.zeros( (K ,K));

for j in range (0,N,1):
	#predicted
	arg= np.amax (R[:,j])
	tmp= np.where(R[:,j]==np.amax(R[:,j]))
	qr= tmp[0]
	#actual
	arg= np.amax (TT[:,j])
	tmp= np.where(TT[:,j]==np.amax(TT[:,j]))
	qt= tmp[0]

	confused[qr[0],qt[0]]= confused[qr[0],qt[0]]+1
	#print(qt[0],qr[0])

print('actual horizontal vs predicted vertical');
for i in range (0,K,1):
	print( confused[i,:])
#exit()






arg= np.amax(R[:,N1])
tmp= np.where(R[:,N1]==np.amax(R[:,N1]))
q= tmp[0]
print (arg,q[0])

arg= np.amax(TT[:,N1])
tmp= np.where(TT[:,N1]==np.amax(TT[:,N1]))
q= tmp[0]

print (arg,q[0])


