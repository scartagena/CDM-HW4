# Three-CLASS LDA
# FISHER DISCRIMINANT
import numpy as np
import matplotlib.pyplot as plt
##########################################
# DISCRIMINATOR parameters
W= np.array([ 0.1366827,   0.40293171, -0.36139352, -0.82967379])
m1=-1.6555031819211683; var1= 0.0558231967342792;pi1 =1.0/3.0;
m2=-0.7331355978532905; var2= 0.04239533003886222;pi2=1.0/3.0;
m3= 1.356049342160755;  var3= 0.02991625225602781;pi3=1.0/3.0;

###########################################
raw1= open('./x1.dat','rt')
raw2= open('./x2.dat','rt')
raw3= open('./x3.dat','rt')
data1=np.loadtxt(raw1,delimiter=" ")
data2=np.loadtxt(raw2,delimiter=" ")
data3=np.loadtxt(raw3,delimiter=" ")

print('data array shape', data1.shape)
print('data array shape', data2.shape)
print('data array shape', data3.shape)
M1=data1.shape[0]
M2=data2.shape[0]
M3=data3.shape[0]

N1=M1//2
N2=M2//2
N3=M3//2

N = N1+N2+N3
NF=data1.shape[1]

print ('N1,N2,N3,NF=',N1,N2,N3,NF)
########################################
x1= np.zeros( (NF,N1));
x2= np.zeros( (NF,N2));
x3= np.zeros( (NF,N3));

for j in range(0,N1,1):
  x1[0:NF,j]=data1[j+N1,0:NF];

for j in range(0,N2,1):
  x2[0:NF,j]=data2[j+N2,0:NF];

for j in range(0,N3,1):
  x3[0:NF,j]=data3[j+N3,0:NF];
######################################

qx1=np.zeros((N1))
for i in range (0,N1):
        qx1[i] =np.dot( x1[:,i],W[:])

qx2=np.zeros((N2))
for i in range (0,N2):
        qx2[i] =np.dot( x2[:,i],W[:])

qx3=np.zeros((N3))
for i in range (0,N3):
        qx3[i] =np.dot( x3[:,i],W[:])
###########################################
# MAXIMUM LIKELYHOOD DISCRIMINATOR
u       = np.zeros((3))
confused= np.zeros( (3,3))
for i in range (0,N1):
        u[0] =-0.5*np.log(2*np.pi*var1)- (qx1[i] - m1)**2/(2.0*var1)+np.log(pi1)
        u[1] =-0.5*np.log(2*np.pi*var2)- (qx1[i] - m2)**2/(2.0*var2)+np.log(pi2)
        u[2] =-0.5*np.log(2*np.pi*var3)- (qx1[i] - m3)**2/(2.0*var2)+np.log(pi3)
        index = np.where(u==np.amax(u))
        q = index[0]
        confused[0,q[0]]= confused[0,q[0]]+1
#        print q[0],u[0],u[1],u[2]


print ('------------------')
for i in range (0,N2):
        u[0] =-0.5*np.log(2*np.pi*var1)- (qx2[i] - m1)**2/(2.0*var1)+np.log(pi1)
        u[1] =-0.5*np.log(2*np.pi*var2)- (qx2[i] - m2)**2/(2.0*var2)+np.log(pi2)
        u[2] =-0.5*np.log(2*np.pi*var3)- (qx2[i] - m3)**2/(2.0*var2)+np.log(pi3)
        index = np.where(u==np.amax(u))
        q = index[0]
        confused[1,q[0]]= confused[1,q[0]]+1
#       print q[0],u[0],u[1],u[2]
print ('------------------')
for i in range (0,N3):
        u[0] =-0.5*np.log(2*np.pi*var1)- (qx3[i] - m1)**2/(2.0*var1)+np.log(pi1)
        u[1] =-0.5*np.log(2*np.pi*var2)- (qx3[i] - m2)**2/(2.0*var2)+np.log(pi2)
        u[2] =-0.5*np.log(2*np.pi*var3)- (qx3[i] - m3)**2/(2.0*var2)+np.log(pi3)
        index = np.where(u==np.amax(u))
        q = index[0]
        confused[2,q[0]]= confused[2,q[0]]+1
#       print q[0],u[0],u[1],u[2]


print('confusion matrix')
for i in range (0,3):
        print ( confused[i,:])


