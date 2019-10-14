# Three-CLASS LDA
# FISHER DISCRIMINANT
import numpy as np
import matplotlib.pyplot as plt

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
pi1= float(N1)/N;
pi2= float(N2)/N;
pi3= float(N3)/N;
print ('N1,N2,N3,NF=',N1,N2,N3,NF)
########################################
x1= np.zeros((NF,N1));
x2= np.zeros((NF,N2));
x3= np.zeros((NF,N3));

for j in range(0,N1,1):
  x1[0:NF,j]=data1[j,0:NF];

for j in range(0,N2,1):
  x2[0:NF,j]=data2[j,0:NF];

for j in range(0,N3,1):
  x3[0:NF,j]=data3[j,0:NF];
######################################
# find columnwise mean
mean1      = np.mean(x1,axis=1)
mean2      = np.mean(x2,axis=1)
mean3      = np.mean(x3,axis=1)
MEAN_OF_TOTAL = (N1*mean1 + N2*mean2 + N3*mean3)/N

print('mean1',mean1)
print('mean2',mean2)
print('mean3',mean3)
#exit()
#####################################
# SW class-covariance (not normalized)
# BISHOP 4.40
######################################
x1_minus_mean1= np.zeros( (NF,N1));
x2_minus_mean2= np.zeros( (NF,N2));
x3_minus_mean3= np.zeros( (NF,N3));

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

print('S1')
for i in range (0,NF,1):
	print (i,S1[i,:])
print('S2')
for i in range (0,NF,1):
	print (i,S2[i,:])
print('S3')
for i in range (0,NF,1):
	print (i,S3[i,:])
######################################
#############################
# SB:Between-class scatter matrix
# BISHOP 4.46 
#############################

SB= np.zeros( (NF,NF));
MEAN_TOTAL = (N1*mean1 + N2*mean2 + N3*mean3)/N

for i in range (0,NF,1):
	for j in range (0,NF,1):
		SB[i,j] = N1*(mean1[i]-MEAN_OF_TOTAL[i])*(mean1[j]-MEAN_OF_TOTAL[j]) \
		+ N2*(mean2[i]-MEAN_OF_TOTAL[i])*(mean2[j]-MEAN_OF_TOTAL[j]) \
		+ N3*(mean3[i]-MEAN_OF_TOTAL[i])*(mean3[j]-MEAN_OF_TOTAL[j])

print ('SB')
for i in range (0,NF,1):
	print (i,SB[i,:])
##################################
# EIGENVALUE/VECTOR APPROACH
# [INVERSE(SW) *SB -I*J ]W =0
##################################

INV_SW_SB = np.linalg.solve(SW,SB);

#find J
value,vec = np.linalg.eig (INV_SW_SB);
print('===============')
print('eigenvalues of INV_SW*SB:')
print(value)
print('eigenvectors of INV_SW*SB:')
print(vec)
print('===============')


#exit();
###########################

# USE BIGGEST EIGENVALUE and its vector
J=np.amax(value)
index = np.where(value==np.amax(value))
q = index[0]
print('maximum eigenvalue and index',J,q[0])

W=vec[:,q[0]];
# UNIT VECTOR SCALING
arg = np.dot(W,W);
arg = np.sqrt(arg)
W=W/arg
print('J=',J,'W=',W)

#exit()

# scatter plot highest cross corr  0,2
plt.subplot(3,1,1)
plt.scatter(x1[0,:],x1[2,:],alpha=1.0,c='r');
plt.scatter(x2[0,:],x2[2,:],alpha=1.0,c='g');
plt.scatter(x3[0,:],x3[2,:],alpha=1.0,c='b');

#plot projection line collinear with W
#qx=np.linspace(-12,12,20)
#plt.plot(qx,W[1]/W[0]*qx,linestyle='-',c='r',label='vector w largest eigenvalue')

#plt.legend()
plt.xlabel('x(0)')
plt.ylabel('x(2)')
#plt.show()
#exit()


plt.subplot(3,1,2)

qy1=np.zeros((N1))
qx1=np.zeros((N1))
for i in range (0,N1):
	qx1[i] =np.dot( x1[:,i],W[:])
plt.scatter(qx1,qy1,alpha=1.0,c='r');

qy2=np.zeros((N2))
qx2=np.zeros((N2))
for i in range (0,N2):
	qx2[i] =np.dot( x2[:,i],W[:])
plt.scatter(qx2,qy2,alpha=1.0,c='g');

qy3=np.zeros((N3))
qx3=np.zeros((N3))
for i in range (0,N3):
	qx3[i] =np.dot( x3[:,i],W[:])
plt.scatter(qx3,qy3,alpha=1.0,c='b');
########################################
#feux-dist
#variance

var1=0
var2=0
var3=0
m1= np.dot(mean1,W);
m2= np.dot(mean2,W);
m3= np.dot(mean3,W);


for i in range(0,NF,1):
	for j in range (0,NF,1):
		var1 = var1+W[i]*S1[i,j]*W[j]/N1
		var2 = var2+W[i]*S2[i,j]*W[j]/N2
		var3 = var3+W[i]*S3[i,j]*W[j]/N3

qxx=np.linspace(-2.5,2.5,200)
plt.plot(qxx,1/np.sqrt(2*var1*np.pi) *np.exp(-(qxx-m1)**2/(2*var1)),linewidth=2,c='r')
plt.plot(qxx,1/np.sqrt(2*var2*np.pi) *np.exp(-(qxx-m2)**2/(2*var2)),linewidth=2,c='g')
plt.plot(qxx,1/np.sqrt(2*var3*np.pi) *np.exp(-(qxx-m3)**2/(2*var3)),linewidth=2,c='b')
plt.show()

#projected mean var
print('projected mean var')
print('class1:mean,var', m1,var1)
print('class2:mean,var', m2,var2)
print('class3:mean,var', m3,var3)

#save W vector
#np.savetxt('./lda.out',W,delimiter=' ')
##################################
# MAXIMUM LIKELYHOOD DISCRIMINATOR
u       = np.zeros((3))
confused= np.zeros( (3,3))
for i in range (0,N1):
	u[0] =-0.5*np.log(2*np.pi*var1)- (qx1[i] - m1)**2/(2.0*var1)+np.log(float(N1)/N)
	u[1] =-0.5*np.log(2*np.pi*var2)- (qx1[i] - m2)**2/(2.0*var2)+np.log(float(N2)/N)
	u[2] =-0.5*np.log(2*np.pi*var3)- (qx1[i] - m3)**2/(2.0*var2)+np.log(float(N3)/N)
	index = np.where(u==np.amax(u))
	q = index[0]
	confused[0,q[0]]= confused[0,q[0]]+1
	print(q[0],u[0],u[1],u[2])

print('------------------')
for i in range (0,N2):
	u[0] =-0.5*np.log(2*np.pi*var1)- (qx2[i] - m1)**2/(2.0*var1)+np.log(float(N2)/N)
	u[1] =-0.5*np.log(2*np.pi*var2)- (qx2[i] - m2)**2/(2.0*var2)+np.log(float(N2)/N)
	u[2] =-0.5*np.log(2*np.pi*var3)- (qx2[i] - m3)**2/(2.0*var2)+np.log(float(N2)/N)

	index = np.where(u==np.amax(u))
	q = index[0]
	confused[1,q[0]]= confused[1,q[0]]+1
#	print q[0],u[0],u[1],u[2]
print('------------------')
for i in range (0,N3):
	u[0] =-0.5*np.log(2*np.pi*var1)- (qx3[i] - m1)**2/(2.0*var1)+np.log(float(N3)/N)
	u[1] =-0.5*np.log(2*np.pi*var2)- (qx3[i] - m2)**2/(2.0*var2)+np.log(float(N3)/N)
	u[2] =-0.5*np.log(2*np.pi*var3)- (qx3[i] - m3)**2/(2.0*var2)+np.log(float(N3)/N)

	index = np.where(u==np.amax(u))
	q = index[0]
	confused[2,q[0]]= confused[2,q[0]]+1
#	print q[0],u[0],u[1],u[2]


print('confusion matrix')
for i in range (0,3):
	print( confused[i,:])
print( 'pi1,pi2,pi3',pi1,pi2,pi3)
