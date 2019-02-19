#!/usr/env python3

## Lab 2
import math, random, scipy, matplotlib
import numpy as np 
from scipy.optimize import minimize
from matplotlib import pyplot as plt

N= 20
C = 100
def_kernel = 'radial'
def_factor = 2
classA = []
classB = []


def generate_data():
    global classA, classB
    classA = np.concatenate((np.random.randn(int(N/4),2)*0.2+[1.5,0.0],np.random.randn(int(N/4),2)*0.2+[-1.5,0.0]))
    classB = np.random.randn(int(N/2),2)*0.2+[0.0,-0.5]
    inputs = np.concatenate((classA,classB))
    targets= np.concatenate((np.ones(int(N/2)),-np.ones(int(N/2))))
    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute,:]
    targets= targets[permute]
    return inputs, targets

def kernel(x1, x2):
	if def_kernel == 'linear':
		return linear_kernel(x1,x2)
	elif def_kernel == 'poly':
		return polynomial_kernel(x1,x2,def_factor)
	elif def_kernel == 'radial':
		return radial_kernel(x1,x2,def_factor)

def linear_kernel(x1,x2):
	return np.dot(x1,x2)

def polynomial_kernel(x1,x2,factor):
	return (np.dot(x1,x2)+1)**factor

def radial_kernel(x1,x2, sigma):
	return math.exp(-(np.linalg.norm(x1-x2)**2)/(2*sigma**2))

def zerofun(alpha):
	return np.dot(alpha,targets)

def objective(alpha):
	net = 0
	for i in range(N):
		for j in range(N):
			net += (alpha[i]*alpha[j]*PCM[i][j])/2
	return net - sum(alpha)

def visualize_data():
    plt.plot([p[0] for p in classA],[p[1] for p in classA], 'b+')
    plt.plot([p[0] for p in classB],[p[1] for p in classB], 'r.')

def calculate_matrix(inputs, targets, N):
	M = []
	for i in range(N):
		A = []
		for j in range(N):
			similarity = kernel(inputs[i],inputs[j])
			A.append(targets[i]*targets[j]*similarity)
		M.append(A)
	return M


def find_non_zero(alpha):
    sv = []
    for i in range(N):
        if alpha[i] > math.pow(10,-5):
           sv.append((alpha[i],inputs[i],targets[i]))
    return sv

def find_b(alphas,sv):
    alpha, s, ts = sv[0]
    net =0
    for i in range(N):
        net += alphas[i]*targets[i]*kernel(s,inputs[i])
    return net - ts

def indicator(x,y,b,alphas):
    s = [x,y]
    net = 0
    for i in range(N):
        net += alphas[i]*targets[i]*kernel(s,inputs[i])
    return net - b

if __name__ == '__main__':
    global PCM, inputs, targets
    inputs, targets = generate_data()
    PCM = calculate_matrix(inputs,targets,N)
    bounds=[(0, C) for b in range(N)]
    constraint = {'type':'eq', 'fun':zerofun}
    ret = minimize(objective, np.zeros(N),bounds=bounds, constraints=constraint)
    alpha = ret['x']
    #print(alpha)
    sv = find_non_zero(alpha)
    b = find_b(alpha,sv)
    xgrid = np.linspace(-5,5)
    ygrid = np.linspace(-4,4)
    grid = np.array([[ indicator(x,y,b,alpha) for x in xgrid] for y in ygrid])
    visualize_data()
    plt.contour(xgrid,ygrid,grid, (-1.0, 0.0, 1.0),colors = ('red','black','blue'), linewidths=(1,3,1))
    plt.show() 
