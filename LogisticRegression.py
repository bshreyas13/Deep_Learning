# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 13:35:44 2021

@author: shrey
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import math


def SplitData(X,y):
    # splitting X and y into training and testing sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1) 
    
    return X_train, X_test, y_train, y_test

def plot_generated(df: pd.DataFrame, x1: str, x2: str, y: str, title: str = '', save: bool = False, figname='figure.png'):
    plt.figure(figsize=(14, 7))
    plt.scatter(x=df[df[y] == 0][x1], y=df[df[y] == 0][x2], label='y = 0')
    plt.scatter(x=df[df[y] == 1][x1], y=df[df[y] == 1][x2], label='y = 1')
    plt.title(title, fontsize=20)
    plt.legend()
    if save:
        plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

## Implementation of Logistic regression with SGD ##


##The sigmoid function adjusts the cost function hypotheses to adjust the algorithm proportionally for worse estimations
def Sigmoid(z):
	G_of_Z = float(1.0 / float((1.0 + math.exp(-1.0*z))))
	return G_of_Z 

##The hypothesis is the linear combination of all the known factors x[i] and their current estimated coefficients theta[i] 
##This hypothesis will be used to calculate each instance of the Cost Function
def Hypothesis(theta, x):
	z = 0
	for i in range(len(theta)):
		z += x[i]*theta[i]
	return Sigmoid(z)

##For each member of the dataset, the result (Y) determines which variation of the cost function is used
##The Y = 0 cost function punishes high probability estimations, and the Y = 1 it punishes low scores
##The "punishment" makes the change in the gradient of ThetaCurrent - Average(CostFunction(Dataset)) greater
def Cost_Function(X,Y,theta,m): 
    sumOfErrors = 0
    sumlog =0 
    H=[]
    for i in range(m): 
        xi = X[i] 
        hi = Hypothesis(theta,xi) 
        H.append(math.log(hi))
        if Y[i] == 1: 
            error = Y[i] * math.log(hi) 
            sumlog += math.log(hi)
        elif Y[i] == 0: 
            error = (1-Y[i]) * math.log(1-hi)
            sumlog += math.log(hi)
            sumOfErrors += error 
    const = -1/m 
    J = const * sumOfErrors
    H = const * sumlog
# 	print ('cost is ', J ) 
    return H,J

##This function creates the gradient component for each Theta value 
##The gradient is the partial derivative by Theta of the current value of theta minus 
##a "learning speed factor aplha" times the average of all the cost functions for that theta
##For each Theta there is a cost function calculated for each member of the dataset
def Cost_Function_Derivative(X,Y,theta,j,m,alpha): 
    sumErrors = 0 
    for i in range(m): 
        xi = X[i] 
        xij = xi[j] 
        hi = Hypothesis(theta,X[i])     
        error = (hi - Y[i])*xij 
        sumErrors += error 
    m = len(Y) 
    constant = float(alpha)/float(m) 
    J = constant * sumErrors
    
    return J

##For each theta, the partial differential 
##The gradient, or vector from the current point in Theta-space (each theta value is its own dimension) to the more accurate point, 
##is the vector with each dimensional component being the partial differential for each theta value
def Gradient_Descent(X,Y,theta,m,alpha):
	new_theta = []
	for j in range(len(theta)):
		CFDerivative = Cost_Function_Derivative(X,Y,theta,j,m,alpha)
		new_theta_value = theta[j] - CFDerivative
		new_theta.append(new_theta_value)
	return new_theta

##The high level function for the LR algorithm which, for a number of steps (num_iters) finds gradients which take 
##the Theta values (coefficients of known factors) from an estimation closer (new_theta) to their "optimum estimation" which is the
##set of values best representing the system in a linear combination model
def Logistic_Regression(X,Y,alpha,theta,num_iters): 
    m = len(Y)
    H=[]
    J=[]
    for x in range(num_iters): 
        new_theta = Gradient_Descent(X,Y,theta,m,alpha) 
        theta = new_theta 
        H_,J_ = Cost_Function(X,Y,theta,m)
        H.append(H_)
        J.append(J_)

    plot_curve(H,J,num_iters)
    
    return theta
## This funtion i used to score the classification accuracy obtained with trained weights 
def score_accuracy(X_test,y_test,theta):
    score=0
    for i in range(len(X_test)):
        prediction = round(Hypothesis(X_test[i],theta))
        answer = y_test[i]
        if prediction == answer:
            score += 1
    #the same process is repeated for the implementation from this module and the scores compared to find the higher match-rate
    my_score = float(score) / float(len(X_test))
    return my_score

def plot_curve(H,J,num_iters):

    plt.plot(list(range(0, len(H))),H, '-b')  
    plt.xlabel('Number of iterations')
    plt.ylabel('log-likelihood')
    plt.show()

    plt.plot(list(range(0, len(J))),J, '-b')  
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.show()

## End of Logistic Regression implementation ##

if __name__ == '__main__':
    
## Genrate simulated 2 class dataset drawn from independent gaussian distributions N(1,0)
## using the make classification tool from sklearn
    X, y = make_classification(
        n_samples=1000, 
        n_features=2, 
        n_redundant=0,
        n_informative = 1,
        n_classes = 2,
        n_clusters_per_class=1,
        random_state=70
    )

    df = pd.concat([pd.DataFrame(X), pd.Series(y)], axis=1)
    df.columns = ['x1', 'x2', 'y']
    
    ## Plot Data
    plot_generated(df=df, x1='x1', x2='x2', y='y', title='Dataset with 2 classes')
    
    ## Logistic regression Sklearn
    clf = LogisticRegression(fit_intercept=True, C = 1e15, solver = 'liblinear')
    X_train, X_test, y_train, y_test = SplitData(X,y)
    clf.fit(X_train, y_train)
    print ("The intercept is: {}. The coefficients are:{}".format(clf.intercept_, clf.coef_))
    print("The classification accuracy on Sim data(in):",clf.score(X_test,y_test)*100)

### Training and testing the Implementation from scratch ##
    m = len(X_train)
    n = len(y_test)
    X_train = np.concatenate((np.ones((m, 1)), X_train), axis=1)
    y_train = y_train.reshape(-1,1)
    X_test = np.concatenate((np.ones((n, 1)), X_test), axis=1)
    y_test = y_test.reshape(-1,1)
    initial_theta = [0,0,0]
    alpha = 0.1
    iterations = 1000
    theta=Logistic_Regression(X_train,y_train,alpha,initial_theta,iterations)
    print("The intercept is: {}. The coefficients are:[{},{}]".format(theta[0],theta[1],theta[2]))
    print("Calssification accuracy of implementation from scratch(in %):",score_accuracy(X_test,y_test,theta)*100)
    
### Iris dataset
    
    iris=load_iris()
    
    ##Vizualization of Key features of dataset 
    ##The indices of the features that we are plotting
    x_index = 0
    y_index = 1

    # this formatter will label the colorbar with the correct target names
    formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

    plt.figure(figsize=(5, 4))
    plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
    plt.colorbar(ticks=[0, 1, 2], format=formatter)
    plt.xlabel(iris.feature_names[x_index])
    plt.ylabel(iris.feature_names[y_index])

    plt.tight_layout()
    plt.show()
    
    ## Prepare data into vector
    X_iris = iris.data 
    y_iris = iris.target.reshape(-1,1)  
    data=np.append(X_iris,y_iris,1)
    indicies = []
    for i in range(data.shape[0]):
        if data[i,4]==0 :
            indicies.append(i)
    indicies= np.asarray(indicies)
    # Remove all samples of 3rd class and Ensure label are 0 and 1       
    data = np.delete(data,indicies,0)
    data = np.where(data==2, 0, data )
    
    X_ir,y_ir = data[:,:data.shape[1]-1], data[:,data.shape[1]-1]
    # X_ir = X_ir/ np.linalg.norm(X_ir)
    X_train, X_test, y_train, y_test = SplitData(X_ir,y_ir)
  
    m=len(X_train)
    n=len(y_test)
    X_train = np.concatenate((np.ones((m, 1)), X_train), axis=1)
    y_train = y_train.reshape(-1,1)
    X_test = np.concatenate((np.ones((n, 1)), X_test), axis=1)
    y_test = y_test.reshape(-1,1)
    
    ## initialize 
    initial_theta = [0]* (X_iris.shape[1]+1)
    alpha = 0.1
    iterations = 1000
    theta=Logistic_Regression(X_train,y_train,alpha,initial_theta,iterations)
    print("The intercept is: {}. The coefficients are:[{},{},{},{}]".format(theta[0],theta[1],theta[2],theta[3],theta[4]))
    print("Calssification accuracy on Iris_dataset(in %):",score_accuracy(X_test,y_test,theta)*100)
    
   