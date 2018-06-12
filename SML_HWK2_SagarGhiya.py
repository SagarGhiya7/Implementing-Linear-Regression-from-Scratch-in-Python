
# coding: utf-8

# In[1]:


# Loading necessary packages
import numpy as np
import pandas as pd
import scipy.io as scipy
import os
import matplotlib.pyplot as plt
import math
get_ipython().magic('matplotlib inline')


# In[2]:


os.chdir("C:/Users/Sagar Ghiya/Desktop")


# In[3]:


#Q1
# The dataset given for Q2 has non linear mapping. So using the same for Q1
data1 = scipy.loadmat('dataset1.mat')


# In[4]:


type(data1)


# In[5]:


data1.keys()


# In[6]:


# Storing data as train and test sets
X_train = np.matrix(data1['X_trn'])
Y_train = np.matrix(data1['Y_trn'])
X_test = np.matrix(data1['X_tst'])
Y_test = np.matrix(data1['Y_tst'])


# In[7]:


#Exploring train data
plt.plot(X_train,Y_train,'o')
plt.xlabel('X_Train')
plt.ylabel('Y_Train')
plt.title("Exploring train data")


# In[8]:


#Exploring test data
plt.plot(X_train,Y_train,'ro')
plt.xlabel('X_Test')
plt.ylabel('Y_Test')
plt.title("Exploring test data")


# In[9]:


#Adding intercept
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))


# In[10]:


# Function to calculate closed form theta for linear regression
def linearreg(X,Y):
    a1 = np.dot(X.T,X)
    a2 = np.dot(X.T,Y)
    theta = np.dot((np.linalg.inv(a1)),a2)
    return(theta)


# In[11]:


theta = linearreg(X_train,Y_train)
theta    


# In[12]:


# Cost function J
def cost_func(X, y, theta):
    err = np.power(((X * theta) - y), 2)
    return np.sum(err) / (2 * len(X))


# In[13]:


# Mini-Batch Stochastic Gradient descent
def sgd(X, y, theta, rho, minibatch, epsilon=0.0001, iters=1500):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = theta.ravel().shape[1]
    cost = [np.inf]
    
    #for i in range(iters):
    while True:
        # for batch in range(0, X.shape[0], minibatch_size):
        for b in range(math.ceil(len(X)/minibatch)):
            # generates random samples without replacement, all unique
            rand = np.random.choice(len(X), size=minibatch, replace = False)

           
            X_mini = X[rand]
            y_mini = y[rand]
            
            error = np.dot(X_mini,theta) - y_mini
            
            for j in range(parameters):
                t = np.multiply(error, X_mini[:,j])
                temp[j,0] = theta[j,0] - ((rho / len(X_mini)) * np.sum(t))

            theta = temp
            
        cost.append(cost_func(X, y, theta))
        
        if (cost[-2]-cost[-1]) > 0 and (cost[-2]-cost[-1]) < epsilon:
            break
            
        
        
    return theta,cost
      
    
    


# In[14]:


theta_sgd = np.matrix(np.zeros((X_train.shape[1],1)))
sgd(X_train,Y_train,theta_sgd, 0.0001,10)


# In[15]:


#Q2
#Function to compute mean squared error.
def MSE(y, predict_y):
    return np.mean(np.power((y - predict_y), 2))


# In[16]:


#n =2
#Adding x^2 term
X_train_2n = np.hstack((X_train, np.power(X_train[:, 1:],2)))
X_test_2n  = np.hstack((X_test, np.power(X_test[:, 1:], 2)))


# In[17]:


# Closed form
theta_2n_closed = linearreg(X_train_2n,Y_train)
y_pred_train_cl_2n = np.dot(X_train_2n,theta_2n_closed)
Train_MSE_closed_2n = MSE(Y_train,y_pred_train_cl_2n)
print("MSE for train data is", Train_MSE_closed_2n)


# In[18]:


y_pred_test_cl_2n = np.dot(X_test_2n,theta_2n_closed)
Test_MSE_closed_2n = MSE(Y_test,y_pred_test_cl_2n)
print("MSE for test data is", Test_MSE_closed_2n)


# In[19]:


print("Closed form Theta for 2 degree polynomial:\n ", theta_2n_closed)


# In[20]:


#n=2 Stochastic Gradient Descent
theta_sgd_2n =  np.matrix(np.zeros((X_train_2n.shape[1],1)))
theta_2d, cost_2d = sgd(X_train_2n,Y_train,theta_sgd_2n,0.0001,10)
cost_2d


# In[21]:


print("theta for 2 degree polynomial sgd\n", theta_2d)


# In[22]:


y_pred_train_2n_sgd = np.dot(X_train_2n, theta_2d)
Train_MSE_sgd_2n = MSE(Y_train,y_pred_train_2n_sgd)
print("MSE for train data is", Train_MSE_sgd_2n)


# In[23]:


y_pred_test_2n_sgd = np.dot(X_test_2n, theta_2d)
Test_MSE_sgd_2n = MSE(Y_test,y_pred_test_2n_sgd)
print("MSE for test data is", Test_MSE_sgd_2n)


# In[24]:


#Closed n=3
#Adding x^3 term
X_train_3n = np.hstack((X_train, np.power(X_train[:, 1:], 2), np.power(X_train[:, 1:], 3)))
X_test_3n = np.hstack((X_test, np.power(X_test[:, 1:], 2), np.power(X_test[:, 1:], 3)))


# In[25]:


theta_3n_closed = linearreg(X_train_3n,Y_train)
y_pred_train_cl_3n = np.dot(X_train_3n,theta_3n_closed)
Train_MSE_closed_3n = MSE(Y_train,y_pred_train_cl_3n)
print("MSE for train data is", Train_MSE_closed_3n)


# In[26]:


y_pred_test_cl_3n = np.dot(X_test_3n,theta_3n_closed)
Test_MSE_closed_3n = MSE(Y_test,y_pred_test_cl_3n)
print("MSE for test data is", Test_MSE_closed_3n)


# In[27]:


print("Closed form Theta for 3 degree polynomial:\n ", theta_3n_closed)


# In[28]:


#Mini batch n=3
theta_sgd_3n =  np.matrix(np.zeros((X_train_3n.shape[1],1)))
theta_3d, cost_3d = sgd(X_train_3n,Y_train,theta_sgd_3n,0.00003,10)
cost_3d


# In[29]:


print("theta for 3 degree polynomial sgd\n", theta_3d)


# In[30]:


y_pred_train_3n_sgd = np.dot(X_train_3n, theta_3d)
Train_MSE_sgd_3n = MSE(Y_train,y_pred_train_3n_sgd)
print("MSE for train data is", Train_MSE_sgd_3n)


# In[31]:


y_pred_test_3n_sgd = np.dot(X_test_3n, theta_3d)
Test_MSE_sgd_3n = MSE(Y_test,y_pred_test_3n_sgd)
print("MSE for test data is", Test_MSE_sgd_3n)


# In[32]:


#Closed n=5
X_train_5n = np.hstack((X_train, np.power(X_train[:, 1:], 2), np.power(X_train[:, 1:], 3), np.power(X_train[:, 1:], 4), np.power(X_train[:, 1:], 5)))
X_test_5n = np.hstack((X_test, np.power(X_test[:, 1:], 2), np.power(X_test[:, 1:], 3), np.power(X_test[:, 1:], 4), np.power(X_test[:, 1:], 5)))


# In[33]:


theta_5n_closed = linearreg(X_train_5n,Y_train)
y_pred_train_cl_5n = np.dot(X_train_5n,theta_5n_closed)
Train_MSE_closed_5n = MSE(Y_train,y_pred_train_cl_5n)
print("MSE for train data is", Train_MSE_closed_5n)


# In[34]:


y_pred_test_cl_5n = np.dot(X_test_5n,theta_5n_closed)
Test_MSE_closed_5n = MSE(Y_test,y_pred_test_cl_5n)
print("MSE for test data is", Test_MSE_closed_5n)


# In[35]:


print("Closed form Theta for 5 degree polynomial:\n ", theta_5n_closed)


# In[36]:


#Mini batch n=5
theta_sgd_5n =  np.matrix(np.zeros((X_train_5n.shape[1],1)))
theta_5d, cost_5d = sgd(X_train_5n,Y_train,theta_sgd_5n,0.00000001,5)
cost_5d


# In[37]:


print("theta for 5 degree polynomial sgd\n", theta_5d)


# In[38]:


y_pred_train_5n_sgd = np.dot(X_train_5n, theta_5d)
Train_MSE_sgd_5n = MSE(Y_train,y_pred_train_5n_sgd)
print("MSE for train data is", Train_MSE_sgd_5n)


# In[39]:


y_pred_test_5n_sgd = np.dot(X_test_5n, theta_5d)
Test_MSE_sgd_5n = MSE(Y_test,y_pred_test_5n_sgd)
print("MSE for test data is", Test_MSE_sgd_5n)


# In[40]:


# Effect of mini batch size on speed
# Ans:-
# As the mini batch size increases the speed of computation increases. Due to increase in step size, the descent is faster and hence the computation is faster.

# Effect of mini-batch size on testing error of solution
# Ans:

# Testing error comes out to be minimum for a particular mini batch size. Values higher and lower than that mini batch size may give testing error more or less than optimal.


# In[41]:


#Q3

data2 = scipy.loadmat("dataset2.mat")


# In[42]:


data2.keys()


# In[43]:


X_R_train = np.matrix(data2['X_trn'])
X_R_test = np.matrix(data2['X_tst'])
Y_R_train = np.matrix(data2['Y_trn'])
                   
Y_R_test = np.matrix(data2['Y_tst'])


# In[44]:


#Exploring train data
plt.plot(X_R_train,Y_R_train,'o')
plt.xlabel('X_Train')
plt.ylabel('Y_Train')
plt.title("Exploring train data")


# In[45]:


#Exploring test data
plt.plot(X_R_test,Y_R_test,'ro')
plt.xlabel('X_Test')
plt.ylabel('Y_Test')
plt.title("Exploring test data")


# In[46]:


#Adding Intercept
X_R_train = np.hstack((np.ones((X_R_train.shape[0], 1)), X_R_train))
X_R_test = np.hstack((np.ones((X_R_test.shape[0], 1)), X_R_test))


# In[47]:


#Function for Ridge regression
def ridgeReg(X, y, lambda1 = 1.0):
    b1 = np.dot(X.T, X) + np.eye(X.shape[1]) * lambda1
    b2 = np.dot(X.T, y)
    theta1 = np.dot(np.linalg.inv(b1), b2)
    return theta1


# In[48]:


# Function for ridge stochastic gradient descent
def sgd_Ridge(X, y, theta, rho, minibatch_size, lambda1=0, threshold=0.0001, iters=1000):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = theta.ravel().shape[1]
    cost = [np.inf]
    
    while True:
        for b in range(math.ceil(len(X)/minibatch_size)):
            # generates random samples without replacement, all unique
            rand = np.random.choice(len(X), size=min(len(X), minibatch_size), replace = False)

            # Get pair of (X, y) of the current minibatch/chunk
            X_mini = X[rand]
            y_mini = y[rand]
            
            error = (X_mini * theta) - y_mini
            
            for j in range(parameters):
                term = np.multiply(error, X_mini[:,j])
                temp[j,0] = theta[j,0] - ((alpha / len(X_mini)) * (np.sum(term) + lambda1 * theta[j,0]))

            theta = temp
            
        cost.append(cost_func(X, y, theta))
        
        
        
        if (cost[-2]-cost[-1]) > 0 and (cost[-2]-cost[-1]) < threshold:
            break
        
    return theta, cost


# In[49]:


# First doing all n for closed
# n = 2
X_R_train_2d = np.hstack((X_R_train, np.power(X_R_train[:, 1:],2))) 
X_R_test_2d = np.hstack((X_R_test, np.power(X_R_test[:, 1:], 2)))


# In[50]:


theta_R_2d_closed = ridgeReg(X_R_train_2d, Y_R_train)

y2_train_predict_2d_closed = np.dot(X_R_train_2d, theta_R_2d_closed)
Train_MSE_closed_2d = MSE(Y_R_train,y2_train_predict_2d_closed)
print("MSE for Training set: ", Train_MSE_closed_2d)

y2_test_predict_2d_closed = np.dot(X_R_test_2d, theta_R_2d_closed)
Test_MSE_closed_2d = MSE(Y_R_test,y2_test_predict_2d_closed)
print("MSE for Test set: ",Test_MSE_closed_2d)


# In[51]:


print("θridge for 2-degree polynomial (Closed-form)")
theta_R_2d_closed


# In[52]:


# n=3(Closed)
X_R_train_3d = np.hstack((X_R_train, np.power(X_R_train[:, 1:], 2), np.power(X_R_train[:, 1:], 3)))
X_R_test_3d = np.hstack((X_R_test, np.power(X_R_test[:, 1:], 2), np.power(X_R_test[:, 1:], 3)))


# In[53]:


theta_R_3d_closed = ridgeReg(X_R_train_3d, Y_R_train)

y3_train_predict_3d_closed = np.dot(X_R_train_3d, theta_R_3d_closed)
Train_MSE_closed_3d = MSE(Y_R_train,y3_train_predict_3d_closed)
print("MSE for Training set: ", Train_MSE_closed_3d)

y3_test_predict_3d_closed = np.dot(X_R_test_3d, theta_R_3d_closed)
Test_MSE_closed_3d = MSE(Y_R_test,y3_test_predict_3d_closed)
print("MSE for Test set: ",Test_MSE_closed_3d)


# In[54]:


print("θridge for 3-degree polynomial (Closed-form)")
theta_R_3d_closed


# In[55]:


#n=5(closed)
X_R_train_5d = np.hstack((X_R_train, np.power(X_R_train[:, 1:], 2), np.power(X_R_train[:, 1:], 3), np.power(X_R_train[:, 1:], 4), np.power(X_R_train[:, 1:], 5)))
X_R_test_5d = np.hstack((X_R_test, np.power(X_R_test[:, 1:], 2), np.power(X_R_test[:, 1:], 3), np.power(X_R_test[:, 1:], 4), np.power(X_R_test[:, 1:], 5)))


# In[56]:


theta_R_5d_closed = ridgeReg(X_R_train_5d, Y_R_train)

y5_train_predict_5d_closed = np.dot(X_R_train_5d, theta_R_5d_closed)
Train_MSE_closed_5d = MSE(Y_R_train,y5_train_predict_5d_closed)
print("MSE for Training set: ", Train_MSE_closed_5d)

y5_test_predict_5d_closed = np.dot(X_R_test_5d, theta_R_5d_closed)
Test_MSE_closed_5d = MSE(Y_R_test,y5_test_predict_5d_closed)
print("MSE for Test set: ",Test_MSE_closed_5d)


# In[57]:


print("θridge for 5-degree polynomial (Closed-form)")
theta_R_5d_closed

