import random
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt 
import numpy as np


def create_dataset():
    # generating random data from given mean and covariance
    first_c =  np.random.multivariate_normal((0,0), [[1,0],[0,1]],50)
    second_c = np.random.multivariate_normal((4,5), [[1,0],[0,1]],50)
    train_dataset = np.concatenate((first_c, second_c), axis = 0)

    # creating y train class
    y_train_class = []
    for iter in range(0,100):
        if iter > 49:
            y_train_class.append(1)
        else:
            y_train_class.append(-1)
    y_train_class = np.array(y_train_class)
    y_train_class = y_train_class.reshape(100,1)
    print(y_train_class.shape)
    print(y_train_class[50])

    # creating y test class
    y_test_class = []
    for iterr in range(0,50):
        if iterr > 24:
            y_test_class.append(1)
        else:
            y_test_class.append(-1)
    y_test_class = np.array(y_test_class)
    
    first_c_test =  np.random.multivariate_normal((0,0), [[1,0],[0,1]],25)
    second_c_test = np.random.multivariate_normal((4,5), [[1,0],[0,1]],25)

    test_dataset = np.concatenate((first_c_test,second_c_test),axis = 0)
    return train_dataset, test_dataset, y_train_class, y_test_class



def req_cost_function(theta,f,p):
    # calculating the cost function
    return (np.sum((multiply_theta(f,theta)-p)**2))




def gradient_decent(theta,f,p,alpha,epochs):
    costs = []
    
    for j in range(epochs):
        z = multiply_theta(f,theta)
        cost = (1/len(f))*(((f.T @(z - (p)))))
        theta = theta - (alpha * cost)
        costs.append(req_cost_function(theta,f,p))
    
    return theta,costs


def multiply_theta(x,theta):
    return np.matmul(x,theta)


def predict_regression(alpha,epochs,features_train,train_y,theta):
    theta_vals, cost_value = gradient_decent(theta,features_train, train_y, alpha, epochs)
    j_value = req_cost_function(theta, features_train, train_y)
    return theta_vals, cost_value, j_value


def linear_regression(train_data, train_class):
    alpha_value = 0.01
    epochs_count = 100
    features_train = np.hstack((np.ones((train_data.shape[0],1)), train_data))
    # counting zeros using numpy array build in function
    theta_values = np.zeros((features_train.shape[1], 1))
    # predict regression
    theta_values, cost, j_value = predict_regression(alpha_value, epochs_count, features_train, train_class, theta_values)
    return theta_values, cost, j_value



def calculate_error(predicted_values, actual_values):
    # calculating total for every row in data
    error_sum_of_total_values = np.sum((predicted_values - actual_values)**2)
    print(error_sum_of_total_values/50)


def model_prediction_by_data(train_dataset,test_dataset,train_class,test_class):
    print(train_dataset)
    theta_vals, cost_val, j_value = linear_regression(train_dataset,train_class)
    print(theta_vals)
    class_predictions = []

    for i in range(0, 50):
        curr_value = theta_vals[0] + (theta_vals[1] * test_dataset[i][0]) + (theta_vals[2] * test_dataset[i][1])
        if curr_value >= 0:
            class_predictions.append(1)
        else:
            class_predictions.append(-1)
    
    x1 = train_dataset[:,0]
    x2 = -((theta_vals[0] + theta_vals[1] * x1) / theta_vals[2])
    
    x1_first_part = train_dataset[0:50,0]
    x2_first_part= train_dataset[0:50,1]
    plt.scatter(x1_first_part, x2_first_part, color = "b")
    
    x1_second_part = train_dataset[50:,0]
    x2_second_part = train_dataset[50:,1]
    plt.scatter(x1_second_part, x2_second_part, color = "r")
    plt.plot(x1, x2)

    class_predictions = np.array(class_predictions)
    calculate_error(class_predictions, test_class)
    plt.show()
    print(r2_score(class_predictions, test_class))


train_dataset, test_dataset, train_class, test_class = create_dataset()
model_prediction_by_data(train_dataset, test_dataset, train_class, test_class)
