import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = np.linspace(60, 300)
X = np.deg2rad(X)
K = np.random.normal(0,0.15)
Y = np.sin(X)+K

plt.scatter(X, Y)
plt.title("Generated Data")
plt.show()

generated_data = np.vstack((X, Y)).T


def prediction_of_linear_regression(x_values, y_values, l_r, epochs):
    m_values = np.shape(x_values)[0]
    n_values = np.shape(x_values)[1]
    
    x_values = np.concatenate((np.ones((m_values,1)),x_values), axis=1)
    w_value = 2*np.random.rand(n_values+1,)-1

    loss_history = []
    
    for iterator in range(epochs):
        y_estimated = x_values.dot(w_value)
        error = y_estimated -y_values 
        cost = np.sum(error ** 2)/len(error)
        gradient = (1 / m_values) * x_values.T.dot(error)
        w_value = w_value - l_r * gradient
        loss_history.append(cost)
    return y_estimated, loss_history, w_value 



y_hat_value, sse_value, w_value=prediction_of_linear_regression(generated_data[:, :-1], generated_data[:, -1], 0.00001, 100)


plt.scatter(X, Y)
plt.plot(y_hat_value)
plt.xlim(np.min(X), np.max(X))
plt.ylim(np.min(Y), np.max(Y))
plt.show()
print("SSE for Linear: {}".format(sse_value[-1]))
print("Coefficients for Linear: {}".format(w_value))


def generate_feat_for_polygon(X, deg):
  x_polynomial = X
  for i in range(2,deg+1):
    x_polynomial=np.vstack((x_polynomial, np.power(X, i)))
  data_of_polynomial = x_polynomial.T
  data_of_polynomial = pd.DataFrame(data_of_polynomial)

  data_of_polynomial = (data_of_polynomial-data_of_polynomial.mean())/data_of_polynomial.std()
  data_of_polynomial = data_of_polynomial.to_numpy()
  return data_of_polynomial 

feat_three = generate_feat_for_polygon(X, 3)
feat_six = generate_feat_for_polygon(X, 6)
feat_nine = generate_feat_for_polygon(X, 9)
feat_twelve = generate_feat_for_polygon(X, 12)
feat_fifteen = generate_feat_for_polygon(X, 15)

y_hat_three, sse_three, w_three = prediction_of_linear_regression(feat_three, generated_data[:, -1], 0.00001, 100)
y_hat_six, sse_six, w_six = prediction_of_linear_regression(feat_six, generated_data[:, -1], 0.00001, 100)
y_hat_nine, sse_nine, w_nine = prediction_of_linear_regression(feat_nine, generated_data[:, -1], 0.00001, 100)
y_hat_twelve, sse_twelve, w_twelve = prediction_of_linear_regression(feat_twelve, generated_data[:, -1], 0.00001, 100)
y_hat_fifteen, sse_fifteen, w_fifteen = prediction_of_linear_regression(feat_fifteen, generated_data[:, -1], 0.00001, 100)


plt.scatter(X, Y)
plt.plot(y_hat_three)
plt.xlim(np.min(X), np.max(X))
plt.ylim(-15, 15)
plt.title("Polynomial three")
plt.show()
print("SSE for Polynomial three: {}".format(sse_three[-1]))
print("Coefficients for Polynomial three: {}".format(w_three))


plt.scatter(X, Y)
plt.xlim(np.min(X), np.max(X))
plt.plot(y_hat_six)
plt.title("Polynomial six")
plt.show()
print("SSE for Polynomial six: {}".format(sse_six[-1]))
print("Coefficients for Polynomial six: {}".format(w_six))


plt.scatter(X, Y)
plt.xlim(np.min(X), np.max(X))
plt.plot(y_hat_nine)
plt.title("Polynomial nine")
plt.show()
print("SSE for Polynomial nine: {}".format(sse_nine[-1]))
print("Coefficients for Polynomial nine: {}".format(w_nine))


plt.scatter(X, Y)
plt.plot(y_hat_twelve)
plt.xlim(np.min(X), np.max(X))
plt.ylim(-15, 15)
plt.title("Polynomial twelve")
plt.show()
print("SSE for Polynomial twelve: {}".format(sse_twelve[-1]))
print("Coefficients for Polynomial twelve: {}".format(w_twelve))


plt.scatter(X, Y)
plt.plot(y_hat_fifteen)
plt.title("Polynomial fifteen")
plt.show()
print("SSE for Polynomial fifteen: {}".format(sse_fifteen[-1]))
print("Coefficients for Polynomial fifteen: {}".format(w_fifteen))


def using_regression_for_linear_prediction(x_values, y_values, reg, lam, l_r, epochs):
    m_values = np.shape(x_values)[0]
    n_values = np.shape(x_values)[1]
    x_values = np.concatenate((np.ones((m_values,1)),x_values), axis=1)
    w_value = 2*np.random.rand(n_values+1,)-1
    loss_history = []
    for iteration in range(epochs):
        y_estimated = x_values.dot(w_value)
        error = y_estimated -y_values 
        if(reg == "first"):
            sigw = np.sum(np.abs(w_value))
            cost = np.sum(error ** 2)/len(error)+(lam/2)*sigw
        else:
            sigw=np.sum(np.square(w_value))
            cost = np.sum(error ** 2)/len(error)+(lam/2)*sigw
        gradient = (1 / m_values) * x_values.T.dot(error)
        w_value = w_value - l_r * gradient
        loss_history.append(cost)
    return y_estimated, loss_history,w_value 


# taking custom lambda values
lambda_values = [1e-10,1e-8,1e-4,1e-2,1,10,20]
# models
linear_regularized_model_one = []
linear_regularized_model_two = []

# filling the model's array with values
for i in lambda_values:
  linear_regularized_model_one.append(using_regression_for_linear_prediction(feat_fifteen, generated_data[:, -1], "first", i, 0.001, 100))
  linear_regularized_model_two.append(using_regression_for_linear_prediction(feat_fifteen, generated_data[:, -1], "second", i, 0.001, 100))

# sse, coefficients for model one
for i in range(0, len(linear_regularized_model_one)):
  print("SSE for L1 regularized model with lambda {}: {}".format(lambda_values[i], linear_regularized_model_one[i][1][-1]))
  print("Coefficients L1 for regularized model with lambda {}: {}".format(lambda_values[i], linear_regularized_model_one[i][2]))
  print("\n\n")

# sse, coefficients for model two
for i in range(0, len(linear_regularized_model_two)):
  print("SSE for L2 regularized model with lambda {}: {}".format(lambda_values[i], linear_regularized_model_two[i][1][-1]))
  print("Coefficients L2 for regularized model with lambda {}: {}".format(lambda_values[i], linear_regularized_model_two[i][2]))
  print("\n\n")

# plots for first linear regularized model
for i in range(0, len(linear_regularized_model_one)):
  plt.figure()
  plt.scatter(X, Y)
  plt.plot(linear_regularized_model_one[i][0])
  plt.title("Lambda of {} for the first linear regularized model".format(lambda_values[i]))
  plt.show()

# plots for second linear regularized model
for i in range(0, len(linear_regularized_model_two)):
  plt.figure()
  plt.scatter(X, Y)
  plt.plot(linear_regularized_model_two[i][0])
  plt.title("Lambda of {} for the second linear regularized model".format(lambda_values[i]))
  plt.show()
