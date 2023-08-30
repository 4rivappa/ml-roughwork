import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# X=np.linspace(np.pi/3, 5*np.pi/3)
X=np.linspace(60, 300)
X=np.deg2rad(X)
K=np.random.normal(0,0.15)
Y=np.sin(X)+K

plt.scatter(X, Y)
plt.title("Generated Data")

data=np.vstack((X, Y)).T


def linear_reg_pred(x, y, l_r, epochs):
    m = np.shape(x)[0] # samples
    n = np.shape(x)[1] # features
    
    x = np.concatenate((np.ones((m,1)),x), axis=1)
    w = 2*np.random.rand(n+1,)-1

    loss_history = []
    
    for current_iteration in range(epochs):
        y_estimated = x.dot(w)
        error = y_estimated - y
        cost = np.sum(error ** 2)/len(error)
        gradient = (1 / m) * x.T.dot(error)
        w = w - l_r * gradient
        loss_history.append(cost)
    return y_estimated, loss_history, w



yhat, sse, w=linear_reg_pred(data[:, :-1], data[:, -1], 0.00001, 100)


plt.scatter(X, Y)
plt.plot(yhat)
plt.xlim(np.min(X), np.max(X))
plt.ylim(np.min(Y), np.max(Y))
print("SSE for Linear: {}".format(sse[-1]))
print("Coefficients for Linear: {}".format(w))


def gen_poly_feat(X, deg):
  X_poly=X
  for i in range(2,deg+1):
    X_poly=np.vstack((X_poly, np.power(X, i)))
  
  poly_data=X_poly.T
  # print(poly_data)

  poly_data=pd.DataFrame(poly_data)
  # poly_data=(poly_data-np.mean(poly_data, axis=1))/np.std(poly_data, axis=1)
  poly_data=(poly_data-poly_data.mean())/poly_data.std()
  poly_data=poly_data.to_numpy()
  return poly_data


feat_3=gen_poly_feat(X, 3)
feat_6=gen_poly_feat(X, 6)
feat_9=gen_poly_feat(X, 9)
feat_12=gen_poly_feat(X, 12)
feat_15=gen_poly_feat(X, 15)


yhat3, sse3, w3=linear_reg_pred(feat_3, data[:, -1], 0.00001, 100)
yhat6, sse6, w6=linear_reg_pred(feat_6, data[:, -1], 0.00001, 100)
yhat9, sse9, w9=linear_reg_pred(feat_9, data[:, -1], 0.00001, 100)
yhat12, sse12, w12=linear_reg_pred(feat_12, data[:, -1], 0.00001, 100)
yhat15, sse15, w15=linear_reg_pred(feat_15, data[:, -1], 0.00001, 100)


plt.scatter(X, Y)
plt.plot(yhat3)
plt.xlim(np.min(X), np.max(X))
plt.ylim(-15, 15)

# plt.ylim(np.min(yhat3), np.max(yhat3))
plt.title("Polynomial 3")
print("SSE for Polynomial 3: {}".format(sse3[-1]))
print("Coefficients for Polynomial 3: {}".format(w3))


plt.scatter(X, Y)

plt.xlim(np.min(X), np.max(X))
# plt.ylim(-15, 15)

# plt.ylim(np.min(Y), np.max(Y))
plt.plot(yhat6)
plt.title("Polynomial 6")
print("SSE for Polynomial 6: {}".format(sse6[-1]))
print("Coefficients for Polynomial 6: {}".format(w6))


plt.scatter(X, Y)
plt.xlim(np.min(X), np.max(X))
# plt.ylim(0, np.max(Y))
plt.plot(yhat9)

# plt.ylim(np.min(yhat3), np.max(yhat3))
plt.title("Polynomial 9")
print("SSE for Polynomial 9: {}".format(sse9[-1]))
print("Coefficients for Polynomial 9: {}".format(w9))


plt.scatter(X, Y)
plt.plot(yhat12)
plt.xlim(np.min(X), np.max(X))
plt.ylim(-15, 15)

# plt.ylim(np.min(yhat3), np.max(yhat3))
plt.title("Polynomial 12")
print("SSE for Polynomial 12: {}".format(sse12[-1]))
print("Coefficients for Polynomial 12: {}".format(w12))


plt.scatter(X, Y)
plt.plot(yhat15)
# plt.xlim(np.min(X), np.max(X))
# plt.ylim(-15, 15)

# plt.ylim(np.min(yhat3), np.max(yhat3))
plt.title("Polynomial 15")
print("SSE for Polynomial 15: {}".format(sse15[-1]))
print("Coefficients for Polynomial 15: {}".format(w15))


def linear_pred_with_reg(x, y, reg, lam, l_r, epochs):
    m = np.shape(x)[0] # samples
    n = np.shape(x)[1] # features
    
    x = np.concatenate((np.ones((m,1)),x), axis=1)
    w = 2*np.random.rand(n+1,)-1

    loss_history = []
    
    for current_iteration in range(epochs):
        y_estimated = x.dot(w)
        error = y_estimated - y
        

        if(reg=="L1"):
            sigw=np.sum(np.abs(w))
            cost = np.sum(error ** 2)/len(error)+(lam/2)*sigw
        if(reg=="L2"):
            sigw=np.sum(np.square(w))
            cost = np.sum(error ** 2)/len(error)+(lam/2)*sigw
        gradient = (1 / m) * x.T.dot(error)
        w = w - l_r * gradient
        loss_history.append(cost)
    return y_estimated, loss_history, w



lambdas=[1e-10,1e-8,1e-4,1e-2,1,10,20]
l1_reg_models=[]
l2_reg_models=[]

for i in lambdas:
  l1_reg_models.append(linear_pred_with_reg(feat_15, data[:, -1], "L1", i, 0.001, 100))
  l2_reg_models.append(linear_pred_with_reg(feat_15, data[:, -1], "L2", i, 0.001, 100))


for i in range(0, len(l1_reg_models)):
  print("SSE for L1 regularized model with lambda {}: {}".format(lambdas[i], l1_reg_models[i][1][-1]))
  print("Coefficients L1 for regularized model with lambda {}: {}".format(lambdas[i], l1_reg_models[i][2]))
  print("\n\n")


for i in range(0, len(l2_reg_models)):
  print("SSE for L2 regularized model with lambda {}: {}".format(lambdas[i], l2_reg_models[i][1][-1]))
  print("Coefficients L2 for regularized model with lambda {}: {}".format(lambdas[i], l2_reg_models[i][2]))
  print("\n\n")


for i in range(0, len(l1_reg_models)):
  plt.figure()
  plt.scatter(X, Y)
  plt.plot(l1_reg_models[i][0])

  plt.title("L1 regularized model with lambda {}".format(lambdas[i]))
  print("\n")


for i in range(0, len(l2_reg_models)):
  plt.figure()
  plt.scatter(X, Y)
  plt.plot(l2_reg_models[i][0])

  plt.title("L2 regularized model with lambda {}".format(lambdas[i]))
  print("\n")
