# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)
from sklearn import datasets, linear_model


# 1-dimensional implementation

# fitting/estimating the model
def one_dim_train(Xtrain, ytrain):
  # transpose X to get XT
  Xtrain_T = Xtrain.T
  # multiply XT by X
  XTX = Xtrain_T.dot(Xtrain)
  # take inverse of XTX
  XTXinv = np.linalg.inv(XTX)
  # multiply XT by y
  XTy = Xtrain_T.dot(ytrain)
  # Betahat = XTXinv * XTy
  Betahat = XTXinv.dot(XTy)
  # calculate intercept: alphahat = ybar - Betahat * Xbar
  ybar = np.mean(ytrain)
  Xbar = np.mean(Xtrain)
  alphahat = ybar - Betahat * Xbar
  alphahat = alphahat[0]
  return Betahat, alphahat

# return predicted y values
def one_dim_test(Xtest, Betahat, alphahat):
  y_pred = alphahat + Betahat[0] * Xtest
  return y_pred

# define training and testing datasets
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_X = diabetes_X[:, np.newaxis, 2]
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# sklearn model: trained on training data, predicting test data
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
sklearn_y_pred = regr.predict(diabetes_X_test)

# our model: trained on training data, predicting test data
Betahat, alphahat = one_dim_train(diabetes_X_train, diabetes_y_train)
y_pred = np.squeeze(one_dim_test(diabetes_X_test, Betahat, alphahat))

# plotting ours and sklearn's models with the data
plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
plt.plot(diabetes_X_test, sklearn_y_pred, color='blue', linewidth=3, alpha=.5, label='sklearn model')
plt.plot(diabetes_X_test, np.squeeze(y_pred), color='red', linewidth=3, alpha=.5, label='our model')

plt.xlabel('BMI')
plt.ylabel('Diabetes Diagnosis')
plt.title('Diabetes Disease Progression vs BMI')
plt.legend()
plt.show()

# compare squared error and predicted y values
error_sklearn = sklearn_y_pred - diabetes_y_test
error = y_pred - diabetes_y_test
squared_error_sklearn = np.square(sklearn_y_pred - diabetes_y_test)
squared_error = np.square(y_pred - diabetes_y_test)
print("squared errors compared:", np.concatenate((np.expand_dims(squared_error_sklearn, 1), np.expand_dims(squared_error, 1)), axis=1))
print("predicted y values compared:", np.concatenate((np.expand_dims(y_pred, 1), np.expand_dims(sklearn_y_pred, 1)), axis=1))


# 2-dimensional implementation

# fitting/estimating the model
def two_dim_train(Xtrain, ytrain):
  # transpose X to get XT
  Xtrain_T = Xtrain.T
  # multiply XT by X
  XTX = Xtrain_T.dot(Xtrain)
  # take inverse of XTX
  XTXinv = np.linalg.inv(XTX)
  # multiply xTy
  XTy = Xtrain_T.dot(ytrain)
  # Betahat = XTXinv * XTy
  Betahat = XTXinv.dot(XTy)
  # calculate intercept: alphahat = ybar - Betahat * xbar
  ybar = np.mean(ytrain)
  Xbar = np.mean(Xtrain)
  alphahat = ybar - Betahat * Xbar
  alphahat = alphahat[0]
  return Betahat, alphahat

# return predicted y values
def two_dim_test(X_test, Betahat, alphahat):
  y_pred = alphahat + Betahat[0] * X_test[:,0] + Betahat[1] * X_test[:,1]
  return y_pred

# define training and testing datasets
diabetes = datasets.load_diabetes()
diabetes_X_df = pd.DataFrame(data=diabetes.data, columns = diabetes.feature_names)
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_X = diabetes_X_df[['bmi','s6']].values
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# sklearn model: trained on training data, predicting test data
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
diabetes_y_pred = regr.predict(diabetes_X_test)

# our model: trained on training data, predicting test data
Betahat, alphahat = two_dim_train(diabetes_X_train, diabetes_y_train)
y_pred = np.squeeze(two_dim_test(diabetes_X_test, Betahat, alphahat))

# plotting ours and sklearn's models with the data (3d graph)
x1plot = np.linspace(diabetes_X_test[:,0].min(),diabetes_X_test[:,0].max())
x2plot = np.linspace(diabetes_X_test[:,1].min(),diabetes_X_test[:,1].max())
x1x1plot, x2x2plot = np.meshgrid(x1plot, x2plot)
plane_mesh = np.array([x1x1plot.flatten(), x2x2plot.flatten()]).T
sklearn_plane = regr.predict(plane_mesh)
plane = alphahat + Betahat[0] * x1x1plot.flatten() + Betahat[1] * x2x2plot.flatten()

ax = plt.axes(projection='3d')
ax.plot(diabetes_X_test[:,0], diabetes_X_test[:,1], diabetes_y_test, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
ax.scatter(x1x1plot.flatten(), x2x2plot.flatten(), sklearn_plane, facecolor=(0,0,0,0), s=5, edgecolor='#70b3f0', label='sklearn model')
ax.scatter(x1x1plot.flatten(), x2x2plot.flatten(), plane, facecolor=(0,0,0,0), s=5, edgecolor='red', label='our model')
ax.scatter(diabetes_X_test[:,0], diabetes_X_test[:,1], diabetes_y_test, color='red')
ax.plot(diabetes_X_test[:,0], diabetes_X_test[:,1], diabetes_y_pred, color='blue', alpha=.5, linestyle='none', marker='o')
ax.plot(diabetes_X_test[:,0], diabetes_X_test[:,1], y_pred, color='red', linestyle='none', marker='o', alpha=.5)

ax.set_xlabel('BMI')
ax.set_ylabel('Glucose Level')
ax.set_zlabel('Diabetes Disease Progression')
plt.title('Diabetes Disease Progression vs BMI & Glucose Level')
plt.legend()
plt.show()

# compare squared error and predicted y values
error_sklearn = diabetes_y_pred - diabetes_y_test
error_ours = y_pred - diabetes_y_test
squared_error_sklearn = np.square(diabetes_y_pred - diabetes_y_test)
squared_error = np.square(y_pred - diabetes_y_test)
print("squared errors compared:", np.concatenate((np.expand_dims(squared_error_sklearn, 1), np.expand_dims(squared_error, 1)), axis=1))
print("predicted y values compared:", np.concatenate((np.expand_dims(y_pred, 1), np.expand_dims(diabetes_y_pred, 1)), axis=1))