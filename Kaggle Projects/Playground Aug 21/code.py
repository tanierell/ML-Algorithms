import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# preprocessing function, use normalization for each column in the concat data
def preprocess(x,y):
    print("preprocessing...")
    std = y.std(axis = 0)
    mean = y.mean(axis = 0)
    x = (x-x.mean(axis = 0)) / x.std(axis = 0)
    y = (y-mean) / std
    return x,y,mean,std

## cost function - root mean square error
def cost(x,y, theta):
  return np.sum(np.power(np.subtract(np.dot(x,theta),y), 2)) / x.shape[0]

## grad descent
def gradient_descent(x, y, theta, alpha, termination):
    theta = theta.copy()
    predError = np.subtract(np.dot(x, theta),y)
    temp = 0.0
    J_last = cost(x, y, theta)
    J_new = J_last + 1
    differ = J_new - J_last
    iterations = 0
    while np.abs(differ) > termination and iterations < 10000:
        predError = np.subtract(np.dot(x, theta),y)
    for i in range(len(theta)):
        derivJ = np.dot(predError,(x[:, i]))
        temp = theta[i] - (alpha/x.shape[0]) * derivJ
        theta[i] = temp
        J_new = cost(x, y, theta)
        differ = J_last - J_new
        J_last = J_new
        iterations += 1
    return theta


# we will be looking for the best alpha
def find_best_alpha(x,y):
    train_size = int(0.8 * x.shape[0])

    x_train = x[:train_size]
    y_train = y[:train_size]
    x_val = x[train_size:]
    y_val = y[train_size:]

    print("finding best alpha...")
    best_alpha = 1
    best_cost = 10000000
    temp_cost = 10000000
    alphas = []
    for i in np.arange(1.02,1.04,0.01):
        alphas.append(i)

    for alpha in alphas:
        print("checking alpha: {}".format(alpha))
        # using grad_desc func to receive theta for this shuffled set
        temp_theta = gradient_descent(x_train, y_train, theta, alpha, 0.001)
        #predict MSE
        temp_cost = cost(x_val, y_val, temp_theta)
        if temp_cost < best_cost:
            best_cost = temp_cost
            best_alpha = alpha


    print("best alpha is: {}".format(best_alpha))
    return best_alpha

def predict(x, theta, mean, std):
    print("predicting loss values...")
    pred = np.dot(x,theta)
    pred = np.multiply(pred,std)
    return pred + mean

def main():
  train_data = pd.read_csv("/kaggle/input/tabular-playground-series-aug-2021/train.csv")
  test_data = pd.read_csv("/kaggle/input/tabular-playground-series-aug-2021/test.csv")

  ### setting concat_data to deal with different number of columns
  concat_data = pd.concat([train_data, test_data])

  concat_data = concat_data.fillna(concat_data.mean())
  # searching for the 'sale price' column in concated data
  loss_index = concat_data.columns.get_loc("loss")

  ### converting to numpy arrays to use slicing etc.. ###
  concat_data = concat_data.to_numpy()
  x = concat_data[:, [col for col in range(concat_data.shape[1]) if col != loss_index]]
  y = concat_data[:, loss_index]
  x,y,mean,std = preprocess(x,y)

  ### creating X and y test and train set ###
  x_train = x[:train_data.shape[0]]
  y_train = y[:train_data.shape[0]]

  x_test = x[train_data.shape[0]:]
  ### adding 1's column and initializing the theta vector ###

  ones_train = np.ones((x_train.shape[0],1))
  ones_test = np.ones((x_test.shape[0],1))
  x_train = np.hstack((ones_train, x_train))
  x_test = np.hstack((ones_test, x_test))

  np.random.seed(42)
  theta = np.random.random(x_train.shape[1])

  # gradient descent inputs are- (x training set, y training set, random theta, computed
  # best alpha, and termination value to stop iterating)

  best_alpha = find_best_alpha(x_train, y_train)
  new_theta = gradient_descent(x_train, y_train, theta, best_alpha, 0.00003)

  pred = predict(x_test, new_theta, mean, std)

  header = ['id', 'loss']
  with open('submission.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for index, row in enumerate(pred):
      writer.writerow([index + 250000, row])
      submission = pd.read_csv('submission.csv')
  print("the testing outputs for house prices per id are: ")
  print(submission)

if __name__ == "__main__":
      main()
