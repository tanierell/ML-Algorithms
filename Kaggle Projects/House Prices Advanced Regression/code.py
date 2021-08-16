import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
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


# we will be looking for the best alpha- by using cross validation.
# shuffling training set
def cross_validation(x,y,k):
    print("running CV...")
    temp_train_set=np.concatenate((x, y), axis=1)
    np.random.shuffle(temp_train_set)
    part_size = int(temp_train_set.shape[0] / k)
    start = 0
    end = 0
    best_alpha = 1
    shuffled_set = []
    best_avg = 10000000
    alphas = []

    '''parting the shuffle to k parts, keeps it in a list'''
    for i in range(k):
        start = part_size * i
        end = part_size * (i + 1)
        shuffled_set.append(temp_train_set[start : end])

    for i in np.arange(0.023, 0.028, 0.0002):
        alphas.append(i)


    np.random.shuffle(alphas)

    for alpha in alphas:
        print("checking alpha: {}".format(alpha))
        acc = []
        for i in range(k):
            acc.append(i)
            start = part_size * i
            end = part_size * (i + 1)
            # load one part of the test_set , and create object for each 4-1 combination
            test_indices = [index for index in range(start,end)]
            train_indices = [index for index in range(temp_train_set.shape[0]) if index not in test_indices]

            test_set = temp_train_set[test_indices]
            train_set = temp_train_set[train_indices]

            # using grad_desc func to receive theta for this shuffled set
            temp_theta = gradient_descent(train_set[: ,:-1], train_set[: ,-1], theta, alpha, 0.0003)

            #predict MSE
            acc[i] = cost(test_set[:, :-1], test_set[: ,-1], temp_theta)
        results = sum(acc) / k
        if results < best_avg:
            best_avg = results
            best_alpha = alpha


    print("best alpha is: {}".format(best_alpha))
    return best_alpha

def predict(x, theta, mean, std):
    print("predicting house price values...")
    pred = np.dot(x,theta)
    pred = np.multiply(pred,std)
    return pred + mean

def main():
  train_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
  test_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
  ### setting concat_data to deal with different number of columns
  concat_data = pd.concat([train_data, test_data])
  # searching for the 'sale price' column in concated data
  sale_price_index = concat_data.columns.get_loc("SalePrice")

  ### converting to numpy arrays to use slicing etc.. ###
  concat_data = concat_data.to_numpy()
  x = concat_data[:, [col for col in range(concat_data.shape[1]) if col != sale_price_index]]
  y = concat_data[:, sale_price_index]
  x,y,mean,std = preprocess(x,y)
  ### creating X and y test and train set ###
  x_train = x[:train_data.shape[0]]
  y_train = y[:train_data.shape[0]]

  x_test = x[train_data.shape[0]:]
  y_test = y[train_data.shape[0]:]
  ### adding 1's column and initializing the theta vector ###

  ones_train = np.ones((x_train.shape[0],1))
  ones_test = np.ones((x_test.shape[0],1))
  x_train = np.hstack((ones_train, x_train))
  x_test = np.hstack((ones_test, x_test))

  np.random.seed(42)
  theta = np.random.random(x_train.shape[1])

  # cross validation inputs are- (x training set, y training set, k = 5 folds)
  # gradient descent inputs are- (x training set, y training set, random theta, computed
  # best alpha, and termination value to stop iterating)


  best_alpha = cross_validation(x_train, y_train.reshape(-1,1), 5)
  new_theta = gradient_descent(x_train, y_train, theta, best_alpha, 0.0003)

  pred = predict(x_test, new_theta,mean,std)

  header = ['id', 'SalePrice']
  with open('submission.csv', 'w', encoding='UTF8', newline='') as f:
  writer = csv.writer(f)
  writer.writerow(header)
  for index, row in enumerate(pred):
    writer.writerow([index + 1461, row])

  submission = pd.read_csv('submission.csv')
  submission.to_csv('submission.csv',index=False)
  print("the testing outputs for house prices per id are: \n")
  print(submission)

if __name__ == "__main__":
    main();
