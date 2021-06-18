# Logistic Regression, Bayes and EM

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# Function for ploting the decision boundaries of a model
def plot_decision_regions(X, y, classifier, resolution=0.01):

    # setup marker generator and color map
    markers = ('.', '.')
    colors = ('blue', 'red')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

# ## Normal distribution pdf
def norm_pdf(data, mu, sigma):
    return np.power(np.e, -np.power(data-mu,2) / (2 * np.power(sigma, 2))) / np.power(2 * np.pi * np.power(sigma, 2), 0.5)


def accuracy(predictions, test_set):
    acc = 0.0
    for index,guess in enumerate(predictions):
        if guess == test_set[index]:
            acc += 1
    return (acc / test_set.shape[0])

def evaluation(X_train, y_train, X_test, y_test, type_algo):
    ''' this func receives training and validation sets, and the algo to use- Naive Bayes or LR
        then evaluates for each the training and validation sets their accuracies
    '''
    type_algo.fit(X_train, y_train)

    algo_prediction_train = type_algo.predict(X_train)
    algo_accuracy_train = accuracy(algo_prediction_train, y_train)
    print("Train Accuracy", algo_accuracy_train)

    algo_prediction_test = type_algo.predict(X_test)
    algo_accuracy_test = accuracy(algo_prediction_test, y_test)
    print("Test Accuracy", algo_accuracy_test)




# Logistic Regression
# Your class should contain the following functions:
# 1. fit - the learning function
# 1. predict - the function for predicting an instance after the fit function was executed

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state
        np.random.seed(random_state)
        self.best_theta = []
        self.c_history = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Updating the theta vector in each iteration using gradient descent.
        Store the theta vector in an attribute of the LogisticRegressionGD object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """

        ### target_i runs on the indexes of y_train with to know each instance's index to get label to y_train
        ##theta = np.random.random(size = 3)
        iterations = 0
        N_train = X.shape[0]
        ones_Array_train = np.ones((N_train,))
        X = np.column_stack((ones_Array_train,X))
        self.best_theta = np.random.random(size=X.shape[1])
        diff = self.cost(X,y)
        self.c_history.append(diff)
        lastJ = diff


        while iterations < self.n_iter and diff >= self.eps:
            self.best_theta -= self.eta  * np.dot(X.T, self.sigmoid(X) - y)
            J = self.cost(X,y)
            self.c_history.append(J)
            diff = abs(lastJ - J)
            lastJ = J
            iterations += 1

    def predict(self, X):
        """Return the predicted class label"""
        N_val = X.shape[0]
        ones_Array_val = np.ones((N_val,))
        X = np.column_stack((ones_Array_val,X))

        predictions = np.zeros((N_val,))
        for i,example in enumerate(X):
            if self.sigmoid(example) > 0.5:
                predictions[i] = 1
        return predictions


    def sigmoid(self, X):
        """" hypothesis for logistic regression """
        return 1 / (1 + np.exp(-np.dot(X, self.best_theta)))

    def cost(self, X, y):
        ans = y * np.log(self.sigmoid(X) + 1.8e-320) + (1-y) * np.log(1-self.sigmoid(X) + 1.8e-320)
        return - np.sum(ans) / len(X)

# ## Expectation Maximization

# Implementation of the Expectation Maximization algorithm for gaussian mixture model.
# The class should hold the distribution params.
# Uses log likelihood as the cost function:
class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state = 1):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state
        self.mu = []
        self.sigma = []
        self.weight = []

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        np.random.seed(self.random_state)
        counter = 0

        for i in range(self.k):
            start = int(i * (data.shape[0] / self.k))
            end = int(start + (data.shape[0] / self.k))
            self.mu.append(np.mean(data[start:end]))
            self.sigma.append(np.std(data[start:end]))
            self.weight.append(1 / self.k)


    def expectation(self, data):
        """
        E step - calculating responsibilities
        """
        ### set 2-d array of probabilities-> rows are instances of data and cols for the k's gaussian ###
        counter = 0
        resp = []

        for i in range(self.k):
            resp.append([])
            for j in range(data.shape[0]):
                resp[i].append(norm_pdf(data[j], self.mu[i], self.sigma[i]) * self.weight[i])
                s = 0
                for l in range(self.k):
                    s += self.weight[l] * norm_pdf(data[j], self.mu[l], self.sigma[l])

                resp[i][j] /= s
        return (resp)


    def maximization(self, data):
        """
        M step - updating distribution params
        """
        counter = 0
        responsibility = self.expectation(data)

        ### setting new weight of each gauss dist. by summing respos. column and divide by num of instances ###
        for index in range(len(self.weight)):
            self.weight[index] = np.sum(responsibility[ : ,index]) / data.shape[0]

        ### setting mu for each gaussian, dont know how instance look like what dimension ###
        for gauss in range(self.k):

            for i,instance in enumerate(data):
                counter += responsibility[i][gauss] * instance
            self.mu[gauss] = counter / (self.weight[gauss] * data.shape[0])
        counter = 0
        ### setting sigma for each gaussian, dont know how instance look like what dimension ###
        for gauss in range(self.k):
            counter = 0
            for i,instance in enumerate(data):
                counter += responsibility[i][gauss] * np.power((instance - self.mu[gauss]),2)
            self.sigma[gauss] = (counter / (self.weight[gauss] * data.shape[0])) ** 0.5


    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        self.init_params(data)
        iterations = 0
        temp_cost = self.cost(data)
        diff = temp_cost

        while iterations < self.n_iter and diff >= self.eps:
            self.maximization(data)
            new_cost = self.cost(data)
            diff = abs(temp_cost - new_cost)
            temp_cost = new_cost
            iterations += 1


    def get_dist_params(self):
        return self.mu, self.sigma, self.weight

    def cost(self, data):
        counter = 0
        temp = 0
        for index,instance in enumerate(data):
            for gauss in range(self.k):
                pdf = norm_pdf(instance, self.mu[gauss], self.sigma[gauss])
                counter += np.log2(self.weight[gauss] * pdf)
        return counter

# Naive Bayes
# For calculating the likelihood use the EM algorithm that you implemented above to find the distribution params. With these params you can calculate the likelihood probability.
# Calculate the prior probability directly from the training set.
class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state = 1):
        self.k = k
        self.random_state = random_state
        self.target0_params = []
        self.target1_params = []
        self.target0_prior = 0.0
        self.target1_prior = 0.0

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """

        ''' we get each class' prior '''
        X0 = X[np.where(y == 0)]
        X1 = X[np.where(y == 1)]

        self.target0_prior = X0.shape[0] / X.shape[0]
        self.target1_prior = X1.shape[0] / X.shape[0]

        ''' fitting each model per his k's gaussian '''
        for i in range(X.shape[1]):
            GMM0 = EM(self.k)
            GMM0.fit(X0[:, i])
            self.target0_params.append(GMM0.get_dist_params())


            GMM1 = EM(self.k)
            GMM1.fit(X1[:, i])
            self.target1_params.append(GMM1.get_dist_params())



    def predict(self, X):
        """Return the predicted class label"""
        predictions = []


        ''' calculate likelihood for each instance, and each gaussian, then returns a product of each class' gaussians '''
        for i in range(X.shape[0]):
            f0_1 = 0
            f0_2 = 0
            f1_1 = 0
            f1_2 = 0

            for j in range(self.k):
                f0_1 +=  self.target0_params[0][2][j] * norm_pdf(X[i][0], self.target0_params[0][0][j], self.target0_params[0][1][j])
                f0_2 +=  self.target0_params[1][2][j] * norm_pdf(X[i][1], self.target0_params[1][0][j], self.target0_params[1][1][j])
                f1_1 +=  self.target1_params[0][2][j] * norm_pdf(X[i][0], self.target1_params[0][0][j], self.target1_params[0][1][j])
                f1_2 +=  self.target1_params[1][2][j] * norm_pdf(X[i][1], self.target1_params[1][0][j], self.target1_params[1][1][j])

            like0 = f0_1 * f0_2
            like1 = f1_1 * f1_2

            if like0 * self.target0_prior > like1 * self.target1_prior:
                predictions.append(0)
            else:
                predictions.append(1)
        return np.asarray(predictions)

def main():

    # make matplotlib figures appear inline in the notebook
    get_ipython().run_line_magic('matplotlib', 'inline')
    plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    # Make the notebook automatically reload external python modules
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    # Ignore warnings
    import warnings
    warnings.filterwarnings('ignore')


    # ## Reading the data
    training_set = pd.read_csv('training_set.csv')
    test_set = pd.read_csv('test_set.csv')
    X_training, y_training = training_set[['x1', 'x2']].values, training_set['y'].values
    X_test, y_test = test_set[['x1', 'x2']].values, test_set['y'].values


    # ## Visualizing the data
    # For the first feature only:
    # 1. For the first 1000 data points plot a histogram for each class on the same graph
    # 1. For all the data points plot a histogram for each class on the same graph

    # For both features:
    # 1. For the first 1000 data points plot a scatter plot where each class has different color
    # 1. For all the data points plot a scatter plot where each class has different color

    ### first feature only ####

    for i in range(2):
        if i == 0:
            class0Rows = np.where(training_set.values[0:1000 , -1] == 0)
            class1Rows = np.where(training_set.values[0:1000 , -1] == 1)
            bins = 20
            title = "first 1000 data points"
        else:
            class0Rows = np.where(training_set.values[ : , -1] == 0)
            class1Rows = np.where(training_set.values[ : , -1] == 1)
            bins = 40
            title = "all data points"



        x0 = X_training[class0Rows[0].tolist(), 0:1]
        x1 = X_training[class1Rows[0].tolist(), 0:1]

        plt.hist(x0, bins, alpha=0.5, label='class 0')
        plt.hist(x1, bins, alpha=0.5, label='class 1')
        plt.legend(loc='upper right')
        plt.title(title)
        plt.show()

    ### both features ###

    for i in range(2):
        if i == 0:
            class0Rows = np.where(training_set.values[0:1000 , -1] == 0)
            class1Rows = np.where(training_set.values[0:1000 , -1] == 1)
            title = "first 1000 data points"
        else:
            class0Rows = np.where(training_set.values[ : , -1] == 0)
            class1Rows = np.where(training_set.values[ : , -1] == 1)
            title = "all data points"


        ax = X_training[class0Rows[0].tolist(), 0:1]
        ay = X_training[class0Rows[0].tolist(), 1:2]

        bx = X_training[class1Rows[0].tolist(), 0:1]
        by = X_training[class1Rows[0].tolist(), 1:2]

        plt.scatter(ax, ay ,s = 1, color='blue')
        plt.scatter(bx, by ,s = 1, color='red')

        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.title(title)
        plt.show()





    # ## Cross Validation
    # Use 5-fold cross validation in order to find the best eps and eta params from the given lists.
    # Shuffle the training set before you split the data to the folds.

    '''shuffling training set'''
    shuffle = training_set.sample(frac = 1)
    shuffle = shuffle.to_numpy()
    shuffled_set = []
    part_size = int(shuffle.shape[0] / 5)
    start = 0
    end = 0
    best_tuple = ()
    objects = []
    pred_list = []

    best_avg = 0

    '''parting the shuffle to 5 parts, keeps it in a list'''
    for i in range(5):
        start = part_size * i
        end = part_size * (i + 1)
        shuffled_set.append(shuffle[start : end])


    etas = [0.05, 0.005, 0.0005, 0.00005, 0.000005]
    epss = [0.01, 0.001, 0.0001, 0.00001, 0.000001]

    for eta in etas:
        for eps in epss:
            acc = []
            for i in range(5):
                ''' load one part of the test_set , and create object for each 4-1 combination'''
                obj = LogisticRegressionGD(eta, 1000, eps, random_state=1)
                objects.append(obj)
                test_set = shuffled_set[i]
                train_set = np.arange(0).reshape(0,3)
                acc.append(i)
                for j in range(5):
                    ''' load the other 4 parts of training_set'''
                    if j != i:
                        train_set = np.append(train_set , shuffled_set[j], axis = 0)
                ''' using fit func to receive best theta for the last object added '''

                objects[-1].fit(train_set[ : , 0:2], train_set[ : ,-1])
                pred_list = objects[-1].predict(test_set[ : , 0:2])

                ''' prediction part to calc accuracy '''
                for index,k in enumerate(pred_list):
                    if k == test_set[index][-1]:
                        acc[i] +=1
                acc[i] = acc[i] / test_set.shape[0]
            results = sum(acc) / 5
            if results > best_avg:
                best_avg = results
                best_tuple = (eta,eps)


    print(best_tuple)













    # ## Model evaluation
    # In this section you will build 2 models and fit them to 2 datasets

    # First 1000 training points and first 500 test points:
    # 1. Use the first 1000 points from the training set (take the first original 1000 points - before the shuffle) and the first 500 points from the test set.
    # 1. Fit Logistic Regression model with the best params you found earlier.
    # 1. Fit Naive Bayes model. Remember that you need to select the number of gaussians in the EM.
    # 1. Print the training and test accuracies for each model.
    # 1. Use the `plot_decision_regions` function to plot the decision boundaries for each model (for this you need to use the training set as the input)
    # 1. Plot the cost Vs the iteration number for the Logistic Regression model
    #
    # Use all the training set points:
    # 1. Repeat sections 2-6 for all the training set points

    ### first 1000 points

    best_eta = best_tuple[0]
    best_eps = best_tuple[1]
    X_train = X_training[0:1000]
    y_train = y_training[0:1000]
    first500_X_test = X_test[0:500]
    first500_y_test = y_test[0:500]

    print("Logistic Regression GD")
    lr = LogisticRegressionGD(eta = best_eta, eps = best_eps)
    evaluation(X_train, y_train,  first500_X_test, first500_y_test, lr)

    print("Naive Base")
    nb = NaiveBayesGaussian(k=2)
    evaluation(X_train, y_train,  first500_X_test, first500_y_test, nb)

    plt.figure()
    plt.title("Logistic Regression - first 1000 points - training set")
    plot_decision_regions(X_train, y_train, lr)
    plt.show()

    plt.figure()
    plt.title("Naive Base plot - first 1000 points - training set");
    plot_decision_regions(X_train, y_train, nb)
    plt.show()

    plt.figure()
    plt.title("Logistic Regression cost as a function of iterations")
    plt.plot(list(range(len(lr.c_history))), lr.c_history)
    plt.xlabel("Iterations")
    plt.ylabel('Loss')
    plt.show()

    best_eta = best_tuple[0]
    best_eps = best_tuple[1]

    ### all data points

    print("Logistic Regression GD")
    lr = LogisticRegressionGD(eta = best_eta, eps=best_eps)
    evaluation(X_training, y_training, X_test, y_test, lr)

    print("Naive Base")
    nb = NaiveBayesGaussian(k=4)
    evaluation(X_training, y_training, X_test, y_test, nb)

    plt.figure()
    plt.title("Logistic Regression - all data points - training set")
    plot_decision_regions(X_training, y_training, lr)
    plt.show()

    plt.figure()
    plt.title("Naive Base plot - all data points - training set");
    plot_decision_regions(X_training, y_training, nb)
    plt.show()

    plt.figure()
    plt.title("Logistic Regression cost as a function of iterations")
    plt.plot(list(range(len(lr.c_history))), lr.c_history)
    plt.xlabel("Iterations")
    plt.ylabel('Loss')
    plt.show()


    ''' EXPLENATION GRAPH '''

    '''
        first 1000 data points-
            as we can see the data is a almost linear seperable. that's why the LoR works well with over 95% success on both
            training anf test set. As for Naive Bayes we can see that by assuming there are 2 gaussians distributies. the success rate is around 95% too.

        all data points-
            as we can see the data points are not linear seperable. when using Naive Bayes  we assume there are 4 gaussians,
            which leads to 90% success, since Gaussian Naive Bayes dont need linear seperable data to classify, and looks for minimal maximum.
            for the LoR, it can't classify perfectly since it not linear sepreable data. then the regression finds the best seperator
            for one group, in this case the left one. the success rate of the other side (this case-right side) is lower bounded
            by the lowest prior(this case its 50%), and the success of the left side is close to the success of the first 1000 points.


    '''

    # We explored two types of models: Naive Bayes using EM, and Logistic regression.
    #     - Generate one dataset that you think Naive Bayes will work better than Logisitc Regression.
    #     - Generate another dataset that you think Logistic Regression will work better than Naive Bayes using EM.


    # Explenation:
    #
    # first we generated a graph which will be predicted better by using LoR. As you can see after running, test and train accuracy of LoR is 100%. Naive Bayes train acc is aroung 89%, and test is arount 0.44%. This data leads to these accuracies by generating each feature dependent to the other. in addition, not only dependent relations, but we made it being linear sepereable. Naive Bayes assumes the data is independent given observation, which is not true here, that why he fails and LoR does not make any assumption on dependencies.
    #
    # second, we genreted data which is not linear seperable- a class "0" which is inside class "1" in an eliptic shape. Naive Bayes accuracies are around 90% and LoR are aroung 60%, since it cannot sperate the data. Naive Bayes will classify the data by its gaussians (for normal distributions)- thats why it doesnt need linear seperability, and LoR cant handle it.



    ''' Better LoR '''

    x0c0 = np.random.normal(1, 2, 1000)
    x1c0 = x0c0 * (-2) + np.random.normal(0, 1, 1000)
    x0c1 = np.random.normal(1, 2, 1000)
    x1c1 = x0c1 * (-2) + np.random.normal(0, 1, 1000) + 10
    mat1 = np.column_stack((x0c0, x1c0))
    mat2 = np.column_stack((x0c1, x1c1))
    x_dataset = np.vstack((mat1, mat2))
    y_dataset = np.hstack((np.zeros(1000).astype(int), np.ones(1000).astype(int)))

    best_eta = best_tuple[0]
    best_eps = best_tuple[1]
    X_train = x_dataset[0:1600]
    y_train = y_dataset[0:1600]
    last400_X_test = x_dataset[1600:2000]
    last400_y_test = y_dataset[1600:2000]


    print("Logistic Regression GD")
    lr = LogisticRegressionGD(eta = best_eta, eps = best_eps)
    evaluation(X_train, y_train,  last400_X_test, last400_y_test, lr)

    print("Naive Base")
    nb = NaiveBayesGaussian(k=2)
    evaluation(X_train, y_train,  last400_X_train, last400_y_train, nb)

    ### VISUALIZATION ###

        ### First feature only ###
    for i in range(2):
        if i == 0:
            class0Rows = np.where(y_dataset[0:1000] == 0)
            class1Rows = np.where(y_dataset[0:1000] == 1)
            bins = 20
            title = "first 1000 data points"
        else:
            class0Rows = np.where(y_dataset == 0)
            class1Rows = np.where(y_dataset == 1)
            bins = 40
            title = "all data points"



        x0 = x_dataset[class0Rows[0].tolist(), 0:1]
        x1 = x_dataset[class1Rows[0].tolist(), 0:1]

        plt.hist(x0, bins, alpha=0.5, label='class 0')
        plt.hist(x1, bins, alpha=0.5, label='class 1')
        plt.legend(loc='upper right')
        plt.title(title)
        plt.show()

        ### both features ###

    for i in range(2):
        if i == 0:
            class0Rows = np.where(y_dataset[0:1000] == 0)
            class1Rows = np.where(y_dataset[0:1000] == 1)
            title = "first 1000 data points"
        else:
            class0Rows = np.where(y_dataset == 0)
            class1Rows = np.where(y_dataset == 1)
            title = "all data points"


        ax = x_dataset[class0Rows[0].tolist(), 0:1]
        ay = x_dataset[class0Rows[0].tolist(), 1:2]

        bx = x_dataset[class1Rows[0].tolist(), 0:1]
        by = x_dataset[class1Rows[0].tolist(), 1:2]

        plt.scatter(ax, ay ,s = 1, color='blue')
        plt.scatter(bx, by ,s = 1, color='red')

        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.title(title)
        plt.show()


    plt.figure()
    plt.title("Logistic Regression - all data points - training set")
    plot_decision_regions(x_dataset, y_dataset, lr)
    plt.show()

    plt.figure()
    plt.title("Naive Base plot - all data points - training set");
    plot_decision_regions(x_dataset, y_dataset, nb)
    plt.show()


    # In[143]:


    ''' Better Naive Bayes using EM '''

    x0 = np.random.normal(0,1,2000)
    x1 = np.random.normal(0,1,2000)
    x_dataset = np.column_stack((x0, x1))
    y_dataset = ((pow(x0, 2) + pow(x1,2)) < 1).astype(int)

    best_eta = best_tuple[0]
    best_eps = best_tuple[1]
    X_train = x_dataset[0:1600]
    y_train = y_dataset[0:1600]
    last400_X_test = x_dataset[1600:2000]
    last400_y_test = y_dataset[1600:2000]

    print("Logistic Regression GD")
    lr = LogisticRegressionGD(eta = best_eta, eps = best_eps)
    evaluation(X_train, y_train,  last400_X_test, last400_y_test, lr)

    print("Naive Base")
    nb = NaiveBayesGaussian(k=2)
    evaluation(X_train, y_train,  last400_X_test, last400_y_test, nb)

    ### VISUALIZATION ###

        ### First feature only ###
    for i in range(2):
        if i == 0:
            class0Rows = np.where(y_dataset[0:1000] == 0)
            class1Rows = np.where(y_dataset[0:1000] == 1)
            bins = 20
            title = "first 1000 data points"
        else:
            class0Rows = np.where(y_dataset == 0)
            class1Rows = np.where(y_dataset == 1)
            bins = 40
            title = "all data points"



        x0 = x_dataset[class0Rows[0].tolist(), 0:1]
        x1 = x_dataset[class1Rows[0].tolist(), 0:1]

        plt.hist(x0, bins, alpha=0.5, label='class 0')
        plt.hist(x1, bins, alpha=0.5, label='class 1')
        plt.legend(loc='upper right')
        plt.title(title)
        plt.show()

        ### both features ###

    for i in range(2):
        if i == 0:
            class0Rows = np.where(y_dataset[0:1000] == 0)
            class1Rows = np.where(y_dataset[0:1000] == 1)
            title = "first 1000 data points"
        else:
            class0Rows = np.where(y_dataset == 0)
            class1Rows = np.where(y_dataset == 1)
            title = "all data points"


        ax = x_dataset[class0Rows[0].tolist(), 0:1]
        ay = x_dataset[class0Rows[0].tolist(), 1:2]

        bx = x_dataset[class1Rows[0].tolist(), 0:1]
        by = x_dataset[class1Rows[0].tolist(), 1:2]

        plt.scatter(ax, ay ,s = 1, color='blue')
        plt.scatter(bx, by ,s = 1, color='red')

        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.title(title)
        plt.show()


    plt.figure()
    plt.title("Logistic Regression - all data points - training set")
    plot_decision_regions(x_dataset, y_dataset, lr)
    plt.show()

    plt.figure()
    plt.title("Naive Base plot - all data points - training set");
    plot_decision_regions(x_dataset, y_dataset, nb)
    plt.show()


if __name__ == "__main__":
    main();
