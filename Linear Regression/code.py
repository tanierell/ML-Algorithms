# # Exercise 1: Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv

def preprocess(X, y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Inputs (n features over m instances).
    - y: True labels.

    Returns a two vales:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    X = (X-X.mean()) / (X.std())
    y = (y-y.mean()) / (y.std())
    return X, y

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an obserbation's actual and
    predicted values for linear regression.

    Input:
    - X: inputs  (n features over m instances).
    - y: true labels (1 value over m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns a single value:
    - J: the cost associated with the current set of parameters (single number).
    """
    m = X.shape[0]
    hypo = np.subtract(np.dot(X,theta),y)
    sig_equation = np.power(hypo,2)
    J = (1/(2*m))*np.sum(sig_equation)
    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using
    the *training set*. Gradient descent is an optimization algorithm
    used to minimize some (loss) function by iteratively moving in
    the direction of steepest descent as defined by the negative of
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns two values:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    J_history = [] # Use a python list to save cost in every iteration
    theta = theta.copy() # avoid changing the original thetas
    derivArr = {} # dict to keep eaach theta's derivative
    temp = {} # temporary dict to handle thetas

    J_history.append(compute_cost(X, y, theta))
    m = X.shape[0]
    i = 1
    while (i < num_iters):
        predError = np.subtract(np.dot(X,theta),y)
        for j in range(len(theta)):
            derivArr[j] = np.dot(predError,(X[:, j:j+1]))
            temp[j] = theta[j] - (alpha/m)*derivArr[j]
            theta[j] = temp[j]
        J = compute_cost(X, y, theta)
        J_history.append(J)
        i += 1

    return theta, J_history

def pinv(X, y):
    """
    Calculate the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).

    Returns two values:
    - theta: The optimal parameters of your model.
    """

    pinv_theta = []
    X_transpose = np.transpose(X)
    mult_by_transpose = X_transpose.dot(X)
    inverse_mat = inv(mult_by_transpose)
    pinvX = inverse_mat.dot(X_transpose)
    pinv_theta = pinvX.dot(y)
    return pinv_theta


# We can use a better approach for the implementation of `gradient_descent`.
#Instead of performing 40,000 iterations, we wish to stop when the improvement of the loss value is smaller than `1e-8` from one iteration to the next.

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the *training set*, but stop
    the learning process once the improvement of the loss value is smaller
    than 1e-8. This function is very similar to the gradient descent
    function you already implemented.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns two values:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    J_history = [] # Use a python list to save cost in every iteration
    theta = theta.copy() # avoid changing the original thetas
    temp = {} # dict to keep eaach theta's derivative
    derivArr = {} # temporary dict to handle thetas

    J_history.append(compute_cost(X, y, theta))
    differ = J_history[0] # keeps differential costs between two different thetas
    lastJ = J_history[0] # keeps last added cost
    m = X.shape[0] #number of instances
    i = 1
    while (abs(differ >= 1e-8) and (i < num_iters)):
        predError = np.subtract(np.dot(X,theta),y)
        for j in range(len(theta)):
            derivArr[j] = np.dot(predError,(X[:, j:j+1]))
            temp[j] = theta[j] - (alpha/m)*derivArr[j]
            theta[j] = temp[j]
        J = compute_cost(X, y, theta)
        differ = lastJ - J
        lastJ = J
        J_history.append(J)
        i += 1
    return theta, J_history


# The learning rate is another factor that determines the performance of our model in terms of speed and accuracy.

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over provided values of alpha and train a model using the
    *training* dataset. maintain a python dictionary with alpha as the
    key and the loss on the *validation* set as the value.

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {key (alpha) : value (validation loss)}
    """

    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}
    J_history={}
    np.random.seed(42)
    theta = np.random.random(size=X_train.shape[1])
    for alpha in alphas:
        temp, J_history[alpha] = efficient_gradient_descent(X_train, y_train, theta, alpha, iterations)
        alpha_dict[alpha] = compute_cost(X_val, y_val, temp)

    return alpha_dict



def forward_selection():
    """
    Train the model using the training set using a single feature.
    Choose the best feature according to the validation set. Next,
    check which feature performs best when added to the feature
    you previously chose. Repeat this process until you reach 4
    features and the bias. Don't forget the bias trick.

    Returns:
    - The names of the best features using forward selection.
    """

    mTrain = X_train.shape[0] #number of instances of X_train
    onesArr_train = np.ones((mTrain,)) #array of ones with mTrain size
    mVal = X_val.shape[0] #number of instances of X_val
    onesArr_val = np.ones((mVal,)) #array of ones with mVal size
    np.random.seed(42)
    best_features = []
    lossDict = {} #dict to keep each thetas
    iterations = 40000
    alpha = best_alpha
    bestJ = 0; #variable which keeps each iteration the best feature
    newTheta = np.random.random(size=2)

    for i in range(4):
        newTheta = np.random.random(size=i+2)
        minVal = 10000000000
        for j in range (1,len(all_features.columns)+1):
            if j not in best_features:
                temp_train_array = np.column_stack((onesArr_train,X_train[: , j:j+1]))
                temp_val_array = np.column_stack((onesArr_val,X_val[:, j:j+1]))
                lossDict[j], J_history = efficient_gradient_descent(temp_train_array, y_train, newTheta, alpha, iterations)
                cost = compute_cost(temp_val_array, y_val, lossDict[j])
                if cost < minVal:
                    minVal = cost
                    bestJ = j
        onesArr_train = np.column_stack((onesArr_train, X_train[:, bestJ:bestJ+1]))
        onesArr_val = np.column_stack((onesArr_val, X_val[:, bestJ:bestJ+1]))
        best_features.append(bestJ)

    best_features = [all_features.columns[k] for k in range(len(all_features.columns)) if ((k+1) in best_features)]
    return best_features


def backward_selection():
    """
    Train the model using the training set using all but one of the
    features at a time. Remove the worst feature according to the
    validation set. Next, remove an additional feature along with the
    feature you previously removed. Repeat this process until you
    reach 4 features and the bias. Don't forget the bias trick.

    Returns:
    - The names of the best features using backward selection.
    """

    np.random.seed(42)
    best_features = all_features.columns
    newTheta = np.random.random(size=17)
    iterations = 1000
    alpha = best_alpha
    lossDict = {} #dict to keep each thetas
    temp_train = X_train
    temp_val = X_val
    newList = []
    worstJ = 0 #variable which keeps each iteration the worst feature
    worst_features = []


    while (len(all_features.columns) - len(worst_features) > 4):
        minVal = 100000000
        for j in range (1,len(all_features.columns)+1): #range 1-17 kolel
            if j not in worst_features:
                newList = [i for i in range (len(all_features.columns)+1) if ((i!=j) and (i not in worst_features))]
                temp_train = X_train[:, newList]
                temp_val = X_val[:, newList]
                lossDict[j], J_history = efficient_gradient_descent(temp_train, y_train, newTheta, alpha, iterations)
                cost = compute_cost(temp_val, y_val, lossDict[j])
                if cost < minVal:
                    minVal = cost
                    worstJ = j
        worst_features.append(worstJ)
        newTheta = np.random.random(size=len(all_features.columns)-len(worst_features)) #updates theta's dim to match X_train's

    best_features = [all_features.columns[k] for k in range(len(all_features.columns)) if (k+1 not in worst_features)]
    return best_features

def main():
    np.random.seed(42)
    # make matplotlib figures appear inline in the notebook
    get_ipython().run_line_magic('matplotlib', 'inline')
    plt.rcParams['figure.figsize'] = (14.0, 8.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    # We will use a dataset containing housing prices in King County, USA. The dataset contains 5,000 observations with 18 features and a single target value - the house price.
    #
    # First, we will read and explore the data using pandas and the `.read_csv` method.
    df = pd.read_csv('data.csv') # Make sure this cell runs regardless of your absolute path
    # ### Data Exploration
    # Start by looking at the top of the dataset using the `df.head()` command.
    df.head(5)
    df.describe()

    # We will start with one variable linear regression by extracting the target column and the `sqft_living` variable from the dataset.
    X = df['sqft_living'].values
    y = df['price'].values
    # ## Preprocessing

    X, y = preprocess(X, y)

    # We will split the data into two datasets:
    # 1. The training dataset will contain 80% of the data and will always be used for model training.
    # 2. The validation dataset will contain the remaining 20% of the data and will be used for model evaluation.

    # training and validation split
    np.random.seed(42)
    indices = np.random.permutation(X.shape[0])
    idx_train, idx_val = indices[:int(0.8*X.shape[0])], indices[int(0.8*X.shape[0]):]
    X_train, X_val = X[idx_train], X[idx_val]
    y_train, y_val = y[idx_train], y[idx_val]

    # ## Data Visualization
    plt.plot(X_train, y_train, 'ro', ms=1, mec='k') # the parameters control the size, shape and color of the scatter plot
    plt.ylabel('Price in USD')
    plt.xlabel('sq.ft')
    plt.show()

    # ## Bias Trick
    #
    # Make sure that `X` takes into consideration the bias $\theta_0$ in the linear model.
    # Add columns of ones as the zeroth column of the features (do this for both the training and validation sets).
    N_train = X_train.shape[0]
    ones_Array_train = np.ones((N_train,))
    N_val = X_val.shape[0]
    ones_Array_val = np.ones((N_val,))
    X_train = np.column_stack((ones_Array_train,X_train))
    X_val = np.column_stack((ones_Array_val,X_val))

    # ## Part 2: Single Variable Linear Regression
    theta = np.array([-1, 2])
    J = compute_cost(X_train, y_train, theta)
    np.random.seed(42)
    theta = np.random.random(size=2)
    iterations = 10000
    alpha = 0.1
    theta, J_history = gradient_descent(X_train ,y_train, theta, alpha, iterations)

    # In the following graph, we visualize the loss as a function of the iterations.
    #This is possible since we are saving the loss value at every iteration in the `J_history` array.
    # This visualization might help you find problems with your code.
    #Notice that since the network converges quickly, we are using logarithmic scale for the number of iterations.
    plt.plot(np.arange(iterations), J_history)
    plt.xscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss as a function of iterations')
    plt.show()

    theta_pinv = pinv(X_train ,y_train)
    J_pinv = compute_cost(X_train, y_train, theta_pinv)


    # We can add the loss value for the theta calculated using the psuedo-inverse to our graph.
    #This is another sanity check as the loss of our model should converge to the psuedo-inverse loss.

    plt.plot(np.arange(iterations), J_history)
    plt.xscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss as a function of iterations')
    plt.hlines(y = J_pinv, xmin = 0, xmax = len(J_history), color='r',
               linewidth = 1, linestyle = 'dashed')
    plt.show()



    alpha_dict = find_best_alpha(X_train, y_train, X_val, y_val, 10000)

    # Obtain the best learning rate from the dictionary `alpha_dict`.
    best_alpha = min(alpha_dict, key=alpha_dict.get)

    # Pick the best three alpha values you just calculated and provide **one** graph with three lines indicating the training loss as a function of iterations (Use 10,000 iterations).

    alpha_dict_test = alpha_dict.copy()
    best_alphas = []
    for i in range(3):
        best = min(alpha_dict_test,key = alpha_dict_test.get)
        best_alphas.append(best)
        del alpha_dict_test[best]
    np.random.seed(42)
    theta1 = np.random.random(size=X_train.shape[1]) #randoms theta with X_train's number of features dim
    iterations = 10000

    colors = ['b', 'g', 'r']
    J_history1 = {}
    temp1 = {}
    label=[]
    for i in range (len(best_alphas)):
        temp1[i], J_history1[i] = gradient_descent(X_train, y_train, theta1, best_alphas[i], iterations)
        label.append('alpha =' + ' ' + str(best_alphas[i]))
        plt.plot(np.arange(iterations), J_history1[i], colors[i], label=label[i])

    plt.xscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss as a function of iterations')
    plt.legend()
    plt.show()


    # This is yet another sanity check. This function plots the regression lines of your model and the model based on the pseudoinverse calculation.
    # Both models should exhibit the same trend through the data.

    plt.figure(figsize=(7, 7))
    plt.plot(X_train[:,1], y_train, 'ro', ms=1, mec='k')
    plt.ylabel('Price in USD')
    plt.xlabel('sq.ft')
    plt.plot(X_train[:, 1], np.dot(X_train, theta), 'o')
    plt.plot(X_train[:, 1], np.dot(X_train, theta_pinv), '-')

    plt.legend(['Training data', 'Linear regression', 'Best theta']);

    # ## Part 2: Multivariate Linear Regression
    #
    # In most cases, you will deal with databases that have more than one feature.
    df = pd.read_csv('data.csv')
    df.head()

    # ## Preprocessing
    X = df.drop(columns=['price', 'id', 'date']).values
    y = df['price'].values
    X, y = preprocess(X, y)

    # training and validation split
    np.random.seed(42)
    indices = np.random.permutation(X.shape[0])
    idx_train, idx_val = indices[:int(0.8*X.shape[0])], indices[int(0.8*X.shape[0]):]
    X_train, X_val = X[idx_train,:], X[idx_val,:]
    y_train, y_val = y[idx_train], y[idx_val]


    # Using 3D visualization, we can still observe trends in the data.
    get_ipython().run_line_magic('matplotlib', 'inline')
    import mpl_toolkits.mplot3d.axes3d as p3
    fig = plt.figure(figsize=(5,5))
    ax = p3.Axes3D(fig)
    xx = X_train[:, 1][:1000]
    yy = X_train[:, 2][:1000]
    zz = y_train[:1000]
    ax.scatter(xx, yy, zz, marker='o')
    ax.set_xlabel('bathrooms')
    ax.set_ylabel('sqft_living')
    ax.set_zlabel('price')
    plt.show()


    # Use the bias trick again
    N_train = X_train.shape[0]
    ones_Array_train = np.ones((N_train,))
    N_val = X_val.shape[0]
    ones_Array_val = np.ones((N_val,))
    X_train = np.column_stack((ones_Array_train,X_train))
    X_val = np.column_stack((ones_Array_val,X_val))

    shape = X_train.shape[1]
    theta = np.ones(shape)
    J = compute_cost(X_train, y_train, theta)

    alpha_dict = find_best_alpha(X_train, y_train, X_val, y_val, 10000) #finds best cost for each alpha
    best_alpha = min(alpha_dict, key=alpha_dict.get) # finds the best alpha for the X_train's features

    np.random.seed(42)
    shape = X_train.shape[1]
    theta = np.random.random(shape)
    iterations = 10000
    theta, J_history = gradient_descent(X_train ,y_train, theta, best_alpha, iterations)


    theta_pinv = pinv(X_train ,y_train)
    J_pinv = compute_cost(X_train, y_train, theta_pinv)

    # We can use visualization to make sure the code works well. Notice we use logarithmic scale for the number of iterations,
    # since gradient descent converges after ~500 iterations.

    plt.plot(np.arange(iterations), J_history)
    plt.xscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss as a function of iterations - multivariate linear regression')
    plt.hlines(y = J_pinv, xmin = 0, xmax = len(J_history), color='r',
               linewidth = 1, linestyle = 'dashed')
    plt.show()

    # ## Part 3: Find best features for regression
    #
    # Adding additional features to our regression model makes it more complicated but does not necessarily improves performance.
    # Use forward and backward selection and find 4 features that best minimizes the loss. First, we will reload the dataset as a dataframe in order to access the feature names.

    columns_to_drop = ['price', 'id', 'date']
    all_features = df.drop(columns=columns_to_drop)
    all_features.head(5)

    # ### Forward Feature Selection
    #
    # Complete the function `forward_selection`. Train the model using a single feature at a time, and choose the best feature using the validation dataset.
    # Next, check which feature performs best when added to the feature you previously chose.
    # Repeat this process until you reach 4 features + bias. You are free to use any arguments you need.

    bestForward = forward_selection()


    # ### Backward Feature Selection
    #
    # Complete the function `backward_selection`. Train the model with all but one of the features at a time and remove the worst feature (the feature that its absence yields the best loss value using the validation dataset).
    # Next, remove an additional feature along with the feature you previously removed.
    #Repeat this process until you reach 4 features + bias. You are free to use any arguments you need.

    bestBackward = backward_selection()

if __name__ == "__main__":
    main()
