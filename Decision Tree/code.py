# Decision Trees

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from sklearn.model_selection import train_test_split

class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, node):
        self.children.append(node)


# ## Building a Decision Tree
class DecisionNode:
    """
    This class will hold everything you require to construct a decision tree.
    The structure of this class is up to you. However, you need to support basic
    functionality as described above. It is highly recommended that you
    first read and understand the entire exercise before diving into this class.
    """
    def __init__(self, feature, value, data, depth):
        self.feature = feature # column index of criteria being tested
        self.value = value
        self.children = []
        self.prev = []
        self.data = data
        self.depth = depth
        values, counts = np.unique(self.data[:,-1], return_counts=True)
        self.classes_count = counts
        self.pred = None

    def add_child(self, node):
        self.children.append(node)

# ## Impurity Measures
def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the gini impurity.
    """

    gini = 0.0
    size = data.shape[0] #size of all given samples

    poison_gini = np.power((np.count_nonzero(data[:,-1] == 'p') / size),2)
    edible_gini = np.power((np.count_nonzero(data[:,-1] == 'e') / size),2)
    gini = 1 - (poison_gini + edible_gini)

    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the entropy of the dataset.
    """
    entropy = 0.0
    size = data.shape[0] #size of all samples

    poison_prop = np.count_nonzero(data[:,-1] == 'p') / size
    edible_prop = np.count_nonzero(data[:,-1] == 'e') / size

    if poison_prop == 0 or edible_prop == 0:
        return 0
    else:
        poison_entropy = -1 * poison_prop * np.log2(poison_prop)
        edible_entropy = -1 * edible_prop * np.log2(edible_prop)
        entropy = poison_entropy + edible_entropy
    return entropy

# ## Goodness of Split
#
# Given a feature the Goodnees of Split measures the reduction in the impurity if we split the data according to the feature.


def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.

    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index.
    - impurity func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns the goodness of split (or the Gain Ration).
    """
    goodness = 0.0
    split_values = []
    rows_indices = []
    impurity_dict = {}
    prop = 0.0
    split_prop = 0.0

    split_values = np.unique(data[:,feature:feature+1])

    if gain_ratio:
        for value in split_values:
            prop = value_prop(data, feature, value)
            split_prop += -1 * prop * np.log2(prop)
            rows_indices = np.where(data[:,feature: feature + 1] == value)
            rows_indices = rows_indices[0].tolist()
            goodness +=  prop * calc_entropy(data[rows_indices, :])

        goodness = (calc_entropy(data) - goodness) / split_prop
        return goodness

    else:
        for value in split_values:
            prop = value_prop(data, feature, value)
            rows_indices = np.where(data[:,feature: feature + 1] == value)
            rows_indices = rows_indices[0].tolist()
            goodness += prop * impurity_func(data[rows_indices , :])

        goodness = impurity_func(data) - goodness
        return (goodness)




def value_prop(data, feature, value):
    prop = np.count_nonzero(data[:, feature: feature + 1] == value) / data.shape[0]

    return prop

def chi_square_test(node):

    feature = node.feature
    data = node.data

    Y = []
    instances = len(data)
    for count in node.classes_count:
        Y.append(count / instances)

    values = np.unique(data[:, feature])
    classes = np.unique(data[:, -1])
    chi_square = 0

    for val in values:
        df = np.count_nonzero(data[:, feature] == val)
        for i in range(len(classes)):
            f = np.count_nonzero((data[:, feature] == val) & (data[:, -1] == classes[i]))
            e = df * Y[i]
            chi_square += ((f-e)**2) / e

    return chi_square

def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset.
    You are required to fully grow the tree until all leaves are pure.

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag
    - chi: chi square p-value cut off (1 means no pruning)
    - max_depth: the allowable depth of the tree

    Output: the root node of the tree.
    """
    root = None

    root = DecisionNode("ROOT", '', data, 0)
    add_pred_node(root)
    nodeQueue = [root]

    while(len(nodeQueue) != 0): #while there are nodes in the queue
        node = nodeQueue.pop(0)
        best_feature, max_val = -1, 0

        if impurity(node.data) > 0 and node.depth < max_depth:
             # Get best feature
            for i in range(len(data[0]) - 1):
                goodness = goodness_of_split(node.data,i,impurity,gain_ratio)
                if goodness > max_val:
                     best_feature, max_val = i, goodness

            if best_feature != -1:
                node.feature = best_feature
                feature_data = node.data[:,node.feature]
                values = np.unique(feature_data)
                chi_val = 0

                # Checking significance of chi
                if chi != 1:
                    chi_val = chi_square_test(node)
                if chi == 1 or chi_val >= chi_table[len(values) - 1][chi]:
                    # Ceate a children node and append it to node queue
                    for v in values:

                        child_node = DecisionNode(None, v, node.data[node.data[:, node.feature] == v], node.depth + 1)
                        add_pred_node(child_node)
                        node.add_child(child_node)
                        nodeQueue.append(child_node)


    return root

def add_pred_node(node):
    classes, counts = np.unique(node.data[:,-1], return_counts=True)
    node.pred = classes[counts.argmax()]

def predict(node, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """
    pred = None

    found = True

    while node.feature != None:
        temp = None
        for child in node.children:
            if child.value == instance[node.feature]:
                temp = child
        if not temp:
            break
        node = temp

    return node.pred

def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0
    count = 0.0
    for instance in dataset:
        if predict(node, instance) == instance[-1]: # if instance prediction equal to it's label
            count += 1

    accuracy = count*100 / len(dataset)

    return accuracy

def prunable_nodes(root, prunQueue):
    prunable = False
    for child in root.children:
        if child.feature is None:
            prunable = True
        else:
            prunable_nodes(child, prunQueue)
    if prunable:
        prunQueue.append(root)


def post_pruning(root):

    best_tree = tree_entropy_gain_ratio
    numOfNodes = [count_nodes(root)]
    accList = [calc_accuracy(root, X_test)]
    nodeCounter = 0
    flag = False
    listOfRoots = []
    counter = 0
    bestJ = 0

    while root.feature is not None:
        prunQueue = []
        prunable_nodes(root, prunQueue)
        prevAcc = accList[-1]
        bestAcc = -1
        maxNode = None

        for node in prunQueue:
            tempNode = node.feature
            node.feature = None
            acc = calc_accuracy(root, X_test)
            if acc > bestAcc:
                bestAcc = acc
                maxNode = node
            node.feature = tempNode

        listOfRoots.append(copy.deepcopy(root))
        if prevAcc > bestAcc and flag == False:
            flag = True
            bestJ = counter-1

        maxNode.feature = None
        nodeCounter = count_nodes(root)
        numOfNodes.append(nodeCounter)
        accList.append(bestAcc)
        counter+=1


    return numOfNodes, accList, listOfRoots[bestJ]


def treeDepth(node):
    if node is None or len(node.children) == 0:
        return 0
    else:
        listOfDepths = []
        for child in node.children:
            listOfDepths.append(treeDepth(child))

        return max(listOfDepths) + 1


def count_nodes(node):

    counter = 1
    if node.feature is not None:
        for child in node.children:
            counter += count_nodes(child)

    return counter

def print_tree(node, depth=0, parent_feature='ROOT', feature_val='ROOT'):

    if depth == 0:
        print("[{}, feature=X{}]".format(parent_feature, node.feature, node.value))

    else:
        if len(node.children) == 0:
            feature = 'leaf]: '
            counts = []
            for i in range(len(node.classes_count)):
                count = node.classes_count[i]
                if count != 0:
                    counts.append({float(i): count})
            feature += str(counts)
        else:
            feature = "feature=X" + str(node.feature) + "]"

        print("{:<{}s}[X{}={}, {}".format(" ", depth*3, parent_feature, feature_val, feature))

    for child_node in node.children:
        print_tree(child_node, depth + 1, node.feature, child_node.value)



def main():

    get_ipython().run_line_magic('matplotlib', 'inline')
    plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    # Ignore warnings
    import warnings
    warnings.filterwarnings('ignore')
    # load dataset
    data = pd.read_csv('agaricus-lepiota.csv')

    # One of the advantages of the Decision Tree algorithm is that almost no preprocessing is required.
    # However, finding missing values is always required.
    data = data.dropna(axis = 0, how = 'any', thresh = None, subset = None, inplace = False)


    # We will split the dataset to `Training` and `Testing` datasets.


    # Making sure the last column will hold the labels
    X, y = data.drop('class', axis=1), data['class']
    X = np.column_stack([X,y])
    # split dataset using random_state to get the same split each time
    X_train, X_test = train_test_split(X, random_state=99)

    print("Training dataset shape: ", X_train.shape)
    print("Testing dataset shape: ", X_test.shape)




    calc_gini(X_train), calc_entropy(X)
    s = goodness_of_split(X_train, 2, calc_gini, False)
    print (s)

    #building decision trees
    # python supports passing a function as an argument to another function.
    tree_gini = build_tree(data=X_train, impurity=calc_gini) # gini and goodness of split
    tree_entropy = build_tree(data=X_train, impurity=calc_entropy) # entropy and goodness of split
    tree_entropy_gain_ratio = build_tree(data=X_train, impurity=calc_entropy, gain_ratio=True) # entropy and gain ratio


    # ## Tree evaluation


    # After building the three trees using the training set, you should calculate the accuracy on the test set.
    # For each tree print the training and test accuracy.
    # Select the tree that gave you the best test accuracy.

    print("Gini Tree Accuracy:")
    print("\tTrain set: " + str(calc_accuracy(tree_gini, X_train)))
    print("\tTest set:  " + str(calc_accuracy(tree_gini, X_test)))
    print("Entropy Tree Accuracy:")
    print("\tTrain set: " + str(calc_accuracy(tree_entropy, X_train)))
    print("\tTest set:  " + str(calc_accuracy(tree_entropy, X_test)))
    print("Entropy Tree (Gain) Accuracy:")
    print("\tTrain set: " + str(calc_accuracy(tree_entropy_gain_ratio, X_train)))
    print("\tTest set:  " + str(calc_accuracy(tree_entropy_gain_ratio, X_test)))


    # ## Post pruning
    #
    # Iterate over all nodes in the tree that have at least a single child which is a leaf.
    # For each such node, replace it with its most popular class.
    # Calculate the accuracy on the testing dataset, pick the node that results in the highest testing accuracy and permanently change it in the tree.
    # Repeat this process until you are left with a single node in the tree (the root).
    # Finally, create a plot of the training and testing accuracies as a function of the number of nodes in the tree. (15 points)

    numOfNodes, accList, best_tree = post_pruning(tree_entropy_gain_ratio)
    accList_percent = [value for value in accList]
    plt.plot(numOfNodes, accList_percent)
    plt.gca().invert_xaxis()
    plt.title("Tree accuracy as function of tree size after pruning")
    plt.xlabel("Number of nodes")
    plt.ylabel("Accuracy %")
    plt.show()


    # ## Chi square pre-pruning
    #
    # Consider the following p-value cut-off values: [1 (no pruning), 0.5, 0.25, 0.1, 0.05, 0.0001 (max pruning)].
    # For each value, construct a tree and prune it according to the cut-off value.
    # Next, calculate the training and testing accuracy. On a single plot, draw the training and testing accuracy as a function of the tuple (p-value, tree depth).
    # Mark the best result on the graph with red circle.

    ### Chi square table values ###
    # The first key is the degree of freedom
    # The second key is the p-value cut-off
    # The values are the chi-statistic that you need to use in the pruning
    chi_table = {1: {0.5 : 0.45,
                     0.25 : 1.32,
                     0.1 : 2.71,
                     0.05 : 3.84,
                     0.0001 : 100000},
                 2: {0.5 : 1.39,
                     0.25 : 2.77,
                     0.1 : 4.60,
                     0.05 : 5.99,
                     0.0001 : 100000},
                 3: {0.5 : 2.37,
                     0.25 : 4.11,
                     0.1 : 6.25,
                     0.05 : 7.82,
                     0.0001 : 100000},
                 4: {0.5 : 3.36,
                     0.25 : 5.38,
                     0.1 : 7.78,
                     0.05 : 9.49,
                     0.0001 : 100000},
                 5: {0.5 : 4.35,
                     0.25 : 6.63,
                     0.1 : 9.24,
                     0.05 : 11.07,
                     0.0001 : 100000},
                 6: {0.5 : 5.35,
                     0.25 : 7.84,
                     0.1 : 10.64,
                     0.05 : 12.59,
                     0.0001 : 100000},
                 7: {0.5 : 6.35,
                     0.25 : 9.04,
                     0.1 : 12.01,
                     0.05 : 14.07,
                     0.0001 : 100000},
                 8: {0.5 : 7.34,
                     0.25 : 10.22,
                     0.1 : 13.36,
                     0.05 : 15.51,
                     0.0001 : 100000},
                 9: {0.5 : 8.34,
                     0.25 : 11.39,
                     0.1 : 14.68,
                     0.05 : 16.92,
                     0.0001 : 100000},
                 10: {0.5 : 9.34,
                      0.25 : 12.55,
                      0.1 : 15.99,
                      0.05 : 18.31,
                      0.0001 : 100000},
                 11: {0.5 : 10.34,
                      0.25 : 13.7,
                      0.1 : 17.27,
                      0.05 : 19.68,
                      0.0001 : 100000}}


    tree_entropy_gain_ratio = build_tree(data=X_train, impurity=calc_entropy, gain_ratio=True)
    dof = np.unique(X_train[:,-1]) #degree of freedon calc
    trees_dict = {1 : tree_entropy_gain_ratio}

    trainAcc = [calc_accuracy(tree_entropy_gain_ratio, X_train)]
    testAcc = [calc_accuracy(tree_entropy_gain_ratio, X_test)]
    tuples = [(1,treeDepth(tree_entropy_gain_ratio))] #tuples of (p value, maximum depth of tree)

    for pVal in chi_table[len(dof) - 1].keys():
        trees_dict[pVal] = build_tree(data=X_train, impurity=calc_entropy, gain_ratio=True, chi = pVal)
        trainAcc.append(calc_accuracy(trees_dict[pVal], X_train))
        testAcc.append(calc_accuracy(trees_dict[pVal], X_test))
        tuples.append(str((pVal, treeDepth(trees_dict[pVal]))))

    best_result = max(testAcc)
    listOfKeys = []
    for key in trees_dict:
        listOfKeys.append(key)


    best_chi = listOfKeys[testAcc.index(best_result)]
    plt.xscale('log', basex=2)
    plt.plot(best_chi, best_result, 'o', color='r')
    plt.plot(listOfKeys, trainAcc, label="Train set")
    plt.plot(listOfKeys, testAcc, label="Test Set")
    plt.xticks(listOfKeys, tuples, rotation='vertical')
    plt.xlabel('Depth')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of train set vs test set')
    plt.legend()
    plt.show()


    # Build the best 2 trees:
    # 1. tree_max_depth - the best tree according to max_depth pruning
    # 1. tree_chi - the best tree according to chi square pruning

    tree_max_depth = best_tree
    tree_chi = build_tree(data=X_train, impurity=calc_entropy, gain_ratio=True, chi = 0.1)

    if count_nodes(tree_max_depth) >= count_nodes(tree_chi):
        tree_to_print = copy.deepcopy(tree_chi)
    else:
        tree_to_print = copy.deepcopy(tree_max_depth)


    # ## Print the tree
    # In each brackets:
    # * The first argument is the parent feature with the value that led to current node
    # * The second argument is the selected feature of the current node
    # * If the current node is a leaf, you need to print also the labels and their counts

    print_tree(tree_to_print)

if __name__ == "__main__":
    main();
