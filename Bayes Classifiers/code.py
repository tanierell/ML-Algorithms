
# Naive Bayes Classifier based on Multi-Normal distribution
# Full Bayes Classifier based on Multi-Normal distribution
# Implement a Discrete Naive Bayes Classifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.

    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.

    Returns the normal distribution pdf according to the given mean and var for the given x.
    """
    # we assume X=(x1,x2...xn) is indepents, means P(x1,x2|y)=P(x1|y)*P(x2|y)
    # so we append a list to keep each xi density number, then returns a product of them


    prod_list = []
    dens = 0.0
    for i in range(len(x)):
        exp_power = -1 * (np.power(np.subtract(x[i],mean[i]),2)) / (2 * np.power(std[i],2))
        dens = (np.power(np.e, exp_power)) / (np.sqrt(2 * np.pi * np.power(std[i],2)))
        prod_list.append(dens)
    return (np.prod(prod_list))


class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.

        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        # to calc mean and std as the theta of the equation further, we calc it for different "given" spotted or not
        # further we calc P(spotted|x) = P(x|spotted)...-> givven spotted gives different std and mean values


        self.mean = []
        self.std = []
        self.target = class_value
        self.dataset = dataset
        rows_given_class = np.where(dataset[: , -1] == self.target)
        rows_given_class = rows_given_class[0].tolist()
        for feature in range(dataset.shape[1]-1):
            self.mean.append(np.mean(dataset[rows_given_class , feature]))
            self.std.append(np.std(dataset[rows_given_class , feature]))


    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """

        occur = np.count_nonzero(self.dataset[:,-1] == self.target)

        return (occur / self.dataset.shape[0])


    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """

        return (normal_pdf(x, self.mean, self.std))


    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        #calcs posterior=P(A|x)=P(x|A)*prior(A)

        return (self.get_instance_likelihood(x) * self.get_prior())

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier.
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.

        """
        # check if P(A|x) > P(B|x) then the instance is A, else B
        if self.ccd0.get_instance_posterior(x) > self.ccd1.get_instance_posterior(x):
            return 0
        return 1

class MultiNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.

        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        self.mean = []
        self.target = class_value
        self.dataset = dataset
        rows_of_class = np.where(dataset[: , -1] == class_value)
        rows_of_class = rows_of_class[0].tolist()
        for feature in range(dataset.shape[1]-1):
            self.mean.append(np.mean(dataset[rows_of_class , feature]))
        self.cov = np.cov(self.dataset[rows_of_class, 0:-1],rowvar=False, bias=True)


    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        occur = np.count_nonzero(self.dataset[:,-1] == self.target)

        return (occur / self.dataset.shape[0])


    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        return multi_normal_pdf(x, self.mean, self.cov)

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.get_prior()


class MaxPrior():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum prior classifier.
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1


    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        if self.ccd0.get_prior() > self.ccd1.get_prior():
            return 0
        return 1


# Implement the **MaxLikelihood** class and build a MaxLikelihood object like you did above with the **MAPClassifier**.
class MaxLikelihood():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum Likelihood classifier.
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        if self.ccd0.get_instance_likelihood(x) > self.ccd1.get_instance_likelihood(x):
            return 0
        return 1

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes
        distribution for a specific class. The probabilites are computed with laplace smoothing.

        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        self.target = class_value
        self.dataset = dataset
        self.class_dataset = dataset[np.where(dataset[:,-1] == self.target)]
        self.class_dataset = np.delete(self.class_dataset, -1, axis=1)

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return (self.class_dataset.shape[0] / self.dataset.shape[0])

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = 1
        for i in range(x.shape[0] - 1):
            V_j_size = len(set(self.class_dataset[:,i]))
            n_i = self.class_dataset.shape[0]
            n_ij = self.class_dataset[np.where(self.class_dataset[:,i] == x[i])].shape[0]
            likelihood *= (n_ij + 1) / (n_i + V_j_size)

        return likelihood



    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.get_prior()



def compute_accuracy(testset, map_classifier):
    """
    Compute the accuracy of a given a testset using a MAP classifier object.

    Input
        - testset: The testset for which to compute the accuracy (Numpy array).
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.

    Ouput
        - Accuracy = #Correctly Classified / #testset size
    """
    counter = 0
    for ins in range(testset.shape[0]):
        if map_classifier.predict(testset[ins,0:-1]) == testset[ins,-1]:
            counter += 1
    return counter / testset.shape[0]

def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.

    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.

    Returns the normal distribution pdf according to the given mean and var for the given x.
    """
    dens = 0.0
    exp_power = -0.5 * (np.transpose(np.subtract(x,mean)).dot(np.linalg.inv(cov)).dot(np.subtract(x,mean)))
    dens = (np.power(np.e, exp_power)) * (np.power((2 * np.pi),(x.shape[0]-1)/2) *(np.power(np.linalg.det(cov),-0.5)))

    return dens


def main():

    # The dataset contains 2 features (**Temperature**, **Humidity**) alongside a binary label (**Spotted**) for each instance.<br>
    #
    # We are going to test 2 different classifiers :
    # * Naive Bayes Classifier
    # * Full Bayes Classifier
    # The datafiles are :
    # - randomammal_train.csv
    # - randomammal_test.csv

    # Load the train and test set into a pandas dataframe and convert them into a numpy array.
    train_set = pd.read_csv('data/randomammal_train.csv').values
    test_set = pd.read_csv('data/randomammal_test.csv').values


    # # Data Visualization
    # scatter plot of the training data where __x__=Temerature and **y**=Humidity. <br>
    not_spotted = test_set[np.where(test_set[:,-1] == 0)]
    spotted = test_set[np.where(test_set[:,-1] == 1)]
    for i in range(len(spotted)):
        spotted[i][1] += 50

    plt.scatter(not_spotted[:,0], not_spotted[:,1], color='blue', s=10)
    plt.scatter(spotted[:,0], spotted[:,1], color='green', s=10)
    plt.xlabel('Temerature')
    plt.ylabel('Humidity')
    plt.show()


    # Naive Model:
    # Build the a NaiveNormalClassDistribution for each class.
    naive_normal_CD_0 = NaiveNormalClassDistribution(train_set, 0)
    naive_normal_CD_1 = NaiveNormalClassDistribution(train_set, 1)


    # Implement the **MAPClassifier** class and build a MAPClassifier object containing the 2 distribution objects you just made above.
    naive_normal_classifier = MAPClassifier(naive_normal_CD_0, naive_normal_CD_1)


    # ### Evaluate model
    # Compute the naive model accuracy and store it in the naive accuracy variable.
    naive_accuracy = compute_accuracy(test_set, naive_normal_classifier)
    print(naive_accuracy)


    # Full Model
    # Build the a MultiNormalClassDistribution for each class.
    multi_normal_CD_0 = MultiNormalClassDistribution(train_set, 0)
    multi_normal_CD_1 = MultiNormalClassDistribution(train_set, 1)


    # build a MAPClassifier object contating the 2 distribution objects you just made above.
    multi_normal_classifier = MAPClassifier(multi_normal_CD_0, multi_normal_CD_1)


    # ### Evaluate model
    # Compute the naive model accuracy and store it in the naive accuracy variable.
    full_accuracy = compute_accuracy(test_set, multi_normal_classifier)
    print(full_accuracy)


    # ## Results
    # Bar plot of accuracy of each model side by side.
    plt.bar(x=['Naive', 'Full'], height=[naive_accuracy, full_accuracy])
    plt.title("Naive vs Full accuracy comparison")
    plt.ylabel("Accuracy")


    # # Comparing Max a posteriori, prior, and likelihood results

    # For each of the classifiers above (naive Bayes and full Bayes, in which we compare posterior probabilities),
    # we explore how classifiers would perform if we compare (1) only prior probabilities or (2) only likelihoods.
    # Run and evaluate the models

    # Repeat the process you did for the MAPClassifier, now for the MaxPrior and MaxLikelihood classifiers:
    # 1. Feed the naive_normal distributions and the multi_normal distributions you made for each class into the new models you made in this section
    # 2. Evaluate the accuracies
    # 3. Plot the results as described in the beginning of this section

    naive_normal_CD_0 = NaiveNormalClassDistribution(train_set, 0)
    naive_normal_CD_1 = NaiveNormalClassDistribution(train_set, 1)
    naive_MAP = MAPClassifier(naive_normal_CD_0, naive_normal_CD_1)
    naive_prior = MaxPrior(naive_normal_CD_0, naive_normal_CD_1)
    naive_likelihood = MaxLikelihood(naive_normal_CD_0, naive_normal_CD_1)
    multi_normal_CD_0 = MultiNormalClassDistribution(train_set, 0)
    multi_normal_CD_1 = MultiNormalClassDistribution(train_set, 1)
    multi_MAP = MAPClassifier(multi_normal_CD_0, multi_normal_CD_1)
    multi_prior = MaxPrior(multi_normal_CD_0, multi_normal_CD_1)
    multi_likelihood = MaxLikelihood(multi_normal_CD_0, multi_normal_CD_1)

    # data to plot
    n_groups = 3
    naive_data = []
    full_data = []

    naive_data.append(compute_accuracy(test_set, naive_MAP))
    naive_data.append(compute_accuracy(test_set, naive_prior))
    naive_data.append(compute_accuracy(test_set, naive_likelihood))

    full_data.append(compute_accuracy(test_set, multi_MAP))
    full_data.append(compute_accuracy(test_set, multi_prior))
    full_data.append(compute_accuracy(test_set, multi_likelihood))

    naive_data = tuple(naive_data)
    full_data = tuple(full_data)

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, naive_data, bar_width,
    alpha=opacity,
    color='b',
    label='Naive Classifier')

    rects2 = plt.bar(index + bar_width, full_data, bar_width,
    alpha=opacity,
    color='g',
    label='Multi Classifier')

    plt.xlabel('Classifiers')
    plt.ylabel('Accuracies')
    plt.title('Accuracy by each classifier')
    plt.xticks(index + bar_width, ('MAP', 'MaxPrior', 'MaxLikelihood'))
    plt.legend()

    plt.tight_layout()
    plt.show()


    # # Discrete Naive Bayes Classifier

    # Data
    # We will try to predict breast cancer again only this time from a different dataset,
    # Load the training set and test set provided for you in the data folder.
    #  - breast_trainset.csv
    #  - breast_testset.csv

    # Load the train and test set into a pandas dataframe and convert them into a numpy array.
    train_set = pd.read_csv('data/breast_trainset.csv').values
    test_set = pd.read_csv('data/breast_testset.csv').values


    # ## Build A Discrete Naive Bayes Distribution for each class
    discrete_naive_CD_0 = DiscreteNBClassDistribution(train_set, 0)
    discrete_naive_CD_1 = DiscreteNBClassDistribution(train_set, 1)

    # build a MAPClassifier object contating the 2 distribution objects you just made above.
    discrete_naive_classifier = MAPClassifier(discrete_naive_CD_0, discrete_naive_CD_1)
    compute_accuracy(test_set, discrete_naive_classifier)

if __name__ == "__main__":
    main()
