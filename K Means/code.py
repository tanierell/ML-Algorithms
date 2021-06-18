# Clustering
#
# In this implementation you will experiment with k-means as an image compression algorithm.
# ## Color image quantization

# You might need to install skimage using `conda install -c conda-forge scikit-image`.
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

# Each centroid is a point in RGB space (color) in the image. This function should uniformly pick `k` centroids from the dataset.
#
# Input: a single image of shape `(num_instances, 3)` and `k`, the number of centroids. Notice we are flattening the image to a two dimentional array.
# Output: Randomly chosen centroids of shape `(k,3)`.
def get_random_centroids(X, k):
    centroids = []
    for i in range(k):
        random_indices = np.random.choice(X.shape[0], size=1, replace=False)
        random_centroid = X[random_indices, :]
        centroids.append(random_centroid)
    centroids = [centroid[0] for centroid in centroids]
    return np.array(centroids)

# Inputs:
# * a single image of shape `(num_instances, 3)`
# * the centroids `(k, 3)`
# * the distance parameter p
#
# output: array of shape `(k, num_instances)` thats holds the distances of all points in RGB space from all centroids

# In[56]:


def lp_distance(X, centroids, p=2):
    X = X.astype('float64')
    in_sum = np.abs(X - centroids[:, np.newaxis]) ** p
    sum = np.sum(in_sum, axis = -1)
    distances =  sum ** (1.0 / p)

    return distances


    return distances

# ## Kmeans algorithm
#
# Calculate the locally optimal centroids as learned in class. At each iteration,
# assign every RGB point to the closest centroids and calculate new centroids by averaging the points that were assigned to every centroid.
# This function stops, when no improvement was made or once max_iter iterations passed.
# A reasonable implementation runs on a Core i7 CPU in less than a minute with `k=16`.


def kmeans(X, k, p , calc_method, max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_features, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.
    Outputs:
    - The calculated centroids
    - The final assignment of all RGB points to the closest centroids
    """
    classes = np.array([])
    temp_centroids = get_random_centroids(X, k)
    i = 0
    diff = True
    distances = np.zeros((k,X.shape[0]))

    while i <= max_iter and diff == True:
        centroids = np.copy(temp_centroids)
        ### creating 2d array of distances from each instance to each centroid, then receive the minimum dist each column ###

        distances = np.array(lp_distance(X, temp_centroids, p=p))
        classes = np.argmin(distances, axis = 0)
        for i in range (k):
            cluster =  X[np.where(classes == i)]
            if len(cluster) > 0:
                temp_centroids[i]  = calc_method(cluster, axis = 0)
        if np.sum(np.abs(temp_centroids - centroids)) < 1e-8:
            diff = False



    return centroids, classes

def calc_inertia(X, centroids, classes):
    inertia = 0
    distances = lp_distance(X, centroids, 2) ** 2
    for i in range (len(centroids)):
        inertia += np.sum(distances[i][np.where(classes==i)])

    return inertia

def main():
    # make matplotlib figures appear inline in the notebook
    get_ipython().run_line_magic('matplotlib', 'inline')
    plt.rcParams['figure.figsize'] = (14.0, 8.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    # load the image and confirm skimage is imported properly.
    image = io.imread('data/small_duck.jpg')
    io.imshow(image)
    io.show()
    print(image.shape)
    # save the dimensions of the image and reshape it for easier manipulation
    rows = image.shape[0]
    cols = image.shape[1]
    image = image.reshape(image.shape[0]*image.shape[1],3)
    print(image.shape)

    k = 7
    centroids = get_random_centroids(image, k)

    # The Minkowski distance is a metric which can be considered as a generalization of the Euclidean and Manhattan distances.
    distances = lp_distance(image, centroids, 5)
    centroids, classes = kmeans(image, k=4, p=1, calc_method = np.mean, max_iter=100)

    # We obtained the locally optimal centroids (colors) from our image. To reconstruct the compressed image, we need to specify the color for each pixel that our algorithm associated with some centroid.
    # The following cell does that for you.
    classes = classes.reshape(rows,cols)
    compressed_image = np.zeros((classes.shape[0],classes.shape[1],3),dtype=np.uint8 )
    for i in range(classes.shape[0]):
        for j in range(classes.shape[1]):
                compressed_image[i,j,:] = centroids[classes[i,j],:]
    io.imshow(compressed_image)
    io.show()


    # ## Hyper parameter tuning
    # Run the algorithm for each of the following values for `p = [1,2,3,4,5]` .
    # Test multiple values for `k=[4,8,16]` using two different cluster calculation criteria: the mean and the median of the cluser.
    # For each test, calculate the intertia and visualize it using graphs. Explain your results.

    '''
        as we can see in the results graphs, we tested 3 k's values- 4,8,16. Since its hard to guess the optimal k value,
        we have been looking for the one that represents "elbow". from the graphs its clearly k=8.
    '''
    p = [1,2,3,4,5]
    k = [4,8,16]

    for i in p:
        inertia_mean = []
        inertia_median = []
        for j in k:
            centroids, classes = kmeans(image, k=j, calc_method=np.mean, p=i)
            inertia_mean.append(calc_inertia(image,centroids, classes))
            centroids, classes = kmeans(image, k=j, calc_method=np.median, p=i)
            inertia_median.append(calc_inertia(image,centroids, classes))
        plt.plot(k, inertia_mean, "bo-", label="Inertia using mean")
        plt.plot(k, inertia_median, "ro-", label="Inertia using median")
        plt.title("Parameter tuning using p=" + str(i) + " and k = {4, 8, 16}")
        plt.xlabel("k", fontsize=14)
        plt.ylabel("Inertia", fontsize=14)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main();
