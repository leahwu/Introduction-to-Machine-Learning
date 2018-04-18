import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt



def read_data():
    path = "./assign2_dataset/"
    train_x = np.loadtxt(path+"page_blocks_train_feature.txt")
    train_y = np.loadtxt(path+"page_blocks_train_label.txt")
    test_x = np.loadtxt(path+"page_blocks_test_feature.txt")
    test_y = np.loadtxt(path+"page_blocks_test_label.txt")

    # normalization
    train_x = np.divide(train_x - train_x.mean(axis=0), train_x.std(axis=0))
    test_x = np.divide(test_x - test_x.mean(axis=0), test_x.std(axis=0))
    return train_x, train_y, test_x, test_y

def know_data(train_y, test_y, K):
    print("\n\nData Summary: \n")
    print("---------------------------------------- training set ----------------------------------------")
    m_train = train_y.shape[0]
    print("total ", m_train, " samples.")
    for k in np.arange(1, K+1):
        num = np.where(train_y == k)[0].shape[0]
        print("label {0}: number of samples = {1}, and the percentage is {2:.3f}".format(k, num, num/m_train))

    print("")
    print("---------------------------------------- test set ----------------------------------------")
    m_test = test_y.shape[0]
    print("total ", m_test, " samples.")
    for k in np.arange(1, K + 1):
        num = np.where(test_y == k)[0].shape[0]
        print("label {0}: number of samples = {1}, and the percentage is {2:.3f}".format(k, num, num / m_test))



def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_regression(x, y, learning_rate, max_iteration):
    """
    :param x: features, numpy array of shape(m, n) where m = # examples, and n = #features
    :param y: labels, numpy array of shape(m, ) value: 0 or 1
    :param learning_rate: learning rate
    :param max_iteration: maximum steps of iterations
    :return:
    """

    # SMOTE, note: note before adding intercepts
    x, y = smote(x, y, 5)

    # Initialize beta with zeros
    m, n = x.shape
    beta = np.zeros(n + 1, )
    error_norms = np.zeros((max_iteration,))

    # Add intercepts, i.e. append 1 to the end of every sample
    x = np.pad(x, ((0,0), (0,1)), mode='constant', constant_values=1)

    for step in np.arange(max_iteration):
        z = np.dot(x, beta)
        pred = sigmoid(z)
        error = y - pred
        error_norm = np.linalg.norm(error) / m
        error_norms[step] = error_norm
        # Calculate the gradient
        grad = -np.dot(x.T, error)


        # Update beta with the gradient
        beta -= learning_rate * grad

        # Print the log-likelihood, error, precision and recall every so often
        # expect to see the ascending log likelihood and the descending error
        if step % 1000 == 0:
            print("The {0}th Iteration -------------------------".format(step+1))
            print("Log Likelihood: ", log_lilkelihood(y, z))
            print("error:", error_norm)

            yhat = np.zeros(y.shape)
            yhat[pred >= 0.5] = 1
            pre, recall = precision_and_recall(yhat, y)
            print("precision = {0: .4f}, recall = {1: .4f}".format(pre, recall))

    plt.plot(np.arange(max_iteration), error_norms)
    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.show()

    return beta


def one_vs_rest(train_x, train_y, test_x, test_y, K, learning_rate=0.0001, num_steps=100000):
    """
    One V.S. Rest
    :param x: features, of shape (m, n)
    :param y: labels, of shape (m,), takes value in 1,2,3,...,K
    :param learning_rate: learning rate
    :param num_steps: maximum iterations steps
    :param K: number of claases for y
    :return:
    """

    m_test, _ = test_x.shape

    # predicted probabilities of y = 1, 2, ..., K
    pred_probs = np.zeros((m_test, K))

    # add interceps
    train_x_pad = np.pad(train_x, ((0, 0), (0, 1)), mode='constant', constant_values=1)
    test_x_pad = np.pad(test_x, ((0, 0), (0, 1)), mode='constant', constant_values=1)

    for k in np.arange(1, K + 1):
        # training

        print("\n---------------------------------------- Start training using the {0}th classifier ----------------------------------------".format(k))

        # chaning into 0-1 values based on y = k or not: y != k -- > 0; y = k -- > 1
        binary_train_y = train_y.copy()
        # NOTE: change to 0 first; otherwise, may cause confusion for next-steps of replacing values
        binary_train_y[binary_train_y != k] = 0
        binary_train_y[binary_train_y == k] = 1

        binary_test_y = test_y.copy()
        binary_test_y[binary_test_y != k] = 0
        binary_test_y[binary_test_y == k] = 1

        beta = logistic_regression(train_x, binary_train_y, learning_rate, num_steps)

        # Predict

        z = np.dot(test_x_pad, beta)
        pred_test = sigmoid(z)
        yhat_test = np.zeros(binary_test_y.shape)
        yhat_test[pred_test >= 0.5] = 1

        pred_probs[:,k-1] = pred_test

        print("\nFinish training\n")

        if (k == 4):
            xx =1
        # Compute the precision and the recall for the training set
        # Use original data / unsmoted ones
        z_in_train = np.dot(train_x_pad, beta)
        yhat_train = np.zeros(train_y.shape)
        pred_train = sigmoid(z_in_train)
        yhat_train[pred_train >= 0.5] = 1
        pre, recall = precision_and_recall(yhat_train, binary_train_y)
        print("On original training set (un-smoted):")
        print("precision = {0: .4f}, recall = {1: .4f}".format(pre, recall))
        err_train = pred_train - binary_train_y
        err_norm_train = np.linalg.norm(err_train) / train_y.shape[0]
        print("error = {0:.4f}\n\n".format(err_norm_train))


        print("On test set: ")
        pre, recall = precision_and_recall(yhat_test, binary_test_y)
        print("precision = {0: .4f}, recall = {1: .4f}".format(pre, recall))
        err_test = pred_test - binary_test_y
        err_norm_test = np.linalg.norm(err_test) / test_y.shape[0]
        print("error = {0:.4f}\n\n".format(err_norm_test))




    # Print training and testing summary:
    # test accuracy
    yhat_k = pred_probs.argmax(axis=1) + 1
    acc = accuracy(yhat_k, test_y, K)
    print("\n\n---------------------------------------------------------------------------------")
    print("Accuracy = {0:.4f}. \n\n".format(acc))



def precision_and_recall(yhat, y):
    """
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    :param yhat: prediction, takes value 0 or 1
    :param y: ground-truth, takes value 0 or 1
    :return: precison value
    """


    pp_idx = np.where(yhat == 1)[0] # predicted positive
    tp = (y[pp_idx] == 1).sum()
    predicted_positive = pp_idx.shape[0]

    pn_idx = np.where(yhat == 0)[0] # predicted negative
    fn = (y[pn_idx] == 1).sum()

    precision = tp / predicted_positive
    recall = tp / (tp + fn)

    return precision, recall


def accuracy(yhat, y, K):
    """
    accuracy = correct_guesses / number of total samples
    :param yhat: predictions, taking value in 1, ..., K
    :param y: gound_truth， taking value in 1, ..., K
    :param K: number of classes for y
    :return: accuracy
    """

    correct_guesses = 0
    for k in np.arange(1, K + 1):
        idx = np.where(yhat == k)[0]
        correct_guesses += (y[idx] == k).sum()

    all = y.shape[0]
    return correct_guesses / all

def smote(x, y, k):
    """
    SMOTE: Synthetic Minority Over-sampling TEchnique
    Original Paper ref: https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume16/chawla02a-html/node6.html
    :param x: features
    :param y: labels
    :param k: k-nearest neighbors
    :return: x, y after oversampling
    """
    # m: number of total samples, n: number of features
    m, n = x.shape

    index1 = np.where(y == 1)[0]
    index0 = np.where(y == 0)[0]

    # y0 = np.extract(y == 0, y)

    x1 = np.take(x, index1, axis=0)
    x0 = np.take(x, index0, axis=0)

    count1 = x1.shape[0]
    count0 = x0.shape[0]
    print("Before smoting, numbers of 2 classes: {0}, {1}".format(count0, count1))
    diff = count1 - count0

    if (diff > 0):
        syn_x0 = knn_and_populate(x0, k, diff)
        x = np.concatenate((x, syn_x0))
        y = np.concatenate((y, np.zeros(syn_x0.shape[0],)))

    else:
        syn_x1 = knn_and_populate(x1, k, -diff)
        x = np.concatenate((x, syn_x1))
        y = np.concatenate((y, np.ones(syn_x1.shape[0], )))

        print("After smoting, numbers of 2 classes: {0}, {1}".format(count0, syn_x1.shape[0]))
    return x, y


def knn_and_populate(xx, k, diff):
    """
    Apply knn first, and then populate (to generate the synthetic samples)
    :param xx: features-samples in the minority class
    :param k: k-nearest neighbors
    :param diff: positive value, difference between number of samples in the minority class and majority class
    :return: N*T synthetic samples. If N < 1, randomize the minority class samples as only a random percent of them will be SMOTEd
    See the definition of N below.
    """
    T, n = xx.shape

    x = xx.copy()
    if k > (T - 1):
        k = T - 1
    N = diff / T
    if N < 1:
        print("random shuffling")
        np.random.shuffle(x)
        T = diff
        N = 1

    N = int(N)

    np.random.seed(17)
    # Used for storing synthetic samples
    synthetic_samples = np.zeros((T*N, n))
    # index for synthetic samples
    new_index = 0
    # Used for storing knn
    nearest_x = np.zeros((T, k))

    # KNN: generate k-nearest neighbors for each sample in x
    for i in np.arange(0, T):
        # KNN, based on the implementation from ref:  https://www.jair.org/media/953/live-953-2037-jair.pdf
        # fixed a bug in the ref code
        dists = cdist(x, np.array((x[i],)))
        idx = np.argpartition(dists, k, axis=0)[1:k+1]

        # Populate: generate N synthetic values for each sample x[i] using x[i]'s k-nearest neighbors
        for n in np.arange(0, N):
            # Randomly select a neighbor among knn
            nn = np.random.randint(low=0, high=k)

            distance = x[idx[nn]] - x[i]
            gap = np.random.uniform(low=0, high=1)
            synthetic_samples[new_index] = x[i] + gap * distance
            new_index += 1

    return synthetic_samples


def log_lilkelihood(y, z):
    ll = np.sum(y * z - np.log(1 + np.exp(z)))
    return ll



# Test functions below


def test_logistic_regression():
    np.random.seed(12)
    num_observations = 5000

    x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], num_observations)

    simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
    simulated_labels = np.hstack((np.zeros(num_observations),
                                  np.ones(num_observations)))

    weights = logistic_regression(simulated_separableish_features, simulated_labels, learning_rate=5e-5, max_iteration=100000)
    return weights


def test_knn_and_populate():
    # 100 samples in the minority class
    # generate 200 more
    x = np.random.rand(100, 10)
    k = 5
    diff = 200
    synthetic_samples = knn_and_populate(x, k, diff)


def test_smote():
    # for x: m = 20, n = 5, for y: pos = 20, neg = 10
    np.random.seed(20)
    x = np.random.rand(30, 5)
    y = np.concatenate((np.zeros(10), np.ones(20)))
    np.random.shuffle(y)
    k = 5
    return smote(x, y, k)

def main():

    train_x, train_y, test_x, test_y = read_data()
    K=5
    know_data(train_y, test_y, K)
    print("\n\n\n")
    one_vs_rest(train_x, train_y, test_x, test_y, K, learning_rate=0.0005, num_steps=10000)


if __name__ == "__main__":
    main()

