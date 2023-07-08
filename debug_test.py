#%%
# importing
import numpy as np
import pandas as pd
from sklearn.datasets import make_spd_matrix
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans



#%% md
# Assignment #0
#%%
# create data set of different number of gaussian groups in different dimension with different means and different covariance matrices. each one should have its own samples and labels

def generate_gaussian_groups(n_groups, n_samples, dim, priors=None, means=None, covs=None):
    if priors is None:
        priors = np.random.dirichlet(np.ones(n_groups))  # Generate random priors if not specified
    assert len(priors) == n_groups, "Number of priors must match the number of groups"
    assert np.isclose(sum(priors), 1), "Priors must sum to 1"

    datasets = []
    for i in range(n_groups):
        if means is not None:
            assert len(means) == n_groups, "Number of means must match the number of groups"
            assert len(means[i]) == dim, "Dimension of means must match the dimension of the data"
            mean = means[i]
        else:
            mean = np.random.uniform(-10, 10, dim)
        if covs is not None:
            assert len(covs) == n_groups, "Number of covariances must match the number of groups"
            assert covs[i].shape == (dim, dim), "Dimension of covariance must match the dimension of the data"
            cov = covs[i]
        else:
            cov = make_spd_matrix(dim)
        samples = int(priors[i] * n_samples)
        data = np.random.multivariate_normal(mean, cov, samples)
        labels = np.full(samples, i)
        df = pd.DataFrame(data, columns=[f'feature_{j}' for j in range(dim)])
        df['label'] = labels
        datasets.append(df)
    # reorder the dataframe with random order
    datasets = pd.concat(datasets).sample(frac=1).reset_index(drop=True) # shuffle rows
    return datasets

def plot_gaussian_groups(datasets):
    fig = plt.figure(figsize=(10,7))
    if len(datasets.shape) == 2:
        ax = fig.add_subplot()
        ax.scatter(datasets['feature_0'], datasets['feature_1'], c=datasets['label'], s=50, cmap='viridis')
        ax.set_xlabel('feature_0')
        ax.set_ylabel('feature_1')
    else:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(datasets['feature_0'], datasets['feature_1'], datasets['feature_2'], c=datasets['label'], s=50, cmap='viridis')
        ax.set_xlabel('feature_0')
        ax.set_ylabel('feature_1')
        ax.set_zlabel('feature_2')
    fig.suptitle('Gaussian Groups (only first features)')
    plt.show()

#%% md
# Assignment #1
#%% md
### Creating the algorithm of EM for GMM
#%%
class GMM:
    def __init__(self, n_components, n_init=1, init_method='random'):
        self.K = n_components  # Number of Gaussian components
        self.n_init = n_init  # Number of times the algorithm will be run with different initializations
        self.init_method = init_method


    def initialize(self, X, i=1, ):
        n_samples, n_features = X.shape

        if self.init_method == 'kmeans':
            # Initialize means using k-means
            kmeans = KMeans(n_clusters=self.K, random_state=0).fit(dataset[dataset.columns.drop('label')])
            self.mu = kmeans.cluster_centers_
            print(f"Means are initialized with K-Means.")
        elif self.init_method == 'random':
            # Initialize means by randomly choosing data points
            indices = np.random.choice(n_samples, size=self.K, replace=False)
            self.mu = X[indices]
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")

        # Initialize covariances to be identity matrices
        self.Sigma = np.stack([np.eye(n_features) for _ in range(self.K)])

        # Initialize priors to be uniform probabilities
        self.weights = np.full(self.K, 1 / self.K)

        print(f"Intitialization #{i}: Initialized means, covariances and weights")

    def e_step(self, X):
        n_samples = X.shape[0]

        # Compute the likelihood
        likelihood = np.zeros((n_samples, self.K))
        for i in range(self.K):
            likelihood[:, i] = multivariate_normal.pdf(X, mean=self.mu[i], cov=self.Sigma[i])

        # Compute the responsibilities using Bayes' rule
        numerator = likelihood * self.weights
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        gamma = numerator / denominator          # responsibility = prior * likelihood / evidence

        return gamma

    def m_step(self, X, gamma):
        n_samples = X.shape[0]

        # Compute the total responsibility assigned to each component
        Nk = gamma.sum(axis=0)

        # Update the means
        self.mu = np.dot(gamma.T, X) / Nk[:, np.newaxis]

        # Update the covariances
        for i in range(self.K):
            diff = X - self.mu[i]
            self.Sigma[i] = np.dot(gamma[:, i] * diff.T, diff) / Nk[i]

        # Update the weights
        self.weights = Nk / n_samples

    def fit(self, X, max_iter=1000):
        for init in range(1, 1+self.n_init):
            self.initialize(X, i=init)
            log_likelihood_old = None

            for i in tqdm(range(max_iter)):  # Maximum of 100 iterations
                # E-step
                gamma = self.e_step(X)

                # M-step
                self.m_step(X, gamma)

                # Compute the log-likelihood
                log_likelihood_new = np.sum(gamma * (np.log(self.weights) + self.log_likelihood(X)))
                if log_likelihood_old is not None and abs(log_likelihood_new - log_likelihood_old) < 1e-3:
                    print("Converged after {} iterations".format(i+1))
                    break
                log_likelihood_old = log_likelihood_new

    def log_likelihood(self, X):
        n_samples = X.shape[0]
        log_likelihood = np.zeros((n_samples, self.K))

        for i in range(self.K):
            log_likelihood[:, i] = multivariate_normal.logpdf(X, mean=self.mu[i], cov=self.Sigma[i])

        return log_likelihood

    def predict(self, X):
        # Perform the E-step with the learned parameters
        gamma = self.e_step(X)

        # Assign each data point to the component that gives it the highest responsibility
        labels = gamma.argmax(axis=1)

        return labels

#%% md
### Creating the dataset
#%%
def split_dataset(dataset, test_size=0.2):
    # Split the dataset into training and testing sets
    if test_size:
        X_train, X_test, y_train, y_test = train_test_split(dataset.drop('label', axis=1), dataset['label'], test_size=test_size, random_state=42)
        return X_train.values, y_train.values, X_test.values, y_test.values
    else:
        X_train = dataset[dataset.columns.drop('label')]
        y_train = dataset['label']
        return X_train.values, y_train.values

#%% md
# Assignment #2
#%%
class WFKM:
    def __init__(self, X, P, K, max_K):
        self.M = len(X)         # number of data points
        self.u = np.zeros((self.M, K))    # membership matrix
        self.K = K              # number of prototypes = CLUSTERS
        self.max_K = max_K
        self.X = X
        self.P = P.copy()

    def find_prototypes(self, max_iter=100):
        # X is the data matrix
        # K is the number of prototypes
        # P is the previous centers (prototypes) matrix - length K
        iter = 0
        u_prev = None
        eps = 0.01

        for iter in tqdm(range(max_iter)):
            self.iter = iter
            u_prev = self.u.copy()

            # calculate membership matrix u
            self.u = self._memberships(self.X, self.P)

            # # calculate membership matrix u
            # for k in range(self.K):
            #     for i in range(self.M):
            #         self.u[k, i] = self._memberships(self.X[i], self.P, k)

            self.P = np.array([np.sum((self.u[:, k] ** 2)[:, None] * self.X, axis=0)
                               / np.sum(self.u[:, k] ** 2) for k in range(self.K)])

            # break if the condition is met
            if not self.cond(self.u, u_prev, eps):
                break

        self.D = self.distances(self.X, self.P)

        return self.P, self.u, self.D

    # def _memberships(self, _x_i, _P, k):
    #     d_inverse = self._inverse_distance(_x_i, _P[k], k)
    #     u_k_i = d_inverse / np.sum([self._inverse_distance(_x_i, _P[j], k) for j in range(len(_P))])
    #     return u_k_i
    def _memberships(self, X, P):
        # Compute inverse distances for all pairs of data points and prototypes
        d_inverse = self._inverse_distance(X, P)

        # Compute memberships for each data point to each prototype
        u = d_inverse / np.sum(d_inverse, axis=1)[:, None]

        return u

    # def _inverse_distance(self, _x_i, _p_k, k):
    #     if self.iter == 0 or self.K == self.max_K:
    #         d = 10 * np.trace(np.cov(self.X))
    #     else:
    #         d = (_p_k - _x_i).T @ (_p_k - _x_i)
    #     # exponential?
    #     return 1 / d
    def _inverse_distance(self, X, P):
        if self.iter == 0 and self.K == self.max_K:
            d = 10 * np.trace(np.cov(self.X)) * np.ones((len(X), len(P)))
        else:
            # Compute squared Euclidean distances for all pairs of data points and prototypes
            diff = X[:, None, :] - P[None, :, :] # M x K x F
            d = np.sum(diff ** 2, axis=2)   # sum on the axis of features
        return 1 / (d + 1e-8)

    # def distances(self, X, P):
    #     return (P - X) @ (P - X).T
    def distances(self, X, P):
        return np.sum((P[None, :, :] - X[:, None, :]) ** 2, axis=-1)


    def cond(self, u, u_prev, eps=0.01):
        if u_prev is None:
            return True
        else:
            return np.linalg.norm(u - u_prev) > eps


class WUOFC:
    def __init__(self, ):
        pass

    def fit(self, X, max_K):
        metrics = {}
        clustering = {}

        P_init = X.mean(axis=0)[None, :]
        for K in range(1, max_K+1):
            print(f'Cluster size - K = {K}')
            # init wfkm with K clusters
            wfkm = WFKM(X, P_init, K, max_K=max_K)
            # find prototypes
            P, U, D = wfkm.find_prototypes(max_iter=100)

            # calculate metrics
            metrics[K] = self.metrics(X, P, U, D, K)
            clustering[K] = self.predict(U)

            # creating the new center
            P_init = np.concatenate((P_init,
                                     np.array(X[np.unravel_index(U.argmin(), U.shape)[1]])[None, :]))

        return metrics, clustering

    def predict(self, U):
        return np.argmax(U, axis=1)

    def _F_calculate(self, U, X, P):
        # Compute F for each data point and each cluster
        # Reshape X, prototypes, and memberships to allow broadcasting
        X_reshaped = X[:, None, :]
        P_reshaped = P[None, :, :]
        U_reshaped = U[:, :, None, None]

        # Compute the difference and outer product for each pair of data point and prototype
        diff = (P_reshaped - X_reshaped)[:, :, :, None]
        F_k = diff @ diff.transpose(0, 1, 3, 2)

        # Compute the weighted sum of F_k for each cluster
        F = np.sum(U_reshaped * F_k, axis=0)

        # Compute the sum of memberships for each cluster
        sum_dominator_k = U.sum(axis=0)

        # Normalize F by the sum of memberships
        F /= sum_dominator_k[:, None, None]

        return F

    def hv(self, F):
        # fuzzy_hypercube_criteria
        return np.sqrt(np.linalg.det(F)).sum(axis=0)

    def pd(self, F, U):
        # partition_density_criteria
        return U.sum(axis=0).sum(axis=0) / self.hv(F)

    def apd_central(self, F, U):
        # average_partition_density_central_criteria
        return (U.sum(axis=0) / np.sqrt(np.linalg.det(F))).mean(axis=0)

    def apd_max(self, F, U, K):
        # average_partition_density_max_criteria
        U_max = []
        for k in range(K):
            argmax_indices = np.argmax(U, axis=1) == k
            U_max.append(U[argmax_indices][:, k].sum(axis=0))
        U_max = np.array(U_max)
        return (U_max / np.sqrt(np.linalg.det(F))).mean(axis=0)

    def np(self, U, D, K):
        # normalized_partition_criteria
        # return (U ** 2 * D.T).sum() * K       # TODO: check it - i've changed it
        return np.einsum('mk,mk->k', D ** 2, U).sum() * K

    def metrics(self,X, P, U, D, K):
        F = self._F_calculate(U, X, P)
        return {
            'hv': self.hv(F),
            'pd': self.pd(F, U),
            'apd_central': self.apd_central(F, U),
            'apd_max': self.apd_max(F, U, K),
            'np': self.np(U, D, K)
        }


train_predictions = {}
train_gt = {}

datasets = ['lines'] # ['random', 'triangle', 'square', 'lines', 'star']
max_clusters = 7

for dataset_name in datasets:
    print('=' * 50)
    print("Working on dataset: ", dataset_name)
    dataset = pd.read_csv(f"dataset_{dataset_name}.csv")
    X_train, y_train = split_dataset(dataset, test_size=0)

    train_predictions[dataset_name] = {}
    train_gt[dataset_name] = y_train

    print('Fitting the model with max clusters of ', max_clusters)
    woufc = WUOFC()   # Initialize the WOUFC
    metrics, clustering = woufc.fit(X_train, max_K=max_clusters)

    # y_train_pred = uofc.predict(X_train)
    # train_predictions[dataset_name][init_method][n_groups] = {'y_pred': y_train_pred, 'model': uofc}
