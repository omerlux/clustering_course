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
import copy


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
    def __init__(self, X, P, K):
        self.M = len(X)         # number of data points
        self.u = np.ones((self.M, K))    # membership matrix
        self.K = K              # number of prototypes = CLUSTERS
        self.X = X
        self.P = P.copy()
        self.q = 2

    def find_prototypes(self, max_iter=100):
        # X is the data matrix
        # K is the number of prototypes
        # P is the previous centers (prototypes) matrix - length K
        eps = 0.01

        for iter in tqdm(range(max_iter)):
            self.iter = iter
            u_prev = self.u.copy()

            # calculate membership matrix u
            for k in range(self.K):
                inverse_d_K_len = self._inverse_distances(k)
                self.u[:, k] = inverse_d_K_len[:, k] / inverse_d_K_len.sum(axis=-1)
                # for i in range(self.M):
                #     self.u[i, k] = self._membership(self.X[i], self.P, k)

            self.P = np.array([np.sum((self.u[:, k] ** 2)[:, None] * self.X, axis=0)
                               / np.sum(self.u[:, k] ** 2) for k in range(self.K)])

            # break if the condition is met
            if not self.cond(self.u, u_prev, eps):
                break

        self.D = self.distances(self.X, self.P)

        return self.P, self.u, self.D

    def _inverse_distances(self, k):
        if self.iter == 0 and k > 0 and k == self.K - 1:
            distances_k_len = np.ones((len(self.X), self.K)) * (10 * np.sum(np.var(self.X.T, axis=0, ddof=1)))      # (10 * np.trace(np.cov(self.X)))
        else:
            distances_k_len = self.distances(self.X, self.P)    # euclidian distance
        inverse_d_k_len = np.where(distances_k_len, (1e-7 + distances_k_len) ** (1 / (1 - self.q)), 0)
        return inverse_d_k_len

    def distances(self, X, P):
        return np.sum((P[None, :, :] - X[:, None, :]) ** 2, axis=-1)


    def cond(self, u, u_prev, eps=0.01):
        if u_prev is None:
            return True
        else:
            return np.linalg.norm(u - u_prev) > eps


class WUOFC:
    def __init__(self, ):
        self.evaluation_results = dict()
        self.clustering_results = dict()
        self.models = dict()
        self.q = 2

    def fit(self, X, max_K):
        P_init = X.mean(axis=0)[None, :]
        for K in range(1, max_K+1):
            print(f'Cluster size - K = {K}')
            # init wfkm with K clusters
            wfkm = WFKM(X, P_init, K)
            # find prototypes
            P, U, D = wfkm.find_prototypes(max_iter=100)

            # calculate metrics
            self.evaluation_results[K] = self.metrics(X, P, U, D, K)
            self.clustering_results[K] = self.predict(U)
            self.models[K] = wfkm

            # creating the new center
            P_init = np.concatenate((P_init,
                                     np.array(X[np.unravel_index(U.argmin(), U.shape)[0]])[None, :]))

        return self.evaluation_results, self.clustering_results

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

    def _central_members_Ik(self, X, Pk, Fk):
        I = []
        Gk = np.linalg.inv(Fk)
        for i in range(X.shape[0]):
            sample_i_valid = True
            for j in range(X.shape[1]):
                cond = (Pk - X[i]) @ Gk[j] * (Pk[j] - X[i, j]) < 1
                if not cond:
                    sample_i_valid = False
                    break
            if sample_i_valid:
                I.append(i)
        return I

    def _maximal_members_Jk(self, U, k):
        J = []
        for i in range(len(U)):
            if U[i, k] == np.max(U[i, :]):
                J.append(i)
        return J

    def _Ck_calculate(self, X, Uk, Pk, Fk):
        Ik = self._central_members_Ik(X, Pk, Fk)
        return np.take(Uk, Ik).sum()

    def _Mk_calculate(self, U, k):
        Jk = self._maximal_members_Jk(U, k)
        return np.take(U[:, k], Jk).sum()

    def _invariant_calculate(self, X, P, U, F, K):
        S_w = np.sum(F, axis=0)
        mu = np.mean(X, axis=0)
        clusters = np.argmax(U, axis=1)
        S_b = sum([list(clusters).count(k) * (P[k] - mu).reshape(-1, 1) @ (P[k] - mu).reshape(-1, 1).T for k in range(K)])
        return np.trace(np.linalg.inv(S_w) @ S_b)

    def metrics(self,X, P, U, D, K):
        print('Calculating metrics...')
        F = self._F_calculate(U, X, P)
        # calculating C_k with the group of central members of each cluster
        C_k = [self._Ck_calculate(X, U[:, k], P[k], F[k]) for k in range(K)]
        # calculating M_k with the group of maximal members of each cluster
        M_k = [self._Mk_calculate(U, k) for k in range(K)]
        # calculating H_k - the hypervolume of each cluster
        H_k = [np.linalg.det(F[k]) ** .5 for k in range(K)]
        # hyper volume criteria
        hypervolume = np.sum(H_k)
        # partition density criteria
        partition_density = np.sum(C_k) / hypervolume
        # average partition density central criteria
        avg_pd_central = np.divide(C_k, H_k).mean()
        # average partition density max criteria
        avg_pd_max = np.divide(M_k, H_k).mean()
        # normalized partition criteria
        normalized_partition = np.einsum('mk,mk->k', D, U ** self.q).sum() * K
        # invariant criteria
        invariant = self._invariant_calculate(X, P, U, F, K)
        return {
            'hv': hypervolume,
            'pd': partition_density,
            'apd_central': avg_pd_central,
            'apd_max': avg_pd_max,
            'np': normalized_partition,
            'inv': invariant
        }


train_predictions = {}
train_gt = {}

datasets = ['lines'] # ['random', 'triangle', 'square', 'lines', 'star']
max_clusters = 3

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

    train_predictions[dataset_name]['metrics'] = metrics
    train_predictions[dataset_name]['pred'] = clustering
    train_predictions[dataset_name]['model'] = woufc


for dataset_name in train_predictions.keys():
    init_methods = list(train_predictions[dataset_name].keys())
    dataset = pd.read_csv(f"dataset_{dataset_name}.csv")
    feat0 = dataset["feature_0"]
    feat1 = dataset["feature_1"]
    n_clusters = [k for k in train_predictions[dataset_name]['pred'].keys() if type(k) == int]
    fig, axs = plt.subplots(1, 1 + len(n_clusters), figsize=(22,4))
    axs[0].scatter(feat0, feat1, c=train_gt[dataset_name], cmap='viridis')
    axs[0].set_title("True labels")
    for i, n_cluster in enumerate(n_clusters, 1):
        axs[i].scatter(feat0, feat1, c=train_predictions[dataset_name]['pred'][n_cluster], s=50, cmap='tab20')
        axs[i].set_title(f"Predicted labels\n{n_cluster} Clusters - ")
    fig.suptitle(f'Dataset {dataset_name}', y=1.1)
    plt.show()

