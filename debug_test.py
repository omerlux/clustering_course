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
class UOFC:
    def __init__(self, n_clusters, init_method='random'):
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.prototypes = None
        self.max_iter = 1000

    def _init_prototypes(self, data):
        if self.init_method == 'random':
            # Initialize prototypes randomly from the data points
            prototypes = data[np.random.choice(data.shape[0], self.n_clusters, replace=False), :]
        elif self.init_method == 'kmeans':
            # Initialize prototypes using K-means
            # initiate the prototypes with k-mean from sklearn:
            prototypes = KMeans(n_clusters=self.n_clusters, init='random', max_iter=1, n_init=1).fit(data).cluster_centers_
            # prototypes = self._kmeans(data)
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")
        return prototypes

    def _F_calculate(self, X, prototypes=None, memberships=None):
        if prototypes is None:
            prototypes = self.prototypes
        if memberships is None:
            memberships = self._memberships(X, prototypes)
        # Compute F for each data point and each cluster
        # Reshape X, prototypes, and memberships to allow broadcasting
        X_reshaped = X[:, None, :]
        prototypes_reshaped = prototypes[None, :, :]
        memberships_reshaped = memberships[:, :, None, None]

        # Compute the difference and outer product for each pair of data point and prototype
        diff = (prototypes_reshaped - X_reshaped)[:, :, :, None]
        F_k = diff @ diff.transpose(0, 1, 3, 2)

        # Compute the weighted sum of F_k for each cluster
        F = np.sum(memberships_reshaped * F_k, axis=0)

        # Compute the sum of memberships for each cluster
        sum_dominator_k = memberships.sum(axis=0)

        # Normalize F by the sum of memberships
        F /= sum_dominator_k[:, None, None]

        return F

    def _distances(self, X, prototypes=None, memberships=None):
        if prototypes is None:
            prototypes = self.prototypes
        if memberships is None:
            memberships = self._memberships(X, prototypes)
        F = self._F_calculate(X, prototypes, memberships)
        distances = np.zeros((self.n_clusters, X.shape[0]))
        for k in range(self.n_clusters):
            a_k = self._memberships(X)[:, k].sum()
            for i in range(X.shape[0]):
                distances[k, i] = ((np.linalg.det(F[k]) ** (1/2)) / a_k) * np.exp(
                    (prototypes[k] - X[i])[:, None].T @ (F[k] ** (-1)) @ (prototypes[k] - X[i])[:, None] / 2
                )
        return distances

    def _memberships(self, data, prototypes=None):
        if prototypes is None:
            prototypes = self.prototypes

        # Compute memberships for each data point to each cluster
        # Reshape data and prototypes to allow broadcasting
        data_reshaped = data[:, None, :]
        prototypes_reshaped = prototypes[None, :, :]

        # Compute the distance for each data point to each prototype
        diff = data_reshaped - prototypes_reshaped
        memberships = np.einsum('ijk,ijk->ij', diff, diff)  # equivalent to summing over the last axis of diff**2

        # Apply the same transformations as in the original function
        m = np.clip(1 / memberships, a_min=-1, a_max=100000)
        m = m / m.sum(axis=1)[:, None]
        return m

    def fit(self, data, max_iter=1000):
        self.max_iter = max_iter
        # Initialize classification prototypes
        self.prototypes = self._init_prototypes(data)

        for i in tqdm(range(self.max_iter)):
            # Step 1: Cluster data using fuzzy K-means with current prototypes
            # Compute memberships for each data point to each cluster
            memberships = self._memberships(data)

            # Step 2: Refine prototypes using fuzzy maximum likelihood estimation
            new_prototypes = np.dot(memberships.T ** 2, data) / (memberships ** 2).sum(axis=0, keepdims=True).T

            # Check for convergence
            if np.allclose(self.prototypes, new_prototypes, atol=1e-2, rtol=1e-2):
                break

            self.prototypes = new_prototypes

    def predict(self, data):
        # Assign each data point to the cluster with the highest membership
        distances = self._distances(data)
        return np.argmax(distances, axis=0)

    def fuzzy_hypercube_criteria(self, X):
        F = self._F_calculate(X)
        hv = np.sqrt(np.linalg.det(F)).sum(axis=0)
        return hv

    def partition_density_criteria(self, X):
        memberships = self._memberships(X)
        pd = memberships.sum(axis=0).sum(axis=0) / self.fuzzy_hypercube_criteria(X)
        return pd

    def average_partition_density_central_criteria(self, X):
        F = self._F_calculate(X)
        memberships = self._memberships(X)
        apd = (memberships.sum(axis=0) / np.sqrt(np.linalg.det(F))).mean(axis=0)
        return apd

    def average_partition_density_max_criteria(self, X):
        F = self._F_calculate(X)
        memberships = self._memberships(X)
        membership_max = []
        for k in range(self.n_clusters):
            argmax_indices = np.argmax(memberships, axis=1) == k
            membership_max.append(memberships[argmax_indices][:, k].sum(axis=0))
        membership_max = np.array(membership_max)
        apd = (membership_max / np.sqrt(np.linalg.det(F))).mean(axis=0)
        return apd

    def normalized_partition_criteria(self, X):
        memberships = self._memberships(X)
        distances = self._distances(X)
        return (memberships ** 2 * distances.T).sum()

#%% md
### Fitting the model
#%%
train_predictions = {}
train_gt = {}

init_methods = ['random', 'kmeans']
datasets = ['lines']
num_of_groups = [4]

for dataset_name in datasets:
    print('=' * 50)
    print("Working on dataset: ", dataset_name)
    dataset = pd.read_csv(f"dataset_{dataset_name}.csv")
    X_train, y_train = split_dataset(dataset, test_size=0)

    train_predictions[dataset_name] = {method: {} for method in init_methods}
    train_gt[dataset_name] = y_train
    for init_method in init_methods:  # number of times to fit the model
        print('-'* 25)
        for n_groups in num_of_groups:  # number of clusters
            print("Fitting model with init_method: ", init_method, " and n_groups: ", n_groups)
            uofc = UOFC(n_clusters=n_groups, init_method=init_method)  # Initialize the UOFC
            uofc.fit(X_train)
            y_train_pred = uofc.predict(X_train)
            train_predictions[dataset_name][init_method][n_groups] = {'y_pred': y_train_pred, 'model': uofc}
#%% md
### Visualizing the results
#%%
for dataset_name in train_predictions.keys():
    init_methods = list(train_predictions[dataset_name].keys())
    dataset = pd.read_csv(f"dataset_{dataset_name}.csv")
    feat0 = dataset["feature_0"]
    feat1 = dataset["feature_1"]
    n_groups = list(train_predictions[dataset_name][init_methods[0]].keys())
    for method in init_methods:
        fig, axs = plt.subplots(1, 1 + len(n_groups), figsize=(22,4))
        axs[0].scatter(feat0, feat1, c=train_gt[dataset_name], cmap='viridis')
        axs[0].set_title("True labels")
        for i, n_group in enumerate(n_groups, 1):
            axs[i].scatter(feat0, feat1, c=train_predictions[dataset_name][method][n_group]['y_pred'], s=50, cmap='tab20')
            axs[i].set_title(f"Predicted labels\nInit {method} | {n_group} Clusters - ")
        fig.suptitle(f'Dataset {dataset_name}', y=1.1)
        plt.show()

#%% md
### Evaluating the results by 6 criteria
#%%
method = 'random'
dataset_name = 'random'
n_groups = 3

X = pd.read_csv(f"dataset_{dataset_name}.csv")
X = X[X.columns.drop('label')]
y_pred = train_predictions[dataset_name][method][n_groups]['y_pred']
uofc = train_predictions[dataset_name][method][n_groups]['model']

#%%
X.shape
#%%
y_pred.shape
#%%
prototypes = uofc.prototypes
prototypes.shape
#%%
memberships = uofc._memberships(X.values)
memberships.shape
#%%
F = uofc._F_calculate(X.values)
F
#%%
distances = np.zeros((n_groups, X.shape[0]))
for k in range(n_groups):
    a_k = uofc._memberships(X.values)[:, k].sum()
    for i in range(X.shape[0]):
        distances[k, i] = ((np.linalg.det(F[k]) ** (1/2)) / a_k) * np.exp(
            (prototypes[k] - X.values[i])[:, None].T @ (F[k] ** (-1)) @ (prototypes[k] - X.values[i])[:, None] / 2
        )
#%%
distances.argmax(axis=0)
#%% md

#%% md

# #%%
# def fuzzy_hypercube_criteria(X, y_pred):
#     K = np.unique(y_pred).shape[0]  # Number of clusters
#     V_HV = 0  # Initialize hypervolume
#     for k in range(K):
#         # Select data points in kth cluster
#         X_k = X[y_pred == k]
#         # Compute covariance matrix
#         F_k = np.cov(X_k, rowvar=False)
#         # Compute hypervolume of kth cluster
#         h_k = np.sqrt(np.linalg.det(F_k))
#         # Add to total hypervolume
#         V_HV += h_k
#     return V_HV
#
# def partition_density_criteria(X, y_pred, memberships):
#     K = np.unique(y_pred).shape[0]  # Number of clusters
#     V_PD = 0  # Initialize partition density
#     h_sum = 0  # Initialize sum of hypervolumes
#     for k in range(K):
#         # Select data points in kth cluster
#         X_k = X[y_pred == k]
#         # Compute covariance matrix and its inverse
#         F_k = np.cov(X_k, rowvar=False)
#         G_k = np.linalg.inv(F_k)
#         # Compute centroid of kth cluster
#         p_k = X_k.mean(axis=0)
#         # Compute memberships of "central members"
#         I_k = ((X_k - p_k) @ G_k * (p_k - X_k)).sum(axis=1) < 1
#         C_k = memberships[y_pred == k][I_k].sum()
#         # Compute hypervolume of kth cluster
#         h_k = np.sqrt(np.linalg.det(F_k))
#         # Add to total partition density and sum of hypervolumes
#         V_PD += C_k
#         h_sum += h_k
#     return V_PD / h_sum
#
# def average_partition_density_central_members_criteria(X, y_pred, memberships):
#     K = np.unique(y_pred).shape[0]  # Number of clusters
#     V_AD = 0  # Initialize average partition density
#     for k in range(K):
#         # Select data points in kth cluster
#         X_k = X[y_pred == k]
#         # Compute covariance matrix and its inverse
#         F_k = np.cov(X_k, rowvar=False)
#         G_k = np.linalg.inv(F_k)
#         # Compute centroid of kth cluster
#         p_k = X_k.mean(axis=0)
#         # Compute memberships of "central members"
#         I_k = ((X_k - p_k) @ G_k * (p_k - X_k)).sum(axis=1) < 1
#         C_k = memberships[y_pred == k][I_k].sum()
#         # Compute hypervolume of kth cluster
#         h_k = np.sqrt(np.linalg.det(F_k))
#         # Add to total average partition density
#         V_AD += C_k / h_k
#     return V_AD / K

# def average_partition_density_maximal_members_criteria(X, y, y_pred):
# #
# # def normalized_by_k_partition_indexes_criteria(X, y, y_pred):
# #
# # def invariant_criteria(X, y, y_pred):
# #%% md
# ### Evaluating the results by 6 criteria
# #%%
# def fuzzy_hypercube_criteria(X, y_pred):
#     K = np.unique(y_pred).shape[0]  # Number of clusters
#     V_HV = 0  # Initialize hypervolume
#     for k in range(K):
#         # Select data points in kth cluster
#         X_k = X[y_pred == k]
#         # Compute covariance matrix
#         F_k = np.cov(X_k, rowvar=False)
#         # Compute hypervolume of kth cluster
#         h_k = np.sqrt(np.linalg.det(F_k))
#         # Add to total hypervolume
#         V_HV += h_k
#     return V_HV
#
# def partition_density_criteria(X, y_pred, memberships):
#     K = np.unique(y_pred).shape[0]  # Number of clusters
#     V_PD = 0  # Initialize partition density
#     h_sum = 0  # Initialize sum of hypervolumes
#     for k in range(K):
#         # Select data points in kth cluster
#         X_k = X[y_pred == k]
#         # Compute covariance matrix and its inverse
#         F_k = np.cov(X_k, rowvar=False)
#         G_k = np.linalg.inv(F_k)
#         # Compute centroid of kth cluster
#         p_k = X_k.mean(axis=0)
#         # Compute memberships of "central members"
#         I_k = ((X_k - p_k) @ G_k * (p_k - X_k)).sum(axis=1) < 1
#         C_k = memberships[y_pred == k][I_k].sum()
#         # Compute hypervolume of kth cluster
#         h_k = np.sqrt(np.linalg.det(F_k))
#         # Add to total partition density and sum of hypervolumes
#         V_PD += C_k
#         h_sum += h_k
#     return V_PD / h_sum
#
# def average_partition_density_central_members_criteria(X, y_pred, memberships):
#     K = np.unique(y_pred).shape[0]  # Number of clusters
#     V_AD = 0  # Initialize average partition density
#     for k in range(K):
#         # Select data points in kth cluster
#         X_k = X[y_pred == k]
#         # Compute covariance matrix and its inverse
#         F_k = np.cov(X_k, rowvar=False)
#         G_k = np.linalg.inv(F_k)
#         # Compute centroid of kth cluster
#         p_k = X_k.mean(axis=0)
#         # Compute memberships of "central members"
#         I_k = ((X_k - p_k) @ G_k * (p_k - X_k)).sum(axis=1) < 1
#         C_k = memberships[y_pred == k][I_k].sum()
#         # Compute hypervolume of kth cluster
#         h_k = np.sqrt(np.linalg.det(F_k))
#         # Add to total average partition density
#         V_AD += C_k / h_k
#     return V_AD / K
#
# def average_partition_density_maximal_members_criteria(X, y, y_pred):
#
# def normalized_by_k_partition_indexes_criteria(X, y, y_pred):
#
# def invariant_criteria(X, y, y_pred):