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


def split_dataset(dataset, test_size=0.2):
    # Split the dataset into training and testing sets
    if test_size:
        X_train, X_test, y_train, y_test = train_test_split(dataset.drop('label', axis=1), dataset['label'], test_size=test_size, random_state=42)
        return X_train.values, y_train.values, X_test.values, y_test.values
    else:
        X_train = dataset[dataset.columns.drop('label')]
        y_train = dataset['label']
        return X_train.values, y_train.values


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
            prototypes = self._kmeans(data)
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")
        return prototypes

    def _kmeans(self, data):
        # Initialize prototypes randomly from the data points
        prototypes = data[np.random.choice(data.shape[0], self.n_clusters, replace=False), :]

        for i in tqdm(range(self.max_iter)):
            # Compute distances from data points to prototypes
            distances = self._cdist(data, prototypes)

            # Assign each data point to the closest prototype
            labels = np.argmin(distances, axis=1)

            # Compute new prototypes as the mean of the data points assigned to each prototype
            new_prototypes = np.array([data[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            # Check for convergence
            if np.allclose(prototypes, new_prototypes, atol=1e-2, rtol=1e-2):
                break

            prototypes = new_prototypes

        return prototypes

    def _cdist(self, X, Y):
        # Compute pairwise distances between rows in X and Y
        return np.sqrt(((X[:, np.newaxis] - Y) ** 2).sum(axis=2))

    def fit(self, data, max_iter=1000):
        self.max_iter = max_iter
        # Initialize classification prototypes
        self.prototypes = self._init_prototypes(data)

        for i in tqdm(range(self.max_iter)):
            # Step 1: Cluster data using fuzzy K-means with current prototypes
            # Compute memberships for each data point to each cluster
            distances = self._cdist(data, self.prototypes)
            memberships = np.clip(1 / distances, a_min=-1, a_max=10000)
            memberships = memberships / memberships.sum(axis=1)[:, None]

            # Step 2: Refine prototypes using fuzzy maximum likelihood estimation
            new_prototypes = np.dot(memberships.T, data) / memberships.sum(axis=0, keepdims=True).T

            # Check for convergence
            if np.allclose(self.prototypes, new_prototypes, atol=1e-2, rtol=1e-2):
                break

            self.prototypes = new_prototypes

    def predict(self, data):
        # Assign each data point to the cluster with the highest membership
        distances = self._cdist(data, self.prototypes)
        memberships = 1 / distances
        memberships = memberships / (1e-8 + memberships.sum(axis=1, keepdims=True))
        return np.argmax(memberships, axis=1)




train_predictions = {}
train_gt = {}

init_methods = ['random', 'kmeans']
datasets = ['random', 'triangle', 'square', 'lines', 'star']

for dataset_name in datasets[:1]:
    print('=' * 50)
    print("Working on dataset: ", dataset_name)
    dataset = pd.read_csv(f"dataset_{dataset_name}.csv")
    X_train, y_train = split_dataset(dataset, test_size=0)
    n_groups = len(set(y_train))
    train_predictions[dataset_name] = {}
    train_gt[dataset_name] = y_train
    for init_method in init_methods:  # number of times to fit the model
        uofc = UOFC(n_clusters=n_groups, init_method=init_method)  # Initialize the UOFC
        uofc.fit(X_train)
        y_train_pred = uofc.predict(X_train)
        train_predictions[dataset_name][init_method] = y_train_pred