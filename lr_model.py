import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from utils import result_to_df

def normalize_network(network):
    min_val = np.min(network)
    max_val = np.max(network)
    network = (network - min_val) / (max_val - min_val)
    return network

def binarize_network(network, threshold=0.5):
    return (network > threshold).astype(int)

def lr_estimated_params(X):
    inferred_network = np.zeros((X.shape[1], X.shape[1]))
    for i, gene in enumerate(X.columns):
        X_other_genes = X.drop(columns=[gene])
        model = LinearRegression()
        model.fit(X_other_genes, X[gene])
        coefficients = model.coef_
        inferred_network[i, :] = np.insert(coefficients, i, 0)
    # normalize the network into  (0, 1) range, here the range includes the negative values
    normalized_network = normalize_network(inferred_network)
    # using the binarize function to convert the network into binary format, with the threshold as the same as diagonal values
    # binary_network = binarize_network(inferred_network, threshold=np.diag(inferred_network))
    
    binary_network = result_to_df(normalized_network)

    return binary_network
