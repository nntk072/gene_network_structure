import numpy as np
from scipy.integrate import odeint
from utils import result_to_df

# Define the SDE model function
def sde_model(y, t, params):
    dydt = np.dot(params, y)
    return dydt

# Define the function to estimate the network structure using SDE model
def sdes_estimated_params(gene_df, y0, t, noise_std):
    # Initialize the parameters matrix with random values
    params = np.random.rand(len(y0), len(y0))

    # Integrate the SDE model
    sol = odeint(sde_model, y0, t, args=(params,))

    # Add noise to the solution
    sol += np.random.normal(0, noise_std, sol.shape)

    # Estimate the network structure by fitting the parameters to the data
    for i in range(len(y0)):
        for j in range(len(y0)):
            A = np.vstack([sol[:, j], np.ones(len(sol[:, j]))]).T
            params[i, j], _ = np.linalg.lstsq(A, gene_df.iloc[:, i], rcond=None)[0]

    # Normalize the network structure from 0 to 1
    params = (params - params.min()) / (params.max() - params.min())

    # Convert to 0 or 1, based on the threshold of the diagonal
    # params = (params > np.diag(params)).astype(int)

    # Convert the estimated parameters to a DataFrame
    params = result_to_df(params)
    return params
