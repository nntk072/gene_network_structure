import numpy as np
from scipy.integrate import odeint
from utils import result_to_df

# Define the SDE model function


def ode_model(y, t, params):
    dydt = np.dot(params, y)
    return dydt

# Define the function to estimate the network structure using ODE model
def odes_estimated_params(gene_df, y0, t):
    # Initialize the parameters matrix with random values
    params = np.random.rand(len(y0), len(y0))

    # Integrate the ODE model
    sol = odeint(ode_model, y0, t, args=(params,))

    # Estimate the network structure by fitting the parameters to the data
    for i in range(len(y0)):
        for j in range(len(y0)):
            A = np.vstack([sol[:, j], np.ones(len(sol[:, j]))]).T
            params[i, j], _ = np.linalg.lstsq(A, gene_df.iloc[:, i], rcond=None)[0]

    # Convert to 0 or 1, based on a threshold
    # threshold = 0.5
    # params = (params > threshold).astype(int)
    # normalize to 0 and 1
    params = (params - params.min()) / (params.max() - params.min())
    # Convert the estimated parameters to a DataFrame
    params_df = result_to_df(params)
    return params_df