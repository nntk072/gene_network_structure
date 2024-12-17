import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from utils import result_to_df

# Define the SDE model function


def sde_minimize_model(y, t, params):
    dydt = np.dot(params, y)
    return dydt

# Define the objective function for optimization


def objective_function(params, gene_df, y0, t, model, noise_std):
    params = params.reshape((len(y0), len(y0)))
    sol = odeint(model, y0, t, args=(params,))
    # Clip the solution to avoid overflow issues
    sol = np.clip(sol, -1e10, 1e10)
    
    # Ensure no NaNs or infinities in the solution
    sol = np.nan_to_num(sol)
    sol += np.random.normal(0, noise_std, sol.shape)

    error = np.sum((gene_df.values - sol)**2)
    return error

# Define the function to estimate the network structure using SDE model


def sdes_minimize_estimated_params(gene_df, y0, t, noise_std):
    # Initialize the parameters matrix with random values
    np.random.seed(42)
    initial_params = np.random.rand(len(y0), len(y0)).flatten()

    # Optimize the parameters to minimize the objective function
    result = minimize(objective_function, initial_params, args=(
        gene_df, y0, t, sde_minimize_model, noise_std), method='trust-constr')
    # Reshape the optimized parameters to a matrix
    optimized_params = result.x.reshape((len(y0), len(y0)))

    # Normalize the parameters in the range from 0 to 1
    optimized_params = (optimized_params - optimized_params.min()) / (optimized_params.max() - optimized_params.min())
    # Convert to 0 or 1, based on the threshold of the diagonal
    # optimized_params = (optimized_params > np.diag(optimized_params)).astype(int)
    # Convert the optimized parameters to a DataFrame
    optimized_params = result_to_df(optimized_params)

    return optimized_params
