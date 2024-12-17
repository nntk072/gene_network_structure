import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import pandas as pd

# Define the ODE model function
def ode_model(y, t, params):
    dydt = np.dot(params, y)
    return dydt

# Define the objective function for optimization
def objective_function(params, gene_df, y0, t):
    params = params.reshape((len(y0), len(y0)))
    sol = odeint(ode_model, y0, t, args=(params,))
    
    # Clip the solution to avoid overflow issues
    sol = np.clip(sol, -1e10, 1e10)
    
    # Ensure no NaNs or infinities in the solution
    sol = np.nan_to_num(sol)
    
    error = np.sum((gene_df.values - sol)**2)
    return error

# Define the function to estimate the network structure using ODE model
def odes_minimize_estimated_params(gene_df, y0, t):
    # Initialize the parameters matrix with random values
    initial_params = np.random.rand(len(y0), len(y0)).flatten()
    
    # Optimize the parameters to minimize the objective function
    result = minimize(objective_function, initial_params, args=(gene_df, y0, t), method='trust-constr')
    
    # Reshape the optimized parameters to a matrix
    optimized_params = result.x.reshape((len(y0), len(y0)))
    
    # Convert to 0 or 1, based on the threshold of the diagonal
    # Normalize from 0 to 1
    optimized_params = (optimized_params - optimized_params.min()) / (optimized_params.max() - optimized_params.min())

    # Convert the optimized parameters to a DataFrame
    optimized_params = pd.DataFrame(optimized_params, columns=gene_df.columns, index=gene_df.columns)

    return optimized_params