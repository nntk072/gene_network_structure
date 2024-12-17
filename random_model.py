import numpy as np
import pandas as pd
from utils import result_to_df

def random_estimated_params():
    # Random inference of network structure
    inferred_network_structure = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            # if i != j:
                # inferred_network_structure[i, j] = np.random.choice(
                #     [0, 1])  # Randomly infer connections
                # 0 or 1 are not good, make it random in range 0 and 1
            inferred_network_structure[i, j] = np.random.uniform(0, 1)
    inferred_network_structure = result_to_df(inferred_network_structure)
    return inferred_network_structure
