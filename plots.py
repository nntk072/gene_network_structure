import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap

from sklearn.metrics import roc_curve, auc


def plot_data(df: pd.DataFrame):
    """Make plot of either raw or discretized measurement data."""
    plt.figure(figsize=(5, 5))
    plt.plot(df.index.values, df.values)
    plt.xlabel("time (min)")
    plt.legend(df.columns.values)
    plt.ylabel("gene expression (arbitrary units)")

    # Calculate min and max values for y-axis
    min_val = df.values.min()
    max_val = df.values.max()

    # Create ticks with a range of 0.02
    ticks = np.arange(0, max_val + 0.02, 0.02)

    # Set y-ticks and labels
    plt.yticks(ticks=ticks, labels=[f"{tick:.2f}" for tick in ticks])

    plt.title("Gene expression data")
    plt.ylabel("gene expression (arbitrary units)")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.show()

def calculate_best_threshold(ground_truth, predicted):
    fpr, tpr, thresholds = roc_curve(ground_truth.values.ravel(), predicted.values.ravel())

    if np.isinf(thresholds).any():
        thresholds = np.where(np.isinf(thresholds), np.nan, thresholds)
   
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    if np.isnan(optimal_threshold):
        optimal_threshold = 0.5
    return optimal_threshold

def plot_confusion_matrix(ground_truth, predicted, title='', ax=None, best_threshold=0.5):
    # Define confusion components
    tp_matrix = (predicted >= best_threshold) & (ground_truth == 1)
    tn_matrix = (predicted < best_threshold) & (ground_truth == 0)
    fp_matrix = (predicted >= best_threshold) & (ground_truth == 0)
    fn_matrix = (predicted < best_threshold) & (ground_truth == 1)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Use masking for plotting
    tp_mask = np.ma.masked_where(~tp_matrix, tp_matrix)
    tn_mask = np.ma.masked_where(~tn_matrix, tn_matrix)
    fp_mask = np.ma.masked_where(~fp_matrix, fp_matrix)
    fn_mask = np.ma.masked_where(~fn_matrix, fn_matrix)

    # Plot each type with clear colors
    ax.imshow(tp_mask, cmap=ListedColormap(['green']), interpolation='none')
    ax.imshow(tn_mask, cmap=ListedColormap(['grey']), interpolation='none')
    ax.imshow(fp_mask, cmap=ListedColormap(['red']), interpolation='none')
    ax.imshow(fn_mask, cmap=ListedColormap(['orange']), interpolation='none')

    # Add titles and axes labels
    ax.set_title(f'Confusion Matrix of {title}, Threshold={best_threshold:.2f}')

    # Set custom tick labels
    ax.set_xticks(np.arange(len(ground_truth.columns)))
    ax.set_yticks(np.arange(len(ground_truth.index)))
    ax.set_xticklabels(ground_truth.columns)
    ax.set_yticklabels(ground_truth.index)

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Create legend
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='Edge found, True',
               markersize=10, markerfacecolor='green'),
        Line2D([0], [0], marker='s', color='w', label='No edge, True',
               markersize=10, markerfacecolor='grey'),
        Line2D([0], [0], marker='s', color='w', label='Edge found, False',
               markersize=10, markerfacecolor='red'),
        Line2D([0], [0], marker='s', color='w', label='No edge, False',
               markersize=10, markerfacecolor='orange')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    return fig, ax


def plot_all_confusion_matrices(ground_truth_network_structure, output_truth_network_structure_list, output_truth_network_structure_description_list):
    num_models = len(output_truth_network_structure_list)
    num_rows = int(np.ceil(np.sqrt(num_models)))
    num_cols = int(np.ceil(num_models / num_rows))

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, 14))

    for i, (output_truth_network_structure, output_truth_network_structure_description) in enumerate(zip(output_truth_network_structure_list, output_truth_network_structure_description_list)):
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col] if num_models > 1 else axs
        best_threshold = calculate_best_threshold(ground_truth_network_structure, output_truth_network_structure)
        plot_confusion_matrix(ground_truth_network_structure, output_truth_network_structure,
                              output_truth_network_structure_description, ax=ax, best_threshold=best_threshold)
    plt.tight_layout()
    plt.show()
