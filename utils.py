import pandas as pd
import numpy as np
from sklearn.metrics import auc
import matplotlib.pyplot as plt


def read_from_csv(filename: str) -> pd.DataFrame:
    """Read data from CSV file and return as DataFrame."""
    return pd.read_csv(filename, sep=';', index_col=0)


def result_to_df(array: np.array):
    """Make array of edges into a neat data frame."""
    genes = ['ASH1', 'CBF1', 'GAL4', 'GAL80', 'SWI5']
    df = pd.DataFrame(array.astype(float))
    df.columns = genes
    df.index = genes
    return df


def evaluate_model(ground_truth: pd.DataFrame, predicted: pd.DataFrame):
    thresholds = np.linspace(0, 1, 100)

    true_positive_rates = []
    false_positive_rates = []

    for threshold in thresholds:
        true_positives = np.sum(
            (predicted >= threshold).values & (ground_truth.values == 1))
        false_positives = np.sum(
            (predicted >= threshold).values & (ground_truth.values == 0))
        true_negatives = np.sum(
            (predicted < threshold).values & (ground_truth.values == 0))
        false_negatives = np.sum(
            (predicted < threshold).values & (ground_truth.values == 1))
        # if it is in series type, then convert it into int32 format before calculating
        true_positives = true_positives.astype(np.int32)
        false_positives = false_positives.astype(np.int32)
        true_negatives = true_negatives.astype(np.int32)
        false_negatives = false_negatives.astype(np.int32)
        tpr = true_positives / \
            (true_positives + false_negatives) if (true_positives +
                                                   false_negatives) > 0 else 0
        fpr = false_positives / \
            (false_positives + true_negatives) if (false_positives +
                                                   true_negatives) > 0 else 0

        true_positive_rates.append(tpr)
        false_positive_rates.append(fpr)

    # Ensure the arrays are 1-dimensional and sorted
    true_positive_rates = np.array(true_positive_rates).flatten()
    false_positive_rates = np.array(false_positive_rates).flatten()

    # Sort the false positive rates and corresponding true positive rates
    sorted_indices = np.argsort(false_positive_rates)
    false_positive_rates = false_positive_rates[sorted_indices]
    true_positive_rates = true_positive_rates[sorted_indices]

    roc_auc = auc(false_positive_rates, true_positive_rates)

    plt.figure()
    plt.plot(false_positive_rates, true_positive_rates, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def calculate_rates(ground_truth, predicted, thresholds):
    # Check if ground truth and predicted has been normalized or not, if not, then normalize them
    if not (ground_truth.values.min() >= 0 and ground_truth.values.max() <= 1):
        ground_truth = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min())
    if not (predicted.values.min() >= 0 and predicted.values.max() <= 1):
        predicted = (predicted - predicted.min()) / (predicted.max() - predicted.min())

    true_positive_rates = []
    false_positive_rates = []

    for threshold in thresholds:
        true_positives = np.sum(
            (predicted >= threshold).values & (ground_truth.values == 1))
        false_positives = np.sum(
            (predicted >= threshold).values & (ground_truth.values == 0))
        true_negatives = np.sum(
            (predicted < threshold).values & (ground_truth.values == 0))
        false_negatives = np.sum(
            (predicted < threshold).values & (ground_truth.values == 1))

        tpr = true_positives / \
            (true_positives + false_negatives) if (true_positives +
                                                   false_negatives) > 0 else 0
        fpr = false_positives / \
            (false_positives + true_negatives) if (false_positives +
                                                   true_negatives) > 0 else 0

        true_positive_rates.append(tpr)
        false_positive_rates.append(fpr)

    true_positive_rates = np.array(true_positive_rates).flatten()
    false_positive_rates = np.array(false_positive_rates).flatten()

    sorted_indices = np.argsort(false_positive_rates)
    return false_positive_rates[sorted_indices], true_positive_rates[sorted_indices]


def evaluate_all_models(ground_truth: pd.DataFrame, predicted_list: list, model_names: list):
    thresholds = np.linspace(0, 1, 100)
    plt.figure()

    for predicted, model_name in zip(predicted_list, model_names):
        fpr, tpr = calculate_rates(ground_truth, predicted, thresholds)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    

