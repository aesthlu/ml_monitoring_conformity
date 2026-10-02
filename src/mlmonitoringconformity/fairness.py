import numpy as np

def demographic_parity_difference(y_pred, sensitive) -> float:
    """
    Calculate the Demographic Parity Difference (DPD) between two groups.

    Parameters:
    y_pred (array-like): The predicted labels (binary).
    sensitive (array-like): The sensitive attribute (binary).

    Returns:
    float: The Demographic Parity Difference value.
    """
    # Ensure inputs are numpy arrays
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    # Calculate the positive prediction rates for each group
    group_0_rate = np.mean(y_pred[sensitive == 0])
    group_1_rate = np.mean(y_pred[sensitive == 1])

    # Calculate the Demographic Parity Difference
    dpd = group_0_rate - group_1_rate

    return dpd

def equal_opportunity_difference(y_true, y_pred, sensitive) -> float:
    """
    Calculate the Equal Opportunity Difference (EOD) between two groups.

    Parameters:
    y_true (array-like): The true labels (binary).
    y_pred (array-like): The predicted labels (binary).
    sensitive (array-like): The sensitive attribute (binary).

    Returns:
    float: The Equal Opportunity Difference value.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    # Calculate the true positive rates for each group
    group_0_tpr = np.mean(y_pred[(sensitive == 0) & (y_true == 1)])
    group_1_tpr = np.mean(y_pred[(sensitive == 1) & (y_true == 1)])

    # Calculate the Equal Opportunity Difference
    eod = group_0_tpr - group_1_tpr

    return eod

