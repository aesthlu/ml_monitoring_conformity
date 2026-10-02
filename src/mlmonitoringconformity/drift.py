import numpy as np
from dataclasses import dataclass


@dataclass
class DriftResult:
    statistic: float
    p_value: float
    drift_detected: bool


def psi(reference, current, bins=10) -> float:
    """
    Calculate the Population Stability Index (PSI) between two distributions.

    Parameters:
    reference (array-like): The reference distribution (e.g., historical data).
    current (array-like): The current distribution (e.g., new data).
    bins (int): The number of bins to use for the histogram.

    Returns:
    float: The PSI value.
    """
    # Create histograms for both distributions
    ref_hist, bin_edges = np.histogram(reference, bins=bins, density=True)
    curr_hist, _ = np.histogram(current, bins=bin_edges, density=True)

    # Avoid division by zero and log of zero by adding a small constant
    ref_hist = np.where(ref_hist == 0, 1e-10, ref_hist)
    curr_hist = np.where(curr_hist == 0, 1e-10, curr_hist)

    # Calculate PSI
    psi_value = np.sum((curr_hist - ref_hist) * np.log(curr_hist / ref_hist))

    return psi_value

def ks_drift(reference, current, alpha=0.05) -> DriftResult:
    """
    Perform the Kolmogorov-Smirnov test to detect drift between two distributions.

    Parameters:
    reference (array-like): The reference distribution (e.g., historical data).
    current (array-like): The current distribution (e.g., new data).
    alpha (float): Significance level for the test.

    Returns:
    DriftResult: A dataclass containing the KS statistic, p-value, and drift detection result.
    """
    from scipy.stats import ks_2samp

    statistic, p_value = ks_2samp(reference, current)
    drift_detected = p_value < alpha

    return DriftResult(statistic=statistic, p_value=p_value, drift_detected=drift_detected)

