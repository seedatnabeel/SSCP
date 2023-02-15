# stdlib
import pickle

# third party
import numpy as np
import pandas as pd
from scipy.stats import sem

from typing import Tuple, List, Dict, Any


def write_to_file(contents: Any, filename: str) -> None:
    """
    > Writes the contents to the specified file.
    Args:
        contents: The contents to write to the file.
        filename: The name of the file to write to.
    """
    with open(filename, "wb") as handle:
        pickle.dump(contents, handle)

def read_from_file(filename):
    """
    > This function loads a file from a pickle
    Args:
      filename: the name of the file to read from
    Returns:
      the pickle file.
    """
    # load file from pickle

    return pickle.load(open(filename, "rb"))


def compute_interval_metrics(lower_bound: float, upper_bound: float, y_true: float) -> Tuple[float, float]:
    """
    > Calculates the coverage and average length of the confidence intervals
    Args:
      lower_bound: lower bound of the confidence interval
      upper_bound: upper bound of the confidence interval
      y_true: actual values

    Returns:
      coverage: the percentage of the actual values that are within the confidence interval
      avg_length: the average length of the confidence interval
    """
    in_the_range = np.sum((y_true >= lower_bound) & (y_true <= upper_bound))
    coverage = in_the_range / len(y_true) * 100
    avg_length = np.mean(abs(upper_bound - lower_bound))
    return coverage, avg_length


def compute_excess(lb: np.ndarray, ub: np.ndarray, true: np.ndarray) -> Tuple[float, float]:
    """
    > Computes the average excess of the true values over the lower and upper bounds
    Args:
      true: the true values of the data
      lb: lower bound
      ub: upper bound
    Returns:
      The mean and the proportion of excess
    """
    true, lb, ub = np.array(true), np.array(lb), np.array(ub)
    excess = []
    for i in range(true.shape[0]):
        if true[i] >= lb[i] and true[i] <= ub[i]:
            excess.append(np.min([true[i] - lb[i], ub[i] - true[i]]))

    return np.nanmean(excess), np.sum(excess) / true.shape[0]


def compute_deficet(lb: np.ndarray, ub: np.ndarray, true: np.ndarray) -> Tuple[float, float]:
    """
    > Computes the average and the proportion of the time that the true value is outside the confidence
    interval
    Args:
      true: the true values of the parameters
      lb: lower bound
      ub: upper bound
    Returns:
      The mean and the proportion of the deficet
    """

    true, lb, ub = np.array(true), np.array(lb), np.array(ub)
    deficet = []
    for i in range(true.shape[0]):
        if true[i] <= lb[i] or true[i] >= ub[i]:
            deficet.append(
                np.min([np.abs(true[i] - lb[i]), np.abs(true[i] - ub[i])]),
            )

    return np.nanmean(deficet), np.sum(deficet) / true.shape[0]


def process_results(results_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    > Takes a list of dictionaries, converts it to a dataframe with metrics computed

    Args:
      results_list: a list of dictionaries, each dictionary is the results of a single run of the model

    Returns:
      A dictionary of dictionaries.
    """
    df = pd.DataFrame(results_list)

    res: Dict[str, Dict[str, float]] = {}

    for model in list(df.columns):

        metrics: Dict[str, float] = {}

        for key in df[model].values[0].keys():
            metric = [mydict[key] for mydict in df[model].values]
            metrics[f"{key}_mean"] = np.nanmean(metric)
            metrics[f"{key}_std"] = np.nanstd(metric)
            metrics[f"{key}_se"] = sem(metric)

        res[model] = metrics

    return res
