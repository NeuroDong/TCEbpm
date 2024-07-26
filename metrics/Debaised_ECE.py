'''
Paper: Verified Uncertainty Calibration
'''


import numpy as np
from typing import List, Tuple, NewType, TypeVar
import torch.nn.functional as F

Data = List[Tuple[float, float]]  # List of (predicted_probability, true_label).
Bins = List[float]  # List of bin boundaries, excluding 0.0, but including 1.0.
BinnedData = List[Data]  # binned_data[i] contains the data in bin i.
T = TypeVar('T')

eps = 1e-6

def split(sequence: List[T], parts: int) -> List[List[T]]:
    assert parts <= len(sequence)
    array_splits = np.array_split(sequence, parts)
    splits = [list(l) for l in array_splits]
    assert len(splits) == parts
    return splits

def get_discrete_bins(data: List[float]) -> Bins:
    sorted_values = sorted(np.unique(data))
    bins = []
    for i in range(len(sorted_values) - 1):
        mid = (sorted_values[i] + sorted_values[i+1]) / 2.0
        bins.append(mid)
    bins.append(1.0)
    return bins

def get_labels_one_hot(labels, k):
    assert np.min(labels) >= 0
    assert np.max(labels) <= k - 1
    num_labels = labels.shape[0]
    labels_one_hot = np.zeros((num_labels, k))
    labels_one_hot[np.arange(num_labels), labels] = 1
    return labels_one_hot

def difference_mean(data : Data) -> float:
    """Returns average pred_prob - average label."""
    data = np.array(data)
    ave_pred_prob = np.mean(data[:, 0])
    ave_label = np.mean(data[:, 1])
    return ave_pred_prob - ave_label

def get_bin_probs(binned_data: BinnedData) -> List[float]:
    bin_sizes = list(map(len, binned_data))
    num_data = sum(bin_sizes)
    bin_probs = list(map(lambda b: b * 1.0 / num_data, bin_sizes))
    assert(abs(sum(bin_probs) - 1.0) < eps)
    return list(bin_probs)

def unbiased_square_ce(binned_data: BinnedData) -> float:
    # Note, this is not the l2 CE. It does not take the square root.
    def bin_error(data: Data):
        if len(data) < 2:
            return 0.0
            # raise ValueError('Too few values in bin, use fewer bins or get more data.')
        biased_estimate = abs(difference_mean(data)) ** 2
        label_values = list(map(lambda x: x[1], data))
        mean_label = np.mean(label_values)
        variance = mean_label * (1.0 - mean_label) / (len(data) - 1.0)
        return biased_estimate - variance
    bin_probs = get_bin_probs(binned_data)
    bin_errors = list(map(bin_error, binned_data))
    return np.dot(bin_probs, bin_errors)

def unbiased_l2_ce(binned_data: BinnedData) -> float:
    return max(unbiased_square_ce(binned_data), 0.0) ** 0.5

def normal_debiased_ce(binned_data : BinnedData, power=1, resamples=1000) -> float:
    bin_sizes = np.array(list(map(len, binned_data)))
    if np.min(bin_sizes) <= 1:
        raise ValueError('Every bin must have at least 2 points for debiased estimator. '
                         'Try adding the argument debias=False to your function call.')
    label_means = np.array(list(map(lambda l: np.mean([b for a, b in l]), binned_data)))
    label_stddev = np.sqrt(label_means * (1 - label_means) / bin_sizes)
    model_vals = np.array(list(map(lambda l: np.mean([a for a, b in l]), binned_data)))
    assert(label_means.shape == (len(binned_data),))
    assert(model_vals.shape == (len(binned_data),))
    ce = plugin_ce(binned_data, power=power)
    bin_probs = get_bin_probs(binned_data)
    resampled_ces = []
    for i in range(resamples):
        label_samples = np.random.normal(loc=label_means, scale=label_stddev)
        # TODO: we can also correct the bias for the model_vals, although this is
        # smaller.
        diffs = np.power(np.abs(label_samples - model_vals), power)
        cur_ce = np.power(np.dot(bin_probs, diffs), 1.0 / power)
        resampled_ces.append(cur_ce)
    mean_resampled = np.mean(resampled_ces)
    bias_corrected_ce = 2 * ce - mean_resampled
    return bias_corrected_ce

def plugin_ce(binned_data: BinnedData, power=2) -> float:
    def bin_error(data: Data):
        if len(data) == 0:
            return 0.0
        return abs(difference_mean(data)) ** power
    bin_probs = get_bin_probs(binned_data)
    bin_errors = list(map(bin_error, binned_data))
    return np.dot(bin_probs, bin_errors) ** (1.0 / power)

def get_top_predictions(probs):
    return np.argmax(probs, 1)

def get_top_probs(probs):
    return np.max(probs, 1)

def get_equal_bins(probs: List[float], num_bins: int=10) -> Bins:
    """Get bins that contain approximately an equal number of data points."""
    sorted_probs = sorted(probs)
    if num_bins > len(sorted_probs):
        num_bins = len(sorted_probs)
    binned_data = split(sorted_probs, num_bins)
    bins: Bins = []
    for i in range(len(binned_data) - 1):
        last_prob = binned_data[i][-1]
        next_first_prob = binned_data[i + 1][0]
        bins.append((last_prob + next_first_prob) / 2.0)
    bins.append(1.0)
    bins = sorted(list(set(bins)))
    return bins

def bin(data: Data, bins: Bins):
    return fast_bin(data, bins)


def fast_bin(data, bins):
    prob_label = np.array(data)
    bin_indices = np.searchsorted(bins, prob_label[:, 0])
    bin_sort_indices = np.argsort(bin_indices)
    sorted_bins = bin_indices[bin_sort_indices]
    splits = np.searchsorted(sorted_bins, list(range(1, len(bins))))
    binned_data = np.split(prob_label[bin_sort_indices], splits)
    return binned_data

def _get_ce(probs, labels, p, debias, num_bins, binning_scheme, mode='marginal'):
    def ce_1d(probs, labels):
        assert probs.shape == labels.shape
        assert len(probs.shape) == 1
        data = list(zip(probs, labels))
        if binning_scheme == get_discrete_bins:
            assert(num_bins is None)
            bins = binning_scheme(probs)
        else:
            bins = binning_scheme(probs, num_bins=num_bins)
        if p == 2 and debias:
            return unbiased_l2_ce(bin(data, bins))
        elif debias:
            return normal_debiased_ce(bin(data, bins), power=p)
        else:
            return plugin_ce(bin(data, bins), power=p)
    if mode != 'marginal' and mode != 'top-label':
        raise ValueError("mode must be 'marginal' or 'top-label'.")
    probs = np.array(probs)
    labels = np.array(labels)
    if not(np.issubdtype(labels.dtype, np.integer)):
        raise ValueError('labels should an integer numpy array.')
    if len(labels.shape) != 1:
        raise ValueError('labels should be a 1D numpy array.')
    if probs.shape[0] != labels.shape[0]:
        raise ValueError('labels and probs should have the same number of entries.')
    if len(probs.shape) == 1:
        # If 1D (2-class setting), compute the regular calibration error.
        if np.min(labels) < 0 or np.max(labels) > 1:
            raise ValueError('If probs is 1D, each label should be 0 or 1.')
        return ce_1d(probs, labels)
    elif len(probs.shape) == 2:
        if np.min(labels) < 0 or np.max(labels) > probs.shape[1] - 1:
            raise ValueError('labels should be between 0 and num_classes - 1.')
        if mode == 'marginal':
            labels_one_hot = get_labels_one_hot(labels, k=probs.shape[1])
            assert probs.shape == labels_one_hot.shape
            marginal_ces = []
            for k in range(probs.shape[1]):
                cur_probs = probs[:, k]
                cur_labels = labels_one_hot[:, k]
                marginal_ces.append(ce_1d(cur_probs, cur_labels) ** p)
            return np.mean(marginal_ces) ** (1.0 / p)
        elif mode == 'top-label':
            preds = get_top_predictions(probs)
            correct = (preds == labels).astype(probs.dtype)
            confidences = get_top_probs(probs)
            return ce_1d(confidences, correct)
    else:
        raise ValueError('probs should be a 1D or 2D numpy array.')

def get_binning_ce(probs, labels, p=2, debias=True, num_bins = 15, binning_mode = "ew", mode='top-label'):
    """Estimate the calibration error of a binned model.

    Args:
        probs: A numpy array of shape (n,) or (n, k). If the shape is (n,) then
            we assume binary classification and probs[i] is the model's confidence
            the i-th example is 1. Otherwise, probs[i][j] is the model's confidence
            the i-th example is j, with 0 <= probs[i][j] <= 1.
        labels: A numpy array of shape (n,). labels[i] denotes the label of the i-th
            example. In the binary classification setting, labels[i] must be 0 or 1,
            in the k class setting labels[i] is an integer with 0 <= labels[i] <= k-1.
        p: We measure the lp calibration error, where p >= 1 is an integer.
        debias: Should we try to debias the estimates? For p = 2, the debiasing
            has provably better sample complexity.
        mode: 'marginal' or 'top-label'. 'marginal' calibration means we compute the
            calibraton error for each class and then average them. Top-label means
            we compute the calibration error of the prediction that the model is most
            confident about.

    Returns:
        Estimated calibration error, a floating point value.
    """

    if binning_mode == "ew":
        return _get_ce(probs, labels, p, debias,num_bins=num_bins, binning_scheme=get_equal_prob_bins, mode=mode)
    elif binning_mode == "em":
        return _get_ce(probs, labels, p, debias,num_bins=num_bins, binning_scheme=get_equal_bins, mode=mode)

def get_equal_prob_bins(probs: List[float], num_bins: int=10) -> Bins:
    return [i * 1.0 / num_bins for i in range(1, num_bins + 1)]

def Debaised_ECE(probs, labels, p=2, debias=True, num_bins = 15, binning_mode = "ew", mode='top-label'):
    """Get the calibration error.

    Args:
        probs: A numpy array of shape (n,) or (n, k). If the shape is (n,) then
            we assume binary classification and probs[i] is the model's confidence
            the i-th example is 1. Otherwise, probs[i][j] is the model's confidence
            the i-th example is j, with 0 <= probs[i][j] <= 1.
        labels: A numpy array of shape (n,). labels[i] denotes the label of the i-th
            example. In the binary classification setting, labels[i] must be 0 or 1,
            in the k class setting labels[i] is an integer with 0 <= labels[i] <= k-1.
        p: We measure the lp calibration error, where p >= 1 is an integer.
        debias: Should we try to debias the estimates? For p = 2, the debiasing
            has provably better sample complexity.
        mode: 'marginal' or 'top-label'. 'marginal' calibration means we compute the
            calibraton error for each class and then average them. Top-label means
            we compute the calibration error of the prediction that the model is most
            confident about.

    Returns:
        Estimated calibration error, a floating point value.
        The method first uses heuristics to check if the values came from a scaling
        method or binning method, and then calls the corresponding function. For
        more explicit control, use lower_bound_scaling_ce or get_binning_ce.
    """
    return get_binning_ce(probs, labels, p, debias, num_bins = num_bins, binning_mode = binning_mode, mode=mode)

if __name__=="__main__":
    import os
    import sys
    parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0,parentdir) 
    from tools.Calibration.utils import Load_z
    from customKing.config.config import get_cfg

    #Original ECE
    task_mode = "Calibration"
    cfg = get_cfg(task_mode)
    testDataset = Load_z(cfg,"valid")
    z_list,label_list = testDataset.z_list,testDataset.label_list.long().numpy()
    confidence_list = F.softmax(z_list, dim=1).numpy()

    ECE_debias_ew = Debaised_ECE(confidence_list,label_list,p=2,debias=True,num_bins = 15, binning_mode = "ew")
    print("ECE_debias_ew:",ECE_debias_ew)

    ECE_debias_em = Debaised_ECE(confidence_list,label_list,p=2,debias=True,num_bins = 15, binning_mode = "em")
    print("ECE_debias_ew:",ECE_debias_em)

