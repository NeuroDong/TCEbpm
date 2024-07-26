'''
Paper: Mitigating Bias in Calibration Error Estimation
'''


import numpy as np

"""Binning methods."""
import abc
import numpy as np
import torch.nn.functional as F


class BinMethod(abc.ABC):
  """General interface for specifying binning method."""

  def __init__(self, num_bins):
    self.num_bins = num_bins

  @abc.abstractmethod
  def compute_bin_indices(self, scores):
    """Assign a bin index for each score.

    Args:
      scores: np.array of shape (num_examples, num_classes) containing the
        model's confidence scores

    Returns:
      bin_indices: np.array of shape (num_examples, num_classes) containing the
        bin assignment for each score
    """
    pass


class BinEqualWidth(BinMethod):
  """Divide the scores into equal-width bins."""

  def compute_bin_indices(self, scores):
    """Assign a bin index for each score assuming equal width bins.

    Args:
      scores: np.array of shape (num_examples, num_classes) containing the
        model's confidence scores

    Returns:
      bin_indices: np.array of shape (num_examples, num_classes) containing the
        bin assignment for each score
    """
    edges = np.linspace(0.0, 1.0, self.num_bins + 1)
    bin_indices = np.digitize(scores, edges, right=False)
    # np.digitze uses one-indexed bins, switch to using 0-indexed
    bin_indices = bin_indices - 1
    # Put examples with score equal to 1.0 in the last bin.
    bin_indices = np.where(scores == 1.0, self.num_bins - 1, bin_indices)
    return bin_indices


class BinEqualExamples(BinMethod):
  """Divide the scores into bins with equal number of examples."""

  def compute_bin_indices(self, scores):
    """Assign a bin index for each score assumes equal num examples per bin.

    Args:
      scores: np.ndarray of shape [N, K] containing the model's confidence

    Returns:
      bin_indices: np.ndarray of shape [N, K] containing the bin assignment for
        each score
    """
    num_examples = scores.shape[0]
    num_classes = scores.shape[1]

    bin_indices = np.zeros((num_examples, num_classes), dtype=int)
    for k in range(num_classes):
      sort_ix = np.argsort(scores[:, k])
      bin_indices[:, k][sort_ix] = np.minimum(
          self.num_bins - 1,
          np.floor((np.arange(num_examples) / num_examples) *
                   self.num_bins)).astype(int)
    return bin_indices

class CalibrationMetric():
  """Class to compute the calibration error.

  Let N = num examples, K = num classes, and B = num bins.
  """

  def __init__(self,
               ce_type="quant",
               num_bins=15,
               bin_method="equal_examples",
               norm=2,
               multiclass_setting="top_label"):
    """Initialize calibration metric class.

    Args:
      ce_type: str describing the type of calibration error to compute.
        em_ece_bin implements equal mass ECE_bin
        ew_ece_bin implements equal width ECE_bin
        em_ece_sweep implements equal mass ECE_sweep
        ew_ece_sweep implements equal width ECE_sweep
      num_bins: int for number of bins.
      bin_method: string for binning technique to use. Must be either
        "equal_width", "equal_examples" or "".
      norm: integer for norm to use to compute the calibration error. Norm
        should be > 0.
      multiclass_setting: string specifying the type of multiclass calibration
        error to compute. Must be "top_label" or "marginal". If "top_label",
        computes the calibration error of the top class. If "marginal", computes
        the marginal calibration error.
    """
    if bin_method not in ["equal_width", "equal_examples", ""]:
      raise NotImplementedError("Bin method not supported.")
    if multiclass_setting not in ["top_label", "marginal"]:
      raise NotImplementedError(
          f"Multiclass setting {multiclass_setting} not supported.")
    if bin_method == "equal_width" or ce_type.startswith("ew"):
      self.bin_method = BinEqualWidth(num_bins)
    elif bin_method == "equal_examples" or ce_type.startswith("em"):
      self.bin_method = BinEqualExamples(num_bins)
    elif bin_method == "None":
      self.bin_method = None
    else:
      raise NotImplementedError(f"Bin method {bin_method} not supported.")

    self.ce_type = ce_type
    self.norm = norm
    self.num_bins = num_bins
    self.multiclass_setting = multiclass_setting
    self.configuration_str = "{}_bins:{}_{}_norm:{}_{}".format(
        ce_type, num_bins, bin_method, norm, multiclass_setting)

  def get_configuration_str(self):
    return self.configuration_str

  def predict_top_label(self, fx, y):
    """Compute confidence scores and correctness of predicted labels.

    Args:
      fx: np.ndarray of shape [N, K] for predicted confidence fx.
      y: np.ndarray of shape [N, K] for one-hot-encoded labels.

    Returns:
      fx_top: np.ndarray of shape [N, 1] for confidence score of top label.
      hits: np.ndarray of shape [N, 1] denoting whether or not top label
        is a correct prediction or not.
    """
    picked_classes = np.argmax(fx, axis=1)
    labels = np.argmax(y, axis=1)
    hits = 1 * np.array(picked_classes == labels, ndmin=2).transpose()
    fx_top = np.max(fx, axis=1, keepdims=True)
    return fx_top, hits

  def compute_error(self, fx, y):
    fx = fx[:, np.newaxis]
    y = y[:, np.newaxis]

    if self.num_bins > 0 and self.bin_method:
      binned_fx, binned_y, bin_sizes, bin_indices = self._bin_data(fx, y)

    if self.ce_type in ["ew_ece_bin", "em_ece_bin"]:
      calibration_error = self._compute_error_all_binned(
          binned_fx, binned_y, bin_sizes)
    elif self.ce_type in ["label_binned"]:
      calibration_error = self._compute_error_label_binned(
          fx, binned_y, bin_indices)
    elif self.ce_type.endswith(("sweep")):
      calibration_error = self._compute_error_monotonic_sweep(fx, y)
    else:
      raise NotImplementedError("Calibration error {} not supported.".format(
          self.ce_type))

    return calibration_error

  def _compute_error_no_bins(self, fx, y):
    """Compute error without binning."""
    num_examples = fx.shape[0]
    ce = pow(np.abs(fx - y), self.norm)
    return pow(ce.sum() / num_examples, 1. / self.norm)

  def _compute_error_all_binned(self, binned_fx, binned_y, bin_sizes):
    """Compute calibration error given binned data."""
    num_examples = np.sum(bin_sizes[:, 0])
    num_classes = binned_fx.shape[1]
    ce = pow(np.abs(binned_fx - binned_y), self.norm) * bin_sizes
    ce_sum = 0
    for k in range(num_classes):
      ce_sum += ce[:, k].sum()
    return pow(ce_sum / (num_examples*num_classes), 1. / self.norm)

  def _compute_error_label_binned(self, fx, binned_y, bin_indices):
    """Compute label binned calibration error."""
    num_examples = fx.shape[0]
    num_classes = fx.shape[1]
    ce_sum = 0.0
    for k in range(num_classes):
      for i in range(num_examples):
        ce_sum += pow(
            np.abs(fx[i, k] - binned_y[bin_indices[i, k], k]), self.norm)
    ce_sum = pow(ce_sum / num_examples, 1. / self.norm)
    return ce_sum

  def _bin_data(self, fx, y):
    """Bin fx and y.

    Args:
      fx: np.ndarray of shape [N, K] for predicted confidence fx.
      y: np.ndarray of shape [N, K] for one-hot-encoded labels.

    Returns:
      A tuple containing:
        - binned_fx: np.ndarray of shape [B, K] containing mean
            predicted score for each bin and class
        - binned_y: np.ndarray of shape [B, K]
            containing mean empirical accuracy for each bin and class
        - bin_sizes: np.ndarray of shape [B, K] containing number
            of examples in each bin and class
    """
    bin_indices = self.bin_method.compute_bin_indices(fx)
    num_classes = fx.shape[1]

    binned_fx = np.zeros((self.num_bins, num_classes))
    binned_y = np.zeros((self.num_bins, num_classes))
    bin_sizes = np.zeros((self.num_bins, num_classes))

    for k in range(num_classes):
      for bin_idx in range(self.num_bins):
        indices = np.where(bin_indices[:, k] == bin_idx)[0]
        # Disable for Numpy containers.
        # pylint: disable=g-explicit-length-test
        if len(indices) > 0:
          # pylint: enable=g-explicit-length-test
          mean_score = np.mean(fx[:, k][indices])
          mean_accuracy = np.mean(y[:, k][indices])
          bin_size = len(indices)
        else:
          mean_score = 0.0
          mean_accuracy = 0.0
          bin_size = 0
        binned_fx[bin_idx][k] = mean_score
        binned_y[bin_idx][k] = mean_accuracy
        bin_sizes[bin_idx][k] = bin_size

    return binned_fx, binned_y, bin_sizes, bin_indices

  def _compute_error_monotonic_sweep(self, fx, y):
    """Compute ECE using monotonic sweep method."""
    fx = np.squeeze(fx)
    y = np.squeeze(y)
    non_nan_inds = np.logical_not(np.isnan(fx))
    fx = fx[non_nan_inds]
    y = y[non_nan_inds]

    if self.ce_type == "em_ece_sweep":
      bins = self.em_monotonic_sweep(fx, y)
    elif self.ce_type == "ew_ece_sweep":
      bins = self.ew_monotonic_sweep(fx, y)
    n_bins = np.max(bins) + 1
    ece, _ = self._calc_ece_postbin(n_bins, bins, fx, y)
    return ece

  def _calc_ece_postbin(self, n_bins, bins, fx, y):
    """Calculate ece_bin after bins are computed and determine monotonicity."""
    ece = 0.
    monotonic = True
    last_ym = -1000
    for i in range(n_bins):
      cur = bins == i
      if any(cur):
        fxm = np.mean(fx[cur])
        ym = np.mean(y[cur])
        if ym < last_ym:  # determine if predictions are monotonic
          monotonic = False
        last_ym = ym
        n = np.sum(cur)
        ece += n * pow(np.abs(ym - fxm), self.norm)
    return (pow(ece / fx.shape[0], 1. / self.norm)), monotonic

  def em_monotonic_sweep(self, fx, y):
    """Monotonic bin sweep using equal mass binning scheme."""
    sort_ix = np.argsort(fx)
    n_examples = fx.shape[0]
    bins = np.zeros((n_examples), dtype=int)

    prev_bins = np.zeros((n_examples), dtype=int)
    for n_bins in range(2, n_examples):
      bins[sort_ix] = np.minimum(
          n_bins - 1, np.floor(
              (np.arange(n_examples) / n_examples) * n_bins)).astype(int)
      _, monotonic = self._calc_ece_postbin(n_bins, bins, fx, y)
      if not monotonic:
        return prev_bins
      prev_bins = np.copy(bins)
    return bins

  def ew_monotonic_sweep(self, fx, y):
    """Monotonic bin sweep using equal width binning scheme."""
    n_examples = fx.shape[0]
    bins = np.zeros((n_examples), dtype=int)
    prev_bins = np.zeros((n_examples), dtype=int)
    for n_bins in range(2, n_examples):
      bins = np.minimum(n_bins - 1, np.floor(fx * n_bins)).astype(int)
      _, monotonic = self._calc_ece_postbin(n_bins, bins, fx, y)
      if not monotonic:
        return prev_bins
      prev_bins = np.copy(bins)
    return bins
  
if __name__=="__main__":
    import os
    import sys
    parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0,parentdir) 
    from tools.Calibration.utils import Load_z
    from customKing.config.config import get_cfg
    import torch

    #Original ECE
    task_mode = "Calibration"
    cfg = get_cfg(task_mode)
    testDataset = Load_z(cfg,"valid")
    z_list,label_list = testDataset.z_list,testDataset.label_list.long()
    softmaxes = F.softmax(z_list, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accs = predictions.eq(label_list)
    confidences,resort_index = torch.sort(confidences)
    accs = accs[resort_index]
    confidences = confidences.numpy()
    accs = accs.numpy()

    Metric_fun = CalibrationMetric(ce_type="em_ece_sweep",num_bins=15,bin_method="equal_examples",norm=2)
    ECE_sweep_em = Metric_fun.compute_error(confidences,accs)
    print("ECE_sweep_em:",ECE_sweep_em)

    Metric_fun = CalibrationMetric(ce_type="ew_ece_bin",num_bins=15,bin_method="equal_width",norm=1)
    ew_ece_bin = Metric_fun.compute_error(confidences,accs)
    print("ew_ece_bin:",ew_ece_bin)
    