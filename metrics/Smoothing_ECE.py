'''
Paper: Smooth ECE: Principled Reliability Diagrams via Kernel Smoothing
'''

import numpy as np
import torch
import torch.nn.functional as F

def smooth_round_to_grid(f, y, eval_points = 1000):
    values = np.zeros(eval_points)
    bins = (f * (eval_points-1)).astype(int).clip(0, eval_points-2)
    frac = f * (eval_points - 1) - bins
    np.add.at(values, bins, (1-frac)*y)
    np.add.at(values, bins + 1, frac*y)
    return values

class BaseKernelMixin:
    def smooth(self, f, y, x_eval, eps = 0.0001):
        ys = self.apply(f, y, x_eval)
        dens = self.apply(f, np.ones_like(y), x_eval) + eps
        ys = ys / dens
        return ys, dens

    def kde(self,
            x, x_eval, debias_boundary=True, bounds=(0, 1), eps=0.00001):
        weights = np.ones_like(x) / len(x)
        density = self.apply(x, weights, x_eval) + eps
        if debias_boundary:
            # This uses an approximation (for non-Gaussian kernels)
            # kde of uniform distribution over (0, 1)
            z = np.linspace(*bounds, 1000)
            bias = self.kde(z, x_eval, debias_boundary=False) + eps
            density /= bias
        return density

def interpolate(t, y):
    """
    Evaluates linear interpolation of a function mapping i/(len[y] - 1) -> y[i] on all points t.
    """
    buckets_cnt = len(y)
    bucket_size = 1 / (buckets_cnt - 1)
    inds = (t * (buckets_cnt - 1)).astype(int).clip(0, buckets_cnt - 2)
    residual = (t - inds * bucket_size) / bucket_size
    return y[inds] * (1 - residual) + y[inds + 1] * residual

class GaussianKernel(BaseKernelMixin):
    def __init__(self, sigma):
        self.sigma = sigma

    def kernel_ev(self, num_eval_points : int):
        t = np.linspace(0, 1, num_eval_points)
        var = self.sigma ** 2
        res = np.exp(-(t - 0.5) * (t - 0.5) / (2*var))
        res /= np.sqrt(2 * np.pi) * self.sigma
        #res *= len(t) / np.sum(res)
        return res

    def convolve(self, values, eval_points):
        ker = self.kernel_ev(eval_points)
        return np.convolve(values, ker, 'same')

    def apply(self, f, y, x_eval, eval_points = None):
        if eval_points is None:
            eval_points = max(2000, round(20 / self.sigma))
#        eval_points = 40_000
        eval_points = (eval_points // 2) + 1
        values = smooth_round_to_grid(f, y, eval_points = eval_points)
        smoothed = self.convolve(values, eval_points)
        return interpolate(x_eval, smoothed)

    def kde(self, x, x_eval):
        weights = np.ones_like(x) / len(x)
        density = self.apply(x, weights, x_eval)
        return density

class ReflectedGaussianKernel(GaussianKernel):
    def convolve(self, values, eval_points):
        ker = self.kernel_ev(eval_points)
        ext_vals = np.concatenate([np.flip(values)[:-1], values, np.flip(values)[1:]])
        return np.convolve(ext_vals, ker, "valid")[eval_points//2 : eval_points//2 + eval_points]

def _get_default_kernel():
    if False:
        return LogitGaussianKernel
    else:
        return ReflectedGaussianKernel

def smooth_ece_interpolated(r_grid, sigma):
    kernel = _get_default_kernel()
    ker = kernel(sigma)
    rs = ker.convolve(r_grid, len(r_grid))
    return np.sum(np.abs(rs)) / len(r_grid)

def search_param(predicate, start=1, refine=10):
    if predicate(start):
        return start
    start, end = 1, 0 
    for _ in range(refine):
        midpoint = (start + end) / 2
        if predicate(midpoint):
            end = midpoint
        else:
            start = midpoint
    return start

def smECE_fast(f, y, eps=0.001, return_width = False):
    ## This version of smECE discretizes the dataset at the appropriate resolution.
    num_eval_points = 200
    r_values = smooth_round_to_grid(f, f - y, eval_points=num_eval_points) / len(f)

    def recalculate_if_necessary(alpha):
        nonlocal num_eval_points
        nonlocal r_values
        recalc = False
        while round(20/alpha) > num_eval_points:
            recalc = True
            num_eval_points *= 4
        if recalc:
            r_values = smooth_round_to_grid(f, f-y, eval_points = num_eval_points) / len(f)

    def check_smooth_ece(alpha):
        recalculate_if_necessary(alpha)
        return alpha < eps or alpha < smooth_ece_interpolated(r_values, alpha)

    sigma = search_param(check_smooth_ece, start=1, refine=10)
    if return_width:
        return smooth_ece_interpolated(r_values, sigma), sigma
    else:
        return smooth_ece_interpolated(r_values, sigma)
    
if __name__=="__main__":
    import os
    import sys
    parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0,parentdir) 
    from tools.Calibration.utils import Load_z
    from customKing.config.config import get_cfg

    task_mode = "Calibration"
    cfg = get_cfg(task_mode)
    testDataset = Load_z(cfg,"valid")
    z_list,label_list = testDataset.z_list,testDataset.label_list
    softmaxes = F.softmax(z_list, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accs = predictions.eq(label_list)
    confidences,resort_index = torch.sort(confidences)
    accs = accs[resort_index]
    confidences = confidences.numpy()
    accs = accs.numpy()
    smECE = smECE_fast(confidences,accs)
    print(smECE)