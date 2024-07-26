'''
Paper: CALIBRATION OF NEURAL NETWORKS USING SPLINES
'''

import numpy as np
import torch
import torch.nn.functional as F

def ensure_numpy(a):
    if not isinstance(a, np.ndarray): a = a.numpy()
    return a

def KS_error(scores, labels):
    # KS stands for Kolmogorov-Smirnov

    # Change to numpy, then this will work
    scores = ensure_numpy (scores)
    labels = ensure_numpy (labels)

    # Sort the data
    order = scores.argsort()
    scores = scores[order]
    labels = labels[order]

    # Accumulate and normalize by dividing by num samples
    nsamples = len(scores)
    integrated_scores = np.cumsum(scores) / nsamples
    integrated_accuracy   = np.cumsum(labels) / nsamples

    # Work out the Kolmogorov-Smirnov error
    KS_error_max = np.amax(np.absolute (integrated_scores - integrated_accuracy))

    return KS_error_max

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
    z_list,label_list = testDataset.z_list,testDataset.label_list
    softmaxes = F.softmax(z_list, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    labels = predictions.eq(label_list)
    KS_e = KS_error(confidences,labels)
    print(KS_e)