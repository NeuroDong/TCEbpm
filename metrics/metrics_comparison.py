from .TCE_BPM import logit_logit,beta_density,generating_data_from_a_distribution,TCE_BPM
import matplotlib.pyplot as plt
import numpy as np
from .naive_ECE import ECE_LOSS_equal_mass
from .ECE_sweep import CalibrationMetric
from .Debaised_ECE import Debaised_ECE
from .KS_ECE import KS_error
from .Smoothing_ECE import smECE_fast
from .LS_ECE import logit_smoothed_ece
import logging
import mpmath

def TCE_compute(true_function,confidence_distribution):
    def f(x):
        return np.abs(true_function.compute(x) - x)*confidence_distribution.compute(x)
    
    mpmath.mp.dps = 15
    result = mpmath.quad(f,[0,1])
    result = float(mpmath.nstr(result,5))
    return result
    
def visual_error_comparation():
    #D1
    Ps_fit_fun = logit_logit()
    alpha = 1.1233
    beta = 0.1147
    confidence_disribution = beta_density(alpha,beta)

    sampling_nums = [i*500 for i in range(1,11)]
    TCE_list = [[] for i in range(len(sampling_nums))]
    ECE_bin_list = [[] for i in range(len(sampling_nums))]
    ECE_sweep_list = [[] for i in range(len(sampling_nums))]
    ECE_debaised_list = [[] for i in range(len(sampling_nums))]
    KS_error_list = [[] for i in range(len(sampling_nums))]
    Smooth_ECE_list = [[] for i in range(len(sampling_nums))]
    LS_ECE_list = [[] for i in range(len(sampling_nums))]
    Estimated_TCE_list = [[] for i in range(len(sampling_nums))]
    
    TCE = TCE_compute(Ps_fit_fun,confidence_disribution)
    n_bins = 15
    run_num = 100
    for epoch  in range(run_num):
        for i in range(len(sampling_nums)):
            confidences = np.random.beta(alpha,beta,size=sampling_nums[i])   # Reference: Section 6 in Mitigating Bias in Calibration Error Estimation
            confidences = np.sort(confidences)
            y_list = generating_data_from_a_distribution(Ps_fit_fun,confidences)
            y_list = np.array(y_list)
            
            TCE_list[i].append(TCE)

            ECE_bin_em = ECE_LOSS_equal_mass(n_bins=n_bins)
            ece_bin_em = ECE_bin_em(confidences,y_list)
            ECE_bin_list[i].append(ece_bin_em)

            Metric_fun = CalibrationMetric(ce_type="em_ece_sweep",num_bins=n_bins,bin_method="equal_examples",norm=1)
            ECE_sweep_em = Metric_fun.compute_error(confidences,y_list)
            ECE_sweep_list[i].append(ECE_sweep_em)

            ECE_debias_em = Debaised_ECE(confidences,y_list,p=1,debias=True,num_bins = n_bins, binning_mode = "em")
            ECE_debaised_list[i].append(ECE_debias_em)

            ks_error = KS_error(confidences,y_list)
            KS_error_list[i].append(ks_error)

            sm_ECE = smECE_fast(confidences,y_list)
            Smooth_ECE_list[i].append(sm_ECE)

            Ls_ECE = logit_smoothed_ece(confidences,y_list)
            LS_ECE_list[i].append(Ls_ECE)

            TCE_bpm = TCE_BPM(confidences,y_list)
            Estimated_TCE_list[i].append(TCE_bpm)

        logging.info(f"The {epoch}-th run is completed, with a total of {run_num} runs.")

    TCE_list = [np.mean(TCEs) for TCEs in TCE_list]
    ECE_bin_list = [np.mean(ECE_bins) for ECE_bins in ECE_bin_list]
    ECE_sweep_list = [np.mean(ECE_sweeps) for ECE_sweeps in ECE_sweep_list]
    ECE_debaised_list = [np.mean(ECE_debaiseds) for ECE_debaiseds in ECE_debaised_list]
    KS_error_list = [np.mean(KS_errors) for KS_errors in KS_error_list]
    Smooth_ECE_list = [np.mean(Smooth_ECEs) for Smooth_ECEs in Smooth_ECE_list]
    LS_ECE_list = [np.mean(LS_ECEs) for LS_ECEs in LS_ECE_list]
    Estimated_TCE_list = [np.mean(Estimated_TCEs) for Estimated_TCEs in Estimated_TCE_list]

    plt.figure(figsize=(10,10))
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H']
    fontsize = 30
    plt.plot(sampling_nums,TCE_list, marker=markers[0],label = "$TCE$")
    plt.plot(sampling_nums,ECE_bin_list, marker=markers[1],label="$ECE_{bin}$")
    plt.plot(sampling_nums,ECE_sweep_list, marker=markers[2],label="$ECE_{sweep}$")
    plt.plot(sampling_nums,ECE_debaised_list, marker=markers[3],label = "$ECE_{debaised}$")
    plt.plot(sampling_nums,KS_error_list, marker=markers[4],label = "KS-error")
    plt.plot(sampling_nums,Smooth_ECE_list, marker=markers[5],label = "smECE")
    plt.plot(sampling_nums,LS_ECE_list, marker=markers[5],label = "LS-ECE")
    plt.plot(sampling_nums,Estimated_TCE_list, marker=markers[6], linewidth = 3, label = "$TCE_{bpm}$ (ours)")
    plt.xlabel('Number of samples',fontname="Times New Roman",fontsize=fontsize)
    plt.ylabel(f'Calibration error (%)(mean of {run_num} runs)',fontname="Times New Roman",fontsize=fontsize)
    plt.legend(prop={"family": "Times New Roman","size":fontsize},loc="upper left",ncol = 2,frameon=False,columnspacing=0.1)
    plt.tick_params(axis='both', labelsize=fontsize)
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax = plt.gca()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)  
        ax.spines[axis].set_color('black')  
    y_max = max([max(TCE_list),max(ECE_bin_list),max(ECE_sweep_list),max(ECE_debaised_list),max(KS_error_list),max(Estimated_TCE_list)])
    y_min = min([min(TCE_list),min(ECE_bin_list),min(ECE_sweep_list),min(ECE_debaised_list),min(KS_error_list),min(Estimated_TCE_list)])
    plt.ylim([y_min,y_max+0.005])

    def format_yaxis(x, pos):
        return f"{x*100:.1f}"

    from matplotlib.ticker import FuncFormatter
    formatter = FuncFormatter(format_yaxis)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.show()