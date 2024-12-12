import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from .naive_ECE import ECE_binAcc_em
from scipy import integrate
from scipy.special import beta as betafn
from scipy.stats import beta as betafit
import mpmath

def logit(s):
    s_clipped = np.clip(s, 1e-10, 1 - 1e-10)
    if isinstance(s,np.ndarray):
        return np.log(s_clipped / (1 - s_clipped))
    else:
        return mpmath.log(s_clipped / (1 - s_clipped))

def sigmoid(s):
    if isinstance(s,np.ndarray):
        return 1/(1+np.exp(-s))
    else:
        return 1/(1+mpmath.exp(-s))

def beta_prior(s,param):
    '''
    Reference: Beyond sigmoids: How to obtain well-calibrated probabilities from binary classifiers with beta calibration
    '''
    if isinstance(s,list):
        s = np.array(s)
    s_clipped = np.clip(s, 1e-10, 1 - 1e-10)
    out = 1/(1+1/((mpmath.exp(param[2]))*((s_clipped**param[0])/((1-s_clipped)**param[1]))))
    return out

class beta_density:
    def __init__(self,alpha = 1.,beta = 1.0) -> None:
        self.alpha = alpha
        self.beta = beta

    def compute(self,s):
        if isinstance(s,list):
            s = np.array(s)
        s_clipped = np.clip(s, 1e-10, 1 - 1e-10)
        out = (s_clipped**(self.alpha-1))*((1-s_clipped)**(self.beta-1))/betafn(self.alpha,self.beta)
        return out

def P_D_1_B(init_params,confidences, accuracies,prior):
    num_bins = [10 + i for i in range(40)]
    min_num_bin = len(confidences) // 100
    max_num_bin = len(confidences) // 20
    step = (max_num_bin-min_num_bin) // 10
    num_bins = range(min_num_bin,max_num_bin+1,step)
    p_D_M = 0.

    for num_bin in num_bins:
        mass_in_bin = len(confidences)//num_bin
        p_D_M_B = 0.
        
        for i in range(num_bin-1):
            Ps = confidences[i*mass_in_bin:(i+1)*mass_in_bin]
            Ps_mean = Ps.mean()
            acc_in_bin = accuracies[i*mass_in_bin:(i+1)*mass_in_bin].mean()
            Ps_mean = prior(Ps_mean,init_params)
            p_D_M_B = p_D_M_B + math.exp(-((Ps_mean-acc_in_bin)**2))
            #p_D_M_B = p_D_M_B + (1/math.exp((Ps[-1]-Ps[0])**0.1)) * math.exp(-((Ps_mean-acc_in_bin)**2))

        p_D_M = p_D_M + p_D_M_B
    return -p_D_M

def Maximum_likelihood_solution(confidences, accuracies, prior):
    Likelihood = P_D_1_B
    a = 5.
    b = 5.
    c = b*math.log(1-0.5)+a*math.log(0.5)
    parm = (a,b,c)
    result = minimize(Likelihood,parm, args=(confidences, accuracies,prior),method="Nelder-Mead",options={'fatol': 1e-10})
    return result.x

def TCE_BMP_compute(prior,param,confidence_distribution,low_density_threshold=None):
    mpmath.mp.dps = 15
    def f1(x):
        return confidence_distribution.compute(x)
    
    if low_density_threshold == None:
        division = 1.0
    else:
        division = mpmath.quad(f1, [0, low_density_threshold])

    def f2(x):
        return np.abs(prior(x,param) - x)*confidence_distribution.compute(x)/division
    
    if low_density_threshold == None:
        result = mpmath.quad(f2, [0,1])
    else:
        result = mpmath.quad(f2, [0,low_density_threshold])

    result = float(mpmath.nstr(result,5))
    return result

def TCE_BPM(confidences, accs, confidence_distribution = None,low_density_threshold = None):
    prior = beta_prior
    param = Maximum_likelihood_solution(confidences, accs,prior)
    if confidence_distribution == None:
        mean = np.mean(confidences)
        var = np.var(confidences, ddof=1)
        a = mean**2 * (1 - mean) / var - mean
        b = a * (1 - mean) / mean
        #confidences_clipped = np.clip(confidences, 1e-10, 1 - 1e-10)
        #a, b, loc, scale = betafit.fit(confidences_clipped,floc=0, fscale=1)
        confidence_distribution = beta_density(a,b)
    TCE = TCE_BMP_compute(prior,param,confidence_distribution,low_density_threshold=low_density_threshold)
    return TCE

def plot_Max_Min_Bands(x_list,y_list_list):
    mean = np.mean(y_list_list,axis=1)
    min_line = np.min(y_list_list,axis=1)
    max_line = np.max(y_list_list,axis=1)
    index = [max_line[i]!=0 for i in range(len(max_line))]
    x_list = np.array(x_list)
    x_list = x_list[index]
    mean = mean[index]
    min_line = min_line[index]
    max_line = max_line[index]

    plt.plot(x_list, mean, linewidth = 4, label='HB\'s mean result (bins from 10 to 50)', color='green')
    # 绘制数据带
    plt.fill_between(x_list, min_line, max_line, color='blue', alpha=0.2, label='HB\'s result range (bins from 10 to 50)')

    plt.xlabel('Confidence',fontname="Times New Roman",fontsize=30)
    #plt.ylabel('Accuracy',fontname="Times New Roman",fontsize=30)
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax = plt.gca()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)  # 设置线条宽度
        ax.spines[axis].set_color('black')  # 设置线条颜色

    plt.legend(prop={"family": "Times New Roman","size":30},loc="upper left")
    plt.ylim([0,1])
    plt.xlim([0,1])
    plt.tick_params(axis='both', labelsize=30)

def visual_TCE_BPM(confidences, y_true):
    prior = beta_prior
    param = Maximum_likelihood_solution(confidences, y_true,prior)
    num_bins = [10 + i for i in range(40)]
    xs = [i/50+0.01 for i in range(50)]
    xs_acc = [[]for i in range(len(xs))]
    for num_bin in num_bins:
        ECE_binAcc_compute = ECE_binAcc_em(num_bin)
        Accs,bin_boundaries = ECE_binAcc_compute(confidences,y_true)
        for k in range(len(xs)):
            h=0
            for i in range(len(bin_boundaries)):
               if xs[k] > bin_boundaries[i] and xs[k]<= bin_boundaries[i+1]:
                   h = 1
                   xs_acc[k].append(Accs[i])
            assert h==1, "wrong!"           
    
    plot_Max_Min_Bands(xs,xs_acc)
    s = [0.+i/1000 for i in range(1000)]
    out = prior(s,param)
    plt.plot(s,out, linewidth = 4,label="Estimated calibration curve")
    plt.legend(prop={"family": "Times New Roman","size":30},framealpha=0.1,loc="upper left")

class logit_logit:
    '''
    link functions is logit
    transform functions is logit
    Reference: Section 6 in Mitigating Bias in Calibration Error Estimation
    '''
    def __init__(self) -> None:
        self.beta0 = -0.88
        self.beta1 = 0.49

    def compute(self,x):
        if isinstance(x,list):
            x = np.array(x)
        p = sigmoid(self.beta0+self.beta1*logit(x))
        return p
    
def generating_data_from_a_distribution(Ps_fit_fun,confidences):
    sample_list = []
    for i in range(len(confidences)):
        Ps = Ps_fit_fun.compute(confidences[i])
        sample = np.random.binomial(1, Ps)
        sample_list.append(sample)
    return sample_list

def visual_fitting_effect():
    Ps_fit_fun = logit_logit()   #true calibration curve
    sampling_num = 10000
    confidences = np.random.beta(1.1233,0.1147,size=sampling_num)   # Reference: Section 6 in Mitigating Bias in Calibration Error Estimation
    confidences = np.sort(confidences)
    y_list = generating_data_from_a_distribution(Ps_fit_fun,confidences)
    y_list = np.array(y_list)

    # plot true distribution and our method
    s = [0.+i/1000 for i in range(1000)]
    out = Ps_fit_fun.compute(s)
    fig = plt.figure(figsize=(11,10))
    plt.plot(s,out, linewidth = 3,label="True calibration curve")
    visual_TCE_BPM(confidences,y_list)
    plt.legend(prop={"family": "Times New Roman","size":30},framealpha=0.1,loc="upper left")
    plt.show()