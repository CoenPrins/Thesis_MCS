import ast
import pandas as pd
import itertools
import interpolate
from scipy.stats import ks_2samp
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import ttest_ind
import time_og_data2
from scipy import stats



"""

Old version still contains some failed statistical analysis tools! 

Like the vs statistical plots of all the means etc. Still handy to keep around, 

Since it can provide some interresting pivot stuff!




Okay you got your method! 


1. H0 path A and B are the same, samples drawn from them are indistinghuisible statistically 
2. Choose test stat = mean difference between the two samples safe this! 
4. Calculate the mean distance between the two! (okay letsgo)
5 Determine distr of test statistic, (resampling normal distribution)
6. 


Concate the two paths X and Y to XY 

do this 10k times 
    draw len(x) samples from X y and compute the mean 
    Draw len(y) samples from y and compute the mean
    Compute the difference between the means and save this into list 

"""


# https://stats.stackexchange.com/questions/524226/bootstrapping-for-groups-of-unequal-size 


# Function to calculate Euclidean distance
def euclidean_dist(x1, x2):
 
    dist_ar = x1 - x2
    return dist_ar, np.mean(dist_ar)


# return bootstrap N mean bootstrapped samples

# this possibly needs a turnaround with N and data ! 
def bootstrap_sample(data, N):
    mean_values = np.empty(N)  

    for i in range(N):
        random_samples = np.random.choice(data, size=len(data), replace=True)
        mean_values[i] = np.mean(random_samples)

    return mean_values

# Function to compute mean Euclidean distance differences
def mean_distance_difference(method1, method2):
    return np.mean(method1 - method2)

# Function to perform the statistical test using bootstrapping
def bootstrap_stat_test(method1, method2, num_bootstraps):

    

    # print(np.mean(method1))
    # print(np.mean(method2))
    bootstrap_means1 = bootstrap_sample(method1, num_bootstraps)
    bootstrap_means2 = bootstrap_sample(method2, num_bootstraps)

    stat_shap, p_val_shap = shapiro(bootstrap_means1)

    print("normal p val", p_val_shap)


    stat_shap, p_val_shap = shapiro(bootstrap_means2)
    
    print("normal p val", p_val_shap)
    
    statistic, p_value = levene(bootstrap_means1, bootstrap_means2)

    print("leven_p", p_value)


    statistic, p_value = ttest_ind(bootstrap_means1, bootstrap_means2)

    print("final", p_value)

    mean1 = np.mean(bootstrap_means1)
    mean2= np.mean(bootstrap_means2)
    print('mean1', mean1)
    print('mean2', mean2)

    std1 = np.std(bootstrap_means1)
    std2 = np.std(bootstrap_means2)


    print("conf calc self 1", mean1 - 2*std1, mean1 + 2*std1 )
    print("conf calc self 2", mean2 - 2*std2, mean2 + 2*std2 )


    confidence_interval1 = np.percentile(bootstrap_means1, [2.5, 97.5])
    confidence_interval2 = np.percentile(bootstrap_means2, [2.5, 97.5])

    print("confidence1", confidence_interval1)
    print("confidence2", confidence_interval2)

        # Plot histograms
    plt.hist(bootstrap_means1, bins=30, alpha=0.5, label='Distribution 1')
    plt.hist(bootstrap_means2, bins=30, alpha=0.5, label='Distribution 2')

    # Add labels and legend
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


    # plt.hist(bootstrap_means, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    # plt.title('Data Distribution')
    # plt.xlabel('Values')
    # plt.ylabel('Density')
    # # plt.show()
  

    # # Calculate a 95% confidence interval
    # confidence_interval = np.percentile(bootstrap_means, [2.5, 97.5])
    # print("con interval", confidence_interval)


    # # Calculate the p-value
    # p_value = np.mean(bootstrap_means >= abs(mean_diff)) + np.mean(bootstrap_means <= -abs(mean_diff))
    # print("p_value", p_value)
    # print()
    # print()
    # print()

    # return confidence_interval, p_value, observed_diff, bootstrap_means



# Function to compute mean Euclidean distance differences
def mean_distance_difference(method1, method2):
    return np.mean(method1 - method2)

# Function to perform the statistical test using bootstrapping
def bootstrap_stat_test_dif(method1, method2, num_bootstraps):


    
    method1 = np.array(method1)
    method2 = np.array(method2)
    
    observed_diff, mean_diff = euclidean_dist(method1, method2)
    # print("observed_diff", observed_diff)
    print("mean_dif", mean_diff)
    # # print("observed_diff", observed_diff)
    # mean_diff = mean_distance_difference(method1, method2)
    # print("mean_diff2", mean_diff)

    # Shifting observed differences to have mean zero
    shift_diff = observed_diff - mean_diff

    # print("shift_dif", shift_diff)
    # stat_shap, p_val_shap = shapiro(shift_diff)

    # print("normal p val shift", p_val_shap)
   

    # Bootstrapping
    bootstrap_means = bootstrap_sample(shift_diff, num_bootstraps)

    # stat_shap, p_val_shap = shapiro(bootstrap_means)

    # print("normal p val", p_val_shap)

    # plt.hist(bootstrap_means, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    # plt.title('Data Distribution')
    # plt.xlabel('Values')
    # plt.ylabel('Density')
    # plt.show()
  

    # Calculate a 95% confidence interval
    confidence_interval = np.percentile(bootstrap_means, [2.5, 97.5])
    print("con interval", confidence_interval)


    # Calculate the p-value
    p_value = np.mean(bootstrap_means >= abs(mean_diff)) + np.mean(bootstrap_means <= -abs(mean_diff))
    print("p_value", p_value)
    print()
    print()
    print()

    return confidence_interval, p_value, observed_diff, bootstrap_means



def bootstrap_sample_concat(method_conc, len_1, len_2, N):
    mean_values = np.empty(N)  

    for i in range(N):


        random_samples1 = np.random.choice(method_conc, size= len_1, replace=True)
        mean1 = np.mean(random_samples1)

        random_samples2 = np.random.choice(method_conc, size= len_2, replace=True)
        mean2 = np.mean(random_samples2)


        mean_diff = abs(mean1 - mean2)

        mean_values[i] = mean_diff




    return mean_values






    return mean_values
def bootstrap_concate(method1, method2, num_bootstraps):

    mean1 = np.mean(method1)
    mean2 = np.mean(method2)

    mean_diff = abs(mean1 - mean2)

    # print(mean_diff)


    method_conc = method1 + method2



    boot_mean_diff = bootstrap_sample_concat(method_conc, len(method1), len(method2), num_bootstraps)


    p_val = np.mean(boot_mean_diff >= abs(mean_diff))
    # print("p_val", p_val)
    # plt.hist(boot_mean_diff, bins=30, alpha=0.5, label='Distribution 1')
    # plt.axvline(mean_diff, color="red")

    # # Add labels and legend
    # plt.xlabel('Values')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.show()
    return p_val


    
    

    




    
# n_og_paths = [time_og_data2.n_og_path1, time_og_data2.n_og_path2, time_og_data2.n_og_path3]
# path_time_ls = [[time_og_data2.start_time1, time_og_data2.end_time1], [time_og_data2.start_time2, time_og_data2.end_time2], [time_og_data2.start_time3, time_og_data2.end_time3]]

# long_path = [n_og_paths, path_time_ls]

# # df= pd.read_csv('data\\results\\full_av_lp2.csv', index_col=0)

# dt_1 = 3
# radio_rssi_1 = "lp2"
# mis_val_mtd_1 = "ignore"
# knn_N_1 = 6
# loc_method_1 = "gaus_top"


# dt_2 = 10
# radio_rssi_2 = "lp2"
# mis_val_mtd_2 = "ignore"
# knn_N_2 = 6
# loc_method_2 = "average"

# plotfigs= False


# path_1 = f"data\\paths\\dt_{dt_1}\\raw\\{radio_rssi_1}\\{mis_val_mtd_1}\\{knn_N_1}\\{loc_method_1}\\path"
# path_2 = f"data\\paths\\dt_{dt_2}\\raw\\{radio_rssi_2}\\{mis_val_mtd_2}\\{knn_N_2}\\{loc_method_2}\\path"




# av_ls_1, best_ls, av_i_error, av_er_t_ls_1, b_er_t_ls = interpolate.accuracy(long_path[0] ,path_1, long_path[1], plt_paths =plotfigs, av_plt=False, best_plt=False, lines_plt=False, filter = None, filename=False)

# av_ls_2, best_ls, av_i_error, av_er_t_ls_2, b_er_t_ls = interpolate.accuracy(long_path[0] ,path_2, long_path[1], plt_paths =plotfigs, av_plt=False, best_plt=False, lines_plt=False, filter = None, filename=False)


# you just take the largest one as is! 


# # this doesn't have to be a problem if you bootstrap by drawing from a distribution
# # By the by they are just distributions of things not 



# # if you just draw from distribution you can create normal distributions from the error, and create in this way a normal error. 

# # of course there will be some spread but that's to be expected. 
# # dt_2 always has to be the larger one! 
# av_er_total_1 = sum(av_er_t_ls_1, [])
# print("av_ls_total", len(av_er_total_1))
 

# av_er_total_2 = sum(av_er_t_ls_2, [])
# print("av_ls_total", len(av_er_total_2))

# # bootstrap_stat_test(av_er_total_1, av_er_total_2, 100000)

# # bootstrap_stat_test_dif (av_er_total_1, av_er_total_2, 100000)



# test = bootstrap_concate(av_er_total_1, av_er_total_2, 100000)
# # problem we can't compare 
# # conf_interval, p_val, observed_dif, bootstrap_mean  = bootstrap_stat_test(data_group1[path].values, data_group2[path].values, 1000)



