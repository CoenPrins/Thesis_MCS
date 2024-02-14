import ast
import pandas as pd
import itertools
from scipy.stats import ks_2samp
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats import shapiro
""""

Old version still contains some failed statistical analysis tools! 

Like the vs statistical plots of all the means etc. Still handy to keep around, 

Since it can provide some interresting pivot stuff!

"""





# Function to calculate Euclidean distance
def euclidean_dist(x1, x2):
 
    dist_ar = x1 - x2
    return dist_ar, np.mean(dist_ar)


# return bootstrap N mean bootstrapped samples
def bootstrap_sample(data, N):
    mean_values = np.empty(N)  

    for i in range(N):
        random_samples = np.random.choice(data, size=len(data), replace=True)
        mean_values[i] = np.mean(random_samples)

    # print(mean_values)
    return mean_values

# Function to compute mean Euclidean distance differences
def mean_distance_difference(method1, method2):
    return np.mean(method1 - method2)

# Function to perform the statistical test using bootstrapping
def bootstrap_stat_test(method1, method2, num_bootstraps):

    # just so the bootstrap doesn't crash for calculiting for identical arrays
    # changing the order shouldn't matter for statistical analsysis. 


    # print("mean_method1", np.mean(method1))
    # print("mean_method2", np.mean(method2))
    # stat_shap, p_val_shap = shapiro(method1)

    # print("normal p val1", p_val_shap)

    # stat_shap, p_val_shap = shapiro(method2)

    # print("normal p val2", p_val_shap)

    # print("method1", method1)
    # print("method2", method2)
    np.random.shuffle(method1)
    np.random.shuffle(method2)

    # print("method1", method1)
    # print("method2", method2)
    # print("method1", type(method1))
    # print("method2", type(method2))
    
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

# convoluted code to convert the mean and 
def convert_to_float_list(string_representation):
    try:
        # Convert the string to a list using ast.literal_eval and convert each element to float
        lst = ast.literal_eval(string_representation)
        return [float(item) for item in lst]
    except (SyntaxError, ValueError):
        # Handle the case where the string cannot be evaluated as a list
        return None
    
def boxplot(df):
    
    sns.set(style="whitegrid")  # optional styling
    plt.figure(figsize=(8, 6))  # optional, set the figure size

    sns.violinplot(data=df)
    plt.title('Boxplot of Group A and Group B')
    plt.show()


# Read the CSV files
df= pd.read_csv('data\\results\\full_av_lp2.csv', index_col=0)



# Convert the 'path1' column from string representation to a list of floats

# path_list = ["path1", "path2", "path3"]

    # print(result_df)

df_av_t10 = df.drop_duplicates(subset='path_av_av', keep='first')
df_av_t10 = df_av_t10[df_av_t10["path_len"] == "long_path"].nsmallest(10, 'path_av_av')
# df_t10 = df.nsmallest(10, 'path_av_av')




df_av_t10.to_csv("data\\results\\table_av_t10.csv")
df_av_t10_latex= df_av_t10[['threshold', 'dt', 'mis_val', 'knn', 'loc_method', 'path1_av', 'path2_av', 'path3_av', 'path_av_av']]


df_1_t5 = df.drop_duplicates(subset='path1_av', keep='first')

# Apply the function to create a new column 'list_of_floats'
df_1_t5['path1_av'] = df_1_t5['path1_av'].apply(convert_to_float_list)


df_1_t5['first_1'] = df_1_t5['path1_av'].apply(lambda x: x[0] if x else None)
df_1_t5 = df_1_t5[df_1_t5["path_len"] == "long_path"].nsmallest(10, 'first_1')
# df_t10 = df.nsmallest(10, 'path_av_av')



df_1_t5.to_csv("data\\results\\table_1_t5.csv")


df_1_t5_latex= df_1_t5[['threshold', 'dt', 'mis_val', 'knn', 'loc_method', 'path1_av', 'path2_av', 'path3_av', 'path_av_av']]


df_2_t5 = df.drop_duplicates(subset='path2_av', keep='first')

# Apply the function to create a new column 'list_of_floats'
df_2_t5['path2_av'] = df_2_t5['path2_av'].apply(convert_to_float_list)



df_2_t5['first_2'] = df_2_t5['path2_av'].apply(lambda x: x[0] if x else None)
df_2_t5 = df_2_t5[df_2_t5["path_len"] == "long_path"].nsmallest(10, 'first_2')
# df_t10 = df.nsmallest(10, 'path_av_av')



df_2_t5.to_csv("data\\results\\table_2_t5.csv")
df_2_t5_latex= df_2_t5[['threshold', 'dt', 'mis_val', 'knn', 'loc_method', 'path1_av', 'path2_av', 'path3_av', 'path_av_av']]



df_3_t5 = df.drop_duplicates(subset='path3_av', keep='first')

# Apply the function to create a new column 'list_of_floats'
df_3_t5['path3_av'] = df_3_t5['path3_av'].apply(convert_to_float_list)


# print(df_3_t5['path3_av'])

df_3_t5['first_3'] = df_3_t5['path3_av'].apply(lambda x: x[0] if x else None)
df_3_t5 = df_3_t5[df_3_t5["path_len"] == "long_path"].nsmallest(10, 'first_3')
# df_t10 = df.nsmallest(10, 'path_av_av')



df_3_t5.to_csv("data\\results\\table_3_t5.csv")

df_3_t5_latex= df_3_t5[['threshold', 'dt', 'mis_val', 'knn', 'loc_method', 'path1_av', 'path2_av', 'path3_av', 'path_av_av']]


"""
Print the top 10 top 5 tables here! (these are important for the plotting of the cool plots)
"""


print("top10 for average path!")
print(df_av_t10_latex.to_latex(index=False))
print()
print()


print("top5 for path1!")
print(df_1_t5_latex.to_latex(index=False))
print()
print()


print("top5 for path2!")
print(df_2_t5_latex.to_latex(index=False))

print()
print()


print("top5 for path3!")
print(df_3_t5_latex.to_latex(index=False))


print()
print()


# the specific 

# Threshold & Penalize 
analyse_columns = ['threshold', 'mis_val']


"""

Threshold 
# """

# thres_df = df.copy()

# thres_df['Path_Average'] = thres_df["path_av_av"]


# thres_df['path1_av'] = thres_df['path1_av'].apply(convert_to_float_list)
# thres_df['Path_1'] = thres_df['path1_av'].apply(lambda x: x[0] if x else None)

# thres_df['path2_av'] = thres_df['path2_av'].apply(convert_to_float_list)
# thres_df['Path_2'] = thres_df['path2_av'].apply(lambda x: x[0] if x else None)


# thres_df['path3_av'] = thres_df['path3_av'].apply(convert_to_float_list)
# thres_df['Path_3'] = thres_df['path3_av'].apply(lambda x: x[0] if x else None)




# # thres_group_df = thres_df.groupby(['threshold', 'mis_val'])[['Path_Average', 'Path_1', 'Path_2', 'Path_3']].agg(['mean', 'std'])


# # print(thres_group_df)


# thres_box_df = pd.DataFrame()
# # test = thres_df[(thres_df['threshold'] == 'lp2') & (thres_df['mis_val'] == "penalty")]['path_av_av'].values
# # print(test)

# thres_box_df["No Threshold, \n Penalty"] = thres_df[(thres_df['threshold'] == 'lp2') & (thres_df['mis_val'] == "penalty")]['path_av_av'].values
# thres_box_df["No Threshold, \n Ignore"] =  thres_df[(thres_df['threshold'] == 'lp2') & (thres_df['mis_val'] == "ignore")]['path_av_av'].values
# thres_box_df["-100 Threshold, \n Penalty"] = thres_df[(thres_df['threshold'] == 'lp2_n100') & (thres_df['mis_val'] == "penalty")]['path_av_av'].values
# thres_box_df["-100 Threshold, \n Ignore"] = thres_df[(thres_df['threshold'] == 'lp2_n100') & (thres_df['mis_val'] == "ignore")]['path_av_av'].values
# thres_box_df["-85 Threshold, \n Penalty"] = thres_df[(thres_df['threshold'] == 'lp2_n85') & (thres_df['mis_val'] == "penalty")]['path_av_av'].values
# thres_box_df["-85 Threshold, \n Ignore"] = thres_df[(thres_df['threshold'] == 'lp2_n85') & (thres_df['mis_val'] == "ignore")]['path_av_av'].values


# test = boxplot(thres_box_df)

# thres_group_test = thres_df.groupby(['threshold', 'mis_val'])[['Path_Average', 'Path_1', 'Path_2', 'Path_3']]


# # Create a list to store the results
# results = []

# # Get unique combinations of threshold and mis_val from the DataFrame
# unique_combinations = thres_group_test.groups.keys()
# print(unique_combinations, "unique thres")
# # Iterate over all combinations of group pairs and paths
# for (threshold1, mis_val1), (threshold2, mis_val2) in itertools.product(unique_combinations, repeat=2):
#     data_group1 = thres_group_test.get_group((threshold1, mis_val1))
#     data_group2 = thres_group_test.get_group((threshold2, mis_val2))

    
#     # Perform Kolmogorov-Smirnov test for each path
#     for path in ['Path_Average', 'Path_1', 'Path_2', 'Path_3']:

#         print("datagroup1pathlen", len(data_group1[path]))
#         print("datagroup1pathlen", len(data_group2[path]))
#         print(len(thres_df))


#         conf_interval, p_val, observed_dif, bootstrap_mean  = bootstrap_stat_test(data_group1[path].values, data_group2[path].values, 1000)

        
#         # Append results to the list
#         results.append({'Group_1': (threshold1, mis_val1), 'Group_2': (threshold2, mis_val2), 'Path': path, 'P-Value': p_val})



# # Create the DataFrame from the list
# p_df_thres = pd.DataFrame(results)

# # Displaying the p-values DataFrame
# # print(p_values_df)

# p_df_thres.to_csv("data\\results\\test_significance.csv")

# p_df_thres_av = p_df_thres[p_df_thres["Path"] == "Path_Average"]

# p_df_thres_1 = p_df_thres[p_df_thres["Path"] == "Path_1"]
# p_df_thres_2 = p_df_thres[p_df_thres["Path"] == "Path_2"]
# p_df_thres_3 = p_df_thres[p_df_thres["Path"] == "Path_3"]


# p_df_thres_av.to_csv("data\\results\\test_Path_average.csv")
# # Displaying the p-values Da

# pivot_df_av = p_df_thres_av.pivot(index='Group_1', columns='Group_2', values='P-Value')
# pivot_df_1 = p_df_thres_1.pivot(index='Group_1', columns='Group_2', values='P-Value')
# pivot_df_2  = p_df_thres_2.pivot(index='Group_1', columns='Group_2', values='P-Value')
# pivot_df_3  = p_df_thres_3.pivot(index='Group_1', columns='Group_2', values='P-Value')


# print(pivot_df_av)
# print(pivot_df_1)
# print(pivot_df_2)
# print(pivot_df_3)



# # print(pivot_df.to_latex(float_format='%.2e'))
# print('heyoooo')
# print(thres_group_df)


"""
Knn

"""


knn_df = df.copy()

knn_df['Path_Average'] = knn_df["path_av_av"]


# print("knn_path3")
# print(knn_df['path3_av'])

knn_df['path1_av'] = knn_df['path1_av'].apply(convert_to_float_list)
knn_df['Path_1'] = knn_df['path1_av'].apply(lambda x: x[0] if x else None)

knn_df['path2_av'] = knn_df['path2_av'].apply(convert_to_float_list)
knn_df['Path_2'] = knn_df['path2_av'].apply(lambda x: x[0] if x else None)



knn_df['path3_av'] = knn_df['path3_av'].apply(convert_to_float_list)



knn_df['Path_3'] = knn_df['path3_av'].apply(lambda x: x[0] if x else None)


# print(knn_df["Path_3"])

knn_group_df = knn_df.groupby(['knn'])[['Path_Average', 'Path_1', 'Path_2', 'Path_3']].agg(['mean', 'std'])

# print(knn_group_df)

knn_box_df = pd.DataFrame()


knn_box_df['k = 1'] = knn_df[knn_df['knn'] == 1]['path_av_av'].values
knn_box_df['k = 3'] = knn_df[knn_df['knn'] == 3]['path_av_av'].values
knn_box_df['k = 6'] = knn_df[knn_df['knn'] == 6]['path_av_av'].values
knn_box_df['k = 12'] = knn_df[knn_df['knn'] == 12]['path_av_av'].values

print("test wtf are these values", knn_box_df['k = 1'])
knn_box = boxplot(knn_box_df)

unique_values = knn_df['knn'].unique()


results = []

# Iterate over all pairs of unique values
for value1, value2 in itertools.product(unique_values, repeat=2):
    data_group1 = knn_df[knn_df['knn'] == value1]
    data_group2 = knn_df[knn_df['knn'] == value2]
    # Perform Kolmogorov-Smirnov test for each path
    for path in ['Path_Average', 'Path_1', 'Path_2', 'Path_3']:
        if path == 'Path_Average':
            pass

        # stat, p_val = ks_2samp(data_group1[path], data_group2[path])
        # test = bootstrap(data_group1[path], data_group2[path])
        conf_interval, p_val, observed_dif, bootstrap_mean  = bootstrap_stat_test(data_group1[path].values, data_group2[path].values, 1000)
        # Append results to the list
        results.append({'Group_1': value1, 'Group_2': value2, 'Path': path, 'P-Value': p_val})

# Create the DataFrame from the list
p_df_knn = pd.DataFrame(results)

print("knn TEST")
print("knn TEST")
print("knn TEST")
print("knn TEST")

# print(p_values_df_knn)

p_df_knn_av = p_df_knn[p_df_knn["Path"] == "Path_Average"]
p_df_knn_1 = p_df_knn[p_df_knn["Path"] == "Path_1"]
p_df_knn_2 = p_df_knn[p_df_knn["Path"] == "Path_2"]
p_df_knn_3 = p_df_knn[p_df_knn["Path"] == "Path_3"]


pivot_df_knn_av = p_df_knn_av.pivot(index='Group_1', columns='Group_2', values='P-Value')
pivot_df_knn_1 = p_df_knn_1.pivot(index='Group_1', columns='Group_2', values='P-Value')
pivot_df_knn_2 = p_df_knn_2.pivot(index='Group_1', columns='Group_2', values='P-Value')
pivot_df_knn_3 = p_df_knn_3.pivot(index='Group_1', columns='Group_2', values='P-Value')

print(pivot_df_knn_av)
print(pivot_df_knn_1)
print(pivot_df_knn_2)
print(pivot_df_knn_3)
print("test done")



"""
LOCMETH
"""




# locmeth_df = df.copy()

# locmeth_df['Path_Average'] = locmeth_df["path_av_av"]


# # print("knn_path3")
# # print(locmeth_df['path3_av'])

# locmeth_df['path1_av'] = locmeth_df['path1_av'].apply(convert_to_float_list)
# locmeth_df['Path_1'] = locmeth_df['path1_av'].apply(lambda x: x[0] if x else None)

# locmeth_df['path2_av'] = locmeth_df['path2_av'].apply(convert_to_float_list)
# locmeth_df['Path_2'] = locmeth_df['path2_av'].apply(lambda x: x[0] if x else None)



# locmeth_df['path3_av'] = locmeth_df['path3_av'].apply(convert_to_float_list)



# locmeth_df['Path_3'] = locmeth_df['path3_av'].apply(lambda x: x[0] if x else None)


# # print(knn_df["Path_3"])

# locmeth_group_df = locmeth_df.groupby(['loc_method'])[['Path_Average', 'Path_1', 'Path_2', 'Path_3']].agg(['mean', 'std'])

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_colwidth', None)

# # print(locmeth_group_df)

# locmeth_box_df = pd.DataFrame()



# locmeth_box_df["Average"] = locmeth_df[locmeth_df['loc_method'] == 'average']['path_av_av'].values
# locmeth_box_df["Gaussian Weighted, \n Standard"] = locmeth_df[locmeth_df['loc_method'] == 'gaus_normal']['path_av_av'].values
# locmeth_box_df["Gaussian Weighted, \n Top 80 Percent"] = locmeth_df[locmeth_df['loc_method'] == 'gaus_top']['path_av_av'].values
# locmeth_box_df["Gaussian, \n Largest Probability"] = locmeth_df[locmeth_df['loc_method'] == 'gaus_top']['path_av_av'].values

# loc_box = boxplot(locmeth_box_df)



# unique_values = locmeth_df['loc_method'].unique()


# results = []

# # Iterate over all pairs of unique values
# for value1, value2 in itertools.product(unique_values, repeat=2):
#     data_group1 = locmeth_df[locmeth_df['loc_method'] == value1]
#     data_group2 = locmeth_df[locmeth_df['loc_method'] == value2]
#     # Perform Kolmogorov-Smirnov test for each path
#     for path in ['Path_Average', 'Path_1', 'Path_2', 'Path_3']:
#         if path == 'Path_Average':
#             pass

#         stat, p_value = ks_2samp(data_group1[path], data_group2[path])
#         # Append results to the list
#         results.append({'Group_1': value1, 'Group_2': value2, 'Path': path, 'P-Value': p_value})

# # Create the DataFrame from the list
# p_df_locmeth = pd.DataFrame(results)



# # print(p_values_df_locmeth)

# p_df_locmeth_av = p_df_locmeth[p_df_locmeth["Path"] == "Path_Average"]
# p_df_locmeth_1 = p_df_locmeth[p_df_locmeth["Path"] == "Path_1"]
# p_df_locmeth_2 = p_df_locmeth[p_df_locmeth["Path"] == "Path_2"]
# p_df_locmeth_3 = p_df_locmeth[p_df_locmeth["Path"] == "Path_3"]


# pivot_df_locmeth_av = p_df_locmeth_av.pivot(index='Group_1', columns='Group_2', values='P-Value')
# pivot_df_locmeth_1 = p_df_locmeth_1.pivot(index='Group_1', columns='Group_2', values='P-Value')
# pivot_df_locmeth_2 = p_df_locmeth_2.pivot(index='Group_1', columns='Group_2', values='P-Value')
# pivot_df_locmeth_3 = p_df_locmeth_3.pivot(index='Group_1', columns='Group_2', values='P-Value')

# print(pivot_df_locmeth_av)
# print(pivot_df_locmeth_1)
# print(pivot_df_locmeth_2)
# print(pivot_df_locmeth_3)

"""
locmeth done

"""