import ast
import pandas as pd
import itertools
from scipy.stats import ks_2samp
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats import shapiro
import bootstrap


"""

IN this file you will, 




Best vs average vs av i needs to be treated as a seperate thing..... DONE!


Make a little closing statement on best vs average. Show convincingly the averages of all. and the instances in term of conversion rate! DONE


Than you prove the validity of taking the 5 different devices! DONE

You can then remove best from the plots! DONE


Create the top 10 tables for path 1 path 2 and path 3 and path average. 
(which contains both the average and best for each option)

These top 10 tables will be bootstrapped against each other for significance. 
And violin plots will be created of the original distributions. 


You will also plot the boxplots of the distributions to compare (violin plots are weird)

For the best overall solution, for all three paths you make the 


Path long 
Path short 
Path error 









-------------------------------------------------------------------------



You will also create 2 more complex tables. 

1 for datatype, threshold and dt (multicolumn)

1 for knn and location method. 



additionally for each subthing you make a tiny plot of all aglomerates together 


"""



# Function to calculate Euclidean distance
def euclidean_dist(x1, x2):
 
    dist_ar = x1 - x2
    return dist_ar, np.mean(dist_ar)


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
df = df[df['path_len'] == 'long_path'] 



"""
Small lines of code for showing that averaging 5 path locations before computing average error yields better results!
"""

# df_5_device = df.copy()

# df_5_device['dif'] = df_5_device['path_av_av'] - df_5_device['path3_av_i_av']
# df_5_device['bool_sum'] = df_5_device['path_av_av'] < df_5_device['path3_av_i_av']

# mean = df_5_device['dif'].mean()
# sum = df_5_device['bool_sum'].sum()
# ratio = sum/len(df)

# print(df_5_device['bool_sum'])
# print("mean",mean)
# print('sum', sum)
# print('ratio', ratio)



"""

all the top 10 shit maybe you need this down the line idk, yes I DO! 


I want to save them in file for the entire length! 

I want to do statistical tests of the top 10, + the first occurence of non lp2, and the first occurence of non, dt10

If those are statistically significant that could be huge! 
"""

# df_av_t = df.drop_duplicates(subset='path_av_av', keep='first')


# df_av_t_csv = df_av_t[df_av_t["path_len"] == "long_path"].nsmallest(len(df_av_t), 'path_av_av')

# df_av_t_latex = df_av_t[df_av_t["path_len"] == "long_path"].nsmallest(10, 'path_av_av')
# # df_t10 = df.nsmallest(10, 'path_av_av')




# df_av_t_csv.to_csv("data\\results\\table_top_path_average.csv")


# print("top10 for average path!")
# df_av_t_latex= df_av_t_latex[['threshold', 'dt', 'mis_val', 'knn', 'loc_method', 'path1_av', 'path2_av', 'path3_av', 'path_av_av']]
# print(df_av_t_latex.to_latex(index=False))
# print()
# print()


# path_ls = ['path1_av', 'path2_av', 'path3_av']

# for path in path_ls:


#     df_i_t = df.drop_duplicates(subset=path, keep='first')

#     # Apply the function to create a new column 'list_of_floats'
#     df_i_t[path] = df_i_t[path].apply(convert_to_float_list)

#     path_mean = path + "_mean"
#     path_std = path+"_std"
#     df_i_t[path_mean] = df_i_t[path].apply(lambda x: x[0] if x else None)
#     df_i_t[path_std] = df_i_t[path].apply(lambda x: x[1] if x else None)

#     # here I want to safe the entire ordened 
#     df_i_t_csv= df_i_t[df_i_t["path_len"] == "long_path"].nsmallest(len(df_i_t[[path]]), path_mean)
#     # df_t10 = df.nsmallest(10, 'path_av_av')

#     df_i_t_csv.to_csv(f"data\\results\\table_top_{path}.csv")

#     df_i_t_latex = df_i_t[df_i_t["path_len"] == "long_path"].nsmallest(10, path_mean)

#     df_i_t_latex= df_i_t_latex[['threshold', 'dt', 'mis_val', 'knn', 'loc_method', 'path1_av', 'path2_av', 'path3_av', 'path_av_av']]
#     print(f"top10 for {path}")
#     print(df_i_t_latex.to_latex(index=False))
#     print()
#     print()





"""
Threshold dt and data big table
# """

# df_big1 = df.copy()



# df_big1['misval_thres'] = df_big1['threshold'] + '_' + df_big1['mis_val']
# df_big1 = df_big1.groupby(['dt', 'misval_thres'])['path_av_av'].agg(['mean', 'std']).reset_index().round(2)
# df_big1['mean_std'] = "\mu= " +  df_big1['mean'].astype(str) + ", \sigma= " + df_big1['std'].astype(str)



# df_big1_pivot = df_big1.pivot(index='dt', columns='misval_thres', values='mean_std')
# col_order = ['lp2_ignore', 'lp2_penalty',  'lp2_n100_ignore', 'lp2_n100_penalty', 'lp2_n85_ignore',
#  'lp2_n85_penalty']

# i_order = ["3s", "5s", "10s"]

# # Reorder columns and index
# df_big1_pivot = df_big1_pivot.reindex(i_order)
# df_big1_pivot = df_big1_pivot.reindex(col_order, axis='columns')


# print("table for dt misval thres path average")
# print(df_big1_pivot.to_latex())
# print()


# i_path_ls = ['path1_av', 'path2_av', 'path3_av']

# for path in i_path_ls:
#     df_big1_i = df.copy()
#     df_big1_i['misval_thres'] = df_big1_i['threshold'] + '_' + df_big1_i['mis_val']


#     df_big1_i[path] = df_big1_i[path].apply(convert_to_float_list)

#     df_big1_i[path] = df_big1_i[path].apply(lambda x: x[0] if x else None)

#     df_big1_i = df_big1_i.groupby(['dt', 'misval_thres'])[path].agg(['mean', 'std']).reset_index().round(2)
#     # $\substack{\mu= 15.22 \\ \sigma= 1.62}$
#     df_big1_i['mean_std'] = "$\substack{\mu= " +  df_big1_i['mean'].astype(str) + ",  \\\ \sigma= " + df_big1_i['std'].astype(str) + "}$"
#     df_big1_i_pivot = df_big1_i.pivot(index='dt', columns='misval_thres', values='mean_std')

#     df_big1_i_pivot = df_big1_i_pivot.reindex(i_order)
#     df_big1_i_pivot = df_big1_i_pivot.reindex(col_order, axis='columns')


    
#     print(f"table for dt misval thres path {path}")
#     print(df_big1_i_pivot.to_latex())
#     print()





# print()
# print()


"""
Loc method and knn big table 

"""

# df_big2 = df.copy()
# knn_col = df_big2['knn'].unique().tolist()
# meth_col = df_big2['loc_method'].unique().tolist()


# df_big2 = df_big2.groupby(['knn', 'loc_method'])['path_av_av'].agg(['mean', 'std']).reset_index().round(2)



# df_big2['mean_std'] = "\mu= " +  df_big2['mean'].astype(str) + ", \sigma= " + df_big2['std'].astype(str)


# df_big2_pivot = df_big2.pivot(index='loc_method', columns='knn', values='mean_std')
# col_order = [1, 3, 6, 12]
# i_order = ['average', 'gaus_normal', 'gaus_80', 'gaus_top']

# # Reorder columns and index
# df_big2_pivot = df_big2_pivot.reindex(i_order)
# df_big2_pivot = df_big2_pivot.reindex(col_order, axis='columns')



# print(f"table for knn loc path average")
# print(df_big2_pivot.to_latex())
# print()
# # print(df_big2.pivot(index='loc_method', columns='knn', values='mean_std').to_latex(float_format="%.2f"))


# i_path_ls = ['path1_av', 'path2_av', 'path3_av']

# for path in i_path_ls:
#     df_big2_i = df.copy()
   

#     df_big2_i[path] = df_big2_i[path].apply(convert_to_float_list)

#     df_big2_i[path] = df_big2_i[path].apply(lambda x: x[0] if x else None)

#     df_big2_i = df_big2_i.groupby(['knn', 'loc_method'])[path].agg(['mean', 'std']).reset_index().round(2)
    
#     df_big2_i['mean_std'] = "\mu= " +  df_big2_i['mean'].astype(str) + ", \sigma= " + df_big2_i['std'].astype(str)


#     df_big2_i_pivot = df_big2_i.pivot(index='loc_method', columns='knn', values='mean_std')

    
#     # Reorder columns and index
#     df_big2_i_pivot = df_big2_i_pivot.reindex(i_order)
#     df_big2_i_pivot = df_big2_i_pivot.reindex(col_order, axis='columns')

#     print(f"table for knn loc path {path}")
#     print(df_big2_i_pivot.to_latex())
#     print()
  


"""

Threshold 

"""

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



# print(pivot_df.to_latex(float_format='%.2e'))
# print('heyoooo')
# print(thres_group_df)


"""
Knn

"""


# knn_df = df.copy()

# knn_df['Path_Average'] = knn_df["path_av_av"]


# # print("knn_path3")
# # print(knn_df['path3_av'])

# knn_df['path1_av'] = knn_df['path1_av'].apply(convert_to_float_list)
# knn_df['Path_1'] = knn_df['path1_av'].apply(lambda x: x[0] if x else None)

# knn_df['path2_av'] = knn_df['path2_av'].apply(convert_to_float_list)
# knn_df['Path_2'] = knn_df['path2_av'].apply(lambda x: x[0] if x else None)



# knn_df['path3_av'] = knn_df['path3_av'].apply(convert_to_float_list)



# knn_df['Path_3'] = knn_df['path3_av'].apply(lambda x: x[0] if x else None)


# # print(knn_df["Path_3"])

# knn_group_df = knn_df.groupby(['knn'])[['Path_Average', 'Path_1', 'Path_2', 'Path_3']].agg(['mean', 'std'])

# # print(knn_group_df)

# knn_box_df = pd.DataFrame()


# knn_box_df['k = 1'] = knn_df[knn_df['knn'] == 1]['path_av_av'].values
# knn_box_df['k = 3'] = knn_df[knn_df['knn'] == 3]['path_av_av'].values
# knn_box_df['k = 6'] = knn_df[knn_df['knn'] == 6]['path_av_av'].values
# knn_box_df['k = 12'] = knn_df[knn_df['knn'] == 12]['path_av_av'].values

# print("test wtf are these values", knn_box_df['k = 1'])
# knn_box = boxplot(knn_box_df)

# unique_values = knn_df['knn'].unique()


# results = []

# # Iterate over all pairs of unique values
# for value1, value2 in itertools.product(unique_values, repeat=2):
#     data_group1 = knn_df[knn_df['knn'] == value1]
#     data_group2 = knn_df[knn_df['knn'] == value2]
#     # Perform Kolmogorov-Smirnov test for each path
#     for path in ['Path_Average', 'Path_1', 'Path_2', 'Path_3']:
#         if path == 'Path_Average':
#             pass

#         # stat, p_val = ks_2samp(data_group1[path], data_group2[path])
#         # test = bootstrap(data_group1[path], data_group2[path])
#         conf_interval, p_val, observed_dif, bootstrap_mean  = bootstrap_stat_test(data_group1[path].values, data_group2[path].values, 1000)
#         # Append results to the list
#         results.append({'Group_1': value1, 'Group_2': value2, 'Path': path, 'P-Value': p_val})

# # Create the DataFrame from the list
# p_df_knn = pd.DataFrame(results)

# print("knn TEST")
# print("knn TEST")
# print("knn TEST")
# print("knn TEST")

# # print(p_values_df_knn)

# p_df_knn_av = p_df_knn[p_df_knn["Path"] == "Path_Average"]
# p_df_knn_1 = p_df_knn[p_df_knn["Path"] == "Path_1"]
# p_df_knn_2 = p_df_knn[p_df_knn["Path"] == "Path_2"]
# p_df_knn_3 = p_df_knn[p_df_knn["Path"] == "Path_3"]


# pivot_df_knn_av = p_df_knn_av.pivot(index='Group_1', columns='Group_2', values='P-Value')
# pivot_df_knn_1 = p_df_knn_1.pivot(index='Group_1', columns='Group_2', values='P-Value')
# pivot_df_knn_2 = p_df_knn_2.pivot(index='Group_1', columns='Group_2', values='P-Value')
# pivot_df_knn_3 = p_df_knn_3.pivot(index='Group_1', columns='Group_2', values='P-Value')

# print(pivot_df_knn_av)
# print(pivot_df_knn_1)
# print(pivot_df_knn_2)
# print(pivot_df_knn_3)
# print("test done")



"""
LOCMETH

i fucked this one up a bit look at the old file for clarifications! 
# """


# # locmeth_box_df = pd.DataFrame()



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
Small overall tables for every thing!
"""


# def overall_table(df_function, column):

#     df_overall = df_function.copy()
    
#     df_overall = df_function.groupby([column])[['Path_Average_mean', 'Path_1_mean', "Path_2_mean", "Path_3_mean"]].agg(['mean', 'std']).reset_index().round(2)
#     # df_overall = df_function.groupby([column])['Path_Average_mean'].agg(['mean', 'std']).reset_index().round(2)
    
#     print(df_overall.to_latex())
    

#     # df_overall['Path Average'] = "$\substack{\mu= " +  df_overall['path_av_av'].astype(str) + "}$"
#     # df_overall['Path 1'] = "$\substack{\mu= " +  df_overall['Path_1_mean'].astype(str) + ",  \\\ \sigma= " + df_overall['Path_1_std'].astype(str) + "}$"
#     # df_overall['Path 2'] = "$\substack{\mu= " +  df_overall['Path_2_mean'].astype(str) + ",  \\\ \sigma= " + df_overall['Path_2_std'].astype(str) + "}$"
#     # df_overall['Path 3'] = "$\substack{\mu= " +  df_overall['Path_3_mean'].astype(str) + ",  \\\ \sigma= " + df_overall['Path_3_std'].astype(str) + "}$"

#     pass

# overview_df = df.copy()

# overview_df['Path_Average_mean'] = overview_df["path_av_av"]

# print(overview_df['Path_Average_mean'])


# overview_df['path1_av'] = overview_df['path1_av'].apply(convert_to_float_list)
# overview_df['Path_1_mean'] = overview_df['path1_av'].apply(lambda x: x[0] if x else None)
# overview_df['Path_1_std'] = overview_df['path1_av'].apply(lambda x: x[1] if x else None)

# print(overview_df['Path_1_mean'])

# overview_df['path2_av'] = overview_df['path2_av'].apply(convert_to_float_list)
# overview_df['Path_2_mean'] = overview_df['path2_av'].apply(lambda x: x[0] if x else None)
# overview_df['Path_2_std'] = overview_df['path2_av'].apply(lambda x: x[1] if x else None)

# overview_df['path3_av'] = overview_df['path3_av'].apply(convert_to_float_list)
# overview_df['Path_3_mean'] = overview_df['path3_av'].apply(lambda x: x[0] if x else None)
# overview_df['Path_3_std'] = overview_df['path3_av'].apply(lambda x: x[1] if x else None)


# test = overall_table(overview_df, 'threshold')
# test = overall_table(overview_df, 'dt')
# test = overall_table(overview_df, 'mis_val')
# test = overall_table(overview_df, 'knn')
# test = overall_table(overview_df, 'loc_method')

"""

the statistical test 


"""

av_ls_1 = sum(av_ls_1, [])
print("av_ls_total", len(av_ls_1))
 

av_ls_2 = sum(av_ls_2, [])
print("av_ls_total", len(av_ls_2))




p_val = bootstrap.bootstrap_concate(av_ls_1, av_ls2, 100000)

"""" 

Statistical test for everything! You are going to use this template for all the fucking things that come trough here! 
"now it's getting fucking exciting BITCH" 
"""

# you make your list of values to compare. 

# you path them up so we can run them through interpolate and safe all their error paths! 

# create a list of these error paths? (including some name or something otherwise it's difficult)




# unique_values = [whatever! (A sort of numbering is good?)]j
# unique values name! 


# results = []

# # Iterate over all pairs of unique values
# for value1, value2 in itertools.product(unique_values, repeat=2):



# You run them all against each other ! 
# you calculate the p_values for each and every one 

#    results.append({'Group_1': value1, 'Group_2': value2, 'Path': path, 'P-Value': p_value})


# p_df_locmeth = pd.DataFrame(results)



# # print(p_values_df_locmeth)
# (this step is only neccesary if you want to produce the entire ritch again!)
# p_df_locmeth_av = p_df_locmeth[p_df_locmeth["Path"] == "Path_Average"]





"""" 

Statistical tests top 10 
"""



""""
Stat test top performing dt
"""


"""
Stat test top performing lp 
"""

