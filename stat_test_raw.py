
import pandas as pd
import itertools

from scipy.stats import ks_2samp
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import interpolate
import time_og_data2
from scipy import stats
import bootstrap 



"""
For the 

"""



    
n_og_paths = [time_og_data2.n_og_path1, time_og_data2.n_og_path2, time_og_data2.n_og_path3]
path_time_ls = [[time_og_data2.start_time1, time_og_data2.end_time1], [time_og_data2.start_time2, time_og_data2.end_time2], [time_og_data2.start_time3, time_og_data2.end_time3]]

long_path = [n_og_paths, path_time_ls]


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




# av_ls1, best_ls, av_i_error, av_er_t_ls_1, b_er_t_ls = interpolate.accuracy(long_path[0] ,path_1, long_path[1], plt_paths =plotfigs, av_plt=False, best_plt=False, lines_plt=False, filter = None, filename=False)

# av_ls2, best_ls, av_i_error, av_er_t_ls_2, b_er_t_ls = interpolate.accuracy(long_path[0] ,path_2, long_path[1], plt_paths =plotfigs, av_plt=False, best_plt=False, lines_plt=False, filter = None, filename=False)




# """

# the statistical test 


# """

# av_ls1 = sum(av_ls1, [])
# print("av_ls_total", len(av_ls1))
 

# av_ls2 = sum(av_ls2, [])
# print("av_ls_total", len(av_ls2))   




# p_val = bootstrap.bootstrap_concate(av_ls1, av_ls2, 100000)

"""" 

Statistical test for everything! You are going to use this template for all the fucking things that come trough here! 
"now it's getting fucking exciting BITCH" 


So for every category we take the top performance! 

And we make a violin plot + a little table! 
"""




# pre prep function 

# you make your list of values to compare. 

# you path them up so we can run them through interpolate and safe all their error paths! 

#


"""
Load in top10 data:
"""


def readrow(df_names, df):
    path_ls = []

# path_1 = f"data\\paths\\dt_{dt_1}\\raw\\{radio_rssi_1}\\{mis_val_mtd_1}\\{knn_N_1}\\{loc_method_1}\\path"
    for index, row in df.iterrows():
        # print(row.index)
        threshold = row['threshold']
        dt = row['dt']
        dt = dt[:-1]
        mis_val = row['mis_val']
        loc_method = row['loc_method']
        knn = row['knn']
        path = f"data\\paths\\dt_{dt}\\raw\\{threshold}\\{mis_val}\\{knn}\\{loc_method}\\path"
        path_ls.append([df_names[index], path])

    return path_ls


# def readrowcol(df_names, df):
#     path_ls = []

# # path_1 = f"data\\paths\\dt_{dt_1}\\raw\\{radio_rssi_1}\\{mis_val_mtd_1}\\{knn_N_1}\\{loc_method_1}\\path"
#     for index, row in df.iterrows():
#         # print(row.index)
#         threshold = row['threshold']
#         dt = row['dt']
#         mis_val = row['mis_val']
#         loc_method = row['loc_method']
#         knn = row['knn']
#         path = f"data\\paths\\dt_{dt}\\raw\\{threshold}\\{mis_val}\\{knn}\\{loc_method}\\path"
#         path_ls.append((df_names[index], path))

#     return path_ls




df= pd.read_csv('data\\results\\table_top_path_average.csv', index_col=0)

top10 = df.head(10).reset_index(drop=True)

# print(top10)

# top10_p_ls =[]

top10_names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
top10_path_ls = readrow(top10_names, top10)

# Get the indices of the rows with the lowest 'path_av_av' for each unique 'knn' value



min_indices = df.groupby('knn')['path_av_av'].idxmin()
best_knn_df = df.loc[min_indices]

knn_names= [1, 3, 6, 12]

# Sorting the DataFrame based on the custom order of the 'City' column
best_knn_df= best_knn_df.sort_values(by='knn', key=lambda x: pd.Categorical(x, categories=knn_names, ordered=True)).reset_index(drop=True)



knn_path_ls = readrow(knn_names, best_knn_df)
# print("knn pathls!!", knn_path_ls)




min_indices = df.groupby('threshold')['path_av_av'].idxmin()
best_threshold_df = df.loc[min_indices]

threshold_names = ['lp2', 'lp2_n100', 'lp2_n85']

# Sorting the DataFrame based on the custom order of the 'City' column
best_threshold_df= best_threshold_df.sort_values(by='threshold', key=lambda x: pd.Categorical(x, categories=threshold_names, ordered=True)).reset_index(drop=True)


threshold_path_ls = readrow(threshold_names, best_threshold_df)
# print("threshold_path_ls", threshold_path_ls)


min_indices = df.groupby('dt')['path_av_av'].idxmin()
best_dt_df = df.loc[min_indices]

# print(best_dt_df['dt'])

dt_names = ['3s', '5s', '10s']
best_dt_df= best_dt_df.sort_values(by='dt', key=lambda x: pd.Categorical(x, categories=dt_names, ordered=True)).reset_index(drop=True)

# print("best_dt_df", best_dt_df['dt'])

# dt_names = ['3', '5', '10']
dt_path_ls = readrow(dt_names, best_dt_df)



# print(dt_path_ls)
min_indices = df.groupby('loc_method')['path_av_av'].idxmin()
best_locmeth_df = df.loc[min_indices]


locmeth_names = ["average", "gaus_normal", "gaus_80", "gaus_top"]
best_locmeth_df= best_locmeth_df.sort_values(by='loc_method', key=lambda x: pd.Categorical(x, categories=locmeth_names, ordered=True)).reset_index(drop=True)



locmeth_path_ls = readrow(locmeth_names, best_locmeth_df)
# print("locmeth", locmeth_path_ls)


top10_path_ls = top10_path_ls
threshold_path_ls = threshold_path_ls
knn_path_ls = knn_path_ls
dt_path_ls = dt_path_ls
locmeth_path_ls = locmeth_path_ls



# def boxplot(path_ls, title, tick_text):

#     boxplot_df= pd.DataFrame()
#     for path in path_ls:
#         print("path", path)
        
#         av_ls, best_ls, av_i_error, av_er_t_ls, b_er_t_ls = interpolate.accuracy(long_path[0],path[1], long_path[1], plt_paths =False, av_plt=False, best_plt=False, lines_plt=False, filter = None, filename=False)
#         av_er_total_ls = sum(av_er_t_ls, [])
#         # col_name = tick_text + "\n" + str(path[0])
#         col_name = tick_text  + str(path[0])
#         boxplot_df[col_name] = pd.Series(av_er_total_ls)
#         print("hallo")

#     print(boxplot_df)
#     sns.set(style="whitegrid")  # optional styling

#     plt.figure(figsize=(8, 6))  # optional, set the figure size

#     sns.boxplot(data=boxplot_df)
    

#     plt.xticks(fontsize=18)  # Change the fontsize to your desired value
#     plt.yticks(fontsize=18)  # Change the fontsize to your desired value
#     plt.savefig(f'data\\figures\\boxplots\\{title}.pdf', bbox_inches='tight')
#     plt.show()


def boxplot123(path_ls, filename, tick_text):

    boxplot_df= pd.DataFrame()

    for i in range(3):

        # locmeth_names = ["Average", "Normal \n Gaussian", "Top 80% \n Gaussian", "Top \n Gaussian"]
        for index, path in enumerate(path_ls):
    
            av_ls, best_ls, av_i_error, av_er_t_ls, b_er_t_ls = interpolate.accuracy(long_path[0],path[1], long_path[1], plt_paths =False, av_plt=False, best_plt=False, lines_plt=False, filter = None, filename=False)
            
            av_er_total_ls = av_er_t_ls[i]
            print("av_er_t_ls", av_er_t_ls)
            av_er_total_ls = bootstrap.bootstrap_sample(av_er_total_ls, 100000)
            # col_name = tick_text + "\n" + str(path[0])
            col_name = tick_text  + str(path[0])
            # col_name = locmeth_names[index]
            boxplot_df[col_name] = pd.Series(av_er_total_ls)
            mean_value = sum(av_er_total_ls)/len(av_er_total_ls)
    


        sns.set(style="whitegrid")  # optional styling

        plt.figure(figsize=(8, 6))  # optional, set the figure size
        plt.ylabel('Error (m)', fontsize=18)
        sns.violinplot(data=boxplot_df)
        

        plt.xticks(fontsize=18)  # Change the fontsize to your desired value
        plt.yticks(fontsize=18)  # Change the fontsize to your desired value
        plt.savefig(f'data\\figures\\boxplots\\raw_{i+ 1}_{filename}.pdf', bbox_inches='tight')
        plt.close()
    # plt.show()

def boxplot(path_ls, filename, tick_text):

    boxplot_df= pd.DataFrame()


    # locmeth_names = ["Average", "Normal \n Gaussian", "Top 80% \n Gaussian", "Top \n Gaussian"]
    for index, path in enumerate(path_ls):
  
        av_ls, best_ls, av_i_error, av_er_t_ls, b_er_t_ls = interpolate.accuracy(long_path[0],path[1], long_path[1], plt_paths =False, av_plt=False, best_plt=False, lines_plt=False, filter = None, filename=False)
        av_er_total_ls = sum(av_er_t_ls, [])

        av_er_total_ls = bootstrap.bootstrap_sample(av_er_total_ls, 100000)
        # col_name = tick_text + "\n" + str(path[0])
        col_name = tick_text  + str(path[0])
        # col_name = locmeth_names[index]
        boxplot_df[col_name] = pd.Series(av_er_total_ls)
        mean_value = sum(av_er_total_ls)/len(av_er_total_ls)
   


    sns.set(style="whitegrid")  # optional styling

    plt.figure(figsize=(8, 6))  # optional, set the figure size
    plt.ylabel('Error (m)', fontsize=18)
    sns.violinplot(data=boxplot_df)
    

    plt.xticks(fontsize=18)  # Change the fontsize to your desired value
    plt.yticks(fontsize=18)  # Change the fontsize to your desired value
    plt.savefig(f'data\\figures\\boxplots\\raw_{filename}.pdf', bbox_inches='tight')
    # plt.show()

def stats(path_ls):
    results = []


    av_er_ls = []

    for path in path_ls:
        av_ls, best_ls, av_i_error, av_er_t_ls, b_er_t_ls = interpolate.accuracy(long_path[0],path[1], long_path[1], plt_paths =False, av_plt=False, best_plt=False, lines_plt=False, filter = None, filename=False)
        av_er_total = sum(av_er_t_ls, [])
        av_er_ls.append((path[0], av_er_total))

    print(av_er_ls, "av_er_ls")
    for value1, value2 in itertools.product(av_er_ls, repeat=2):
       
       p_val = bootstrap.bootstrap_concate(value1[1], value2[1], 100000)

       results.append({'Group_1': value1[0], 'Group_2': value2[0], 'P-Value': p_val})

    
    p_df = pd.DataFrame(results)
    pivot_df= p_df.pivot(index='Group_1', columns='Group_2', values='P-Value')
    print(pivot_df)
    print(pivot_df.to_latex())
    print()
    print()


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


print("top10")

# test = boxplot(top10_path_ls, "top10", "")

# test = stats(top10_path_ls)


thres_names = ["None", "-100", "-85"]

for index, element in enumerate(threshold_path_ls):
    element[0] = thres_names[index]



print("threshold")
# threshold_stat = stats(threshold_path_ls)
threshold_box = boxplot123(threshold_path_ls, "threshold", "")

# print("dt")

# dt = stats(dt_path_ls)
# dt = boxplot(dt_path_ls, "dt ", "dt =")


# print("knn")
# knn_stat = stats(knn_path_ls)
# knn_box = boxplot(knn_path_ls, "knn", "k = ")




# locmeth_names = ["Average", "Gaussian \n Standard", "Gaussian \n Top 80 %%", "Gaussian Top"]


# for index, element in enumerate(locmeth_path_ls):
#     element[0] = locmeth_names[index]

# locmeth  = stats(locmeth_path_ls)
# locmeth  = boxplot(locmeth_path_ls, "locmeth", "")



# # print(locmeth_path_ls)
# # print(knn_path_ls)
# # print(dt_path_ls)



"""
The average top10 values go here! written out 
"""



