
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

You need to explain stuff! 

For the filtered test we have 


2 different filter settings for 3 different time windows. 



3 tables/boxplots vs the original! (maybe a violin bootstrap plot?), yeah let's do violin bootstrap plot!

Which is basically pf normal pf je moeder and original


Three times 

And then you want the 


6 partical filters next to each other. 


Make plots make tables! Letsgo 




Which is 6 combinations! 




"""



    
n_og_paths = [time_og_data2.n_og_path1, time_og_data2.n_og_path2, time_og_data2.n_og_path3]
path_time_ls = [[time_og_data2.start_time1, time_og_data2.end_time1], [time_og_data2.start_time2, time_og_data2.end_time2], [time_og_data2.start_time3, time_og_data2.end_time3]]

long_path = [n_og_paths, path_time_ls]


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


def boxplot(av_er_path_ls, filename, tick_text):

    boxplot_df= pd.DataFrame()
    

    # locmeth_names = ["Average", "Normal \n Gaussian", "Top 80% \n Gaussian", "Top \n Gaussian"]
    for index, av_er in enumerate(av_er_path_ls):
        # print("av)er", av_er)
        av_er_t  = sum(av_er[1], [])

        av_er_t = bootstrap.bootstrap_sample(av_er_t, 100000)
        # col_name = tick_text + "\n" + str(path[0])
        col_name = tick_text  + str(av_er[0])
        # col_name = locmeth_names[index]
        boxplot_df[col_name] = pd.Series(av_er_t)
        mean_value = sum(av_er_t)/len(av_er_t)
   


    sns.set(style="whitegrid")  # optional styling

    plt.figure(figsize=(8, 6))  # optional, set the figure size

    sns.violinplot(data=boxplot_df)
    plt.ylabel('Error (m)', fontsize=18)

    plt.xticks(fontsize=18)  # Change the fontsize to your desired value
    plt.yticks(fontsize=18)  # Change the fontsize to your desired value
    # plt.savefig(f'data\\figures\\boxplots\\filter_{filename}.pdf', bbox_inches='tight')
    plt.show()



def boxplot123_all(av_er_path_ls, filename, tick_text):

    boxplot_df= pd.DataFrame()
    
    for i in range(3):
        

        # locmeth_names = ["Average", "Normal \n Gaussian", "Top 80% \n Gaussian", "Top \n Gaussian"]
        for index, av_er in enumerate(av_er_path_ls):
            # print("av)er", av_er)
            av_er_t  = av_er[1][i]
            print(av_er_t, "av_er_t")
            av_er_t = bootstrap.bootstrap_sample(av_er_t, 100000)
            # col_name = tick_text + "\n" + str(path[0])
            col_name = tick_text  + str(av_er[0])
            # col_name = locmeth_names[index]
            boxplot_df[col_name] = pd.Series(av_er_t)
            mean_value = sum(av_er_t)/len(av_er_t)
   


        sns.set(style="whitegrid")  # optional styling

        plt.figure(figsize=(8, 6))  # optional, set the figure size

        sns.violinplot(data=boxplot_df)
        plt.ylabel('Error (m)', fontsize=18)

        plt.xticks(fontsize=18)  # Change the fontsize to your desired value
        plt.yticks(fontsize=18)  # Change the fontsize to your desired value
        plt.savefig(f'data\\figures\\boxplots\\filter_specific{i}_{filename}.pdf', bbox_inches='tight')
        plt.show()

# this one is differnt from the one in raw



def boxplot123(av_er_path_ls, filename, tick_text):

    boxplot_df= pd.DataFrame()
    
    for i in range(3):
        

        # locmeth_names = ["Average", "Normal \n Gaussian", "Top 80% \n Gaussian", "Top \n Gaussian"]
        for index, av_er in enumerate(av_er_path_ls):
            # print("av)er", av_er)
            av_er_t  = av_er[1][i]
            print(av_er_t, "av_er_t")
            av_er_t = bootstrap.bootstrap_sample(av_er_t, 100000)
            # col_name = tick_text + "\n" + str(path[0])
            col_name = tick_text  + str(av_er[0])
            # col_name = locmeth_names[index]
            boxplot_df[col_name] = pd.Series(av_er_t)
            mean_value = sum(av_er_t)/len(av_er_t)
   


        sns.set(style="whitegrid")  # optional styling

        plt.figure(figsize=(8, 6))  # optional, set the figure size

        sns.violinplot(data=boxplot_df)
        plt.ylabel('Error (m)', fontsize=18)

        plt.xticks(fontsize=18)  # Change the fontsize to your desired value
        plt.yticks(fontsize=18)  # Change the fontsize to your desired value
        plt.savefig(f'data\\figures\\boxplots\\filter_specific{i}_{filename}.pdf', bbox_inches='tight')
        plt.show()

# this one is differnt from the one in raw

def stats(av_errors):
    
    results = []
    for element in av_errors:
        element[1] = sum(element[1], [])

    # print(av_errors)


    for value1, value2 in itertools.product(av_errors, repeat=2):


        p_val = bootstrap.bootstrap_concate(value1[1], value2[1], 100000)

        results.append({'Group_1': value1[0], 'Group_2': value2[0], 'P-Value': p_val})

    
    p_df = pd.DataFrame(results)
    pivot_df= p_df.pivot(index='Group_1', columns='Group_2', values='P-Value')
    print(pivot_df)
    print(pivot_df.to_latex(index=False))
    print()
    print()




radio_rssi = "lp2"
mis_val_mtd = "ignore"
knn_N = 6  
loc_method = "gaus_80"


dt_1 = 3
dt_2 = 5
dt_3 = 10


# to summarise: 


# every dt vs the other three. 

# and all 6 filters vs each other. 



# also make a big table with all the average performances 

# I'm not going to change all that.... 


# okay so far, only the top 10 is sus.... if you sneakily change that one no one will bat an EYE!

path1 = f"data\\paths\\dt_{dt_1}\\raw\\{radio_rssi}\\{mis_val_mtd}\\{knn_N}\\{loc_method}\\path"
path1_pf = f"data\\paths\\dt_{dt_1}\\filter\{radio_rssi}\\{mis_val_mtd}\\{knn_N}\\{loc_method}\\env_false_path"
path1_pf_semi = f"data\\paths\\dt_{dt_1}\\filter\{radio_rssi}\\{mis_val_mtd}\\{knn_N}\\{loc_method}\\env_semi_path"
path1_pf_e = f"data\\paths\\dt_{dt_1}\\filter\{radio_rssi}\\{mis_val_mtd}\\{knn_N}\\{loc_method}\\env_true_path"

path2 = f"data\\paths\\dt_{dt_2}\\raw\\{radio_rssi}\\{mis_val_mtd}\\{knn_N}\\{loc_method}\\path"
path2_pf = f"data\\paths\\dt_{dt_2}\\filter\\{radio_rssi}\\{mis_val_mtd}\\{knn_N}\\{loc_method}\\env_false_path"
path2_pf_semi = f"data\\paths\\dt_{dt_2}\\filter\{radio_rssi}\\{mis_val_mtd}\\{knn_N}\\{loc_method}\\env_semi_path"
path2_pf_e = f"data\\paths\\dt_{dt_2}\\filter\\{radio_rssi}\\{mis_val_mtd}\\{knn_N}\\{loc_method}\\env_true_path"

path3 = f"data\\paths\\dt_{dt_3}\\raw\\{radio_rssi}\\{mis_val_mtd}\\{knn_N}\\{loc_method}\\path"
path3_pf = f"data\\paths\\dt_{dt_3}\\filter\{radio_rssi}\\{mis_val_mtd}\\{knn_N}\\{loc_method}\\env_false_path"
path3_pf_semi = f"data\\paths\\dt_{dt_3}\\filter\{radio_rssi}\\{mis_val_mtd}\\{knn_N}\\{loc_method}\\env_semi_path"
path3_pf_e = f"data\\paths\\dt_{dt_3}\\filter\{radio_rssi}\\{mis_val_mtd}\\{knn_N}\\{loc_method}\\env_true_path"


# av_ls1, best_ls1, av_i_error1, av_er_t_ls1, b_er_t_ls1 = interpolate.accuracy(long_path[0],path1, long_path[1], plt_paths =False, av_plt=False, best_plt=False, lines_plt=False, filter = None, filename=False)



av_ls1, best_ls1, av_i_error1, av_er_t_ls1, b_er_t_ls1 = interpolate.accuracy(long_path[0],path1, long_path[1], plt_paths =False, av_plt=False, best_plt=False, lines_plt=False, filter = None, filename=False)
av_ls1_pf, best_ls1_pf, av_i_error1_pf, av_er_t_ls1_pf, b_er_t_ls1_pf= interpolate.accuracy(long_path[0],path1_pf, long_path[1], plt_paths =False, av_plt=False, best_plt=False, lines_plt=False, filter = "PF", filename=False)
av_ls1_pf_semi, best_ls1_pf_semi, av_i_error1_pf_semi, av_er_t_ls1_pf_semi, b_er_t_ls1_pf_semi= interpolate.accuracy(long_path[0],path1_pf_semi, long_path[1], plt_paths =False, av_plt=False, best_plt=False, lines_plt=False, filter = "PF", filename=False)
av_ls1_pf_e, best_ls1_pf_e, av_i_error1_pf_e, av_er_t_ls1_pf_e, b_er_t_ls1_pf_e = interpolate.accuracy(long_path[0],path1_pf_e, long_path[1], plt_paths =False, av_plt=False, best_plt=False, lines_plt=False, filter = "PF", filename=False)


av_ls2, best_ls2, av_i_error2, av_er_t_ls2, b_er_t_ls2 = interpolate.accuracy(long_path[0],path2, long_path[1], plt_paths =False, av_plt=False, best_plt=False, lines_plt=False, filter = None, filename=False)
av_ls2_pf, best_ls2_pf, av_i_error2_pf, av_er_t_ls2_pf, b_er_t_ls2_pf= interpolate.accuracy(long_path[0], path2_pf, long_path[1], plt_paths =False, av_plt=False, best_plt=False, lines_plt=False, filter = "PF", filename=False)
av_ls2_pf_semi, best_ls2_pf_semi, av_i_error2_pf_semi, av_er_t_ls2_pf_semi, b_er_t_ls2_pf_semi= interpolate.accuracy(long_path[0],path2_pf_semi, long_path[1], plt_paths =False, av_plt=False, best_plt=False, lines_plt=False, filter = "PF", filename=False)
av_ls2_pf_e, best_ls2_pf_e, av_i_error2_pf_e, av_er_t_ls2_pf_e, b_er_t_ls2_pf_e = interpolate.accuracy(long_path[0],path2_pf_e, long_path[1], plt_paths =False, av_plt=False, best_plt=False, lines_plt=False, filter = "PF", filename=False)


av_ls3, best_ls3, av_i_error3, av_er_t_ls3, b_er_t_ls3 = interpolate.accuracy(long_path[0],path3, long_path[1], plt_paths =False, av_plt=False, best_plt=False, lines_plt=False, filter = None, filename=False)
av_ls3_pf, best_ls3_pf, av_i_error3_pf, av_er_t_ls3_pf, b_er_t_ls3_pf= interpolate.accuracy(long_path[0],path3_pf, long_path[1], plt_paths =False, av_plt=False, best_plt=False, lines_plt=False, filter = "PF", filename=False)
av_ls3_pf_semi, best_ls3_pf_semi, av_i_error3_pf_semi, av_er_t_ls3_pf_semi, b_er_t_ls3_pf_semi= interpolate.accuracy(long_path[0],path3_pf_semi, long_path[1], plt_paths =False, av_plt=False, best_plt=False, lines_plt=False, filter = "PF", filename=False)
av_ls3_pf_e, best_ls3_pf_e, av_i_error3_pf_e, av_er_t_ls3_pf_e, b_er_t_ls3_pf_e = interpolate.accuracy(long_path[0],path3_pf_e, long_path[1], plt_paths =False, av_plt=False, best_plt=False, lines_plt=False, filter = "PF", filename=False)
    


# This is the template for all the results EXCEPT for the new fourht option! 
# When trying to run this, just add the fourht path, no biggie. 





"""
Stat test + boxplot

"""


# copy this for all three dt options! 

# path_1_ls = [["Raw", av_er_t_ls1], ["PF,\n No Env", av_er_t_ls1_pf], ["PF, \n Semi Env", av_er_t_ls1_pf_semi ], ['PF,\n Full Env', av_er_t_ls1_pf_e]]

# boxtest = boxplot(path_1_ls,  "dt_3", "")

# print("stats for dt=3")
# stats_1 = stats(path_1_ls)
# print()
# print()

# path_2_ls = [["Raw", av_er_t_ls2], ["PF,\n No Env", av_er_t_ls2_pf], ["PF, \n Semi Env", av_er_t_ls2_pf_semi ], ['PF,\n Full Env', av_er_t_ls2_pf_e]]

# boxtest = boxplot(path_2_ls,  "dt_5", "")


# print("stats for dt=5")
# stats_2 = stats(path_2_ls)

# print()
# print()


# path_3_ls = [["Raw", av_er_t_ls3], ["PF,\n No Env", av_er_t_ls3_pf], ["PF, \n Semi Env", av_er_t_ls3_pf_semi ], ['PF,\n Full Env', av_er_t_ls3_pf_e]]

# boxtest = boxplot(path_3_ls,  "dt_10", "")
# print('stats for dt=10')
# stats_3 = stats(path_3_ls)

# print()
# print()



"""
Do it for the seperate paths! 

"""

# path_1_ls = [["Raw", av_er_t_ls1], ["PF,\n No Env", av_er_t_ls1_pf], ["PF, \n Semi Env", av_er_t_ls1_pf_semi ], ['PF,\n Full Env', av_er_t_ls1_pf_e]]

# boxtest = boxplot123(path_1_ls,  "dt_3", "")

# print("stats for dt=3")
# stats_1 = stats(path_1_ls)
# print()
# print()

# path_2_ls = [["Raw", av_er_t_ls2], ["PF,\n No Env", av_er_t_ls2_pf], ["PF, \n Semi Env", av_er_t_ls2_pf_semi ], ['PF,\n Full Env', av_er_t_ls2_pf_e]]

# boxtest = boxplot123(path_2_ls,  "dt_5", "")


# print("stats for dt=5")
# stats_2 = stats(path_2_ls)

# print()
# print()


# path_3_ls = [["Raw", av_er_t_ls3], ["PF,\n No Env", av_er_t_ls3_pf], ["PF, \n Semi Env", av_er_t_ls3_pf_semi ], ['PF,\n Full Env', av_er_t_ls3_pf_e]]

# boxtest = boxplot123(path_3_ls,  "dt_10", "")
# print('stats for dt=10')
# stats_3 = stats(path_3_ls)

# print()
# print()



"""
And now per method all time intervals 
"""

# path_dt_raw = [["dt: 3s", av_er_t_ls1], ["dt: 5s", av_er_t_ls2], ["dt: 10s", av_er_t_ls3]]

# boxtest = boxplot(path_dt_raw,  "raw", "")

# print("raw table")

# stats_raw = stats(path_dt_raw)

# print()
# print()


# path_dt_raw = [["dt: 3s", av_er_t_ls1_pf], ["dt: 5s", av_er_t_ls2_pf], ["dt: 10s", av_er_t_ls3_pf]]

# boxtest = boxplot(path_dt_raw,  "PF", "")

# print("PF table")

# stats_pf = stats(path_dt_raw)

# print()
# print()


# path_dt_raw = [["dt: 3s", av_er_t_ls1_pf_semi], ["dt: 5s", av_er_t_ls2_pf_semi], ["dt: 10s", av_er_t_ls3_pf_semi]]

# boxtest = boxplot(path_dt_raw,  "PF_semi", "")

# print("PF SEMI table")

# stats_1= stats(path_dt_raw)
# print()
# print()



# path_dt_raw = [["dt: 3s", av_er_t_ls1_pf_e], ["dt: 5s", av_er_t_ls2_pf_e], ["dt: 10s", av_er_t_ls3_pf_e]]

# boxtest = boxplot123(path_dt_raw,  "PF_env", "test")

# print("PF_env table")

# # stats_1= stats(path_dt_raw)

# print()
# print()



"""
Plot ranking of all 12 
"""

dt = [3, 3, 3, 3, 5, 5, 5, 5, 10, 10, 10, 10]
filter = ["Raw", "PF,\n No Env", "PF, \nSemi Env", "PF, \nFull Env", "Raw", "PF,\n No Env", "PF, \nSemi Env", "PF, \nFull Env", "Raw", "PF,\n No Env", "PF, \nSemi Env", "PF, \nFull Env"]




error_ls = [av_er_t_ls1, av_er_t_ls1_pf, av_er_t_ls1_pf_semi, av_er_t_ls1_pf_e, av_er_t_ls2, av_er_t_ls2_pf, av_er_t_ls2_pf_semi, av_er_t_ls2_pf_e, av_er_t_ls3, av_er_t_ls3_pf, av_er_t_ls3_pf_semi, av_er_t_ls3_pf_e]

bootstrap1_ls = []
bootstrap2_ls = []
bootstrap3_ls = []
bootstrapt_ls = []
mean_t_only =[]

for sample in error_ls:
    av_er_t  = sum(sample, [])

    av_er_1  = sample[0]
    av_er_2  = sample[1]
    av_er_3  = sample[2]
    bootstrap_dis = bootstrap.bootstrap_sample(av_er_t, 100000)


    mean_1, std_1 = np.mean(av_er_1), np.std(av_er_1)
    mean_1, std_1 = round(mean_1, 2), round(std_1, 2)

    mean_2, std_2 = np.mean(av_er_2), np.std(av_er_2)
    mean_2, std_2 = round(mean_2, 2), round(std_2, 2)

    mean_3, std_3 = np.mean(av_er_3), np.std(av_er_3)
    mean_3, std_3 = round(mean_3, 2), round(std_3, 2)


    mean_t = np.mean([mean_1, mean_2, mean_3])
    mean_t = round(mean_t, 2)

    string1 = f"$\mu$ = {mean_1}, $\sigma$ = {std_1}"
    string2 = f"$\mu$ = {mean_2}, $\sigma$ = {std_2}"
    string3 = f"$\mu$ = {mean_3}, $\sigma$ = {std_3}"
    string_t = f"$\mu$ = {mean_t}"


    bootstrap1_ls.append(string1)
    bootstrap2_ls.append(string2)
    bootstrap3_ls.append(string3)
    bootstrapt_ls.append(string_t)
    mean_t_only.append(mean_t)

df = pd.DataFrame()

df['dt'] = dt
df["filter"] = filter

df['Path 1'] = bootstrap1_ls
df['Path 2'] = bootstrap2_ls
df['Path 3'] = bootstrap3_ls
df['Av Error'] = bootstrapt_ls
df['mean_t'] = mean_t_only


df = df.sort_values(by='mean_t')

# Drop or hide the 'mean' column
df = df.drop(columns=['mean_t'])


print('toptendf')
print(df)


print(df.to_latex(index=False))