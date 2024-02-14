import interpolate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time_og_data2
import time



"""
For each path it is confirmed that the path is actually walked in 20 seconds of the 2 minutes. 


In order to find this  iwant to run the path for every time interval within the 120/180 interval that was given. 


This roughly amounts to 100 runs for a 2 minute interval and 160 runs for a 180. 

I want to run this data for many different settings. Maybe for all combinations? Lol.

ideally I would like to run the data for a tester for all 5 paths, instead of for the average. 



In general I want to test best paths vs average of 5 to see how the difference holds up. 

ACTUALLY I think it would make a lot of sense to look at Average path vs average of single paths vs best. 


This would require however to change interpolate probably, or slightly rework it with ANOTHER toggle.  


Print all these paths. 


Then
"""


og1 = time_og_data2.og_path1
og2 = time_og_data2.og_path2
og3 = time_og_data2.og_path3

time1 = [time_og_data2.start_time1, time_og_data2.end_time1 - pd.Timedelta(seconds= 20)]
time2 = [time_og_data2.start_time2, time_og_data2.end_time2 - pd.Timedelta(seconds= 20)]
time3 = [time_og_data2.start_time3, time_og_data2.end_time3 - pd.Timedelta(seconds= 20)]


print('len og1', len(og1))

# path = f"data\\paths\\dt_3\\raw\\lp2_n85\\penalty\\3\\average\\path"

# duration of time window true path given by researchers
path_t = 20

# # calculates sampling rate (amount of samples in path per time walking)
og_s1 = len(og1)/path_t
og_s2 = len(og2)/path_t
og_s3 = len(og3)/path_t





# # total number of seconds in a path 
dt1 =  (time1[1] - time1[0]).total_seconds()
dt2  =  (time2[1] - time2[0]).total_seconds()
dt3 =  (time3[1] - time3[0]).total_seconds()


og_n_l1 = int(og_s1 * dt1)
og_n_l2 = int(og_s2 * dt2)

og_n_l3 = int(og_s3 * dt3)


print("len1", og_n_l1)
print("len2", og_n_l2)
print('len3', og_n_l3)


def time_window_entire(time, og, og_n, og_s, path, path_name):

    time_start = pd.to_datetime(time[0])

    time_end = pd.to_datetime(time[1])
    path_time_ls = [[time_start, time_end]]



    t_mean_ls = []

    plt_av_ls = []
    plt_best_ls = []
    plt_av_ir_ls = []
    t_ls = []
    og_ls =[]
    path_ls = []
    before_ls = []
    after_ls = [og[-1] for _ in range(og_n - len(og))]
    new_og = before_ls + og + after_ls


    path_number = new_og
    # path_name = f"path_{path_number}"

    for i in range(og_n - len(og)):

    
        av_ls, best_ls, av_i_error, av_t_error_ls, av_b_error_ls = interpolate.accuracy([path_number],path, path_time_ls , plt_paths=False, av_plt=False, best_plt=False, lines_plt=False, filter=None)
    
        plt_av_ls.append(av_ls[0][0])
        plt_best_ls.append(best_ls[0][0])
        plt_av_ir_ls.append(av_i_error[0])


        seconds = i / og_s
        # print("secondds", seconds)
        # print("timestart", time_start)
        t_walk_s = time_start + pd.Timedelta(seconds= seconds)

        # print("timead", time_add)
        
        t_ls.append(t_walk_s)
        path_ls.append(path_name)
        t_mean_ls.append([av_ls[0], (t_walk_s, t_walk_s + pd.Timedelta(seconds=20)), new_og])
        og_ls.append(new_og)
        before_ls.append(og[0])
        after_ls.pop(0)
        new_og = before_ls + og + after_ls
        path_number = new_og

    


 
    t_mean_ls.sort(key=lambda x: x[0])
    

    best_sol = t_mean_ls[0]
    
    # best_time = t_mean_ls[1]
    # best_start = best_time[0]
    # best_end = best_time[1]

    # print(type(best_start))
    # print(type(best_end))
    # print(f"path= {path_name}")
    # print(f"best_start=",best_start)
    # print(f"best_end", best_end)
    # print("best_sol", best_sol)
    
    # this is for the test values in the beginning
    

    # print("df", df)
    #  [[3.4321681702311837, 2.4090786667802573, 13.728672680924735], (Timestamp('2021-04-08 18:48:45'), Timestamp('2021-04-08 18:49:05'))], 

    # plt.plot(t_ls, plt_av_ls)
    # plt.show()
    # plt.savefig(f"data\\time_window\\{path_name}")

    # min_val = [0, 0]
    # for i in range(len(plt_av_ls)):
    #     if av_ls[i] < min_val[0]:
    #         min_val[0] = av_ls[i]
    #         min_val[1] = t_ls[i]

    # print(f"min_val for path {path_name}")
    # print(f"min_val", min_val[0])
    # print(f"min_val", min_val[1])
    # print()
    # print()
    return plt_av_ls, t_ls, path_ls, og_ls, best_sol


# # best_start -= pd.Timedelta(seconds=15)
# # best_end += pd.Timedelta(seconds = 10)
# path_n_df = av_df[(av_df['time'] >= best_start) & (av_df['time'] < best_end)]
# t_vals, a_vals = interpolate.accuracy(og3 ,path_n_df, [best_start, best_end], plot_toggle=True, both_toggle=True, lines_toggle=True, filter = None)
# # path = f"data\\paths\\dt_10\\filter\\lp2\\ignore\\6\\gaus_normal\\path0"
# # path = "junk\\fake_og"
# av_df = pd.read_csv(f"{path}.csv", index_col=0) 
# av_df['time'] = pd.to_datetime(av_df['time'], format='mixed')
# av_df["av_time"] = av_df["time"]

# print(av_df)




record_start = time.time()



# column_names = 


best_og_ls = []
best_er_ls = []
best_start_ls = []
best_path_ls = []

dt_ls = ["dt_5"]
mis_val_ls = ["ignore"]
knn_N = [6]
locmeths = ["gaus_normal"]

first = True
whole_ls = []


# this is the mean_ls


df_best = pd.DataFrame(columns=['error', 'time', 'path', 'n_og'])
df_total = pd.DataFrame(columns=['error_mean','error_std', 'time', 'path', 'n_og'])

# we add dt locmeth misval nsize

# dt_col 
plt_av_ls_total_whole =[]
for dt in dt_ls:
    for mis_val in mis_val_ls:
        for n_size in knn_N:
            for locmeth in locmeths:

                path = f"data\\paths\\{dt}\\raw\\lp2\\{mis_val}\\{n_size}\\{locmeth}\\path"


                time_ls = [[time1, og1, og_n_l1, og_s1, path, "path1"], [time2, og2, og_n_l2, og_s2, path, "path2"], [time3, og3, og_n_l3, og_s3, path, "path3"]]
                
                # time_ls = [[time1, og1, og_n_l1, og_s1, path, "path1"]]
                # Initialize DataFrame with columns
                
                plt_av_ls_total_path = []
                if first:
                    
                    t_ls_total = []
                    path_ls_total = []
                    n_og_ls_total = []

                for item in time_ls:
                    plt_av_ls, t_ls, path_ls, new_og_ls, best_sol =  time_window_entire(item[0], item[1], item[2], item[3], item[4], item[5])
                    best_er, best_times, best_og = best_sol
                    best_start = best_times[0]

                    plt_av_ls_total_path += plt_av_ls
                    print(f'for {item[5]} the length of av_ls', len(plt_av_ls))
                    
                    if first:
                        t_ls_total += t_ls
                        path_ls_total += path_ls
                        n_og_ls_total += new_og_ls


                    best_og_ls.append(best_og)
                    best_er_ls.append(best_er)
                    best_start_ls.append(best_start)
                    best_path_ls.append(item[5])
                plt_av_ls_total_whole.append(plt_av_ls_total_path)
                first = False

        

# we add dt locmeth misval nsize
df_best['error'] = best_er_ls
df_best['time'] = best_start_ls
df_best['path'] = best_path_ls
df_best['n_og'] = best_og_ls

        # whole_ls.append(plt_av_ls_total)


# print("plt_av_ls", plt_av_ls_total)

# print("df_best", df_best)

total_means = np.mean(plt_av_ls_total_whole, axis=0)
total_std_devs = np.std(plt_av_ls_total_whole, axis=0)


df_total['error_mean'] = total_means
df_total['error_std'] = total_std_devs
df_total['time'] = t_ls_total
df_total['path'] = path_ls_total
df_total['n_og'] = n_og_ls_total






df_total.to_csv(f"data\\time_window\\small_total.csv")
df_best.to_csv(f"data\\time_window\\small_proper_best.csv")
record_end = time.time()




# print("time elapsed= ", record_end - record_start)




"""

PENALTY ENTIRE 

path= path1
best_start= [13.410051097025294, 7.428965614160382, 482.76183949291055]
best_end (Timestamp('2021-04-08 18:44:14.838709677'), Timestamp('2021-04-08 18:44:34.838709677'))

path= path2
best_start= [4.893872669993479, 2.332699031701852, 117.4529440798435]
best_time (Timestamp('2021-04-08 18:52:43.333333333'), Timestamp('2021-04-08 18:53:03.333333333'))

path= path3
best_start= [7.299921730307743, 3.7660837536830614, 262.79718229107874]
best_time (Timestamp('2021-04-08 18:48:52.727272727'), Timestamp('2021-04-08 18:49:12.727272727'))


IGNORE ENTIRE: 




"""



"""
path 1: (3.9341759174174173, (Timestamp('2021-04-08 18:44:41'), Timestamp('2021-04-08 18:45:01')))

after the end time it seems rougly located at the end so that's good, before that though it gets pulled to the middle! 

THIS IS EXCELLENT!!!

path 2: (Timestamp('2021-04-08 18:53:32'), Timestamp('2021-04-08 18:53:52'))) for this one it doesn't matter much tho


path 3: 

"""