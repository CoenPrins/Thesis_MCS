import interpolate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time_og_data2




"""
For each path it is confirmed that the path is actually walked in 20 seconds of the 2 minutes. 


In order to find this  iwant to run the path for every time interval within the 120/180 interval that was given. 


This roughly amounts to 100 runs for a 2 minute interval and 160 runs for a 180. 

I want to run this data for many different settings. Maybe for all combinations? Lol.

ideally I would like to run the data for a tester for all 5 paths, instead of for the average. 



In general I want to test best paths vs average of 5 to see how the difference holds up. 

ACTUALLY I think it would make a lot of sense to look at Average path vs average of single paths vs best. 


This would require however to change interpolate probably, or slightly rework it with ANOTHER toggle.  
"""


og1 = time_og_data2.og_path1
og2 = time_og_data2.og_path2
og3 = time_og_data2.og_path3

time_1 = [time_og_data2.start_time1, time_og_data2.end_time1]
time_2 = [time_og_data2.start_time2, time_og_data2.end_time2]
time_3 = [time_og_data2.start_time3, time_og_data2.end_time3]




path = f"data\\paths\\dt_5\\raw\\lp2\\ignore\\6\\gaus_normal\\path"

# av_df = interpolate.create_av_df(path,KF=False, PF=False)
# time_start = time_1[0]
# time_start = pd.to_datetime(time_start)
# time_end = time_start + pd.Timedelta(seconds=20)
# path_number = og1
# path_name = f"path_{path_number}"


# time_start = time_2[0]
# time_start = pd.to_datetime(time_start)
# time_end = time_start + pd.Timedelta(seconds=20)
# path_number = og2
# path_name = f"path_{path_number}"



time_start = time_3[0]
time_start = pd.to_datetime(time_start)
time_end = time_start + pd.Timedelta(seconds=20)
path_number = og3
path_name = f"path_{path_number}"



t_mean_ls = []
plt_av_ls = []
plt_best_ls = []
plt_av_ir_ls = []
t_ls = []



while time_end <= time_3[1]:

    # path_n_df = av_df[(av_df['time'] >= time_start) & (av_df['time'] < time_end)]
    # print("early av_df", path_n_df['time'])
    path_time_ls = [[time_start, time_end]]
    # print(len(path_time_ls))
    av_ls, best_ls, av_i_error, placeholder, placeholder2 = interpolate.accuracy([path_number],path, path_time_ls , plt_paths=False, av_plt=False, best_plt=False, lines_plt=False, filter=None)
   
    plt_av_ls.append(av_ls[0][0])
    plt_best_ls.append(best_ls[0][0])
    plt_av_ir_ls.append(av_i_error[0])
    t_ls.append(time_start)
    t_mean_ls.append([av_ls[0], (time_start, time_end)])
    time_start += pd.Timedelta(seconds=1)
    time_end += pd.Timedelta(seconds=1)



print("done one run!!!")
t_mean_ls.sort(key=lambda x: x[0])

print("t_mean_ls", t_mean_ls)
best_sol = t_mean_ls[0]
print(best_sol)
best_time = best_sol[1]


best_start = best_time[0]
best_end = best_time[1]

print(type(best_start))
print(type(best_end))


#  [[3.4321681702311837, 2.4090786667802573, 13.728672680924735], (Timestamp('2021-04-08 18:48:45'), Timestamp('2021-04-08 18:49:05'))], 

plt.plot(t_ls, plt_av_ls, color="green")

plt.xlabel("Time", fontweight='bold')
plt.ylabel('Error (m)', fontweight='bold')

plt.show()
 
# best_start = pd.to_datetime('2021-04-08 18:44:36')
# best_end = best_start + pd.Timedelta(seconds=25)
# path_time_ls = [[best_start, best_end]]

# av_ls, best_ls, av_i_error = interpolate.accuracy([og1],path, path_time_ls , plt_paths=True, av_plt=True, best_plt=False, lines_plt=True, filter=None)


"""
path 1: (3.9341759174174173, (Timestamp('2021-04-08 18:44:41'), Timestamp('2021-04-08 18:45:01')))

after the end time it seems rougly located at the end so that's good, before that though it gets pulled to the middle! 

THIS IS EXCELLENT!!!

path 2: (Timestamp('2021-04-08 18:53:32'), Timestamp('2021-04-08 18:53:52'))) for this one it doesn't matter much tho


path 3: 

"""