import pandas as pd 
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import time_og_data2


"""
This code plots the data entries per key point for the radiomap for different minimum threshold 
And the data entries per time interval for different parameter settings. 

"""


df_radio_raw = pd.read_csv("data\\radio_maps\\csv\\lp.csv")
df_radio2= pd.read_csv("data\\radio_maps\\csv\\lp2.csv")
df_radio2_n100 = pd.read_csv("data\\radio_maps\\csv\\lp2_n100.csv")
df_radio2_n85 = pd.read_csv("data\\radio_maps\\csv\\lp2_n85.csv")

df_radio_n_list = [df_radio_raw, df_radio2, df_radio2_n85]


data_ls = ["lp","lp2", "lp2_n85"]
    
# create plots for different data files 
for index, df_radio in enumerate(df_radio_n_list):

    key_un_r = df_radio["key_point_num"].unique()
    keys_mac_n = []
    # calculate entries per key number
    for key in key_un_r: 

        mask = (df_radio['key_point_num'] == key) & (df_radio['mean_rssi'] != -100)

        mac_ls = df_radio[mask]
    
        
        keys_mac_n.append(len(mac_ls))
        


    hist = plt.bar(key_un_r, keys_mac_n)
    print(f"mean_key_va {data_ls[index]}", np.mean(key_un_r))
    print(f"sum_key_va {data_ls[index]}", np.sum(key_un_r))


    plt.ylabel("Non-zero Measurements", fontsize = 15)
    plt.xlabel("Key_n", fontsize = 15)
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize= 13)
    plt.tight_layout()
   
    # plt.savefig(f"data\\figures\\data_entries\\radio_map\\test\\data_entries_for_{data_ls[index]}.pdf", format='pdf')
    # plt.show()



dt_ls = ["dt_3", "dt_5", "dt_10"]
data_ls = ["lp","lp2", "lp2_n85"]
dt_int= [3, 5, 10]


t_p1 = [time_og_data2.start_time1, time_og_data2.end_time1]

t_p2 = [time_og_data2.start_time2, time_og_data2.end_time2]

t_p3 = [time_og_data2.start_time3, time_og_data2.end_time3]


tp_ls = [t_p1, t_p2, t_p3]
path_clr_ls = ["red", 'blue', 'green']

# plot for all different parameter combinations 
for dt_string in dt_ls:
    
    for data in data_ls:
        
        basetime = None
        time_ref = 0
        df_ls = []
        #load the paths into list and find the most complete timeframe to use as reference
        for i in range(5):

            df_path = pd.read_csv(f"C:\\Users\\coen_\\OneDrive\\Documents\\ma_thesis\\MAIN\\data\\dt\\{dt_string}\\{data}\\dt{i}.csv")
            df_ls.append(df_path)
            time = df_path['time'].unique()
            
            # some df_paths have missing data points, choose the most complete one
            if len(time) > time_ref:
                basetime = time
                time_ref = len(time)
                basetime = sorted(basetime)

        av_entries_ls = []

        # calculate average entry for all 5 detection devices
        for timeslot in basetime:
            entries_n = []
            for df in df_ls:

                mask = df['time'] == timeslot
                un_time = df[mask]
                entries_n.append(len(un_time))
            av_entries_ls.append(np.mean(entries_n))




        basetime = [entry.split(' ')[1] for entry in basetime]
     
        # seperate the three paths into different plots using indexes
        for i in range(len(tp_ls)):
            
        
            start_time = pd.to_datetime(tp_ls[i][0]).time()
            end_time = pd.to_datetime(tp_ls[i][1]).time()

            s_index = None
            e_index = None
            for j in range(len(basetime)):
                
                dt_t = pd.to_datetime(basetime[j]).time()


                if dt_t <= start_time:
                
                    s_index = j 
                    

            for j in range(len(basetime)):
                dt_t = pd.to_datetime(basetime[j]).time()

                if dt_t <= end_time:
                
                    e_index = j 
                    
            path_basetime = basetime[s_index:e_index+1]

    
            path_av_entries_ls = av_entries_ls[s_index:e_index+1]


            plt.figure()
            bar = plt.bar(path_basetime, path_av_entries_ls, color=path_clr_ls[i])
            
            print(f"mean_key_valeu for  {dt_string} and {data}", np.mean(path_av_entries_ls))
            print(f"sum_key_valeu for  {dt_string} and {data}", np.sum(path_av_entries_ls))


            len_bt = int(len(path_basetime)/2)
            tick_locations = [path_basetime[0], path_basetime[len_bt], path_basetime[-1]]

            # Set the ticks on the x-axis
            plt.xticks(tick_locations, fontsize=18)
            plt.yticks(fontsize=18)
            plt.ylabel("Entry Number", fontsize = 23)
            plt.xlabel("Time (s)", fontsize = 23)

            plt.tight_layout()
            plt.savefig(f"data\\figures\\data_entries\\dts\\{dt_string}\\data_entries{data}_path{i}.pdf", format='pdf')
        
            # plt.show()