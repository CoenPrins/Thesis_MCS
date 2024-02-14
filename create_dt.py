"""

time_og_data2
Organizes the datafile into time interval data, 

which is default 10 seconds, but can also be other amounts. 
5-10-15-20-30

Df_radio is read in but only for removing the non matching signals...

this creates 5 files for every btmac (receiving device), you can specify stuff!

"""


import pandas as pd 
from datetime import datetime
import numpy as np
import create_path
import time_og_data2

def only_paths(df):
   

    df['time'] = pd.to_datetime(df['time'], format='mixed')

    


    mask = ((df['time'] >= time_og_data2.start_time1) & (df['time'] <= time_og_data2.end_time1)) | \
        ((df['time'] >= time_og_data2.start_time2) & (df['time'] <= time_og_data2.end_time2)) | \
        ((df['time'] >= time_og_data2.start_time3) & (df['time'] <= time_og_data2.end_time3))

    # Create a new DataFrame with the filtered rows
    filtered_df = df[mask]
    
    return filtered_df

def create_dt(rssi_path, radio_path, dt_val, path_toggle, knn_N, mis_val_method, loc_method):
    # print("starting up a round of create_dt!!!")
    df_rssi = pd.read_csv(f"data\\rssi\\{rssi_path}.csv", index_col=0)
    df_radio = pd.read_csv(f"data\\radio_maps\\csv\\{radio_path}.csv",  index_col=0)
    
    #set time to pandas datetime

    df_rssi['time'] = pd.to_datetime(df_rssi['time'], format='mixed')


    # only keep data that's within the timewindow of true paths 
    if path_toggle:
        df_rssi = only_paths(df_rssi)

    
    # get all the unique btmac numbers from rssi file 
    btmac_rssi = df_rssi['btmac'].unique()
   
    
    # create a list containing the 5 different btmacs as elements 
    btmacs = []
    for btmac in btmac_rssi:
        df_btmac = df_rssi.loc[df_rssi['btmac']== btmac].copy()
        
        btmacs.append(df_btmac)
        
        
    

    count = 0
    path_interval_ls  = [[time_og_data2.start_time1, time_og_data2.end_time1], [time_og_data2.start_time2, time_og_data2.end_time2], [time_og_data2.start_time3, time_og_data2.end_time3]]
    df_ls = []
    for btmac in btmacs:
        append_df = pd.DataFrame(data=None, columns=btmac.columns)
        append_df["meas_time"] = None
        btmac["meas_time"] = None
        

        # look at the three different paths seperately
        for path_times in path_interval_ls:
            d_0 = path_times[0]
            d_max = path_times[1]
            dt = dt_val
            d_delt = d_0 +  pd.to_timedelta(dt, unit='s')
            
            while d_delt <= d_max: 
                mask = (btmac['time'] >= d_0) & (btmac['time'] < d_delt)
             
                df_interval = btmac.loc[mask].copy()
          
                # in case there are no signals received within the time interval
                if len(df_interval) > 0:
                    mac_un_dt = df_interval['mac'].unique()
                    df_dt = df_interval.drop_duplicates(subset='mac')
                   
                    # if there are multiple signals with the same mac take the average.
                    # there is no variance in the rssis data only in the radio map


                    # take all the unique macs and assign them average rssi and time
                    for mac in mac_un_dt:
                        means_rssi = df_interval.groupby('mac')['rssi'].mean()
                        means_time = df_interval.groupby('mac')['time'].mean()
                        
                        
                        df_dt.loc[df_dt['mac'] == mac, 'rssi'] = means_rssi[mac]
                        df_dt.loc[df_dt['mac'] == mac, 'meas_time'] = means_time[mac]
                



                    # time signifies interval, meas_time unique time
                    df_dt = df_dt.assign(time=d_0)
                    
                    append_df = pd.concat([append_df, df_dt])
                    append_df = append_df.reset_index(drop=True)
                    d_delt += pd.to_timedelta(dt, unit='s')
                    d_0 += pd.to_timedelta(dt, unit='s')

                # skip if no entries 
                else:
                    print("interval with zero entry!! time=" , d_0, 'count=', count)
                    d_delt += pd.to_timedelta(dt, unit='s')
                    d_0 += pd.to_timedelta(dt, unit='s')

        
        print()
        
        append_df.to_csv(f"data\\dt\\dt_{dt_val}\\{rssi_path}\\dt{count}.csv")
        df_ls.append(append_df)

     
        count += 1
    
    
    
    create_path.create_path(df_ls, knn_N, mis_val_method, loc_method, rssi_path, radio_path, dt_val)
        

       


# dt_ls = [5, 10, 15, 30]

# # these are not neccesarily the combinations!
# rssi_ls = [["lp.csv", 'rssi_lp'], ["lp2.csv", "rssi_lp2"], ["lp2_n85.csv", "rssi_lp2_n85"], ["lp2_n100.csv", "rssi_lp2_n100"]]

# radio_path = "data\\radio_maps\\csv\\radio_map_2.csv"
# for dt in dt_ls:
#     for rssi in rssi_ls:
#             run = create_dt(rssi[0], radio_path, rssi[1],  dt)


"""
First we create the 
"""



# dt_ls = [10]
# radio_rssi_path_ls= ["lp2"]
# mis_val_methods_ls = ["penalty", "ignore"]
# knn_N_ls  = [1, 3, 6, 12]
# loc_method_ls = ["average", "gaus_normal", "gaus_80", "gaus_top"]

# for dt in dt_ls:
#     for path in radio_rssi_path_ls:
#         for mis_val_method in mis_val_methods_ls:
#                 for knn_N in knn_N_ls:
#                     for loc_method in loc_method_ls:
#                         runtest = create_dt(path, path, dt_val=dt, path_toggle=True, knn_N = knn_N, mis_val_method=mis_val_method, loc_method=loc_method)





# you still need to run the 
dt_ls = [3]
radio_rssi_path_ls= ["lp", "lp2", "lp2_n85", "lp2_n100"]
mis_val_methods_ls = ["penalty", "ignore"]
knn_N_ls  = [1, 3, 6, 12]
loc_method_ls = ["average", "gaus_normal", "gaus_80", "gaus_top"]
# make this enumerate combo! 
for dt in dt_ls:
    for path in radio_rssi_path_ls:
        for mis_val_method in mis_val_methods_ls:
                for knn_N in knn_N_ls:
                    for loc_method in loc_method_ls:
                        runtest = create_dt(path, path, dt_val=dt, path_toggle=True, knn_N = knn_N, mis_val_method=mis_val_method, loc_method=loc_method)


# dt_ls = [5]
# radio_rssi_path_ls= ["lp2_n100"]
# mis_val_methods_ls = ["penalty", "ignore"]
# knn_N_ls  = [1, 3, 6, 12]
# loc_method_ls = ["average", "gaus_normal", "gaus_80", "gaus_top"]
# # make this enumerate combo! 
# for dt in dt_ls:
#     for path in radio_rssi_path_ls:
#         for mis_val_method in mis_val_methods_ls:
#                 for knn_N in knn_N_ls:
#                     for loc_method in loc_method_ls:
#                         runtest = create_dt(path, path, dt_val=dt, path_toggle=True, knn_N = knn_N, mis_val_method=mis_val_method, loc_method=loc_method)


# dt_ls = [10]


# radio_rssi_path_ls= ["lp"]
# mis_val_methods_ls = ["penalty", "ignore"]
# knn_N = 6
# loc_method = "gaus_normal"
# no_ls = ["_no_6a", "_no_44", "_no_a6","_no_ac","_no_c6","_no_fe"]
# # make this enumerate combo! 
# for dt in dt_ls:
#     for path in radio_rssi_path_ls:
#         for mis_val_method in mis_val_methods_ls:
#             for no in no_ls:
#                 path_alter = path + no
#                 runtest = create_dt(path_alter, path_alter, dt_val=dt, path_toggle=True, knn_N = knn_N, mis_val_method=mis_val_method, loc_method=loc_method)




#OG start times

# start_time1 = pd.to_datetime("04/08/2021 18:44:00")  
# end_time1 = pd.to_datetime("04/08/2021 18:47:00")    

# start_time2 = pd.to_datetime("04/08/2021 18:52:00")  
# end_time2 = pd.to_datetime("04/08/2021 18:54:00")    

# start_time3 = pd.to_datetime("04/08/2021 18:48:00")  
# end_time3 = pd.to_datetime("04/08/2021 18:51:00") 


   # append_df = pd.DataFrame(columns=btmac)
        # append_df = pd.DataFrame().reindex_like(btmac)

        # the saving of the index here is helpfull but causes the an index of na to form, 
        # sloppy code but we remove the na later