import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import pickle
import math
from explore_AP_KP import hospital_img_template
import pandas as pd
import kalman_filter
import time_og_data2

# this one should matter! since we are going to remove AP points 
# but for now we can't do anything, since you don't have the locations! 
ap_loc  = np.load('C:\\Users\\coen_\\OneDrive\\Documents\\ma_thesis\\files\\iBeacon_data\\summer_2_floor_04_08\\points_wifi_2.npy')
img_hospital = plt.imread("data\\environment\\map_2floor_bw.png") 

# which radiodict doesn't matter here, since it's only about the keys 
with open("data\\radio_maps\\dict\\lp.pkl", 'rb') as fp: 
    radiodict= pickle.load(fp)
keys = radiodict.keys()



def format_latex(values):
    mean_val, std_dev = values
    return f"$\\mu = {mean_val}$, $\\sigma = {std_dev}$"

def time_difference(start_time, mid_time, end_time):

    time_dif = mid_time - start_time
    time_dif = time_dif.total_seconds()
    total_time = end_time - start_time
    total_time = total_time.total_seconds()

    fraction = time_dif/total_time


    return fraction



def av_time_calc(timestamps):
    # Convert the list of timestamps to datetime objects
    datetime_objects = pd.to_datetime(timestamps, format='mixed')

    # Find the minimum timestamp in the list
    min_time = min(datetime_objects)

    # Calculate the time differences and store them in a list
    time_diffs = [(time - min_time).total_seconds() for time in datetime_objects]
    
    # Calculate the mean of the time differences
    mean_seconds = np.sum(time_diffs) / len(time_diffs)
    
    # Calculate the average time by adding the mean time difference to the minimum time
    av_time = min_time + pd.to_timedelta(mean_seconds, unit='s')

    return av_time



# with open("data\\radio_maps\\dict\\dict_sigma_radio_2_lp.pkl", 'rb') as fp: 
#     radio_s_dict= pickle.load(fp)

# keys are the same for radio_dict and radio_s_dict

"""
For plotting 
"""


# image of blueprint hospital 


# you have to substract these from positions to make it work with image 

hard_dx = 235
hard_dy = 76

# ratio points to meters 



# def format_meters(value, _):
#     meters = value * conversion_factor
#     return f'{meters:.2f} m'



def eucl_d(x_int, y_int, x_rssi, y_rssi):
    euc_d_ls = []
    for i in range(len(x_int)): 
        euc_d_ls.append(math.sqrt(abs(x_int[i]- x_rssi[i])**2) + math.sqrt(abs(y_int[i] - y_rssi[i])**2))

    return euc_d_ls


def short_d(x_int, y_int, x_rssi, y_rssi):
    short_d_ls = []
    for i in range(len(x_rssi)):
        # Initialize the minimum distance to a large value
        min_distance = float('inf')
    
        for j in range(len(x_int)):
            # Calculate distance between (x[i], y[i]) and (ref_x[j], ref_y[j])
            distance = math.sqrt(abs(x_int[j]- x_rssi[i])**2) + math.sqrt(abs(y_int[j] - y_rssi[i])**2)
        
        # Update the minimum distance if the current distance is smaller
            min_distance = min(min_distance, distance)
        short_d_ls.append(min_distance)
    
    return short_d_ls


        

# Given path

#These are test paths 04-08 corresponding with rssi map


def accuracy(og_path, rssi_df, time_interval, plot_toggle, both_toggle, lines_toggle, filter):
    # Separate x and y coordinates


    if filter == "PF":
        print("test PF")
        x2 = rssi_df['x_pf']
        y2 = rssi_df['y_pf']
    
    if filter == "KF":
        print("test KF")
        x2 = rssi_df['x_kf']
        y2 = rssi_df['y_kf']
    
    if filter == None:
        # print("test filter no workie")
        x2 = rssi_df['x']
        y2 = rssi_df['y']

    x, y = zip(*og_path)
    
    time_av = rssi_df['av_time']
    print("time av=", time_av)
    end_time = time_interval[1]
    begin_time = time_interval[0]
    total_seconds = (end_time -  begin_time).total_seconds()
    # this is probably going to be changed because it is a DF file! 

    # this is temporary language change this
    # print("rssi_path", rssi_path)
    # print("type rssi", type(rssi_path))
    # x2, y2 = rssi_path["x"], rssi_path["y"]
   
    x = list(map(lambda x: x - hard_dx, x))
    y = list(map(lambda x: x - hard_dy, y))

    # Create RegularGridInterpolator
    interp = RegularGridInterpolator((np.arange(len(x)),), np.array([x, y]).T, method="linear")
    

    # Define the grid for interpolation
    grid_x = np.linspace(0, len(x) - 1, 1000)  
    
    interpolated_path = interp(grid_x)
    
    # Separate x and y coordinates of the interpolated path
    interpolated_x, interpolated_y = interpolated_path.T
    

    x2 = list(map(lambda x: x - hard_dx, x2))
    y2 = list(map(lambda x: x - hard_dy, y2))

    # remove creating x2 interpolate and y2 interpolate here, it confuses everything! 
    
    x_match = []
    y_match = []


    for i, time in enumerate(time_av):

        seconds_passed = (time - begin_time).total_seconds()
        ratio = seconds_passed/total_seconds
        print("seconds passed", seconds_passed)
        print("total seconds", total_seconds)
        print("ratio!!!", ratio)
        ratio_indice = ratio * len(interpolated_x)
        if ratio_indice >= len(interpolated_x):
            ratio_indice = -1
        ratio_indice = int(ratio_indice)
        print("ratio indice = ", ratio_indice)
        x_match.append(interpolated_x[ratio_indice])
        y_match.append(interpolated_y[ratio_indice])


    #returns a list of all euclidean distances 
    t_dist_ls = eucl_d(x_match, y_match, x2, y2)
    a_dist_ls = short_d(x_match, y_match, x2, y2)
    # convert into meters
    meters_conversion = 0.13333
    t_dist_ls = [item * meters_conversion for item in t_dist_ls]
    a_dist_ls = [item * meters_conversion for item in a_dist_ls]
    t_mean_value = np.mean(t_dist_ls)
    t_std_value = np.std(t_dist_ls)
    t_sum_value = np.sum(t_dist_ls)
    a_mean_value = np.mean(a_dist_ls)
    a_std_value = np.std(a_dist_ls)
    a_sum_value = np.sum(a_dist_ls)




    if plot_toggle:
        
        fig, ax = hospital_img_template(img_hospital, ap_loc, radiodict, ap_toggle=True, finger_toggle=False)
        ax.plot(x , y, 'o', label='Original Path')
        ax.plot(interpolated_x, interpolated_y, label='Interpolated Path')

        if both_toggle:
            interp2 = RegularGridInterpolator((np.arange(len(x2)),), np.array([x2, y2]).T, method="linear")
            grid_x2 = np.linspace(0, len(x2)-1, 1000)
            interpolated_path2 = interp2(grid_x2)
            interpolated_x2, interpolated_y2 = interpolated_path2.T
            ax.plot(x2, y2, 'o', label="RSSI Path")
            ax.plot(interpolated_x2, interpolated_y2, label="interpolated RSSI Path" )
            # plt.title(f'Both original and RSSI paths for {path_name}')

        if lines_toggle:
            for i in range(len(x2)):
                if i ==0:
                    ax.plot([x_match[i], x2[i]], [y_match[i], y2[i]], color="gray", label="comparison lines")
                else:
                    ax.plot([x_match[i], x2[i]], [y_match[i], y2[i]], color="gray")


        plt.legend()
        plt.tight_layout()
        plt.show()
    return [t_mean_value, t_std_value, t_sum_value], [a_mean_value, a_std_value, a_sum_value]
   


def create_av_df(path, PF, KF):

    df_ls = []
    un_time_set = set()

# Create a list of all the data from the 5 different recording devices and create list of all the unique timepoints
    for i in range(5):
        rssi_data = pd.read_csv(f"{path}{i}.csv", index_col=0) 
        df_ls.append(rssi_data)
        unique_timestamps = rssi_data['time'].unique()
        un_time_set.update(unique_timestamps)
    x_av_ls = []
    y_av_ls = []

    PF_x_av_ls = [] 
    PF_y_av_ls = []
    KF_x_av_ls = []
    KF_y_av_ls = []

    time_av_ls = []
    un_time_ls = list(un_time_set)
    un_time_ls = sorted(un_time_ls)

    for time in un_time_ls:
        time_ls = []
        x_ls = []
        y_ls = []
        PF_x_ls = [] 
        PF_y_ls = []
        KF_x_ls = []
        KF_y_ls = []

        for df in df_ls:

            if time in df['time'].values:

                time_val = df.loc[df['time'] == time, 'av_time'].values[0]
            
                
                x_val = df.loc[df['time'] == time, 'x'].values[0]
                y_val = df.loc[df['time'] == time, 'y'].values[0]
                time_ls.append(time_val)
                x_ls.append(x_val)
                y_ls.append(y_val)
                
                if PF:
                    x_val = df.loc[df['time'] == time, 'x_pf'].values[0]
                    y_val = df.loc[df['time'] == time, 'y_pf'].values[0]
                    PF_x_ls.append(x_val)
                    PF_y_ls.append(y_val)
                

                if KF:
                    x_val = df.loc[df['time'] == time, 'x_kf'].values[0]
                    y_val = df.loc[df['time'] == time, 'y_kf'].values[0]
                    KF_x_ls.append(x_val)
                    KF_y_ls.append(y_val)


        av_df = pd.DataFrame(data=None, columns= df.columns)
        time_av = av_time_calc(time_ls)
        time_av_ls.append(time_av)

        x_av = np.sum(x_ls)/len(x_ls)
        y_av = np.sum(y_ls)/len(y_ls)
        x_av_ls.append(x_av)
        y_av_ls.append(y_av)

        if PF:
            print("yeah yeah should work")
            x_av_pf = np.sum(PF_x_ls)/len(PF_x_ls)
            y_av_pf = np.sum(PF_y_ls)/len(PF_y_ls)
            PF_x_av_ls.append(x_av_pf)
            PF_y_av_ls.append(y_av_pf)

        if KF:
            x_av_pf = np.sum(KF_x_ls)/len(KF_x_ls)
            y_av_pf = np.sum(KF_y_ls)/len(KF_y_ls)
            KF_x_av_ls.append(x_av_pf)
            KF_y_av_ls.append(y_av_pf)

        

    av_df['time'] = un_time_ls
    av_df['av_time'] = time_av_ls
    av_df['x'] = x_av_ls
    av_df['y'] = y_av_ls
    
    av_df['time'] = pd.to_datetime(av_df['time'], format='mixed')
    
    if PF:
        av_df['x_pf'] = PF_x_av_ls
        av_df['y_pf'] = PF_y_av_ls
    if KF:
        av_df['x_kf'] = KF_x_av_ls
        av_df['y_kf'] = KF_y_av_ls

    return av_df

            


# av_df = create_av_df(path, PF=True, KF=False)
# print(av_df)
# print(av_df.columns)
# av_df.to_csv("junk\\test_av_df_filter.csv")


path_time_ls = [[time_og_data2.start_time1, time_og_data2.end_time1], [time_og_data2.start_time2, time_og_data2.end_time2], [time_og_data2.start_time3, time_og_data2.end_time3]]
og_paths = [time_og_data2.og_path1, time_og_data2.og_path2, time_og_data2.og_path3]


# path = f"data\\paths\\dt_10\\filter\\lp2\\ignore\\6\\gaus_normal\\path0"
av_df = pd.read_csv(f"{path}.csv", index_col=0) 
av_df['time'] = pd.to_datetime(av_df['time'], format='mixed')
av_df["av_time"] = av_df["time"]

# print(av_df)
# for i in range(len(path_time_ls)):
#     print(type(path_time_ls[i][1]))
#     path_n_df = av_df[(av_df['time'] >= path_time_ls[i][0]) & (av_df['time'] <= path_time_ls[i][1])]
#     t_vals, a_vals = accuracy(og_paths[i] ,path_n_df, path_time_ls[i], plot_toggle=True, both_toggle=True, lines_toggle=False, filter = None)
#     t_mean, t_std, t_sum = t_vals
#     a_mean, a_std, a_sum = a_vals
#     print("t_mean=", t_mean, "t_std=", t_std, "t_sum=", t_sum)
#     print("a_mean=", a_mean, "a_std=", a_std, "a_sum=", a_sum)




# for i in range(5):
#     av_df= pd.read_csv(f"{path}{i}.csv", index_col=0)
#     av_df['time'] = pd.to_datetime(av_df['time'], format='mixed')
#     for i in range(len(path_time_ls)):
#         print(type(path_time_ls[i][1]))
#         print(type(path_time_ls[i]))
#         path_n_df = av_df[(av_df['time'] >= path_time_ls[i][0]) & (av_df['time'] <= path_time_ls[i][1])]
#         mean, std, sum = accuracy(og_paths[i] ,path_n_df, path_time_ls[i], plot_toggle=True, both_toggle=True, lines_toggle=False)
#         print("mean=", mean, "std=", std, "sum=", sum)
    



# # print(x2, xkal)
# # print(y2, ykal)
# fig, ax = hospital_img_template(img_hospital, ap_loc, keys, ap_toggle=True, finger_toggle=False)
# test = accuracy(og_path3 ,rssi_path, plot_toggle=True, both_toggle=True, lines_toggle=False, path_name='path 1')


# this is the one you need to check you fool! this is the one from the picture! Try this one with only the weird one removed!!!!
# rssi_data  = pd.read_csv("data\\paths\\presentation\\raw\\RSSI_lp.csv_0_method_penalty_n_6_loc_gaus_normal.csv", index_col=0) 


# x_ls = []
# y_ls = []
# path_n = 1
# for i in range(5):
#     rssi_data = pd.read_csv(f"data\\paths\\presentation\\raw\\INTERNET_RSSI_lp_{i}_method_penalty_n_6_loc_gaus_normal.csv", index_col=0) 
    
#     rssi_data = sep_paths(rssi_data)
#     path = rssi_data[path_n]
#     print("pathlength", len(path))
    # unique_timestamps = path['time'].unique()
    # un_time_set.update(unique_timestamps)
    
    # x2, y2 = path["x"], path["y"]


    # # Initialize an empty set

    # # x_meas = x2.to_numpy()
    # # x_kal= kalman_filter.runfilter(x_meas, "X-position")
    # # x_ls.append(x_kal)
    # # y_meas = y2.to_numpy()
    # # y_kal= kalman_filter.runfilter(y_meas, "Y-position")
    # # y_ls.append(y_kal)

    # # without filter

    # x_meas = x2.to_numpy()
    # x_ls.append(x_meas)
    # y_meas = y2.to_numpy()   
    # y_ls.append(y_meas)
