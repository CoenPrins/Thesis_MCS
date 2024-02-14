import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import pickle
import math
from explore_AP_KP import hospital_img_template
import pandas as pd
import kalman_filter
import time_og_data2
import time


first = True
# this one should matter! since we are going to remove AP points 
# but for now we can't do anything, since you don't have the locations! 
ap_loc  = np.load('C:\\Users\\coen_\\OneDrive\\Documents\\ma_thesis\\files\\iBeacon_data\\summer_2_floor_04_08\\points_wifi_2.npy')
img_hospital = plt.imread("data\\environment\\map_2floor_bw.png") 

# which radiodict doesn't matter here, since it's only about the keys 
with open("data\\radio_maps\\dict\\lp.pkl", 'rb') as fp: 
    radiodict= pickle.load(fp)
keys = radiodict.keys()



def format_latex(values):

    if len(values) == 2:
        mean_val, std_dev = values
        string = f"$\\mu = {mean_val}$, $\\sigma = {std_dev}$"
    else:
        mean_val = values
        string = f"$\\mu = {mean_val}$"
    return string

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


"""
Simplify it, 

You only have the normal interpolated error. 


You calculate it using the average, but also for all the paths, the shortest path from the 5 you select. 

You return both!
"""

# def 



def ac_calc(og_path, rssi_df, t_delta, filter):

    if filter == "PF":
        x2 = rssi_df['x_pf']
        y2 = rssi_df['y_pf']
    
    if filter == "KF":
        x2 = rssi_df['x_kf']
        y2 = rssi_df['y_kf']
    
    if filter == None:
        x2 = rssi_df['x']
        y2 = rssi_df['y']


    x2 = list(map(lambda x: x - hard_dx, x2))
    y2 = list(map(lambda x: x - hard_dy, y2))

    x, y = zip(*og_path)
    x = list(map(lambda x: x - hard_dx, x))
    y = list(map(lambda x: x - hard_dy, y))
    
    time_av = rssi_df['av_time']
    # print("length time_av", len(time_av))
    begin_time = t_delta[0]
    end_time = t_delta[1]
    total_seconds = (end_time -  begin_time).total_seconds()
    # this is probably going to be changed because it is a DF file! 



    # Create RegularGridInterpolator
    interp = RegularGridInterpolator((np.arange(len(x)),), np.array([x, y]).T, method="linear")
    

    # Define the grid for interpolation
    grid_x = np.linspace(0, len(x) - 1, 1000)  
    
    interpolated_path = interp(grid_x)
    
    # Separate x and y coordinates of the interpolated path
    interpolated_x, interpolated_y = interpolated_path.T

    
    # print("interpolated_x", interpolated_x)
    # print("interpolated_y", interpolated_y)
    
    
    # remove creating x2 interpolate and y2 interpolate here, it confuses everything! 
    
    x_match = []
    y_match = []

    for i, time in enumerate(time_av):

        seconds_passed = (time - begin_time).total_seconds()
        ratio = seconds_passed/total_seconds
 
        ratio_indice = ratio * len(interpolated_x)
        if ratio_indice >= len(interpolated_x):
            ratio_indice = -1
        ratio_indice = int(ratio_indice)

        x_match.append(interpolated_x[ratio_indice])
        y_match.append(interpolated_y[ratio_indice])


    #returns a list of all euclidean distances 
    t_dist_ls = eucl_d(x_match, y_match, x2, y2)
    # convert into meters

    meters_conversion = 0.13333
    t_dist_ls = [item * meters_conversion for item in t_dist_ls]

    # print("length t_dist_ls", len(t_dist_ls))
    # print('x2', x2)
    # print("y2", y2)
    # print("x_match", x_match)
    # print('y_match', y_match)
    # print("interx", interpolated_x)
    # print("intery", interpolated_y)

    return [t_dist_ls, [x, y], [x2, y2], [x_match, y_match], [interpolated_x, interpolated_y]]





def plot_path(error_ls, lines_plt, ax, legend_name):

    rssi_x, rssi_y = error_ls[2][0], error_ls[2][1]
    x_match, y_match = error_ls[3][0], error_ls[3][1]
    

    
    

    interp2 = RegularGridInterpolator((np.arange(len(rssi_x)),), np.array([rssi_x, rssi_y]).T, method="linear")
    grid_x2 = np.linspace(0, len(rssi_x)-1, 1000)
    interpolated_path2 = interp2(grid_x2)
    interpolated_x2, interpolated_y2 = interpolated_path2.T
    ax.plot(rssi_x, rssi_y, 'o', label=f"RSSI {legend_name} Path Points ")
    ax.plot(interpolated_x2, interpolated_y2, label=f"Interpolated RSSI {legend_name} Path" )
    # plt.title(f'Both original and RSSI paths for {path_name}')

    if lines_plt:
        for i in range(len(rssi_x)):
            if i ==0:
                ax.plot([x_match[i], rssi_x[i]], [y_match[i], rssi_y[i]], color="gray", label="comparison lines")
            else:
                ax.plot([x_match[i], rssi_x[i]], [y_match[i], rssi_y[i]], color="gray")

    return ax

    

    



def accuracy(path_ls, rssi_path, path_time_ls, plt_paths, av_plt, best_plt, lines_plt, filter, filename):
    # Separate x and y coordinates

    if filter == "PF":
        PF = True
        KF = False
    if filter == "KF":
        PF = False
        KF = True
    if filter == None:
        PF = False
        KF = False

    
    av_df = create_av_df(rssi_path, PF, KF)
    
    
    av_er_ls =[]
    best_er_ls =[]
    av_ind_er_ls = []
    # print("path_ls", len(path_ls))
    for i in range(len(path_ls)):
        # print("av_er_ls",av_er_ls)
        t_delta = [path_time_ls[i][0], path_time_ls[i][1]]
        rssi_av_df = av_df[(av_df['time'] >= t_delta[0] ) & (av_df['time'] < t_delta[1])]
        og_path = path_ls[i]
        # print("calculating average ac_calc!")
        av_error = ac_calc(og_path, rssi_av_df, t_delta, filter)

        av_er_ls.append(av_error)

        individual_error_ls = []
        total_error = 0
        for j in range(0, 5):
            single_df = pd.read_csv(f"{rssi_path}{j}.csv")
            single_df['time'] = pd.to_datetime(single_df['time'], format='mixed')
            single_df['av_time'] = pd.to_datetime(single_df['av_time'], format='mixed')
            rssi_single_df = single_df[(single_df['time'] >= t_delta[0] ) & (single_df['time'] < t_delta[1])]
            # print("calculating single ac_calc!")
            single_error = ac_calc(og_path, rssi_single_df, t_delta, filter)
            individual_error_ls.append(single_error)
            total_error += np.mean(single_error[0])

        individual_error_ls.sort(key=lambda x: np.mean(x[0]))
        best_error = individual_error_ls[0]
        best_er_ls.append(best_error)
        av_individual_error = total_error/5
        av_ind_er_ls.append(av_individual_error)

        if plt_paths:
            fig, ax = hospital_img_template(img_hospital, ap_loc, radiodict, ap_toggle=True, finger_toggle=False)
            x, y = av_error[1][0], av_error[1][1]
            interpolated_x, interpolated_y = av_error[4][0], av_error[4][1]
            ax.plot(x , y, 'o', label='True Path Points')
            ax.plot(interpolated_x, interpolated_y, label='Interpolated True Path')


            if av_plt:
               ax =  plot_path(av_error, lines_plt, ax= ax, legend_name= "Average")
            
            if best_plt:
                ax = plot_path(best_error, lines_plt, ax= ax, legend_name = "Best")

            plt.legend()
            plt.tight_layout()
            path_number = i + 1
            if filename:
                plt.savefig(f"data\\figures\\{filename}_path{path_number}.pdf", bbox_inches='tight' )
            plt.show()
            
    

    av_ls = []
    best_ls = []
    av_er_t_ls = []
    b_er_t_ls = []
    for path in av_er_ls:
        # print("length path av er", len(path))
        av_ls.append([np.mean(path[0]), np.std(path[0]), np.sum(path[0])])
        av_er_t_ls.append(path[0])
    
    for path in best_er_ls:
        # print("length path best er", len(path))
        best_ls.append([np.mean(path[0]), np.std(path[0]), np.sum(path[0])])
        b_er_t_ls.append(path[0])

    return(av_ls, best_ls, av_ind_er_ls, av_er_t_ls, b_er_t_ls)
    

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

            



# we alrady have a function that works very well, 
# The only thing left to do is to implement 


"""
for entire timewindow using time windowed true path

"""

# path_time_ls = [[time_og_data2.start_time1, time_og_data2.end_time1], [time_og_data2.start_time2, time_og_data2.end_time2], [time_og_data2.start_time3, time_og_data2.end_time3]]
# og_paths = [time_og_data2.n_og_path1, time_og_data2.n_og_path2, time_og_data2.n_og_path3]


"""
for smalltimewindow using original true path
"""

# path_time_ls = [[time_og_data2.n_start_time1, time_og_data2.n_end_time1], [time_og_data2.n_start_time2, time_og_data2.n_end_time2], [time_og_data2.n_start_time3, time_og_data2.n_end_time3]]
# og_paths = [time_og_data2.og_path1, time_og_data2.og_path2, time_og_data2.og_path3]


# path = f"data\\paths\\dt_3\\raw\\lp2\\ignore\\6\\gaus_normal\\path"

# # # av_df = pd.read_csv(f"{path}.csv", index_col=0) 

# start_time = time.time()
# av_ls, best_ls, av_i_error, av_er_t_ls, b_er_t_ls = accuracy(og_paths ,path, path_time_ls, plt_paths =True, av_plt=True, best_plt=False, lines_plt=False, filter = None)
# end_time = time.time()


# print(av_ls)

"""
Here you have the barebones model for how you can plot the error through time! 

missing! 


For paticle filter I think the initialisation is interesting I thing

"""

# for i in range(len(av_er_t_ls)):
#     len_av_er = len(av_er_t_ls[i])
#     len_b_er = len(b_er_t_ls[i])
    

#     # print(len_av_er)
#     # print(len_b_er)

#     av_t_ls = time_og_data2.generate_datetime_dataset(path_time_ls[i][0], path_time_ls[i][1], len_av_er, small_strings=False)
#     b_t_ls = time_og_data2.generate_datetime_dataset(path_time_ls[i][0], path_time_ls[i][1], len_b_er, small_strings=False)
#     # print("length t list", len(t_ls))

#     plt.plot(av_t_ls, av_er_t_ls[i], label="Error Average path")
#     plt.plot(b_t_ls, b_er_t_ls[i], label="Error Best path")
#     plt.legend()
#     plt.show()
# print("av_er_t", av_er_t_ls)
# print("elapse time", end_time - start_time)
# print("error av path 1", av_ls[0])
# print("error av path 2", av_ls[1])
# print("error av path 3",av_ls[2])

# print("error best path 1", best_ls[0])
# print("error best  path 2", best_ls[1])
# print("error best  path 3",best_ls[2])


# print("error best path 1", av_i_error[0])
# print("error best  path 2", av_i_error[1])
# print("error best  path 3",av_i_error[2])


# plt 


# path_time_ls = [[time_og_data2.start_time1, time_og_data2.end_time1], [time_og_data2.start_time2, time_og_data2.end_time2], [time_og_data2.start_time3, time_og_data2.end_time3]]
# og_paths = [time_og_data2.og_path1, time_og_data2.og_path2, time_og_data2.og_path3]


# path = f"data\\paths\\dt_5\\raw\\lp2\\ignore\\6\\gaus_normal\\path"

# # av_df = pd.read_csv(f"{path}.csv", index_col=0) 

# av_df = create_av_df(path, PF = False, KF = False)

# for i in range(len(path_time_ls)):
#     path_n_df = av_df[(av_df['time'] >= path_time_ls[i][0]) & (av_df['time'] < path_time_ls[i][1])]
#     t_vals = accuracy(og_paths[i] ,path_n_df, path_time_ls[i], plot_toggle=True, both_toggle=True, lines_toggle=False, filter = None)
#     t_mean, t_std, t_sum = t_vals


