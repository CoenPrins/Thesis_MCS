import pandas as pd 
from datetime import datetime
import numpy as np
import copy


df_0 = pd.read_csv("C:\\Users\\coen_\\OneDrive\\Documents\\ma_thesis\\OG_NN\\fp_agent_traj_sigma\\extra_bracket\\create_paths_test0.csv")
df_1 = pd.read_csv("C:\\Users\\coen_\\OneDrive\\Documents\\ma_thesis\\OG_NN\\fp_agent_traj_sigma\\extra_bracket\\create_paths_test1.csv")
df_2 = pd.read_csv("C:\\Users\\coen_\\OneDrive\\Documents\\ma_thesis\\OG_NN\\fp_agent_traj_sigma\\extra_bracket\\create_paths_test2.csv")
df_3 = pd.read_csv("C:\\Users\\coen_\\OneDrive\\Documents\\ma_thesis\\OG_NN\\fp_agent_traj_sigma\\extra_bracket\\create_paths_test3.csv")
df_4 = pd.read_csv("C:\\Users\\coen_\\OneDrive\\Documents\\ma_thesis\\OG_NN\\fp_agent_traj_sigma\\extra_bracket\\create_paths_test4.csv")

df_ls = [df_0, df_1, df_2, df_3, df_4]

path_1 = []
path_2 = []
path_3 = []



# test_path_1_04_08.txt ( 04/08/2021 18:44:00- 18:47:00)
# test_path_2_04_08.txt ( 04/08/2021 18:52:00- 18:54:00)
# test_path_3_04_08.txt ( 04/08/2021 18:48:00- 18:51:00)

count = 0
for df in df_ls:
        
    df['time'] = pd.to_datetime(df['time'], format='mixed')
    print("big test", len(df))

    df_path_1 = df[(df['time'] > "04/08/2021 18:44:00") & (df['time'] < "04/08/2021 18:47:00")]
    path_1.append(df_path_1)
    # print("path_1 ", count, df_path_1['x'])
    df_path_2 = df[(df['time'] > "04/08/2021 18:52:00") & (df['time'] < "04/08/2021 18:54:00")]
    path_2.append(df_path_2)
    # print("path_2 ", count, df_path_2['x'])
    df_path_3 = df[(df['time'] > "04/08/2021 18:48:00") & (df['time'] < "04/08/2021 18:51:00")]
    # print(type(df_path_3))
    path_3.append(df_path_3)
    
    # print("path_3 ", count, df_path_3['x'])
    count += 1 
paths = [path_1, path_2, path_3]




template = pd.DataFrame(data=None, columns= ['time', 'av_x', 'av_y', 'dt0_x','dt0_y', 'dt1_x', 'dt1_y', 'dt2_x','dt2_y', 'dt3_x', 'dt3_y', 'dt4_x', 'dt4_y'])

final_paths = []

# never code like this again! always do it in a loop, think of it like this
# what if i want to tweak something???
for path in paths:
    print("path", len(path))
    temp_df = copy.deepcopy(template)
    
    temp_df['time'] = path[0]['time'].values
    temp_df['dt0_x'] = path[0]['x'].values
    temp_df['dt0_y'] = path[0]['y'].values
    temp_df['dt1_x'] = path[1]['x'].values
    temp_df['dt1_y'] = path[1]['y'].values
    temp_df['dt2_x'] = path[2]['x'].values
    temp_df['dt2_y'] = path[2]['y'].values
    temp_df['dt3_x'] = path[3]['x'].values
    temp_df['dt3_y'] = path[3]['y'].values
    # temp_df['dt4_x'] = path[4]['x'].values
    # temp_df['dt4_y'] = path[4]['y'].values

    temp_df['av_x'] = temp_df[['dt0_x', 'dt1_x', 'dt2_x', 'dt3_x', 'dt4_x']].mean(axis=1)
    temp_df['av_y'] = temp_df[['dt0_y', 'dt1_y', 'dt2_y', 'dt3_y', 'dt4_y']].mean(axis=1)
    final_paths.append(temp_df)
    print(final_paths, "FINAL paths")

for i, path in enumerate(final_paths):
    path.to_csv(f"av_map//w//timewarp//create_paths_test{i}.csv")