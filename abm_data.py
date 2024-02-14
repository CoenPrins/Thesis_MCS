import physical_agent
import pandas as pd
import time_og_data2
import time 




# print(df)
def abm_data(df, dt, obj_env, part_env):
    paths = ['path_1', "path_2", "path_3"]
    og_toggle = False
    RSSI_toggle = True
    PF_N = 1000
    error = 5/0.1333

    wp_path = "data\\distance\\waypoints.pkl"
    dist_path = 'data\\distance\\dist_ls.pkl'

    df['time'] = pd.to_datetime(df['time'], format='mixed')
    df['x'] =  df['x'].round().astype(int)
    df['y'] =  df['y'].round().astype(int)
    df = df.drop(columns=['ax', 'ay', 'bx', 'by', 'cx', 'cy'])
    # print("this is df", df)

    
    
    df_ls = []
    for path_n in paths:

        if path_n == "path_1":
            df_c = df[((df['time'] >= time_og_data2.start_time1) & (df['time'] <= time_og_data2.end_time1))].copy()
            df_l =  len(df[((df['time'] >= time_og_data2.start_time1) & (df['time'] <= time_og_data2.end_time1))])
        if path_n == "path_2":
            df_c = df[((df['time'] >= time_og_data2.start_time2) & (df['time'] <= time_og_data2.end_time2))].copy()
            df_l = len(df[((df['time'] >= time_og_data2.start_time2) & (df['time'] <= time_og_data2.end_time2))])
        if path_n == "path_3":
            df_c = df[((df['time'] >= time_og_data2.start_time3) & (df['time'] <= time_og_data2.end_time3))].copy()
            df_l = len(df[((df['time'] >= time_og_data2.start_time3) & (df['time'] <= time_og_data2.end_time3))].copy())


        n_steps = df_l - 1
        # n_steps = 10
        # print(len(df))
        # print("after time df", df)

        model = physical_agent.IndoorModel_fp(env_map_path = bg_path, RSSI_df = df_c, path = path_n, og_cords=og_toggle, RSSI=RSSI_toggle, PF = True, PF_N =PF_N, obj_env = obj_env, part_env = part_env, RSSI_error= error, dt = dt, wp_path = wp_path, dist_path = dist_path)

        model.run_model(n_steps)

        data_model = model.data_collector.get_model_vars_dataframe().to_numpy().flatten()
        data_agent = model.data_collector.get_agent_vars_dataframe()


        hard_dx = 235
        hard_dy = 76
        df_trans = data_agent.copy()
        df_trans['x'] = df_trans['x'] + hard_dx
        df_trans['y'] = df_trans['y'] + hard_dy
        df_trans['x_pf'] = df_trans['x_pf'] + hard_dx
        df_trans['y_pf'] = df_trans['y_pf'] + hard_dy
        df_trans['time'] = df_c['time'].values
        df_trans['av_time'] = df_c['av_time'].values
        df_ls.append(df_trans)
        # print()
        # print('df_translen', len(df_trans))
        # print()

    result_df = pd.concat(df_ls, ignore_index=True)



    result_df = result_df.sort_values(by='time')

    # print("this is the result_df", result_df)

    # elapsed_time = end_time - start_time
    # print("elapsed time =", elapsed_time)
    return(result_df)




bg_path = 'data\\environment\\map_2floor_PF_bw.png'



dt_ls = [3, 5, 10]
data_ls = ["lp2"]
mis_val_ls = ["ignore"]
knn_n_ls = [6]
selection_ls = ["gaus_80"]

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns


start_time = time.time()
for dt in dt_ls:    
    for data in data_ls:
        for mis_val in mis_val_ls:
            for knn_n in knn_n_ls:
                for selection_mtd in selection_ls:
                        for i in range(0,5):
                            # print(i)
                            
                            file_path = f"data\\paths\\dt_{dt}\\raw\\{data}\\{mis_val}\\{knn_n}\\{selection_mtd}\\path{i}.csv"
                            df = pd.read_csv(file_path)
                            # result = abm_data(df, dt, environment = True)
                            # result.to_csv(f"data\\paths\\dt_{dt}\\filter\\{data}\\{mis_val}\\{knn_n}\\{selection_mtd}\\env_true_path{i}.csv")
                            # result = abm_data(df, dt, environment = False)
                            # result.to_csv(f"data\\paths\\dt_{dt}\\filter\\{data}\\{mis_val}\\{knn_n}\\{selection_mtd}\\env_false_path{i}.csv")
                            result = abm_data(df, dt, obj_env = True, part_env = False)
                            result.to_csv(f"data\\paths\\dt_{dt}\\filter\\{data}\\{mis_val}\\{knn_n}\\{selection_mtd}\\env_semi_path{i}.csv")

end_time = time.time()
print("elapsed time =", end_time - start_time)

# print(result)


# pd.set_option('display.max_rows', None)  # Show all rows
# pd.set_option('display.max_columns', None)  # Show all columns
# df = "data//paths//lp2//raw//6//gaus_normal//"
# dt = 5
# test = abm_data(df, dt)
# # file_path = "junk\\fake_og.csv"


# df = pd.read_csv(file_path)

# print(df)

# print(df['x'])
# dt = 5
# result = abm_data(df, dt)


#     datas = model.data_collector.get_model_vars_dataframe()
#     print(data_model)

#     okay so this is the data for the 
#     print(len(data_agent))
#     print("this is data agent", data_agent)
#     print("this is good data agent", df_trans)

# print(data_agent['x_pf'])
# result_df.to_csv("")

"""


for 500 particles...


Okay we will run it for 1000 particles (as is standard). 

You are gonna run all the other stuff, tonight and tomorrow and then run the particle filter for only the best settings (but for ALL time points)

those you will compare. 

One run (for dt f)

PARTICLE = 1000

(one run is 470 seconds ) = 8 minutes. 

So you just run all the time options for this during your break = (half an hour apr.)
Ask someone to supervise. 


PARTICLE = 500
dt = 5: elapsed time = 210.6834614276886 (for this time you could run apr 120 different combinations for the particle filter!)
dt = 10 : 160 seconds 
dt = 15 :  86.80309867858887
dt = 20 = 80 seconds

"""