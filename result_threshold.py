import interpolate
import pandas as pd
import time_og_data2

og_path1 = time_og_data2.og_path1
og_path2 = time_og_data2.og_path2
og_path3 = time_og_data2.og_path3


path_time_ls = [[time_og_data2.start_time1, time_og_data2.end_time1], [time_og_data2.start_time2, time_og_data2.end_time2], [time_og_data2.start_time3, time_og_data2.end_time3]]
path_names = ["path 1", "path 2", "path 3"]
og_paths = [og_path1, og_path2, og_path3]


dt_ls = [3, 5, 10]


radio_rssi_path_ls= ["lp", "lp2", "lp2_n85", "lp2_n100"]
mis_val_mtd_ls = ["penalty", "ignore"]
knn_N = 6
loc_method = "gaus_normal"



# dt_ls = [10]


# radio_rssi_path_ls= ["lp2"]
# mis_val_mtd_ls = ["ignore"]
# knn_N = 6
# loc_method = "gaus_normal"




# path = "data\\paths\\dt_10\\raw\\lp\\penalty\\6\\gaus_normal\\path"

for radio_rssi in radio_rssi_path_ls:
    df_template = pd.DataFrame(data=None, columns= ["dt", "mis_val_method", "path1", "path2", "path3" ])
    dt_df_ls = []
    mis_val_mtd_df_ls = []
    path_1t_df_ls = []
    path_2t_df_ls = []
    path_3t_df_ls = []

    path_1g_df_ls = []
    path_2g_df_ls = []
    path_3g_df_ls = []
    for dt in dt_ls:
        for mis_val_mtd in mis_val_mtd_ls:


            path = f"data\\paths\\dt_{dt}\\raw\\{radio_rssi}\\{mis_val_mtd}\\{knn_N}\\{loc_method}\\path"
            
            # definte average df based on time and av general distance
            av_df = interpolate.create_av_df(path, PF=False, KF=False)

    
            for i in range(len(path_time_ls)):
                # print(type(path_time_ls[i][1]))
                path_n_df = av_df[(av_df['time'] >= path_time_ls[i][0]) & (av_df['time'] <= path_time_ls[i][1])]
                distances = interpolate.accuracy(og_paths[i] ,path_n_df, path_time_ls[i], plot_toggle=False, both_toggle=True, lines_toggle=False, filter = None)
                
                mean_t, std_t, sum_t =  distances[0]
                # mean_t, std_t = round(mean_t, 3), round(std_t, 3)
                mean_g, std_g, sum_g = distances[1]
                # mean_g, std_g = round(mean_g, 3), round(std_g, 3)
                
                    # print("mean=", mean, "std=", std, "sum=", sum)
                if i ==0:
                    path_1t_df_ls.append([mean_t, std_t])
                    path_1g_df_ls.append([mean_g, std_g])
                if i == 1:
                    path_2t_df_ls.append([mean_t, std_t])
                    path_2g_df_ls.append([mean_g, std_g])
                if i == 2:
                    path_3t_df_ls.append([mean_t, std_t])
                    path_3g_df_ls.append([mean_g, std_g])

            
            dt_df_ls.append(f"{dt}s")
            mis_val_mtd_df_ls.append(mis_val_mtd)

    df = df_template.copy()
    df["dt"] = dt_df_ls
    df["mis_val_method"] = mis_val_mtd_df_ls
    df["path1"] = path_1t_df_ls
    df["path2"] = path_2t_df_ls 
    df["path3"] = path_3t_df_ls
    # df.to_csv(f"data\\results\\{radio_rssi}_{knn_N}_{loc_method}_t.csv")

    df['path1'] = df['path1'].apply(lambda x: interpolate.format_latex(x))
    df['path2'] = df['path2'].apply(lambda x: interpolate.format_latex(x))
    df['path3'] = df['path3'].apply(lambda x: interpolate.format_latex(x))

    print(f"this is the print latex for dist = t")
    print(df.to_latex())
    print()

    df = df_template.copy()
    df["dt"] = dt_df_ls
    df["mis_val_method"] = mis_val_mtd_df_ls
    df["path1"] = path_1g_df_ls
    df["path2"] = path_2g_df_ls 
    df["path3"] = path_3g_df_ls
    # df.to_csv(f"data\\results\\{radio_rssi}_{knn_N}_{loc_method}_g.csv")

    df['path1'] = df['path1'].apply(lambda x: interpolate.format_latex(x))
    df['path2'] = df['path2'].apply(lambda x: interpolate.format_latex(x))
    df['path3'] = df['path3'].apply(lambda x: interpolate.format_latex(x))

    print(f"this is the print latex for dist = g")
    print(df.to_latex())
