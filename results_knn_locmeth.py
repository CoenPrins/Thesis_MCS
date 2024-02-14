import interpolate
import pandas as pd
import time_og_data2

""""
I think this just creates the GIANT LATEX file 

Additionally I need the input here changed. 

That is actually very minor. the only thing you need is the 
definitive changed OG paths, so OG1_window instead of yadayada. 


Also for plotting you ofcourse run it with the correct time, 
so that needs to be saved somewhere. 

"""
og_path1 = time_og_data2.og_path1
og_path2 = time_og_data2.og_path2
og_path3 = time_og_data2.og_path3


path_time_ls = [[time_og_data2.start_time1, time_og_data2.end_time1], [time_og_data2.start_time2, time_og_data2.end_time2], [time_og_data2.start_time3, time_og_data2.end_time3]]
path_names = ["path 1", "path 2", "path 3"]
og_paths = [og_path1, og_path2, og_path3]


# dt_ls = [5, 10, 15, 20]


dt_ls = [10]
radio_rssi_path_ls= ["lp"]
mis_val_methods_ls = ["penalty", "ignore"]
knn_N_ls  = [3, 6, 12]
loc_methods_ls = ["average", "gaus_normal", "gaus_80", "gaus_top"]
# path = "data\\paths\\dt_10\\raw\\lp\\penalty\\6\\gaus_normal\\path"

for loc_method in loc_methods_ls:
    df = pd.DataFrame(data=None, columns= ["loc_method","Knn", "mis_val_method", "path1", "path2", "path3" ])
    loc_methods_df_ls = []
    knn_n_df_ls = []
    mis_val_mtd_df_ls = []
    path_1_df_ls = []
    path_2_df_ls = []
    path_3_df_ls = []
    for dt in dt_ls:
        for mis_val_mtd in mis_val_methods_ls:
            for radio_rssi in radio_rssi_path_ls:
                for knn in knn_N_ls:


                    path = f"data\\paths\\dt_{dt}\\raw\\{radio_rssi}\\{mis_val_mtd}\\{knn}\\{loc_method}\\path"
                    
                    av_df = interpolate.create_av_df(path)

            
                    for i in range(len(path_time_ls)):
                        # print(type(path_time_ls[i][1]))
                        path_n_df = av_df[(av_df['time'] > path_time_ls[i][0]) & (av_df['time'] < path_time_ls[i][1])]
                        mean, std, sum = interpolate.accuracy(og_paths[i] ,path_n_df, path_time_ls[i], plot_toggle=False, both_toggle=True, lines_toggle=False, path_name= path_names[i])
                        mean, std = round(mean, 3), round(std, 3)
                        # print("mean=", mean, "std=", std, "sum=", sum)
                        if i ==0:
                            path_1_df_ls.append([mean, std])
                        if i == 1:
                            path_2_df_ls.append([mean, std])
                        if i == 2:
                            path_3_df_ls.append([mean, std])
                    loc_methods_df_ls.append(loc_method)
                    mis_val_mtd_df_ls.append(mis_val_mtd)
                    # print("knn", knn)
                    # print("knn_n_df_ls", knn_n_df_ls)
                    knn_n_df_ls.append(knn)
                    # print("knn_n_df_ls after append", knn_n_df_ls)

        #         break
        #     break
        # break
            


    df["loc_method"] = loc_methods_df_ls
    df["mis_val_method"] = mis_val_mtd_df_ls
    df["path1"] = path_1_df_ls
    df["path2"] = path_2_df_ls 
    df["path3"] = path_3_df_ls
    df['Knn'] = knn_n_df_ls
    # df.to_csv(f"data\\results\\{radio_rssi_path_ls[0]}_loc_mtd{loc_method}_dt{dt_ls[0]}.csv")

    df['path1'] = df['path1'].apply(lambda x: interpolate.format_latex(x))
    df['path2'] = df['path2'].apply(lambda x: interpolate.format_latex(x))
    df['path3'] = df['path3'].apply(lambda x: interpolate.format_latex(x))
    print(df.to_latex())