import interpolate
import pandas as pd
import time_og_data2

og_paths = [time_og_data2.og_path1, time_og_data2.og_path2, time_og_data2.og_path3]
n_path_time_ls = [[time_og_data2.n_start_time1, time_og_data2.n_end_time1], [time_og_data2.n_start_time2, time_og_data2.n_end_time2], [time_og_data2.n_start_time3, time_og_data2.n_end_time3]]

short_path = [og_paths, n_path_time_ls]

n_og_paths = [time_og_data2.n_og_path1, time_og_data2.n_og_path2, time_og_data2.n_og_path3]
path_time_ls = [[time_og_data2.start_time1, time_og_data2.end_time1], [time_og_data2.start_time2, time_og_data2.end_time2], [time_og_data2.start_time3, time_og_data2.end_time3]]

long_path = [n_og_paths, path_time_ls]


both_path_ls = [short_path, long_path]
both_path_string_ls = ["short_path","long_path"]

plotfigs= False
# these you can edit depending on what you want!
dt_ls = [3, 5, 10]


radio_rssi_path_ls= ["lp2", "lp2_n85", "lp2_n100"]
mis_val_mtd_ls = ["penalty", "ignore"]


knn_N_ls = [1, 3, 6, 12]
loc_method_ls = ["average", "gaus_80", "gaus_normal", "gaus_top"]


df = pd.DataFrame(data=None, columns= None)
longshort_ls = []
radio_rssi_ls = []
dt_df_ls = []
mis_val_mtd_df_ls = []
knn_N_df_ls = []
loc_method_df_ls = []

path_1_df_av_ls = []
path_2_df_av_ls = []
path_3_df_av_ls = []

path_1_df_b_ls = []
path_2_df_b_ls = []
path_3_df_b_ls = []


path_1_df_av_i_ls = []
path_2_df_av_i_ls = []
path_3_df_av_i_ls = []

for path_len_index, path_len in enumerate(both_path_ls):
    for radio_rssi in radio_rssi_path_ls:
        for dt in dt_ls:
            for mis_val_mtd in mis_val_mtd_ls:
                for knn_N in knn_N_ls:
                    for loc_method in loc_method_ls:


                        path = f"data\\paths\\dt_{dt}\\raw\\{radio_rssi}\\{mis_val_mtd}\\{knn_N}\\{loc_method}\\path"
                        av_ls, best_ls, av_i_error, av_er_t_ls, b_er_t_ls = interpolate.accuracy(path_len[0] ,path, path_len[1], plt_paths =plotfigs, av_plt=False, best_plt=False, lines_plt=False, filter = None)
                        # print("av_ls", av_ls)
                        # print()
                        # print("best_ls", best_ls)
                        # print()
                        # print("av_i_error", av_i_error)
                        # print()
                        # print("av_er_t_ls", av_er_t_ls)
                        # print()
                        # print("b_er_t_ls", b_er_t_ls)
                        # print()
                
                        for i in range(len(av_ls)):
                            print("i=", i)
                            # print(type(path_time_ls[i][1]))
                        
                            # print("mean=", mean, "std=", std, "sum=", sum)


                            if i ==0:
                                path_1_df_av_ls.append([av_ls[i][0], av_ls[i][1]])
                                path_1_df_b_ls.append([best_ls[i][0], best_ls[i][1]])
                                path_1_df_av_i_ls.append([av_i_error[i]])
                            if i == 1:
                                path_2_df_av_ls.append([av_ls[i][0], av_ls[i][1]])
                                path_2_df_b_ls.append([best_ls[i][0], best_ls[i][1]])
                                path_2_df_av_i_ls.append([av_i_error[i]])
                            if i == 2:
                                path_3_df_av_ls.append([av_ls[i][0], av_ls[i][1]])
                                path_3_df_b_ls.append([best_ls[i][0], best_ls[i][1]])
                                path_3_df_av_i_ls.append([av_i_error[i]])

                        # # so we know whatsup
                        longshort_ls.append(both_path_string_ls[path_len_index])
                        radio_rssi_ls.append(radio_rssi)
                        dt_df_ls.append(f"{dt}s")
                        mis_val_mtd_df_ls.append(mis_val_mtd)
                        knn_N_df_ls.append(knn_N)
                        loc_method_df_ls.append(loc_method)


df["path_len"] = longshort_ls
df["threshold"] = radio_rssi_ls
df["dt"] = dt_df_ls
df["mis_val"] = mis_val_mtd_df_ls
df['knn'] = knn_N_df_ls
df["loc_method"] = loc_method_df_ls


path_av_av = []
path_b_av = []
path_av_i_av = []
for i in range(len(path_1_df_av_ls)):
    path_av_av.append((path_1_df_av_ls[i][0] + path_2_df_av_ls[i][0] + path_3_df_av_ls[i][0])/3)
    path_b_av.append((path_1_df_b_ls[i][0] + path_2_df_b_ls[i][0] + path_3_df_b_ls[i][0])/3)
    path_av_i_av.append((path_1_df_av_i_ls[i][0] + path_2_df_av_i_ls[i][0] + path_3_df_av_i_ls[i][0])/3)



df["path1_av"] = path_1_df_av_ls
df["path2_av"] = path_2_df_av_ls
df["path3_av"] = path_3_df_av_ls
df["path_av_av"] = path_av_av

df["path1_b"] = path_1_df_b_ls
df["path2_b"] = path_2_df_b_ls
df["path3_b"] = path_3_df_b_ls
df["path_b_av"] = path_b_av

df["path1_av_i"] = path_1_df_av_i_ls
df["path2_av_i"] = path_2_df_av_i_ls
df["path3_av_i"] = path_3_df_av_i_ls
df["path3_av_i_av"] = path_av_i_av  

df.to_csv(f"data\\results\\full_av_lp2.csv")

df['path1_av'] = df['path1_av'].apply(lambda x: interpolate.format_latex(x))
df['path2_av'] = df['path2_av'].apply(lambda x: interpolate.format_latex(x))
df['path3_av'] = df['path3_av'].apply(lambda x: interpolate.format_latex(x))


df['path1_b'] = df['path1_b'].apply(lambda x: interpolate.format_latex(x))
df['path2_b'] = df['path2_b'].apply(lambda x: interpolate.format_latex(x))
df['path3_b'] = df['path3_b'].apply(lambda x: interpolate.format_latex(x))

df['path1_av_i'] = df['path1_av_i'].apply(lambda x: interpolate.format_latex(x))
df['path2_av_i'] = df['path2_av_i'].apply(lambda x: interpolate.format_latex(x))
df['path3_av_i'] = df['path3_av_i'].apply(lambda x: interpolate.format_latex(x))

print(df.to_latex())