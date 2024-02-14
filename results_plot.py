import interpolate
import pandas as pd
import time_og_data2
import kalman_filter
import matplotlib.pyplot as plt
import numpy as np


"""
Incorporate the 

"""

og_path1 = time_og_data2.og_path1
og_path2 = time_og_data2.og_path2
og_path3 = time_og_data2.og_path3

og_paths = [og_path1, og_path2, og_path3]

# these are under scrutiny! 

n_og_path1 = time_og_data2.og_path1
n_og_path2 = time_og_data2.og_path2
n_og_path3 = time_og_data2.og_path3

n_og_paths = [n_og_path1, n_og_path2, n_og_path3]

path_time_ls = [[time_og_data2.start_time1, time_og_data2.end_time1], [time_og_data2.start_time2, time_og_data2.end_time2], [time_og_data2.start_time3, time_og_data2.end_time3]]

n_path_time_ls= [[time_og_data2.n_start_time1, time_og_data2.n_end_time1], [time_og_data2.n_start_time2, time_og_data2.n_end_time2], [time_og_data2.n_start_time3, time_og_data2.n_end_time3]]

path_names = ["path 1", "path 2", "path 3"]


# dt_ls = [5, 10, 15, 20]

# radio_rssi_path_ls= ["lp", "lp_n85", "lp_n100"]
# mis_val_mtd_ls = ["penalty", "ignore"]
# knn_N = 6
# loc_method = "gaus_normal"

# change these settings to your best solutions. 




# p2 10s ignore 6 gaus 80

# appenxid 

# In tables and data you give the top per dt, the top per penalty, 

# This is not so difficult if you have 


# Okay you just need to run this one twice for the short AND long sessions. 


# put them in a new map



# then you run it once in a new map for all the 



dt_ls = [10]
radio_rssi = ["lp2"]
mis_val_mtd = ["ignore"]
knn_N = 6
loc_method = "gaus_80"
# path = "data\\paths\\dt_10\\filter\\lp2\\ignore\\6\\gaus_80\\env_false_path"
path_ls = ["env_false_path", "env_semi_path", "env_true_path"]
path_ls = ["env_false_path"]
# path_ls = ["path"]
plotfigs = True

for radio_rssi in radio_rssi:
    df = pd.DataFrame(data=None, columns= ["dt", "mis_val_method", "path1", "path2", "path3" ])
    dt_df_ls = []
    mis_val_mtd_df_ls = []
    path_1_df_ls = []
    path_2_df_ls = []
    path_3_df_ls = []
    for dt in dt_ls:
        for path_ad in path_ls:

            path = f"data\\paths\\dt_{dt_ls[0]}\\filter\{radio_rssi}\\{mis_val_mtd[0]}\\{knn_N}\\{loc_method}\\"
            print("path_ad=", path_ad)
            path_file = path + path_ad
            
            av_ls, best_ls, av_i_error, av_er_t_ls, b_er_t_ls = interpolate.accuracy(n_og_paths ,path_file, path_time_ls, plt_paths =plotfigs, av_plt=True, best_plt=False, lines_plt=False, filter = None, filename=f"best_sol\\2by2\\filter_{path_ad}")
            # av_ls, best_ls, av_i_error, av_er_t_ls, b_er_t_ls = interpolate.accuracy(og_paths ,path_file, n_path_time_ls, plt_paths =plotfigs, av_plt=True, best_plt=False, lines_plt=True, filter = 'PF', filename=f"best_sol\\best_filter_solution\\short_{path_ad}")

            # av_ls, best_ls, av_i_error, av_er_t_ls, b_er_t_ls = interpolate.accuracy(n_og_paths ,path, path_time_ls, plt_paths =plotfigs, av_plt=False, best_plt=True, lines_plt=False, filter = None)


            colors_av = ['red', 'blue', 'green']
            colors_b = ['pink', 'lightblue', 'lightgreen']
            # for index, path_error in enumerate(av_er_t_ls):

            #     t = time_og_data2.generate_datetime_dataset(path_time_ls[index][0], path_time_ls[index][1], len(path_error), small_strings=True)
         
            #     plt.plot(t, path_error, color=colors[index])
            #     plt.show()

                        
            for i in range(len(av_er_t_ls)):
                len_av_er = len(av_er_t_ls[i])
                len_b_er = len(b_er_t_ls[i])
                

                # print(len_av_er)
                # print(len_b_er)

                av_t_ls = time_og_data2.generate_datetime_dataset(path_time_ls[i][0], path_time_ls[i][1], len_av_er, small_strings=False)
                # b_t_ls = time_og_data2.generate_datetime_dataset(path_time_ls[i][0], path_time_ls[i][1], len_b_er, small_strings=False)
                # print("length t list", len(t_ls))

                plt.plot(av_t_ls, av_er_t_ls[i], color=colors_av[i])
                # plt.plot(b_t_ls, b_er_t_ls[i], color=colors_b[i], label="Error Best path")
                # plt.legend(fontsize=18)
                plt.xlabel('Time', fontsize= 18)
                plt.ylabel('Error', fontsize=18)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.tight_layout()
                path_n = i + 1
                # plt.savefig(f"data\\figures\\best_sol\\best_filter_solution\\error{path_n}_{path_ad}.pdf")
                plt.show()

           
            # av_ls, best_ls, av_i_error, av_er_t_ls, b_er_t_ls = interpolate.accuracy(og_paths ,path, n_path_time_ls, plt_paths =plotfigs, av_plt=True, best_plt=False, lines_plt=True, filter = None, filename="best_sol\\short ")



        

            # print(av_er_t_ls)
            

            
            # plt.plot(av_er_t_ls,)
            # plt.show()

    # df["dt"] = dt_df_ls
    # df["mis_val_method"] = mis_val_mtd_df_ls
    # df["path1"] = path_1_df_ls
    # df["path2"] = path_2_df_ls 
    # df["path3"] = path_3_df_ls
    # df.to_csv(f"data\\results\\{radio_rssi}_{knn_N}_{loc_method}.csv")

    # df['path1'] = df['path1'].apply(lambda x: interpolate.format_latex(x))
    # df['path2'] = df['path2'].apply(lambda x: interpolate.format_latex(x))
    # df['path3'] = df['path3'].apply(lambda x: interpolate.format_latex(x))
    # print(df.to_latex())