import pandas
import interpolate 
import time_og_data2

dt = 10
radio_rssi = "lp2"
mis_val_mtd = "ignore"
knn_N = 6
loc_method = "gaus_80"


n_og_path1 = time_og_data2.og_path1
n_og_path2 = time_og_data2.og_path2
n_og_path3 = time_og_data2.og_path3

n_og_paths = [n_og_path1, n_og_path2, n_og_path3]


path_time_ls = [[time_og_data2.start_time1, time_og_data2.end_time1], [time_og_data2.start_time2, time_og_data2.end_time2], [time_og_data2.start_time3, time_og_data2.end_time3]]


path = f"data\\paths\\dt_{dt}\\raw\\{radio_rssi}\\{mis_val_mtd}\\{knn_N}\\{loc_method}\\path"

av_ls, best_ls, av_i_error, av_er_t_ls, b_er_t_ls = interpolate.accuracy(n_og_paths ,path, path_time_ls, plt_paths = True, av_plt=False, best_plt=False, lines_plt=False, filter = None, filename="best_sol\\longnoimport")
