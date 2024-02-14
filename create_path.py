
import pandas as pd 
from datetime import datetime
import numpy as np
import csv
import pickle 
import sys
import math


# tha mac numbers is wrong! for now. 
"""
implementation of mac ls might be wrong, i thinkg it should be ALL
not just the one in the file, but we will not see and assume this is good!
"""



def av_time_cal(df, time):
    df['time'] = pd.to_datetime(df['time'], format='mixed')
    df['meas_time']= pd.to_datetime(df['meas_time'], format='mixed')
    min_time = df['time'].min(skipna=True)
    mask = df['time'] == time

    filtered_df = df[mask]


    time_diffs = (filtered_df['meas_time'] - min_time).dt.total_seconds()

    mean_seconds = time_diffs.mean()
    av_time = min_time + pd.to_timedelta(mean_seconds, unit='s')

    return av_time


def gaus_p(x, mu, sigma):
    # calculate gaussian probabilty of two signals
   
    prob = (1/(sigma * math.sqrt(2 * math.pi)))*math.exp(-0.5*(((x - mu)/sigma)**2))
    return prob


def eucl(eucl_ls):
    # calculate euclidean distance between two signal patterns
    abs_val =0
    for dist in eucl_ls:
        abs_val += dist**2

    abs_val = math.sqrt(abs_val)
    return abs_val




def create_path(df_ls, knn_N, mis_val_method, loc_method, rssi_name, radio_name, dt_val):

    print()
    print()
    print()
    print(f"now running for dt= {dt_val}, path = {rssi_name},  mis_val ={mis_val_method}, knn= {knn_N}, loc method= {loc_method}")
    print()
    # radiodict = radiodict
    with open(f"data\\radio_maps\\dict\\{radio_name}.pkl", 'rb') as fp: 
        radiodict= pickle.load(fp)


    with open(f"data\\radio_maps\\dict\\sigma_{radio_name}.pkl", 'rb') as fp: 
        radio_s_dict= pickle.load(fp)


    keys = radiodict.keys()


    # go through df_ls (each element is btmac number)
    for i in range(len(df_ls)):
        # print("i =", i)
        df_path = df_ls[i]
        
        df_path['time'] = pd.to_datetime(df_path['time'], format='mixed')
        timeseries = df_path['time'].unique()
        mac_ls = df_path['mac'].unique()
        

        # initiate lists destined for df columns
        time_col = []
        av_time_col = []
        x_col = []
        y_col = []
        ax_col = []
        ay_col =[]
        bx_col = []
        by_col = []
        cx_col = []
        cy_col = []
       
        for time in timeseries:
            
            timedata = df_path[df_path['time'] == time]
        
        
            
            all_keys_eucl = []
            if mis_val_method == "penalty":          
                for key_n in keys:
                    key_abs_ls = []
                    dict_key = radiodict[key_n]

                    for mac in mac_ls:
                     
                        if mis_val_method == "penalty":
                            if mac in dict_key and mac in timedata['mac'].values:
                                abs_d = abs(dict_key[mac] - timedata[timedata['mac'] == mac]['rssi'].values[0])
                        
                                key_abs_ls.append(abs_d)
                                
                            elif mac in dict_key or mac in timedata['mac'].values:
                                min_val = -100
                                if mac in dict_key:
                                    abs_d_penalty = abs(dict_key[mac] - min_val)
                                   
                                    key_abs_ls.append(abs_d_penalty)
                                if mac in timedata['mac'].values:
                                    abs_d_penalty = abs(timedata[timedata['mac'] == mac]['rssi'].values[0] - min_val)

                                    
                                    key_abs_ls.append(abs_d_penalty)
                    
                    if len(key_abs_ls) > 0:
                        key_eucl = eucl(key_abs_ls)
                        if key_eucl >0:
                            all_keys_eucl.append((key_n, key_eucl))

                    


            elif mis_val_method == "ignore":
                for key_n in keys:
                    key_abs_ls = []
                    dict_key = radiodict[key_n]

                    for mac in mac_ls:
                
                            if mac in dict_key and mac in timedata['mac'].values:
                                abs_d = abs(dict_key[mac] - timedata[timedata['mac'] == mac]['rssi'].values[0])
                                key_abs_ls.append(abs_d)


                    # for ignore you take the average value while for penalty you don't
                    if len(key_abs_ls) > 0:
                        key_eucl = eucl(key_abs_ls)
                        key_eucl = key_eucl/len(key_abs_ls)
                        if key_eucl > 0:
                            all_keys_eucl.append((key_n, key_eucl))


            # Sort key points based on euclidean distance from small to big
            all_keys_eucl.sort(key=lambda x: x[1])

            # only keep the k nearest neighbors
            if len(all_keys_eucl) >= knn_N:
                all_keys_eucl = all_keys_eucl[:knn_N]


            else:
                print("ALARM NOT ENOUGH NEIGHBOURS")
                print("for time =", time)
                print('length=', len(all_keys_eucl))
            
            if loc_method == "average":

                
                pos_ls = []

                sum_pos = [0, 0] 
                sum_div = len(all_keys_eucl)
                for element in all_keys_eucl:
                    key_cor = radiodict[element[0]]["pos"]
                    sum_pos[0] += key_cor[0]
                    sum_pos[1] += key_cor[1]
                    pos_ls.append(key_cor)

                sum_pos = [sum_pos[0]/sum_div, sum_pos[1]/sum_div]
               
                pos_final = [sum_pos, pos_ls]
                


                time_col.append(time)
                av_time = av_time_cal(df_ls[i], time)
                av_time_col.append(av_time)
                x_col.append(sum_pos[0])
                y_col.append(sum_pos[1])
                if len(pos_ls) >= 3:
                    ax_col.append(pos_ls[0][0])
                    ay_col.append(pos_ls[0][1])
                    bx_col.append(pos_ls[1][0])
                    by_col.append(pos_ls[1][1])
                    cx_col.append(pos_ls[2][0])
                    cy_col.append(pos_ls[2][1])
                else:
                    ax_col.append(None)
                    ay_col.append(None)
                    bx_col.append(None)
                    by_col.append(None)
                    cx_col.append(None)
                    cy_col.append(None)



            if loc_method == "gaus_normal" or loc_method == "gaus_80" or loc_method == "gaus_top":
            
                total_prob = 0
                key_prob_ls = []

                # the difference in mac loop is because for k-NN we need to loop over 
                # all possible mac values to detect mismatches, here we only want to go
                # over the mac values
                time_mac = timedata['mac'].values


                # dit loopje moet enkel de key points in 
                for key in all_keys_eucl:
                    key_n = key[0]
                    dict_key = radiodict[key_n]
                    dict_key_sigma = radio_s_dict[key_n]
                    key_prob = 0
                    for mac in time_mac:
                        if mac in dict_key:
                            x = timedata[timedata['mac'] == mac]['rssi'].values[0]
                            mu = dict_key[mac]
                            sigma = dict_key_sigma[mac]

                            if sigma != 0:
                                prob = gaus_p(x, mu, sigma)
                                key_prob += prob

                    if key_prob > 0:
                        key_n_entry = [key_n, key_prob]
                
                        key_prob_ls.append(key_n_entry)
                        total_prob += key_prob
                    else: 
                        pass
            


                #normalising key probability
                for j in range(len(key_prob_ls)):

            
                    key_prob_ls[j][1] = key_prob_ls[j][1]/ total_prob
        
        
                
                key_prob_ls.sort(key=lambda x: x[1], reverse=True)
                
                if loc_method == "gaus_top":

                    
                    if len(key_prob_ls) >0:
                        top_k = key_prob_ls[0][0]
                        top_k_pos = radiodict[top_k]["pos"]
                        time_col.append(time)
                        av_time = av_time_cal(df_ls[i], time)
                        av_time_col.append(av_time)
                        x_col.append(top_k_pos[0])
                        y_col.append(top_k_pos[1])

                        ax_col.append(None)
                        ay_col.append(None)
                        bx_col.append(None)
                        by_col.append(None)
                        cx_col.append(None)
                        cy_col.append(None)

                if loc_method == "gaus_80":
                    
                    if len(key_prob_ls) >0:
                        top_80 = []
                        prob_sum = 0
                        
                        index = 0
                        while prob_sum < 0.80:
                            prob = key_prob_ls[index]
                            prob_sum += prob[1]
                            top_80.append(prob)

                            index += 1

                        #normalise so result is still 1
                        for prob_80 in top_80:
                            prob_80[1] = prob_80[1] /prob_sum

                        
                        
                        key_prob_ls = top_80


                if loc_method == "gaus_normal" or loc_method == "gaus_80":

                    sum_w_pos = [0, 0]

                    for element in key_prob_ls:
                        key_pos = radiodict[element[0]]["pos"]
                        sum_w_pos[0] += key_pos[0] * element[1]
                        sum_w_pos[1] += key_pos[1] * element[1]

                    if sum_w_pos[0] == 0 and sum_w_pos[1] == 0:
                        
                        print("key_prob_ls", key_prob_ls)

                    else:
                        
                        time_col.append(time)
                        av_time= av_time_cal(df_ls[i], time)
                        av_time_col.append(av_time)
                        x_col.append(sum_w_pos[0])
                        y_col.append(sum_w_pos[1])
                        
                        if len(key_prob_ls) >= 3:
                            top3 = key_prob_ls[:3]
                            n1_key= top3[0][0]
                            n1_pos = radiodict[n1_key]["pos"]
                            n2_key= top3[1][0]
                            n2_pos = radiodict[n2_key]["pos"]

                            n3_key= top3[2][0]
                            n3_pos = radiodict[n3_key]["pos"]

                            ax_col.append(n1_pos[0])
                            ay_col.append(n1_pos[1])
                            bx_col.append(n2_pos[0])
                            by_col.append(n2_pos[1])
                            cx_col.append(n3_pos[0])
                            cy_col.append(n3_pos[1])
                        else:
                            ax_col.append(None)
                            ay_col.append(None)
                            bx_col.append(None)
                            by_col.append(None)
                            cx_col.append(None)
                            cy_col.append(None)





        coords = pd.DataFrame(data=None, columns= ['time', 'av_time', 'x', 'y', 'ax', 'ay', 'bx', 'by', 'cx', 'cy'])
        coords['time'] = time_col
        coords['x'] = x_col
        coords['y'] = y_col 
        coords['ax'] = ax_col
        coords['ay'] = ay_col
        coords['bx'] = bx_col
        coords['by'] = by_col
        coords['cx'] = cx_col
        coords['cy'] = cy_col
        
        coords['av_time'] = av_time_col
        
        coords.to_csv(f"data\\paths\\dt_{dt_val}\\raw\\{rssi_name}\\{mis_val_method}\\{knn_N}\\{loc_method}\\path{i}.csv")



 