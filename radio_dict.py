import pandas as pd 
from datetime import datetime
import numpy as np
import csv
import pickle 


# import required module
import os
# assign directory


def convert_to_dict(directory):
    # iterate over files in
    # that directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            filename = filename[:-4]  
            df_radio = pd.read_csv(f)

            keys= df_radio['key_point_num'].unique()
        
            radiodict = {}
            radiodict_sigma = {}
            

            # # reaaallly bad syntax here there is a build in panda function for this 
            for key in keys:
                value = {}
                value_sigma = {}
                # df_mac_loop = df_radio.loc[df_radio['key_point_num'] == key]
                df_mac_loop = df_radio[df_radio['key_point_num'] == key]
                df_mac_loop = df_mac_loop[["mac", "mean_rssi", "x_pos", "y_pos" ,"var_rssi"]]
                # absolute blashphemy 
                value["pos"] = (df_mac_loop.iloc[0]['x_pos'], df_mac_loop.iloc[0]['y_pos'])
                for i in range(len(df_mac_loop)):
                    # value = value? very clear syntax bro 
                    
                    value[df_mac_loop.iloc[i]['mac']] = df_mac_loop.iloc[i]['mean_rssi']
                    value_sigma[df_mac_loop.iloc[i]['mac']] = df_mac_loop.iloc[i]['var_rssi']

                
                # print("value i",  key)
                radiodict[key] = value
                radiodict_sigma[key] = value_sigma
            dict_directory  = "data\\radio_maps\\dict"
            with open(f'{dict_directory}\\{filename}.pkl', 'wb') as fp:
                pickle.dump(radiodict, fp)

            with open(f'{dict_directory}\\sigma_{filename}.pkl', 'wb') as fp:
                pickle.dump(radiodict_sigma, fp)
            print(f'dictionary {filename} saved successfully to file')

# directory = "data\\radio_maps\\csv"
# test = convert_to_dict(directory)