import pandas as pd 
from datetime import datetime
import numpy as np
import radio_dict
"""

"""
# df_rssi are the data collected during the online phase 
# df_radio contains the data of the radio map (offline phase)
df_rssi = pd.read_csv("data\\rssi\\original\\rssi_2.csv", index_col=0)
df_radio = pd.read_csv("data\\radio_maps\\original\\radio_map_2.csv", index_col=0)


df_rssi = df_rssi.drop(['Unnamed: 0', 'x', 'y', 'z', 'bat'], axis=1)

#set time to pandas datetime

df_rssi['time'] = pd.to_datetime(df_rssi['time'], format='mixed')

#create easily accesible file containing coordinates of key points for floor 2

k_points = df_radio.loc[df_radio['key_point_num'].unique()]
k_points = k_points[['x_pos','y_pos']]
k_points.to_csv("data//environment//kp_cor.csv")


floor_mac= ['50:d4:f7:b9:c0:c2', '50:d4:f7:b9:c4:44', '50:d4:f7:b9:b1:aa', '50:d4:f7:ff:ec:e0', '50:d4:f7:b9:aa:cc', 
            '50:d4:f7:b9:a6:de', '50:d4:f7:ff:df:56', '50:d4:f7:b9:b1:c0', '50:d4:f7:b9:c4:dc']

mac_rssi = df_rssi['mac'].unique()
mac_radio = df_radio['mac'].unique()
mac_radio_n = df_radio['mac_num'].unique()
btmac_rssi = df_rssi['btmac'].unique()

print("unique_mac_radio", mac_radio)
print("unique mac rssi", mac_rssi)


# exclude non overlapping mac numbers from dataset. 

mac_lap = set(mac_rssi) & set(mac_radio)
df_radio_lp = df_radio.loc[df_radio['mac'].isin(mac_lap)]
df_rssi_lp = df_rssi.loc[df_rssi['mac'].isin(mac_lap)]


print("unique_mac_radio all floors",df_rssi_lp['mac'].unique())
print("unique mac rssi all floors",df_radio_lp['mac'].unique())

df_radio_9= df_radio[df_radio['mac'].isin(['50:d4:f7:b9:c0:c2', '50:d4:f7:b9:c4:44', '50:d4:f7:b9:b1:aa', '50:d4:f7:ff:ec:e0', '50:d4:f7:b9:aa:cc', 
            '50:d4:f7:b9:a6:de', '50:d4:f7:ff:df:56', '50:d4:f7:b9:b1:c0', '50:d4:f7:b9:c4:dc'])]

mac_radio9 = df_radio_9['mac'].unique()

print("length mac_radio9", mac_radio9)

df_radio_lp2 = df_radio_lp[df_radio_lp['mac'].isin(['50:d4:f7:b9:c0:c2', '50:d4:f7:b9:c4:44', '50:d4:f7:b9:b1:aa', '50:d4:f7:ff:ec:e0', '50:d4:f7:b9:aa:cc', 
            '50:d4:f7:b9:a6:de', '50:d4:f7:ff:df:56', '50:d4:f7:b9:b1:c0', '50:d4:f7:b9:c4:dc'])]

df_rssi_lp2 = df_rssi_lp[df_rssi_lp['mac'].isin(['50:d4:f7:b9:c0:c2', '50:d4:f7:b9:c4:44', '50:d4:f7:b9:b1:aa', '50:d4:f7:ff:ec:e0', '50:d4:f7:b9:aa:cc', 
            '50:d4:f7:b9:a6:de', '50:d4:f7:ff:df:56', '50:d4:f7:b9:b1:c0', '50:d4:f7:b9:c4:dc'])]
# print()

print(df_radio_lp2)
print("unique_mac_radio floor 2 ",df_rssi_lp2['mac'].unique())
print("unique mac rssi floor 2",df_radio_lp2['mac'].unique())







df_radio_lp2_n100 = df_radio_lp2.loc[df_radio_lp['mean_rssi'] != -100]
df_rssi_lp2_n100 = df_rssi_lp2.loc[df_rssi_lp['rssi'] != -100]

# # exclude -85

df_radio_lp2_n85 = df_radio_lp2.loc[df_radio_lp['mean_rssi'] > -85]
df_rssi_lp2_n85 = df_rssi_lp2.loc[df_rssi_lp['rssi'] > -85]






print("len df", len(df_rssi))
print("len radio", len(df_radio))
print("unique mac rssi values", len(df_rssi['mac'].unique()))
print("unique mac radio values", len(df_radio['mac'].unique()))

print()

print("len dflap", len(df_rssi_lp))
print("len radiolap", len(df_radio_lp))
print("unique mac rssi values", len(df_rssi_lp['mac'].unique()))
print("unique mac radio values", len(df_radio_lp['mac'].unique()))

print()

df_rssi_lp.to_csv("data\\rssi\\lp.csv")
df_radio_lp.to_csv("data\\radio_maps\\csv\\lp.csv")
print("len dflap2", len(df_rssi_lp2))
print("len radiolap2", len(df_radio_lp2))
print("unique mac rssi values", len(df_rssi_lp2['mac'].unique()))
print("unique mac radio  values", len(df_radio_lp2['mac'].unique()))
print()

df_rssi_lp2.to_csv("data\\rssi\\lp2.csv")
df_radio_lp2.to_csv("data\\radio_maps\\csv\\lp2.csv")

print("no -100")
print("len df", len(df_rssi_lp2_n100))
print("len radio", len(df_radio_lp2_n100))
print("unique mac rssi values", len(df_rssi_lp2_n100['mac'].unique()))
print("unique mac radio values", len(df_radio_lp2_n100['mac'].unique()))
print()

df_rssi_lp2_n100.to_csv("data\\rssi\\lp2_n100.csv")
df_radio_lp2_n100.to_csv("data\\radio_maps\\csv\\lp_n100.csv")


print("no -85")
print("len df", len(df_rssi_lp2_n85))
print("len radio", len(df_radio_lp2_n85))
print("unique mac rssi values", len(df_rssi_lp2_n85['mac'].unique()))
print("unique mac radio values", len(df_radio_lp2_n85['mac'].unique()))
print()


df_rssi_lp2_n85.to_csv("data\\rssi\\lp2_n85.csv")
df_radio_lp2_n85.to_csv("data\\radio_maps\\csv\\lp2_n85.csv")


# only radio map for maybe creating a better radiomap or someh 



df_radio_9.to_csv("data\\radio_maps\\csv\\lp9_AP.csv")

# create a dictionary version of all the files directory
# dict files get placed in "data\\radio_maps\\dict"

directory = "data\\radio_maps\\csv"
test = radio_dict.convert_to_dict(directory)