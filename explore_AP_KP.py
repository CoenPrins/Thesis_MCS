import pandas as pd 
import numpy as np 
import pickle
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib



"""
This file produces a table containing the hospital template for plotting the hospital floor

ap analysis produces a table of AP properties 

plot all coverage plots the coverage of all APs at the same time 

plot specific ap plots the coverage of AP points seperately 


ap point numbers plots the ap points locations + their numbers

"""

ap_loc9  = np.load('C:\\Users\\coen_\\OneDrive\\Documents\\ma_thesis\\files\\iBeacon_data\\summer_2_floor_04_08\\points_wifi_2.npy')
ap_loc8= ap_loc9
ap_loc8 = np.delete(ap_loc8, 7, axis=0)
data_name8 = "lp2"
data_name9 = 'lp9_AP'

# # define unique AP's of dataset
df_AP8 = pd.read_csv(f'data\\radio_maps\\csv\\{data_name8}.csv', index_col=0)['mac'].unique()
df_AP9= pd.read_csv(f'data\\radio_maps\\csv\\{data_name9}.csv', index_col=0)['mac'].unique()




# hardcoded ordered AP 
df_AP9 = ['50:d4:f7:b9:c0:c2', '50:d4:f7:b9:c4:44', '50:d4:f7:b9:b1:aa', '50:d4:f7:ff:ec:e0', 
             '50:d4:f7:b9:aa:cc', '50:d4:f7:b9:a6:de', '50:d4:f7:ff:df:56', '50:d4:f7:b9:b1:c0', '50:d4:f7:b9:c4:dc']


# # hardcoded ordered AP ls
# df_AP8 = ['50:d4:f7:b9:c0:c2', '50:d4:f7:b9:c4:44', '50:d4:f7:b9:b1:aa', '50:d4:f7:ff:ec:e0', 
#              '50:d4:f7:b9:aa:cc', '50:d4:f7:b9:a6:de', '50:d4:f7:ff:df:56', '50:d4:f7:b9:c4:dc']


# print("df_AP", df_AP)


with open(f"data\\radio_maps\\dict\\{data_name8}.pkl", 'rb') as fp: 
    radiodict8= pickle.load(fp)


with open(f"data\\radio_maps\\dict\\sigma_{data_name8}.pkl", 'rb') as fp: 
    radio_s_dict8= pickle.load(fp)



with open(f"data\\radio_maps\\dict\\{data_name9}.pkl", 'rb') as fp: 
    radiodict9= pickle.load(fp)


with open(f"data\\radio_maps\\dict\\sigma_{data_name9}.pkl", 'rb') as fp: 
    radio_s_dict9= pickle.load(fp)

# print(radiodict)

# keys are the same for radio_dict and radio_s_dict
keys = radiodict8.keys()

# print(keys)

"""
For plotting 
"""
# image of blueprint hospital 

img_hospital = plt.imread("data\\environment\\map_2floor_bw.png") 
# you have to substract these from positions to make it work with image 

hard_dx = 235
hard_dy = 76

# ratio points to meters 
conversion_factor = 0.13



# small conversion function for converting x/ylables from points to meters 
def format_meters(value, _):
    meters = value * conversion_factor
    return f'{meters:.2f} m'


def hospital_img_template(img, ap_loc, radiodict, ap_toggle, finger_toggle):

    keys = radiodict.keys()
    fig, ax = plt.subplots()
    meters_conversion = 0.13333
    
    # 
    if ap_toggle:
        for ap in ap_loc:
            ax.plot(ap[0] - hard_dx, ap[1] - hard_dy, '*', color = 'white', mec = 'black', markersize = 3.7**2 )


    if finger_toggle:
        all_pos_x = []
        all_pos_y = []
    
        for key in keys:
            all_pos_x.append(radiodict[key]['pos'][0] - hard_dx)
            all_pos_y.append(radiodict[key]['pos'][1] - hard_dy)
            ax.plot(all_pos_x, all_pos_y, ".", color="blue", markersize = 3.3**2)
        
    ax.imshow(img)

    """
    you can switch the uncomment for tighter plots
    """

    # ax.set_xticks([])
    # ax.set_yticks([])

    ax.xaxis.set_major_formatter(FuncFormatter(format_meters))
    ax.yaxis.set_major_formatter(FuncFormatter(format_meters))
    # ax.set_xticks([10, 20, 30], fontsize=12)
    

    """
        Uncomment this maybe?! 
    """
  
    # ax.grid("on")

    return fig, ax

# researching 3 criteria 
# distinct values 
# standard deviation
# map coverage 
# All printed in a latex format in terminal and saved to pandas csv
def ap_analysis(df, keys, radiodict, namestring):
 
    mac_df = pd.DataFrame(data=None, columns= ['mac number', "nonzero entries", "distinct values", "key point coverage %", "standard deviation"])



    for mac in df:
        print()
        print()
        print(mac)
       
    

        count = 0
        rssi_ls = []
        rssi_set = set()
        for key in keys:
            mac_dict = radiodict[key]
            # print(test)       
            rssi = mac_dict[mac]
            if rssi > -100:
                count += 1
                rssi_ls.append(rssi)
                rssi_set.add(int(rssi))
        new_row = {"mac number": mac, "nonzero entries": count, "distinct values":len(rssi_set), "key point coverage %": count/len(keys)*100, "standard deviation":np.std(rssi_ls)}
        mac_df = mac_df._append(new_row, ignore_index=True)
    mac_df.to_csv(f"data\\environment\\{namestring}.csv")
    print(mac_df.to_latex())



# plots the coverage of all APs at the same time 
def plot_all_coverage(keys, dict, img, ap_loc, ap_ls, data_name, num_colors):

    # Get the "hot" colormap
    cmap = matplotlib.colormaps.get_cmap("Reds")

    # Generate evenly spaced values from 0 to 1 to sample the colormap
    color_values = np.linspace(0.3, 1, num_colors)

    # Get the colors from the colormap
    colors = cmap(color_values)

    key_mac_n = np.zeros(len(keys))
    for i in range(len(keys)):
        for item in dict[i].items():
            if item[0] in ap_ls:
                # print("item", item)
                
                if isinstance(item[1], np.floating):
                    
                    if item[1] > -100:
                        key_mac_n[i] += 1 
               

    coverage_val_range = num_colors
    coverage_ls = [[[], []] for _ in range(coverage_val_range)]

    print("key_mac_n", key_mac_n)

    # Re arrange data structure for matplotlib application
    # Apply dy dx modifiers for plotting

    # index of key_mac_n corresponds with keypoin key value in 
    #radio dictionary
    max_val = 0
    for i in range(len(key_mac_n)):
        
        temp_pos = dict[i]["pos"]
        coverage_lvl = int(key_mac_n[i])
        if coverage_lvl > max_val:
            max_val = coverage_lvl
        # print(temp_pos, "  ", coverage_lvl)
        coverage_ls[coverage_lvl -1][0].append(temp_pos[0] - hard_dx)
        coverage_ls[coverage_lvl -1][1].append(temp_pos[1] - hard_dy)
        
    print("coverage_ls", coverage_ls)
    print(len(coverage_ls))
    print("max_val=", max_val)


    # Function to format the tick labels to display meters
    fig, ax = hospital_img_template(img, ap_loc, dict, ap_toggle=True, finger_toggle=False)
    
    # colors = ['grey', 'olive', 'green', 'cyan', 'blue', 'purple', 'pink', 'yellow', 'orange', 'red']
    # colors = ["red", "orange", "yellow", "pink", "purple", "blue", "cyan", "green", "olive", "grey"]
    for i in range(max_val):
        ax.plot(coverage_ls[i][0], coverage_ls[i][1], '.', color=colors[i], markersize = 3.3**2, label= i +1)

    
    # ax.set_title("AP coverage of Key Points", weight='bold', fontsize=18)
    legend_properties = {'weight':'bold'}
    ax.legend(title="Number of AP signals", fontsize = 25, prop=legend_properties)

    # plt.savefig("data\\figures\\ALL_coverage.png")
    # plt.savefig(f"data\\figures\\ap_spread\\full_spread_{data_name}.pdf", bbox_inches='tight', pad_inches=0.1)

    plt.tight_layout()
    plt.show()


# plot signal spread of specific APs 
def plot_specific_ap(df, radio_dict, img, ap_loc, data_name):



    cmap = matplotlib.colormaps.get_cmap("Reds")
    num_colors = 10

    # Generate evenly spaced values from 0 to 1 to sample the colormap
    color_values = np.linspace(0.1, 1, num_colors)

    # Get the colors from the colormap
    colors = cmap(color_values)
    colors = cmap(color_values)[::-1]
    
    for index, mac in enumerate(df):


        keys = radio_dict.keys()
        ap = [ap_loc[index]]
        fig_copy, ax_copy = hospital_img_template(img, ap, radio_dict, ap_toggle=True, finger_toggle=True)
        # ax_copy.plot(all_pos_x, all_pos_y, ".", color="gray", markersize = 3.3**2)
    

        labels = ['RSSI -55 -50', 'RSSI -60 -55', 'RSSI -65 -60', 'RSSI -70 -65', 'RSSI -75 -70', 'RSSI -80 -75', 'RSSI -85 -80', 'RSSI -90 -85', 'RSSI -95 -90', 'RSSI -100 -95']

        dot_color = colors[2]
        min_val_rssi = -100
        max_val_rssi = -50
        for key in keys:
            mac_dict = radio_dict[key]
            # print(test)       
            rssi = mac_dict[mac]
            if rssi > min_val_rssi:
                if rssi > max_val_rssi:
                    dot_color = colors[0]

                else: 
                    ratio = (abs(rssi - max_val_rssi))/(abs(min_val_rssi - max_val_rssi))
                    color_i = int(ratio*len(colors))
                    # print(ratio, color_i, "checkup")
                    dot_color = colors[color_i]
                    # dot_color = colors[0]
                    


                x_pos = mac_dict['pos'][0] - hard_dx
                y_pos = mac_dict['pos'][1] - hard_dy

                ax_copy.plot(x_pos, y_pos, ".", color=dot_color, markersize = 3.3**2)



        mac_pathname = mac 
        mac_pathname = mac_pathname.replace(":", "_") 

    
        plt.savefig(f"data\\figures\\ap_spread\\{mac_pathname}_AP_SPREAD_{data_name}.pdf", bbox_inches='tight', pad_inches=0.1)


        plt.close(fig_copy)  # Close the main figure
        
        # Save the legend separately

        if index == 0:
            legend_fig, legend_ax = plt.subplots()
            legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]
            legend_ax.legend(handles=legend_patches, loc='upper left')
            legend_ax.axis('off')  # Hide the axes of the legend figure
            legend_fig.savefig(f"data\\figures\\Legend_{mac_pathname}.pdf", bbox_inches='tight', pad_inches=0.1)
            plt.close(legend_fig)  # Close the legend fig
   


# plots the Ap points with numbers 
def ap_points_number(img_hospital, ap_loc, radiodict):
    fig, ax = hospital_img_template(img_hospital, ap_loc, radiodict, ap_toggle=True, finger_toggle=False)


    
    ax.text(ap_loc[0][0] - hard_dx, ap_loc[0][1] - hard_dy, "1", fontsize=16, weight='bold', color='red', ha='right', va='top')
    ax.text(ap_loc[1][0] - hard_dx, ap_loc[1][1] - hard_dy, "2", fontsize=16, weight='bold', color='red', ha='right', va='top')
    ax.text(ap_loc[2][0] - hard_dx, ap_loc[2][1] - hard_dy, "3", fontsize=16, weight='bold', color='red', ha='right', va='top')
    ax.text(ap_loc[3][0] - hard_dx, ap_loc[3][1] - hard_dy, "4", fontsize=16, weight='bold', color='red', ha='right', va='top')
    ax.text(ap_loc[4][0] - hard_dx, ap_loc[4][1] - hard_dy, "5", fontsize=16, weight='bold', color='red', ha='right', va='top')
    ax.text(ap_loc[5][0] - hard_dx, ap_loc[5][1] - hard_dy, "6", fontsize=16, weight='bold', color='red', ha='right', va='top')
    ax.text(ap_loc[6][0] - hard_dx, ap_loc[6][1] - hard_dy, "7", fontsize=16, weight='bold', color='red', ha='right', va='top')
    ax.text(ap_loc[7][0] - hard_dx, ap_loc[7][1] - hard_dy, "8", fontsize=16, weight='bold', color='red', ha='right', va='top')
    ax.text(ap_loc[8][0] - hard_dx, ap_loc[8][1] - hard_dy, "9", fontsize=16, weight='bold', color='red', ha='right', va='top')

    
    plt.savefig(f"data\\figures\\ap_spread\\numbers.pdf", bbox_inches='tight', pad_inches=0.1)
    plt.show()


def kp_points_number(img_hospital, keys, radiodict, ap_loc):
    fig, ax = hospital_img_template(img_hospital, ap_loc, radiodict, ap_toggle=False, finger_toggle=True)


    for key in keys:

        x, y = radiodict[key]['pos']
        print("xy", x, y)
        ax.text(x - hard_dx, y - hard_dy, key, fontsize=8, weight='bold', color='red', ha='right', va='top')
    # ax.text(ap_loc[1][0] - hard_dx, ap_loc[1][1] - hard_dy, "2", fontsize=16, weight='bold', color='red', ha='right', va='top')
    # ax.text(ap_loc[2][0] - hard_dx, ap_loc[2][1] - hard_dy, "3", fontsize=16, weight='bold', color='red', ha='right', va='top')
    # ax.text(ap_loc[3][0] - hard_dx, ap_loc[3][1] - hard_dy, "4", fontsize=16, weight='bold', color='red', ha='right', va='top')
    # ax.text(ap_loc[4][0] - hard_dx, ap_loc[4][1] - hard_dy, "5", fontsize=16, weight='bold', color='red', ha='right', va='top')
    # ax.text(ap_loc[5][0] - hard_dx, ap_loc[5][1] - hard_dy, "6", fontsize=16, weight='bold', color='red', ha='right', va='top')
    # ax.text(ap_loc[6][0] - hard_dx, ap_loc[6][1] - hard_dy, "7", fontsize=16, weight='bold', color='red', ha='right', va='top')
    # ax.text(ap_loc[7][0] - hard_dx, ap_loc[7][1] - hard_dy, "8", fontsize=16, weight='bold', color='red', ha='right', va='top')
    # ax.text(ap_loc[8][0] - hard_dx, ap_loc[8][1] - hard_dy, "9", fontsize=16, weight='bold', color='red', ha='right', va='top')

    
    plt.savefig(f"data\\figures\\key_sim\\numbers.pdf", bbox_inches='tight', pad_inches=0.1)
    plt.show()




# test1 = ap_analysis(df_AP9, keys, radiodict9, "testname")


    

# test2_8 = plot_all_coverage(keys, radiodict8, img_hospital, ap_loc8, df_AP8, data_name8, num_colors = 7)
# test2_9 = plot_all_coverage(keys, radiodict9, img_hospital, ap_loc9, df_AP9, data_name9, num_colors = 7)

# test3 = plot_specific_ap(df_AP9, radiodict9, img_hospital, ap_loc9, data_name9)



# test4 = ap_points_number(img_hospital, ap_loc9, radiodict9)
        
# test5 = kp_points_number(img_hospital, keys, radiodict9, ap_loc9)
    


# test, ax = hospital_img_template(img_hospital, ap_loc9, radiodict9, ap_toggle= True, finger_toggle = True)
# plt.tight_layout()
# plt.savefig('data\\figures\\environment_hospital.pdf', bbox_inches='tight')
# # plt.savefig(data)
# plt.show()