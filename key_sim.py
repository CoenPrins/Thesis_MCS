import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle 
from scipy.stats import ks_2samp
import explore_AP_KP
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap



"""
This file produces Jaccard similarity plots for both the simple and strict 
Jaccard coefficient criteria. 

Additionally it plots the average similarity of key points. 

"""
hard_dx = 235
hard_dy = 76


df_kp_cor = pd.read_csv('C:\\Users\\coen_\\OneDrive\\Documents\\ma_thesis\\MAIN\\data\\environment\\kp_cor.csv')


img_hospital = plt.imread("data\\environment\\map_2floor_bw.png") 

ap_loc  = np.load('C:\\Users\\coen_\\OneDrive\\Documents\\ma_thesis\\files\\iBeacon_data\\summer_2_floor_04_08\\points_wifi_2.npy')

dataset = "lp2"
with open(f'data\\radio_maps\\dict\\{dataset}.pkl', 'rb') as fp:
    radiodict = pickle.load(fp)

num_colors = 10


# cmap = cm.get_cmap("YlGnBu")
cmap = matplotlib.colormaps.get_cmap("hot")
bottom_half_cmap = LinearSegmentedColormap.from_list("bottom_half", cmap(np.linspace(0, 0.5, 256)))

# Generate evenly spaced values from 0 to 1 to sample the colormap
color_values = np.linspace(0, 1, num_colors)

# Get the colors from the colormap
colors = cmap(color_values)


with open(f'data\\radio_maps\\dict\\sigma_{dataset}.pkl', 'rb') as fp:
    radiodict_sigma = pickle.load(fp)

def jaccard_set(list1, list2):
    """Define Jaccard Similarity function for two sets"""
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union





def jaccard_gaus_set(dmu_i, dmu_j, dsigma_i, dsigma_j, n100_dmu_i, n100_dmu_j):


    """Define Jaccard Similarity function for two sets"""
    n = 1000
    np.random.seed(42)
    # keys_i = n100_dmu_i
    # keys_j = n100_dmu_j
    lap_keys = set(n100_dmu_i)  & set(n100_dmu_j)
    intersection = 0
    for mac_key in lap_keys:
            dist_i = np.random.normal(dmu_i[mac_key], abs(dsigma_i[mac_key]), n)
            dist_j = np.random.normal(dmu_j[mac_key], abs(dsigma_j[mac_key]), n)
            result = ks_2samp(dist_i, dist_j)
            p = result.pvalue
            if p >= 0.05:
                intersection += 1
    
    
    union = (len(n100_dmu_i) + len(n100_dmu_j)) - intersection
    # print("return value", float(intersection) / union)
    return float(intersection) / union

# print(radiodict_sigma)


# Step 1: Create a DataFrame with your signal strength data
# Replace this example data with your actual dataset

# Step 2: Calculate pairwise cosine similarities without sklearn
def custom_sim(i, j, dict_mu, dict_sigma):
    
    dmu_i = dict_mu[i]
    dsigma_i = dict_sigma[i]
    dmu_j= dict_mu[j]
    dsigma_j = dict_sigma[j]
   
    lap_keys = set(dmu_i.keys())  & set(dmu_j.keys())
    lap_keys.discard('pos')
    
    
    if len(lap_keys) != 0:
        n = 100 
        sim = 0
        for mac_key in lap_keys:
            dist_i = np.random.normal(dmu_i[mac_key], abs(dsigma_i[mac_key]), n)
            dist_j = np.random.normal(dmu_j[mac_key], abs(dsigma_j[mac_key]), n)
            result = ks_2samp(dist_i, dist_j)
            p = result.pvalue
            if p >= 0.05:
                sim += 1
        # sim = sim/len(lap_keys)
    else:
        sim = 0
    return sim


def gaus_jac(i, j, dict_mu, dict_sigma):
    
    dmu_i = dict_mu[i]
    dsigma_i = dict_sigma[i]
    dmu_j= dict_mu[j]
    dsigma_j = dict_sigma[j]
   
    
    if 'pos' in dmu_i:
        del dmu_i['pos']
    if 'pos' in dmu_j:
        
        del dmu_j['pos']

    if 'pos' in dsigma_i:
        
        del dmu_i['pos']
    if 'pos' in dsigma_j:
        
        del dmu_j['pos']

    dmu_i_keys = dmu_i.keys()
    dmu_j_keys = dmu_j.keys()
    # print(dmu_i_keys, "dmu_i_keys")
    # print(dmu_j_keys, "dmu_j_keys")
 
    n100_dmu_i_keys = []
    for mac_key in dmu_i_keys:
        if dmu_i[mac_key] > -100:
            n100_dmu_i_keys.append(mac_key)

    
    n100_dmu_j_keys = []
    for mac_key in dmu_j_keys:
        if dmu_j[mac_key] > -100:
            n100_dmu_j_keys.append(mac_key)
   
    sim = jaccard_gaus_set(dmu_i, dmu_j, dsigma_i, dsigma_j, n100_dmu_i_keys, n100_dmu_j_keys)
    return sim
    
    # if len(lap_keys) != 0:
    #     n = 100 
    #     sim = 0
    #     for key in lap_keys:
    #         dist_i = np.random.normal(dmu_i[key], abs(dsigma_i[key]), n)
    #         dist_j = np.random.normal(dmu_j[key], abs(dsigma_j[key]), n)
    #         result = ks_2samp(dist_i, dist_j)
    #         p = result.pvalue
    #         if p >= 0.05:
    #             sim += 1
    #     # sim = sim/len(lap_keys)
    # else:
    #     sim = 0
    # return sim

def classic_jac(i, j, dict_mu):
    
    dmu_i = dict_mu[i]
    dmu_j= dict_mu[j]
  
    if 'pos' in dmu_i:

        del dmu_i['pos']
    if 'pos' in dmu_j:

        del dmu_j['pos']


    dmu_i_keys = dmu_i.keys()
    dmu_j_keys = dmu_j.keys()
 
    n100_dmu_i_keys = []
    for mac_key in dmu_i_keys:
        if dmu_i[mac_key] > -100:
            n100_dmu_i_keys.append(mac_key)

    
    n100_dmu_j_keys = []
    for mac_key in dmu_j_keys:
        if dmu_j[mac_key] > -100:
            n100_dmu_j_keys.append(mac_key)
    
    # # print()
    # # print("j list", n100_dmu_i_keys)
    # # print(" i list", n100_dmu_j_keys)
    # # print()
    # # print()
    sim = jaccard_set(n100_dmu_i_keys, n100_dmu_j_keys)

    return sim

def custom_sim(i, j, dict_mu, dict_sigma):
    
    dmu_i = dict_mu[i]
    dsigma_i = dict_sigma[i]
    dmu_j= dict_mu[j]
    dsigma_j = dict_sigma[j]
   
    lap_keys = set(dmu_i.keys())  & set(dmu_j.keys())
    lap_keys.discard('pos')
    
    
    if len(lap_keys) != 0:
        n = 100 
        sim = 0
        for key in lap_keys:
            dist_i = np.random.normal(dmu_i[key], abs(dsigma_i[key]), n)
            dist_j = np.random.normal(dmu_j[key], abs(dsigma_j[key]), n)
            result = ks_2samp(dist_i, dist_j)
            p = result.pvalue
            if p >= 0.05:
                sim += 1
        # sim = sim/len(lap_keys)
    else:
        sim = 0
    return sim

        
def create_simdata(radiodict, radiodict_sigma, df_kp_cor, sim_method):
    n_datapoints = len(radiodict)
    pairwise_similarities = np.zeros((n_datapoints, n_datapoints))

    for i in range(n_datapoints):
        for j in range(n_datapoints):
            if sim_method == "custom_sim":
                similarity = custom_sim(i, j, radiodict, radiodict_sigma)
            if sim_method == "classic_jac":
                similarity = classic_jac(i, j, radiodict)
                
            if sim_method == "gaus_jac":
                similarity = gaus_jac(i, j, radiodict, radiodict_sigma)
            pairwise_similarities[i, j] = similarity

    # Step 3: Create a heatmap to visualize the similarities

    


    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.0)
    sns.set_style("whitegrid")

    # Create a DataFrame from the similarity matrix for visualization
    similarity_df = pd.DataFrame(pairwise_similarities)
    # print("similiarity_df", similarity_df)
    similarity_ls = []

    for i in range(len(similarity_df)):
        
        similarity_ls.append(similarity_df[i].mean())

    # print(max(similarity_ls))
    df_kp_cor['av_sim'] = similarity_ls
    print("sim_df", similarity_df)

    df_kp_cor.to_csv(f"data\\environment\\kp_pos_color{sim_method}_{dataset}.csv")
    # Create a heatmap with color coding
    heatmap = sns.heatmap(similarity_df, cmap=cmap, annot=False, cbar=True, linewidths=0.1, square=True)
    cbar = heatmap.collections[0].colorbar

    # Set font size for the colorbar
    cbar.ax.tick_params(labelsize=20)
    
    heatmap.invert_yaxis()

    # heatmap.set_yticks([0, similarity_df.shape[0] - 1])
    # heatmap.set_yticklabels([similarity_df.index[0], similarity_df.index[-1]])
    
    y_axis_labels = [similarity_df.index[-1]]
    plt.yticks(ticks=[similarity_df.shape[0] - 1], labels=y_axis_labels, rotation=0, fontsize=20)  # Adjust rotation as needed
    heatmap.figure.axes[-1].set_ylabel('Jaccard Coefficient', size=20)
    # plt.ylabel('Accuracy %', size=20)

    heatmap.set_xticks([0, similarity_df.shape[0] - 1])
    heatmap.set_xticklabels([similarity_df.index[0], similarity_df.index[-1]], fontsize=20)
    # plt.title('Similarity Between Data Points (Cosine Similarity)')
    plt.tight_layout()
    plt.savefig(f"data\\key_sim_jac_{sim_method}.png", format="png")
    plt.show()
    print("yodeldoe")

    fig_copy, ax_copy = explore_AP_KP.hospital_img_template(img_hospital, ap_loc, radiodict, ap_toggle =True, finger_toggle = False)

    x_cor_kp = df_kp_cor['x_pos']
    y_cor_kp = df_kp_cor['y_pos']
    av_cor_kp = df_kp_cor['av_sim']
    for i in range(len(df_kp_cor)):
        av_sim_str = av_cor_kp[i]
        av_sim_indice = int(av_sim_str*len(colors))
        ax_copy.plot(x_cor_kp[i] - hard_dx, y_cor_kp[i] - hard_dy, '.', color=colors[av_sim_indice], markersize = 3.3**2)


    vmin = av_cor_kp.max()
    vmax = av_cor_kp.min()
    norm = plt.Normalize(vmin, vmax)  # Adjust the min and max values as needed
    sm = plt.cm.ScalarMappable(cmap=bottom_half_cmap, norm=norm)
    cbar = plt.colorbar(sm, orientation='horizontal', label='Average Similiarity')
    # x_fp_plot = df_kp_cor
    # ax_copy.set_title("TEstingtitle")
    plt.tight_layout()
    plt.savefig(f"data\\keyp_sim_floor{sim_method}.pdf", format="pdf")
    plt.show()
    print('yodeldie!')


test = create_simdata(radiodict, radiodict_sigma, df_kp_cor, sim_method="gaus_jac")