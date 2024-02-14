
import pandas as pd


def transform_paths(df, path_list):

    for path_column in path_list:
        df[path_column] = result_df[path_column].apply(eval)
        mean = df[path_column].apply(lambda x: x[0])
        df[path_column] = mean

    return df
# Read the CSV files
df_lp_t = pd.read_csv('data\\results\\lp_6_gaus_normal_t.csv', index_col=0)
df_lp2_t = pd.read_csv('data\\results\\lp2_6_gaus_normal_t.csv', index_col=0)
df_lp2_n85_t = pd.read_csv('data\\results\\lp2_n85_6_gaus_normal_t.csv', index_col=0)
df_lp2_n100_t = pd.read_csv('data\\results\\lp2_n100_6_gaus_normal_t.csv', index_col=0)


df_lp_g = pd.read_csv('data\\results\\lp_6_gaus_normal_g.csv', index_col=0)
df_lp2_g = pd.read_csv('data\\results\\lp2_6_gaus_normal_g.csv', index_col=0)
df_lp2_n85_g = pd.read_csv('data\\results\\lp2_n85_6_gaus_normal_g.csv', index_col=0)
df_lp2_n100_g = pd.read_csv('data\\results\\lp2_n100_6_gaus_normal_g.csv', index_col=0)

data_ls = [[df_lp_t, df_lp2_t, df_lp2_n85_t, df_lp2_n100_t],[df_lp_g, df_lp2_g, df_lp2_n85_g, df_lp2_n100_g]]

data_names =["time based distance", "general distance"]
for index, data in enumerate(data_ls):

    
    print()
    print()
    print()
    print(f"the results for {data_names[index]}")
    print(f"the results for {data_names[index]}")
    print(f"the results for {data_names[index]}")
    print()
    df_lp = data[0]
    df_lp2 = data[1]
    df_lp2_n85 = data[2]
    df_lp2_n100 = data[3]


    df_lp['origin'] = 'lp'
    df_lp2['origin'] = 'lp2'
    df_lp2_n85['origin'] = 'lp2_n85'
    df_lp2_n100['origin'] = "lp2_n100"

    # Concatenate the dataframes
    result_df = pd.concat([df_lp, df_lp2, df_lp2_n85, df_lp2_n100], ignore_index=True)

    # Convert the 'path1' column from string representation to a list of floats

    path_list = ["path1", "path2", "path3"]
    result_df = transform_paths(result_df, path_list)
    result_df["av_path"] = (result_df["path1"] + result_df["path2"] + result_df["path3"])/3
    # print(result_df)


    lp_df = result_df.groupby('origin')['path1'].mean().reset_index()
    lp_df["path2"] = result_df.groupby('origin')['path2'].mean().reset_index()["path2"]
    lp_df["path3"] = result_df.groupby('origin')['path3'].mean().reset_index()["path3"]
    lp_df["av_path"] = result_df.groupby('origin')['av_path'].mean().reset_index()["av_path"]

    print("lp_df", lp_df)
    print(lp_df.to_latex(index=False))

    mis_val_df = result_df.groupby('mis_val_method')['path1'].mean().reset_index()
    mis_val_df["path2"] = result_df.groupby('mis_val_method')['path2'].mean().reset_index()["path2"]
    mis_val_df["path3"] = result_df.groupby('mis_val_method')['path3'].mean().reset_index()["path3"]
    mis_val_df["av_path"] = result_df.groupby('mis_val_method')['av_path'].mean().reset_index()["av_path"]

    print('mis val')

    print(mis_val_df.to_latex(index=False))
    


    # initialise df with dt and the mean values for path 1 for every dt 
    df_mean_val = result_df.groupby('dt')['path1'].mean().reset_index()

    # add column of mean value of other paths/average paths, we sort by dt but only add the path column
    df_mean_val["path2"] = result_df.groupby('dt')['path2'].mean().reset_index()["path2"]
    df_mean_val["path3"] = result_df.groupby('dt')['path3'].mean().reset_index()["path3"]
    df_mean_val["av_path"] = result_df.groupby('dt')['av_path'].mean().reset_index()["av_path"]

    # print(mean_val)
    print("mean_val")
    print(df_mean_val.to_latex(index=False))

    min_path_row = result_df.loc[result_df['av_path'].idxmin()]
    # Print the result
    print("minimal average comb")
    print(min_path_row)
    min_path_row

