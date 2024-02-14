
import pandas as pd




def transform_paths(df, path_list):


    for path_column in path_list:
        df[path_column] = df[path_column].apply(eval)
        mean = df[path_column].apply(lambda x: x[0])
        df[path_column] = mean

    return df
# Read the CSV files

path_list = ["path1", "path2", "path3"]


df_lp_1 = pd.read_csv('data\\results\\lp_dt10_test_knn_1.csv', index_col=0)

df_lp_1 = transform_paths(df_lp_1, path_list)

df_lp_1['av_path'] = (df_lp_1["path1"] + df_lp_1["path2"] + df_lp_1["path3"])/3
del df_lp_1['Knn']
print(df_lp_1.to_latex(index=False))
print()

df_loc_av = pd.read_csv('data\\results\\lp_loc_mtdaverage_dt10.csv', index_col=0)
df_loc_gaus_n = pd.read_csv('data\\results\\lp_loc_mtdgaus_80_dt10.csv', index_col=0)
df_loc_gaus_80 = pd.read_csv('data\\results\\lp_loc_mtdgaus_normal_dt10.csv', index_col=0)
df_loc_gaus_top = pd.read_csv('data\\results\\lp_loc_mtdgaus_top_dt10.csv', index_col=0)


df_loc_av["origin"] = 'lp_av'
df_loc_gaus_n["origin"] = 'lp_gaus_n'
df_loc_gaus_80["origin"] = 'lp_gaus_80'
df_loc_gaus_top["origin"] = 'lp_gaus_t'





# Concatenate the dataframes
result_df = pd.concat([df_loc_av, df_loc_gaus_n, df_loc_gaus_80, df_loc_gaus_top], ignore_index=True)



# # Convert the 'path1' column from string representation to a list of floats


result_df = transform_paths(result_df, path_list)
result_df["av_path"] = (result_df["path1"] + result_df["path2"] + result_df["path3"])/3
# print()
# print(result_df)


met_df = result_df.groupby('origin')['path1'].mean().reset_index()
met_df['path2'] = result_df.groupby('origin')['path2'].mean().reset_index()['path2']
met_df['path3'] = result_df.groupby('origin')['path3'].mean().reset_index()['path3']
met_df['av_path'] = result_df.groupby('origin')['av_path'].mean().reset_index()['av_path']

print(met_df.to_latex(index=False))


mis_val_df = result_df.groupby('mis_val_method')['path1'].mean().reset_index()
mis_val_df['path2'] = result_df.groupby('mis_val_method')['path2'].mean().reset_index()['path2']
mis_val_df['path3'] = result_df.groupby('mis_val_method')['path3'].mean().reset_index()['path3']
mis_val_df['av_path']= result_df.groupby('mis_val_method')['av_path'].mean().reset_index()['av_path']


print(mis_val_df.to_latex(index=False))



knn_df = result_df.groupby('Knn')['path1'].mean().reset_index()
knn_df['path2'] = result_df.groupby('Knn')['path2'].mean().reset_index()['path2']
knn_df['path3'] = result_df.groupby('Knn')['path3'].mean().reset_index()['path3']
knn_df['av_path'] = result_df.groupby('Knn')['av_path'].mean().reset_index()['av_path'] 



print(knn_df.to_latex(index=False))


min_path1_row = result_df.loc[result_df['av_path'].idxmin()]
# Print the result

print("hallo")
print(min_path1_row)
min_path1_row


# mean_values_dt1 = result_df.groupby('dt')['path1'].mean().reset_index()

# mean_values_dt2 = result_df.groupby('dt')['path2'].mean().reset_index()
# mean_values_dt3 = result_df.groupby('dt')['path3'].mean().reset_index()
# mean_values_dt_average = result_df.groupby('dt')['av_path'].mean().reset_index()
# print(mean_values_dt1)
# print(mean_values_dt2)
# print(mean_values_dt3)
# print(mean_values_dt_average)

