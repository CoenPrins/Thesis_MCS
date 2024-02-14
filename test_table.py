import pandas as pd


df= pd.read_csv('data\\results\\full_full.csv', index_col=0)

df["av_path_av"] = df[['path1_av', 'path2_av', 'path3_av']].apply(lambda row: sum(x[0] for x in row) / 3, axis=1)

print(df["av_path_av"])
