import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# df data
dt_ls = ["dt_3", "dt_5"]
data_ls = ["lp"]
time_ref = 0

for dt_string in dt_ls:
    basetime = None
    df_ls = []

    # Load the paths into a list and find the most complete timeframe to use as reference
    for i in range(5):
        df_path = pd.read_csv(f"C:\\Users\\coen_\\OneDrive\\Documents\\ma_thesis\\MAIN\\data\\dt\\{dt_string}\\{data_ls[0]}\\dt{i}.csv")
        df_ls.append(df_path)
        time = df_path['time'].unique()

        # Some df_paths have missing data points, choose the most complete one
        if len(time) > time_ref:
            basetime = time
            time_ref = len(time)
            basetime = sorted(basetime)

    av_entries_ls = []
    for timeslot in basetime:
        entries_n = []
        for df in df_ls:
            mask = df['time'] == timeslot
            un_time = df[mask]
            entries_n.append(len(un_time))
        av_entries_ls.append(np.mean(entries_n))

    # Set the figure size to fullscreen
    plt.figure(figsize=(10, 6))

    # Convert basetime to datetime
    basetime = pd.to_datetime(basetime)

    plt.bar(basetime, av_entries_ls)
    plt.xticks(rotation=45, ha='right', fontsize=13)
    plt.ylabel("Entry Number", fontsize=15)
    plt.xlabel("Timeslots", fontsize=15)
    plt.title(f"Data Entries for {dt_string}", fontsize=16)
    plt.show()