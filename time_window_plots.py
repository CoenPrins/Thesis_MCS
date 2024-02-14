import interpolate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time_og_data2
import time
import seaborn as sns 

# you got the basic premis down, only difference is you need to change the labels 
# I think if you first make sure the spacing is correct, you can just perform the same 
# transformation on both the time AND the numeric and then it's ALLGOOD!
# print("barteljaap")
# df = sns.load_dataset('iris') 


# df_best = pd.read_csv('data\\time_window\\final_proper_best.csv', index_col=0)
# df_total = pd.read_csv('data\\time_window\\final_proper_total.csv', index_col=0)


df_best = pd.read_csv('data\\time_window\\final_proper_best.csv', index_col=0)
df_total = pd.read_csv('data\\time_window\\final_proper_total.csv', index_col=0)


df_copy = df_best.copy()





df_best['time'] = pd.to_datetime(df_best['time'], format='mixed')



# Extract hours, minutes, and seconds from the datetime column
# df_best['time'] = df_best['time'].dt.time

df_best['time'] = pd.to_numeric(df_best['time'], errors='coerce')

print(df_best)
ax = sns.boxplot(x=df_best["time"], y=df_best["path"])

ax = sns.stripplot(x=df_best["time"], y=df_best["path"], color="black", size=4, jitter=True)

# selected_ticks = [df['time'].iloc[0], df['time'].iloc[2], df['time'].iloc[-1]]
min_max_times = df_best.groupby('path')['time'].agg(['min', 'max']).reset_index()

# Create equally spaced ticks for each path
ticks_per_path = 5
selected_ticks = []
for path, min_val, max_val in zip(min_max_times['path'], min_max_times['min'], min_max_times['max']):
    path_ticks = np.linspace(min_val, max_val, ticks_per_path)
    selected_ticks.extend(path_ticks)



def generate_equally_spaced_timestamps(start, end, num_points=5):
    time_diff = (end - start) / (num_points - 1)  # Adjust for including the start and end
    return [(start + i * time_diff).strftime('%H:%M:%S') for i in range(num_points)]

# Generate 5 equally spaced timestamps for each time period
timestamps1 = generate_equally_spaced_timestamps(time_og_data2.start_time1, time_og_data2.end_time1)
timestamps2 = generate_equally_spaced_timestamps(time_og_data2.start_time2, time_og_data2.end_time2)
timestamps3 = generate_equally_spaced_timestamps(time_og_data2.start_time3, time_og_data2.end_time3)



ticks = timestamps1  + timestamps2  + timestamps3

print(ticks)



plt.xticks(ticks=selected_ticks, labels=[str(ts) for ts in selected_ticks], rotation=45, ha='right')
ax.set_xticklabels(ticks)

plt.tight_layout()
# plt.show()




plt.close()

total_t1 = df_total[df_total["path"] == "path1"]["time"]
total_er1 = df_total[df_total["path"] == "path1"]["error_mean"]
total_std1 = df_total[df_total["path"] == "path1"]["error_std"]
total_og1 = df_total[df_total["path"] == "path1"]["n_og"]


std_plus1 = total_er1 + total_std1
std_minus1 = total_er1 - total_std1



timestamps1 = time_og_data2.generate_datetime_dataset(time_og_data2.start_time1, time_og_data2.end_time1, 5, small_strings=True)
timestamps2 = time_og_data2.generate_datetime_dataset(time_og_data2.start_time2, time_og_data2.end_time2, 5,  small_strings=True)
timestamps3 = time_og_data2.generate_datetime_dataset(time_og_data2.start_time3, time_og_data2.end_time3, 5, small_strings=True)
print("timestamp", type(timestamps1))

print("timestamp0", type(timestamps1[0]))

def get_spaced_items(input_list, num_items):
    print("hello hello test test")
    print("input list", type(input_list))
    
    if num_items <= 2 or len(input_list) <= 2:
        return input_list  # Not enough items to select
    
    spaced_ls = [input_list[0]] + [input_list[i * (len(input_list) - 1) // (num_items - 1)] for i in range(1, num_items - 1)] + [input_list[-1]]

    return spaced_ls




total_t1_ls = total_t1.tolist()
std_plus1_ls = std_plus1.tolist()
std_minus1_ls = std_minus1.tolist()
ticks1 = get_spaced_items(total_t1_ls, 5)

plt.plot(total_t1, total_er1, color="red")

plt.fill_between(total_t1_ls, std_plus1_ls, std_minus1_ls, alpha=0.2, color="red")
plt.xticks(ticks=ticks1, labels=timestamps1)

plt.xlabel("Time", fontweight='bold')
plt.ylabel('Error (m)', fontweight='bold')

# ax.set_xticklabels(ticks)
# plt.xticks(ticks=timestamps1)
# ax.set_xticklabels(ticks)

plt.tight_layout()
plt.show()


total_t2 = df_total[df_total["path"] == "path2"]["time"]
total_er2 = df_total[df_total["path"] == "path2"]["error_mean"]
total_std2 = df_total[df_total["path"] == "path2"]["error_std"]
total_og2 = df_total[df_total["path"] == "path2"]["n_og"]




std_plus2 = total_er2 + total_std2
std_minus2 = total_er2 - total_std2


total_t2_ls = total_t2.tolist()
std_plus2_ls = std_plus2.tolist()
std_minus2_ls = std_minus2.tolist()
ticks2 = get_spaced_items(total_t2_ls, 5)



plt.plot(total_t2, total_er2, color="blue")

plt.fill_between(total_t2, std_plus2, std_minus2, alpha=0.2, color="blue")
plt.xticks(ticks=ticks2, labels=timestamps2)

plt.xlabel("Time", fontweight='bold')
plt.ylabel('Error (m)', fontweight='bold')


plt.show()


total_t3 = df_total[df_total["path"] == "path3"]["time"]
total_er3 = df_total[df_total["path"] == "path3"]["error_mean"]
total_std3 = df_total[df_total["path"] == "path3"]["error_std"]
total_og3 = df_total[df_total["path"] == "path3"]["n_og"]


std_plus3 = total_er3 + total_std3
std_minus3 = total_er3 - total_std3



total_t3_ls = total_t3.tolist()
std_plus3_ls = std_plus3.tolist()
std_minus3_ls = std_minus3.tolist()
ticks3 = get_spaced_items(total_t3_ls, 5)



plt.plot(total_t3, total_er3, color="green")

plt.fill_between(total_t3, std_plus3, std_minus3, alpha=0.2, color="green")
plt.xticks(ticks=ticks3, labels=timestamps3)
# plt.plot(total_t1, std_plus)
# plt.plot(total_t1, std_minus)


plt.xlabel("Time", fontweight='bold')
plt.ylabel('Error (m)', fontweight='bold')
plt.show()
print(total_t1)


index_1 = total_er1.idxmin()

print("start_time1",total_er1.min())


print("start_time1",total_t1.loc[index_1])

print("n_og1", total_og1[index_1])

index_2 = total_er2.idxmin()

print("start_time2",total_er2.min())


print("start_time2",total_t2.loc[index_2])

print("n_og2", total_og2[index_2])



index_3 = total_er3.idxmin()

print("start_time1",total_er3.min())


print("start_time1",total_t3.loc[index_3])
print("n_og3", total_og3[index_3])

# print("start_time2",total_er2.min())

# print("start_time3",total_er3.min())



# total_t2
# total_er2




