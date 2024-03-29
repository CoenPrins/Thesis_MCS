import pandas as pd 
import math
import matplotlib.pyplot as plt
# time intervals of true paths 

start_time1 = pd.to_datetime("04/08/2021 18:44:00")  
end_time1 = pd.to_datetime("04/08/2021 18:47:00")   

start_time2 = pd.to_datetime("04/08/2021 18:52:00")  
end_time2 = pd.to_datetime("04/08/2021 18:54:00")

start_time3 = pd.to_datetime("04/08/2021 18:48:00") 
end_time3 = pd.to_datetime("04/08/2021 18:51:00") 


# replace these values 

n_start_time1 = pd.to_datetime("04/08/2021 18:44:37")  
n_end_time1 = pd.to_datetime("04/08/2021 18:44:57")   

n_start_time2 = pd.to_datetime("04/08/2021 18:52:46")  
n_end_time2 = pd.to_datetime("04/08/2021 18:53:06")

n_start_time3 = pd.to_datetime("04/08/2021 18:48:55") 
n_end_time3 = pd.to_datetime("04/08/2021 18:49:15") 





# coordinates of true paths 

og_path1 = [(317, 261), (317, 261), (324, 251), (330, 245), (337, 239), (345, 231),
        (354, 223), (361, 215), (366, 209), (373, 202), (379, 194), (385, 189),
        (391, 185), (398, 181), (408, 179), (420, 179), (433, 177), (445, 178),
        (459, 179), (474, 178), (487, 178), (495, 177), (507, 176), (516, 172),
        (520, 168), (521, 163), (521, 158), (517, 154), (503, 151), (493, 151), (489, 151)]


og_path2 = [(491, 177),(490, 177),(480, 176),
        (471, 174),(468, 171),(466, 164),
        (466, 159),(472, 153),(486, 149),
        (493, 151),(501, 151),(512, 153),
        (518, 157),(519, 164),(516, 171),
        (509, 177),(501, 177),(492, 177)]



og_path3 = [(402, 235),(401, 234),(399, 231),
        (395, 228),(391, 224),(388, 222),
        (384, 217),(381, 214),(380, 210),
        (378, 204),(381, 194),(386, 189),
        (393, 184),(400, 181),(408, 181),
        (418, 180),(426, 180),(431, 179),
        (439, 177),(446, 177),(454, 173),
        (460, 168),(462, 161),(460, 154),
        (454, 152),(447, 152),(438, 151),
        (430, 151),(421, 150),(415, 150),
        (409, 150),(402, 150),(397, 150)]

# replace this
n_og_path1 = [(317, 261), (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), 
         (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), 
         (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), 
         (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), 
         (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), 
         (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), 
         (317, 261), (317, 261), (317, 261), (317, 261), (317, 261), (324, 251), (330, 245), (337, 239), (345, 231),
           (354, 223), (361, 215), (366, 209), (373, 202), (379, 194), (385, 189), (391, 185), (398, 181), (408, 179), 
           (420, 179), (433, 177), (445, 178), (459, 179), (474, 178), (487, 178), (495, 177), (507, 176), (516, 172), 
           (520, 168), (521, 163), (521, 158), (517, 154), (503, 151), (493, 151), (489, 151), (489, 151), (489, 151), 
           (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151),
             (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), 
             (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), 
             (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151),
               (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), 
               (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), 
               (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), 
               (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), 
               (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), 
               (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), 
               (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151),
               (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), 
               (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151),
                 (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), 
                 (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), 
                 (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151),
                   (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), 
                   (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), 
                   (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), 
                   (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), 
                   (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151)]



# replace this 
n_og_path2 = [(491, 177), (491, 177), (491, 177), (491, 177), (491, 177), (491, 177), (491, 177), (491, 177), (491, 177), (491, 177), 
       (491, 177), (491, 177), (491, 177), (491, 177), (491, 177), (491, 177), (491, 177), (491, 177), (491, 177), (491, 177), 
       (491, 177), (491, 177), (491, 177), (491, 177), (491, 177), (491, 177), (491, 177), (491, 177), (491, 177), (491, 177), 
       (491, 177), (491, 177), (491, 177), (491, 177), (491, 177), (491, 177), (491, 177), (491, 177), (491, 177), (491, 177), 
       (491, 177), (491, 177), (490, 177), (480, 176), (471, 174), (468, 171), (466, 164), (466, 159), (472, 153), (486, 149), 
       (493, 151), (501, 151), (512, 153), (518, 157), (519, 164), (516, 171), (509, 177), (501, 177), (492, 177), (492, 177), 
       (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), 
       (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), 
       (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), 
       (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), 
       (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), (492, 177)]


# replace this 
n_og_path3 = [(402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), 
       (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), 
       (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), 
       (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), 
       (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), 
       (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), 
       (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), 
       (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (402, 235), (401, 234), (399, 231), (395, 228), (391, 224), 
       (388, 222), (384, 217), (381, 214), (380, 210), (378, 204), (381, 194), (386, 189), (393, 184), (400, 181), (408, 181), (418, 180), (426, 180), 
       (431, 179), (439, 177), (446, 177), (454, 173), (460, 168), (462, 161), (460, 154), (454, 152), (447, 152), (438, 151), (430, 151), (421, 150), 
       (415, 150), (409, 150), (402, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), 
       (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), 
       (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), 
       (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), 
       (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), 
       (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), 
       (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), 
       (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), 
       (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), 
       (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), 
       (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), 
       (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), 
       (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), 
       (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), 
       (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150)]


# print("len3", len(n_og3))
def sparse_path(og_path):
    
    sparse = []
    sparse.append(og_path[0])
    sparse.append(og_path[0])
    for i in range(len(og_path)-2):
        sparse.append(og_path[-1])
    return sparse



# sparse_1 = sparse_path(og_path1)


# print(sparse_1)
# print("length og", len(og_path1))
# print("length sparse", len(sparse_1))

# sparse_2 = sparse_path(og_path2)


# print(sparse_2)
# print("length og", len(og_path2))
# print("length sparse", len(sparse_2))

# sparse_3 = sparse_path(og_path3)


# print(sparse_3)
# print("length og", len(og_path3))
# print("length sparse", len(sparse_3))



sparse_1 = [(317, 261), (317, 261), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), 
            (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), 
            (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151), (489, 151)]



sparse_2 = [(491, 177), (491, 177), (492, 177), (492, 177), (492, 177), (492, 177), 
            (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), 
            (492, 177), (492, 177), (492, 177), (492, 177), (492, 177), (492, 177)]

sparse_3 = [(402, 235), (402, 235), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), 
            (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), 
            (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), 
            (397, 150), (397, 150), (397, 150), (397, 150), (397, 150), (397, 150)]







def generate_datetime_dataset(start_time, end_time, n, small_strings):
    # Create a datetime range with equally spaced points
    datetime_points = pd.date_range(start=start_time, end=end_time, periods=n)
    
    if small_strings:
        # Create a new list to store formatted strings
        formatted_strings = []
        
        for dt_point in datetime_points:
            formatted_string = str(dt_point).split(' ')[1].split('.')[0]
            formatted_strings.append(formatted_string)
            
        return formatted_strings
    # If you want to convert the datetime points to a list, you can use:
    # datetime_points_list = datetime_points.tolist()

    return datetime_points


def calculate_distance(path):
    total_distance = 0
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_distance += distance
    return total_distance * 0.133

# distance = calculate_distance(og_path1)
# print("Total distance of the path:", distance)


# distance_1 = calculate_distance(og_path1)
# print("dis 1", distance_1)
# distance_1 = calculate_distance(n_og_path1)
# print("dis 1", distance_1)

# distance_2 = calculate_distance(og_path2)
# print("dis 2", distance_2)
# distance_2 = calculate_distance(n_og_path2)
# print("dis 2", distance_2)

# distance_3 = calculate_distance(og_path3)
# print("dis 3", distance_3)
# distance_3 = calculate_distance(n_og_path3)
# print("dis 3", distance_3)

# output
# dis 1 37.750053590213845
# dis 1 37.750053590213845
# dis 2 17.877276064875605
# dis 2 17.877276064875605
# dis 3 28.628282222443328
# dis 3 28.628282222443328



