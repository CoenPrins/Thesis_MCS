from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
import base64
import pickle 

bg_path = 'data\\environment\\map_2floor_PF_bw.png'


""" 
Maybe change it from a list of dicts to a dictionary of dictionaries? 
Oh no a list is just as good, you only need to substract all the ID's by 1, so that the list index 
is tied to the waypoint index!

"""
class Node():
    """This is the template we are going to use, however we will use a dictionary, 
    This dictionary is saved in a pkl file, and loaded in when the abm is initialised, 
    copies are made per path that can be altered, what is the downside of this? 
    The downside is all waypoints are loaded from the beginning, I don't think that is neccesarily 
    bad. For the pathing I definetly agree!  q
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.connections = []

        self.g = 0

    def __eq__(self, other):
        return self.position == other.position
    

template = [0, (0, 0), []]
# tuples are for inside dictionary for speed, list is for clarity in writing and will nog be like this in final product (will be list of dicts)
# i entered these by hand, but realized it would be better if it started at zero so I transformed it using code 
wp20_ls = [[1, (70, 194), [2]],[2, (79, 185), [1,3]],[3, (114, 150), [2, 4, 5, 6]],[4, (97, 131), [3]],[5, (135, 169), [3]],[6, (137, 126), [3,7,8]],[7, (119, 110), [6]],[8, (176, 101), [6, 9, 10]],[9, (176, 120), [8]],[10, (203, 101), [8,11,20,21]]
           ,[11, (203, 127), [10]],[12, (149, 80), [14]],[13, (159, 63), [14]],[14, (160, 77), [12, 13, 16]],[15, (195, 62), [16]],[16, (195, 74), [14, 15, 17]],[17, (217, 74), [16, 18, 19]],[18, (217, 60), [17]],[19, (230, 76), [17, 20, 26]],[20, (230, 95), [10, 19, 21]]]

wp40_ls = [[21, (239, 101), [22, 10, 20, 23]],[22, (239, 128), [21]],[23, (274, 101), [21, 24, 31, 32]],[24, (289, 95), [23, 31, 25]],[25, (289, 76), [24, 26, 28, 29]],[26, (262, 76), [19, 27, 25]],[27, (262, 59), [26]],[28, (289, 59), [25]],[29, (319, 76), [25, 30]],[30, (319, 63), [29]]
,[31, (310, 101), [23, 24, 36]],[32, (274, 128), [23, 33]],[33, (305, 133), [32, 34]],[34, (314, 133), [33, 35]],[35, (345, 128), [34, 36]],[36, (345, 101), [31, 35, 37, 45]],[37, (352, 95), [36, 38, 45]],[38, (352, 79), [37, 39, 42]],[39, (341, 73), [40, 38]],[40, (341, 60), [39]]]

wp61_ls= [[41, (366, 59), [42]],[42, (366, 73), [38, 41, 43]],[43, (390, 73), [42, 44]],[44, (390, 60), [43]],[45, (390, 101), [36, 37, 46, 47]],[46, (390, 120), [45]],[47, (424, 104), [45, 48]],[48, (456, 133), [47, 49]],[49, (466, 117), [48, 55, 54, 50]],[50, (451, 93), [49, 51, 52]]
          ,[51, (462, 83), [50]],[52, (436, 80), [50, 53]],[53, (446, 69), [52]],[54, (481, 103), [49]],[55, (489, 135), [49, 56, 57]],[56, (503, 122), [55]],[57, (510, 154), [55, 58, 59]],[58, (521, 140), [57]],[59, (522, 169), [57, 60, 61]],[60, (509, 184), [59]],[61, (541, 184), [59, 62]],[62, (552, 172), [61]]]




def update_list(wp_ls):
    updated_list = []
    for wp_numb, (pos_x, pos_y), connections in wp_ls:
        updated_connections = [conn - 1 for conn in connections]
        updated_element = [wp_numb - 1, (pos_x, pos_y), updated_connections]
        updated_list.append(updated_element)
    print(updated_list)
    return updated_list

wp20_ls = update_list(wp20_ls)
wp40_ls = update_list(wp40_ls)
wp61_ls = update_list(wp61_ls)



dict_ls = []
# blueprint 








wp_ls = wp20_ls + wp40_ls + wp61_ls

print(len(wp_ls))

dict_ls = []
for wp in wp_ls:
    dict = {
    'id': wp[0],
    'parent': None, 
    'pos': (wp[1][0], wp[1][1]), 
    'conns':wp[2],
    'g': 0,
    }
    dict_ls.append(dict)



for dict in dict_ls:
    conns_copy = dict['conns']
    new_conns = {}
    self_pos = dict['pos']
    # print('selfpos', self_pos)
    
    for conn in conns_copy:
        
        if conn <= len(dict_ls):
            conn_pos = dict_ls[conn ]['pos']
            # print('con pos', conn_pos)
            conn_dist = np.sqrt(abs(self_pos[0] - conn_pos[0])**2 + abs(self_pos[1] - conn_pos[1])**2)
            new_conns[conn] = conn_dist
            
    dict['conns'] = new_conns
    print()

print()
print()
# print(dict_ls)

# print(dict_ls)


img = image.imread(bg_path)
shape = (img.shape[0], img.shape[1])
dist_per_pix = 1.0
max_x = shape[1]*dist_per_pix - 1
max_y = shape[0]*dist_per_pix - 1
        # print("self max", self.max_x)
wall_mask = np.zeros(shape, dtype=bool)
for i in range(shape[0]):
    for j in range(shape[1]):
        wall_mask[i,j] = img[i,j,0] < 0.1

def is_wall(x, y):
    i = int(y/dist_per_pix)
    j = int(x/dist_per_pix)
    return wall_mask[i, j]


# print("hello there")


def draw_line(x1, y1, x2, y2):
    # Bresenham's line algorithm
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        points.append((x1, y1))

        if x1 == x2 and y1 == y2:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return points

def cross_wall(x1, y1, x2, y2):
    line_points = draw_line(x1, y1, x2, y2)
        
    for x, y in line_points:
        print(x,y)
        if is_wall(x, y):
            
            return True  # Line crosses a wall

    return False  # Line does not cross a wall

pos_wp = []
id_ls = []
for dict in dict_ls:
    x, y = dict['pos']
    plt.scatter(x, y, color='blue', marker='o', label='Dots')
    # plt.text(x, y, dict['id'], fontsize=12, ha='right')
    conns = dict['conns']
    for key in conns:
        conn_x, conn_y = dict_ls[key]['pos']
        plt.plot([x, conn_x], [y, conn_y], color='lightcoral', alpha=0.4, linewidth=2.5)

    # pos_wp.append(dict['pos'])
    # id_ls.append(dict['id'])
    
# pos_wp = np.array(pos_wp)
# print(pos_wp)


# plt.scatter(pos_wp[:, 0], pos_wp[:, 1], color='blue', marker='o', label='Dots')
# plt.plot(line[:,0], line[:,1], color=clr_line)

# for id_ls, x, y in zip(id_ls, pos_wp[:, 0], pos_wp[:, 1]):
#     plt.text(x, y, id_ls, fontsize=12, ha='right')
plt.imshow(img)
plt.show()
print(dict_ls)
# dict_directory  = "data\\distance"


# with open(f'{dict_directory}\\waypoints.pkl', 'wb') as fp:
#                 pickle.dump(dict_ls, fp)
