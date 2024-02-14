from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np

import pickle
import time
import random

"""
Runtime for a thousand is currently 1.5 seconds, drawing the lines takes a lot of brain power. 
You could do something about this... but nah. 
One possible way would be creating a massive dictionary but that is not feasible. 

For now you need to check it for single implementations, are the paths logical?

Typically 50 out of a thousand are correct in RANDOM rnad int, in normal situations this will be far more skewed toward 
direct 

"""


bg_path = 'data\\environment\\map_2floor_PF_bw.png'

with open(f"data\\distance\\waypoints.pkl", 'rb') as fp: 
        wp_ls = pickle.load(fp)


with open(f'data\\distance\\dist_ls.pkl', 'rb') as fp:
                dist_ls = pickle.load(fp)

with open(f'data\\distance\\path_ls.pkl', 'rb') as fp:
                path_ls = pickle.load(fp)


bg_path = 'data\\environment\\map_2floor_PF_bw.png'



# create image of blueprint AND determine wall positions
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
    # Returns true if location is in a wall
    i = int(y/dist_per_pix)
    j = int(x/dist_per_pix)
    return wall_mask[i, j]



def random_pos(max_x, max_y):
    # Returns a random position not inside a wall
    wall = True
    while wall == True:
        x = random.randrange(0, max_x)
        y = random.randrange(0, max_y)
        wall = is_wall(x, y)
    
    return(x, y)
    


def draw_line(x1, y1, x2, y2):
    # Draws line using bresenham's line propagation
    # Returns points 
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
    # Draws line using Bresenham 
    # Returns True if it crosses a wall

    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    wall = False
    while not wall:
        points.append((x1, y1))

        if x1 == x2 and y1 == y2:
            break

        if is_wall(x1, y1):
            wall = True
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return wall


start_pos =  random_pos(max_x, max_y)
end_pos = random_pos(max_x, max_y)


start_pos = (int(46.3), int(165.66425088999185))
end_pos = (425, 53)

# print("start_pos", start_pos)
# print("end_pos", end_pos)
# # start_pos = (122, 146)
# end_pos = (121, 168)

# start_pos = (70, 194)
# # end_pos =  (79, 185)
# end_pos = (121, 168)

# start_pos = (109.4260124815229, 168.08262877479464)
# start_pos = (122, 148)
# end_pos = (141, 180)
start_ls = []
end_ls = []
direct = None

if not cross_wall(start_pos[0], start_pos[1], end_pos[0], end_pos[1]):
        dist =  np.sqrt(abs(start_pos[0] - end_pos[0])**2 + abs(start_pos[1] - end_pos[1])**2)
        direct = dist
     
else:
    # find waypoints with unbroken lines to start and endpoint

    if is_wall(end_pos[0], end_pos[1]):
        for dict in wp_ls:

            if not cross_wall(dict['pos'][0], dict['pos'][1], start_pos[0], start_pos[1]):
                dist =  np.sqrt(abs(dict['pos'][0] - start_pos[0])**2 + abs(dict['pos'][1] - start_pos[1])**2)
                start_ls.append((dict['id'], dist))

                
            
            dist =  np.sqrt(abs(dict['pos'][0] - end_pos[0])**2 + abs(dict['pos'][1] - end_pos[1])**2)
            end_ls.append((dict['id'], dist))

        print("end_ls before sort", end_ls)
        end_ls = sorted(end_ls, key=lambda x: x[1])
        print("end_ls after sort", end_ls)
        end_ls = [end_ls[0]]
        print("only top end ls", end_ls)

            # find waypoints with unbroken lines to start and endpoint
    else:
        for dict in wp_ls:
            if not cross_wall(dict['pos'][0], dict['pos'][1], start_pos[0], start_pos[1]):
                dist =  np.sqrt(abs(dict['pos'][0] - start_pos[0])**2 + abs(dict['pos'][1] - start_pos[1])**2)
                start_ls.append((dict['id'], dist))

                
            if not cross_wall(dict['pos'][0], dict['pos'][1], end_pos[0], end_pos[1]):
                dist =  np.sqrt(abs(dict['pos'][0] - end_pos[0])**2 + abs(dict['pos'][1] - end_pos[1])**2)
                end_ls.append((dict['id'], dist))


if direct is not None:
   
    final_dist = dist
    point_ar = np.array([start_pos, end_pos])
    
    plt.plot(point_ar[:, 0], point_ar[:,1], color='orange')

else:
    # find the shortest dijkstra path that connects to end and start point
    fast_path_ls = []
    for i in range(len(start_ls)):
        for j in range(len(end_ls)):
         
            dist = start_ls[i][1] + dist_ls[start_ls[i][0]][end_ls[j][0]] + end_ls[j][1]
            fast_path_ls.append((dist, path_ls[start_ls[i][0]][end_ls[j][0]]))

    fast_path_ls = sorted(fast_path_ls, key=lambda x: x[0])

    final_wp_path = fast_path_ls[0][1]
    final_dist = fast_path_ls[0][0]
    print("final solution", fast_path_ls[0])

    point_ls = []
    point_ls.append(start_pos)

    # plot fastest path
    for wp in final_wp_path:
        point_ls.append(wp_ls[wp]['pos'])

    point_ls.append(end_pos)

    point_ar = np.array(point_ls)
    print("point ar wp", point_ar)
    plt.plot(point_ar[:, 0], point_ar[:,1], color='orange',  linewidth=4)

plt.plot(start_pos[0], start_pos[1], marker='o' ,color = 'green', markersize=10)
plt.plot(end_pos[0], end_pos[1], marker='o' , color='red', markersize=10)

# plot all waypoints 
for dict in wp_ls:
    x, y = dict['pos']
    plt.scatter(x, y, color='blue', marker='o', label='Dots', s=100)
    # plt.text(x, y, dict['id'], fontsize=12, ha='right')
    conns = dict['conns']
    for key in conns:
        conn_x, conn_y = wp_ls[key]['pos']
        # plt.plot([x, conn_x], [y, conn_y], color='lightcoral', alpha=0.4)

plt.imshow(img)
plt.show()
