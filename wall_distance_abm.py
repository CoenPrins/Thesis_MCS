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

def dijkstra(start_pos, end_pos):


    start_ls = []
    end_ls = []
    direct = False

    if not cross_wall(start_pos[0], start_pos[1], end_pos[0], end_pos[1]):
            dist =  np.sqrt(abs(start_pos[0] - end_pos[0])**2 + abs(start_pos[1] - end_pos[1])**2)
            final_dist = dist
            direct = True
    else:
        # find waypoints with unbroken lines to start and endpoint
        for dict in wp_ls:
            

            if not cross_wall(dict['pos'][0], dict['pos'][1], start_pos[0], start_pos[1]):
                dist =  np.sqrt(abs(dict['pos'][0] - start_pos[0])**2 + abs(dict['pos'][1] - start_pos[1])**2)
                start_ls.append((dict['id'], dist))

                
            if not cross_wall(dict['pos'][0], dict['pos'][1], end_pos[0], end_pos[1]):
                dist =  np.sqrt(abs(dict['pos'][0] - end_pos[0])**2 + abs(dict['pos'][1] - end_pos[1])**2)
                end_ls.append((dict['id'], dist))

    if not direct:
        # find the shortest dijkstra path that connects to end and start point
        fast_path_ls = []
        for i in range(len(start_ls)):
            for j in range(len(end_ls)):
            
                dist = start_ls[i][1] + dist_ls[start_ls[i][0]][end_ls[j][0]] + end_ls[j][1]
                fast_path_ls.append(dist)

        fast_path_ls = sorted(fast_path_ls)
        final_dist = fast_path_ls[0]
        #return final distance

    return final_dist
        
start_time = time.time()

# start_pos =  random_pos(max_x, max_y)
# end_pos = random_pos(max_x, max_y)

# start_pos = (122, 148)
# end_pos = (121, 168)

start_pos = (122, 148)
end_pos = (141, 180)
test = dijkstra(start_pos, end_pos)
end_time = time.time()
time_elapsed = end_time - start_time
print("time elapsed=", time_elapsed)