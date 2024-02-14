from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np

import pickle
import time
import random
bg_path = 'data\\environment\\map_2floor_PF_bw.png'


"""
TO DO:

Make lines draw to all possible waypoints!
How would drawing lines be best? you can also check them
while they are being drawn? 
Test which way would be quicker. (I think there is actually no way checking during 
drawing is NOT quicker)

Then you need to implement. 

You actually need to do a lot.... (FUCK)


"""
with open(f"data\\distance\\waypoints.pkl", 'rb') as fp: 
        wp_ls = pickle.load(fp)


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



def random_pos(max_x, max_y):
    
    wall = True
    while wall == True:
        x = random.randrange(0, max_x)
        y = random.randrange(0, max_y)
        wall = is_wall(x, y)
    
    return(x, y)
    

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
        # print(x,y)
        if is_wall(x, y):
            
            return True  # Line crosses a wall

    return False  # Line does not cross a wall
    

"""
showcasing the line function for two points!
"""
# dots_x = [100, 200]
# dots_y = [50, 150]
# dots_x = [80, 150 ]
# dots_y = [80, 150]



# line = draw_line(dots_x[0], dots_y[0], dots_x[1], dots_y[1])
# line = np.array(line)

# # Plot the image

# clr_line = "pink"

# # Plot the dots on top of the image
# plt.scatter(dots_x, dots_y, color='blue', marker='o', label='Dots')

# if cross_wall(dots_x[0], dots_y[0], dots_x[1], dots_y[1]):
#     clr_line = "red"
# else:
#     clr_line = "green"

# plt.plot(line[:,0], line[:,1], color=clr_line)


# plt.imshow(img)

# plt.show()

"""
plotting red all possible non wall options!
For all of these we will 
"""


# x = np.arange(0, max_x + 1)
# y = np.arange(0, max_y + 1)
# xx, yy = np.meshgrid(x, y)

# # Plot only non-wall coordinates
# plt.scatter(xx[~wall_mask], yy[~wall_mask], marker='s', s=1, color='red')

# plt.imshow(img)
# plt.show()



""""
drawing lines to waypoints, currenlty red lines are commented out, also 
random_point isn't plotted

 """
# random_point = random_pos(max_x, max_y)
random_point = (316, 155)
print(random_point)


pos_ls = []
for dict in wp_ls:
    pos_ls.append(dict['pos'])


print(pos_ls)
start_time = time.time()
for pos in pos_ls:
    line = draw_line(pos[0], pos[1], random_point[0], random_point[1])
    line = np.array(line)
    plt.plot(random_point[0], random_point[1], marker='o' ,color = 'green', markersize = 15)
        
    if cross_wall(pos[0], pos[1], random_point[0], random_point[1]):
        clr_line = "red"
        # plt.plot(line[:,0], line[:,1], color=clr_line)
    else:
        clr_line = "orange"
        plt.plot(line[:,0], line[:,1], color=clr_line, linewidth=6)
    

print("--- %s seconds ---" % (time.time() - start_time))        



for dict in wp_ls:
    x, y = dict['pos']
    plt.scatter(x, y, color='blue', marker='o', label='Dots', s=180)
    # plt.text(x, y, dict['id'], fontsize=12, ha='right')
    conns = dict['conns']
    for key in conns:
        conn_x, conn_y = wp_ls[key]['pos']
        # plt.plot([x, conn_x], [y, conn_y], color='lightcoral', alpha=0.4)
plt.imshow(img)
plt.show()


""""
I plotted all the disonants from all possible points, 
now you know you always have a line!

 """



# x = np.arange(0, max_x + 1)
# y = np.arange(0, max_y + 1)


# pos_ls = []
# for dict in wp_ls:
#     pos_ls.append(dict['pos'])

# invalid_pos =[]
# for i in x:
#     for j in y:
#         line = False
#         if not is_wall(i, j):
#             for pos in pos_ls:
#                 if not cross_wall(i, j,pos[0],pos[1]):
#                     line = True
#             if line == False:
#                 invalid_pos.append((i, j))

#invalid pos now shows zero! woop woop
# print("invalid_pos", invalid_pos)

# invalid_pos = np.array(invalid_pos)
# plt.scatter(invalid_pos[:, 0], invalid_pos[:,1], s=1)


# plt.imshow(img)
# plt.show()
