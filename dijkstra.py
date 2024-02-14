
"""
notes:

Currenlty you don't choose fastes distance but rather suboptimal distance. 



Notes: by pre computaing the solutions you go from roughly 1.5-2 seconds per 1000 particles step to 0.0
"""

    
import pickle 
import time
import copy

with open(f"data\\distance\\waypoints.pkl", 'rb') as fp: 
        wp_ls = pickle.load(fp)


def dijkstra(wp_ls, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # psuedo code 
    print()
    print(f"path from {start} to {end}")
    # this is just a letter, that corresponds to a list of waypoints
    start_wp = start 
    end_wp = end 
    wp_dk_ls = copy.deepcopy(wp_ls)

    # Initialize both open and closed list
    open_ls = []
    closed_ls = []
    open_ls.append(start_wp)

    # Add the start node
    end_score = float('inf')
    # Loop until you find the end
    while len(open_ls) > 0:

        # Get the current node
        
        

        current_wp = open_ls.pop(0)

        closed_ls.append(current_wp)

        # Found the goal now you look back just like your bfa
        if current_wp == end_wp:
            if wp_dk_ls[current_wp]['g'] < end_score:
                end_score = wp_dk_ls[current_wp]['g'] 
        print()
        # print("open",open_ls)
        # print("closed",closed_ls)
        # print("curent_wp", current_wp)
        # print("end_score", end_score)
        # print("looking for chilldren")

       
        children = wp_dk_ls[current_wp]['conns']

        # print("chilldren", children.keys())
        for conn_wp, dist in children.items():
            print(conn_wp, dist)
            if conn_wp in closed_ls:
                pass
                # print('closed list! no search')
            else:
         
                dlt_i = False
                child_g = wp_dk_ls[current_wp]['g'] + dist
                # print("child_g", child_g)
                if child_g > end_score:
                    pass
                    # print("longer than solution")
                else:
                    better_alt = False
                    for i in range(len(open_ls)):
                        if int(open_ls[i]) == int(conn_wp):
                            if child_g < wp_dk_ls[open_ls[i]]['g']:
                                
                                # print('we poppin and loccking yo')
                                open_ls.pop(i)
                                # better_alt = True
                                break
                            else:
                                better_alt = True
                                
                                break
                    
                    if better_alt:
                        pass
                        # print("longer than alternative path won't be explored")
                    else:
                        if len(open_ls) > 0:   
                            inserted = False
                            for j in range(len(open_ls)):
                                if child_g < wp_dk_ls[open_ls[j]]['g']:
                                    open_ls.insert(j, conn_wp)
                                    wp_dk_ls[conn_wp]['g'] = child_g
                                    wp_dk_ls[conn_wp]['parent'] = current_wp
                                    # print('added at j',j, "values", conn_wp, child_g)
                                    # print(open_ls)
                                    inserted = True
                                    break   
                            if not inserted:
                                # print('hoihoi')
                                open_ls.append(conn_wp)
                                wp_dk_ls[conn_wp]['g'] = child_g
                                wp_dk_ls[conn_wp]['parent'] = current_wp
                                # print('added at end', conn_wp, child_g)
                                # print(open_ls)
                                
                        else:
                            open_ls.append(conn_wp)
                            wp_dk_ls[conn_wp]['g'] = child_g
                            wp_dk_ls[conn_wp]['parent'] = current_wp
                            # print('added len0', conn_wp, child_g)
                            # print(open_ls)
    path = []
    path.append(end_wp)
    parent = wp_dk_ls[end_wp]['parent']
    total_dist = wp_dk_ls[end_wp]['g']
    # print("end_wp", end_wp)
    # print("first parent", parent)
    while parent is not None:
        path.append(parent)
        
        parent = wp_dk_ls[parent]['parent']
        # print("path", path)
        # print('parent', parent)
        
    #     backtrack = wp_dk_ls[parent]['id']
    path = path[::-1]

    print("returned path", path)
    print("returned_dist", total_dist)
    return path, total_dist




# we make two dictionaries, one containing the paths and one with the total distance. 




# path, dist = dijkstra(wp_ls, 18, 30)
# path, dist = dijkstra(wp_ls, 5, 2)
# path, dist = dijkstra(wp_ls, 3, 2)
# path, dist = dijkstra(wp_ls, 5, 2)
# path, dist = dijkstra(wp_ls, 6, 2)
# path, dist = dijkstra(wp_ls, 7, 2)
# path, dist = dijkstra(wp_ls, 8, 2)
# print("returned path", path)
# print("distance total", dist)


start_time = time.time()

dist_ls = []
path_ls = []


for i in range(len(wp_ls)):
    dct_dist = {'id': i}
    dct_path = {'id': i}
    for j in range(len(wp_ls)):
        path, dist = dijkstra(wp_ls, i, j)
        dct_dist[j] = dist
        dct_path[j] = path
    print(" done", i)
    dist_ls.append(dct_dist)
    path_ls.append(dct_path)

print("--- %s seconds ---" % (time.time() - start_time))


print("dist_ls", dist_ls)
print("path_ls", path_ls)

dict_directory  = "data\\distance"

with open(f'{dict_directory}\\dist_ls.pkl', 'wb') as fp:
                pickle.dump(dist_ls, fp)

with open(f'{dict_directory}\\path_ls.pkl', 'wb') as fp:
                pickle.dump(path_ls, fp)


