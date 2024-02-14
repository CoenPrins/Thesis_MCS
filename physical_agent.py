import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np
import json
import re 
import pandas as pd 
from datetime import datetime
import time_og_data2
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
import pickle
import time
# import random




def gaus_p(x, mu, sigma):
    # sigma = math.sqrt(sigma)
   
    prob = (1/(sigma * np.sqrt(2 * np.pi)))*np.exp(-0.5*(((x - mu)/sigma)**2))
    return prob



class AgentEnvironmentMap:
    def __init__(self, bw_image_path, wp_ls, dist_ls, dist_per_pix=1.0): 
        self.wp_ls = wp_ls
        self.dist_ls = dist_ls   
        self.img = image.imread(bw_image_path)
        self.shape = (self.img.shape[0], self.img.shape[1])
        self.dist_per_pix = dist_per_pix
        self.max_x = self.shape[1]*dist_per_pix - 1
        self.max_y = self.shape[0]*dist_per_pix - 1
        # print("self max", self.max_x)
        # print("self max_y", self.max_y)
        self.wall_mask = np.zeros(self.shape, dtype=bool)

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.wall_mask[i,j] = self.img[i,j,0] < 0.1

    def is_wall(self, x, y):
        i = int(y/self.dist_per_pix)
        j = int(x/self.dist_per_pix)
        return self.wall_mask[i, j]
    
        
    def cross_wall(self, x1, y1, x2, y2):
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

            if self.is_wall(x1, y1):
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
    

    def generate_PF_ar(self, N, pos, radius, obj_env):
        PF_ls = []
        N_nowall = 0
        while N_nowall < N:
            # create random angle and radius
            theta = np.random.uniform(0, 2*np.pi)
            r = np.sqrt(np.random.uniform(0, 1)) * radius

            # Move x and y to random location within circle
            x = pos[0] + r * np.cos(theta)
            y = pos[1] + r * np.sin(theta)
        
            # add x, y and weight (which is all equal)
            if 0 <= x <= self.max_x and 0 <= y <= self.max_y:
                if not self.is_wall(x, y):
                    PF_ls.append([int(x), int(y), 1/N])
                    N_nowall += 1
    #         print(f'#{self.unique_id} is moving to ({new_x}, {new_y}) fowards waypoint #{self.next_waypoint_index}')            
    #         self.model.space.move_agent(self, (new_x, new_y))
        w_total_sum = 0
        for Particle in PF_ls:
            # ENVIRONMENT TOGGLE HERE
            if obj_env:
                p_dist = self.dijkstra((Particle[0], Particle[1]), (pos[0], pos[1]))
            else:
                p_dist = np.sqrt(abs(Particle[0] - pos[0])**2 + abs(Particle[1] - pos[1])**2)
            
                
            p_prob = gaus_p(x=p_dist, mu=0, sigma=radius)
            Particle[2] = p_prob
            w_total_sum += p_prob
            
            
        PF_ar = np.array(PF_ls)
        PF_ar[:, 2] /= w_total_sum
        
        # print("PF_AR=", PF_ar)

        return PF_ar
    

    def dijkstra(self, start_pos, end_pos):

        start_ls = []
        end_ls = []
        direct = False
        if not self.cross_wall(start_pos[0], start_pos[1], end_pos[0], end_pos[1]):
                dist =  np.sqrt(abs(start_pos[0] - end_pos[0])**2 + abs(start_pos[1] - end_pos[1])**2)
                final_dist = dist
                direct = True
        else:
            # if end point is inside a wall, you ignore walls for line to waypoints
            if  self.is_wall(end_pos[0], end_pos[1]):
                for dict in self.wp_ls:

                    if not self.cross_wall(dict['pos'][0], dict['pos'][1], start_pos[0], start_pos[1]):
                        dist =  np.sqrt(abs(dict['pos'][0] - start_pos[0])**2 + abs(dict['pos'][1] - start_pos[1])**2)
                        start_ls.append((dict['id'], dist))

                        
                    
                    dist =  np.sqrt(abs(dict['pos'][0] - end_pos[0])**2 + abs(dict['pos'][1] - end_pos[1])**2)
                    end_ls.append((dict['id'], dist))
                
                end_ls = sorted(end_ls, key=lambda x: x[1])
        
                end_ls = [end_ls[0]]

            # find waypoints with unbroken lines to start and endpoint
            else:
                for dict in self.wp_ls:
                    

                    if not self.cross_wall(dict['pos'][0], dict['pos'][1], start_pos[0], start_pos[1]):
                        dist =  np.sqrt(abs(dict['pos'][0] - start_pos[0])**2 + abs(dict['pos'][1] - start_pos[1])**2)
                        start_ls.append((dict['id'], dist))

                        
                    if not self.cross_wall(dict['pos'][0], dict['pos'][1], end_pos[0], end_pos[1]):
                        dist =  np.sqrt(abs(dict['pos'][0] - end_pos[0])**2 + abs(dict['pos'][1] - end_pos[1])**2)
                        end_ls.append((dict['id'], dist))

        if not direct:
            # find the shortest dijkstra path that connects to end and start point
            fast_path_ls = []
            for i in range(len(start_ls)):
                for j in range(len(end_ls)):
                
                    dist = start_ls[i][1] + self.dist_ls[start_ls[i][0]][end_ls[j][0]] + end_ls[j][1]
                    fast_path_ls.append(dist)

            fast_path_ls = sorted(fast_path_ls)
   
            final_dist = fast_path_ls[0]
            

        return final_dist

    def to_ascii1(self):
        return '\n'.join([''.join(['#' if m else '.' for m in ln]) for ln in self.wall_mask])    

# If PF_N is your check
    # ONLY ask for self_PF_N, self_PF_ar and self.w_PF if you first checked if PF_N

class PhysicalAgent(Agent):
    def __init__(self, unique_id, model,real_path, PF, PF_N, waypoints):
        super().__init__(unique_id, model)
        
        self.is_moving = True
        self.real_path = real_path

        
        self.waypoints = waypoints
        print("waypoints", waypoints)
        # print("waypoint_type", type(waypoints)) # list
        # print("waypoint[0] type", type(waypoints[0])) #tuple 
        # print("waypoint[0][0] type", type(waypoints[0][0])) #integer
        # self.model.space.move_agent(self, self.waypoints[0])
        self.next_waypoint_index = 1
        self.is_moving = True

        if PF:
            self.PF_N = PF_N
            print("succes I print", self.PF_N)
            # print("test_pos=",test_pos)
            # dit hele lijstje zit in het verdomhoekje!!!!
            test_pos = self.waypoints[0]
            # print("test_pos=",test_pos)
            
            self.error = model.error
            self.dt = self.model.dt
            self.obj_env = model.obj_env
            self.part_env = model.part_env
            # average walking speed in coordinates
            self.v = 10.7
           
            self.PF_ar = self.model.env_map.generate_PF_ar(PF_N, test_pos, self.error, self.obj_env)
            
            x = self.PF_ar[:, 0]
            y = self.PF_ar[:, 1]
            weight = self.PF_ar[:, 2]
            # print("sum(weight)", sum(weight))

            self.w_PF = (sum(x*weight), sum(y*weight))
        else:
            self.PF_N = False
            
    


    def get_points_to_show(self):
        return {'agent': self.pos, 'final_target': self.waypoints[-1], 'next_target': self.waypoints[self.next_waypoint_index]}


    def step(self):


        # regardless of wether the agent has filter or not the actual position remains the path that has been given, 
        # agents with a self.PF_N value simply also store a PF_ar of all particles pos and weight and a position of the average 
        # position (wihch is all the positions times their weight!)
        new_x = self.waypoints[self.next_waypoint_index][0]
        new_y = self.waypoints[self.next_waypoint_index][1]


        # ENVIRONMENT TOGGLE! 

        # Here you want to give each particle 5 options 
        # Every option you pick a spot in a sphere of roughly max distance 
        # See if it's euclidean is less than that

        # if not try again, if 5 times fail, that particle gets probabilty of 0. 

        # check if probability of zero doesn't break anything, in terms of weighting. 

        if self.PF_N:

            new_PF_ls = []
            weights = self.PF_ar[:, -1]

            while len(new_PF_ls) < self.PF_N:
                chosen_index = np.random.choice(len(self.PF_ar), p=weights)
                random_P = np.copy(self.PF_ar[chosen_index])
                theta = np.random.uniform(0, 2*np.pi)
                

            # Move x and y to random location within circle
                for i in range(1):
                    valid_point = False

                    # this seems wildly inaccurate.... 
                    r = np.sqrt(np.random.uniform(0, 1)) * self.v * self.dt 
                    new_P_x = random_P[0] + r * np.cos(theta)
                    new_P_y = random_P[1] + r * np.sin(theta)


                    
                    # print("new_P_x", new_P_x)
                    # print("new_P_Y", new_P_y)
                    if 0 <= new_P_x <= self.model.env_map.max_x and 0 <= new_P_y <= self.model.env_map.max_y:
        
                        if not self.model.env_map.is_wall(int(new_P_x), int(new_P_y)):

                            if self.part_env:
                                # print("let's try some dijkstra!!!")
                                # print("new_p_xy", new_P_x, new_P_y)
                                # print("random_pxy", random_P[0], random_P[1])
                                dijkstra_d = self.model.env_map.dijkstra((int(new_P_x), int(new_P_y)), (random_P[0], random_P[1]))
                                # print("dijkstra_d calculate", dijkstra_d)
                                if dijkstra_d <= r:
                                    new_PF_ls.append([int(new_P_x), int(new_P_y), 1/self.PF_N])
                                    valid_point = True
                                     
                            else:
                                new_PF_ls.append([int(new_P_x), int(new_P_y), 1/self.PF_N])
                                valid_point = True


                    if valid_point:
                        break


            w_total_sum = 0
            for Particle in new_PF_ls:
                
                if self.obj_env:
                    # print("jojojojo")
                    p_dist = self.model.env_map.dijkstra((Particle[0], Particle[1]), (new_x, new_y))
                else:
                    p_dist = np.sqrt(abs(Particle[0] - new_x)**2 + abs(Particle[1] - new_y)**2)
                p_prob = gaus_p(x=p_dist, mu=0, sigma=self.error)
                Particle[2] = p_prob
                w_total_sum += p_prob
            
            self.PF_ar = np.array(new_PF_ls)
            # print("length of self.PF_ar", len(self.PF_ar))
            self.PF_ar[:, 2] /= w_total_sum
            # self.av_pf_x = np.mean(x_pf)
            # self.av_pf_y = np.mean(y_pf)

            weight = self.PF_ar[:, 2]
            # print("sum(weight) stepcheck", sum(weight))


            
            # # alter x coordinate 
            # self.PF_ar[:, 0] += 1

            # # alter y coordinate 
            # self.PF_ar[:, 1] +=1
            

            # w_pf = (altered x coordinates * weight, altered y coordinates* weight) 
            
            self.w_PF = (sum(self.PF_ar[:, 0]*self.PF_ar[:, 2]), sum(self.PF_ar[:, 1]*self.PF_ar[:, 2]))
            
            # print(self.w_PF)
            


       
        
        if self.next_waypoint_index < len(self.waypoints) - 1:
                    self.next_waypoint_index += 1
                    print("waypoint index!", self.next_waypoint_index)
        else:
            self.is_moving = False

        self.model.space.move_agent(self, (new_x, new_y))

    

    
class IndoorModel_fp(Model):

    """
    This code runs for 1 agents that chooses the json paths randomly super hard work jk

    This code will no longer run with Json, but rather just read in csv panda files. 
    This code will have usersetter parmeters for paths & settings. For now it can just have the one setting! 
    the order doesn't seem to matter! 
    """
    def __init__(self, env_map_path, RSSI_df, path, og_cords, RSSI, PF, PF_N, obj_env, part_env, RSSI_error, dt, wp_path, dist_path):
        self.hard_dx = 235
        self.hard_dy = 76
        self.path = env_map_path

        with open(wp_path, 'rb') as fp: 
            wp_ls = pickle.load(fp)


        with open(dist_path, 'rb') as fp:
            dist_ls = pickle.load(fp)  
        self.env_map = AgentEnvironmentMap(env_map_path, wp_ls, dist_ls)
        self.space = ContinuousSpace(self.env_map.max_x, self.env_map.max_y, False)
        self.schedule = RandomActivation(self)
        self.error = RSSI_error
        self.PF = PF
        self.obj_env = obj_env
        self.part_env = part_env
        self.PF_N = PF_N
        self.dt = dt


       

        
        # realy ugly quickfix because i simply duplicated an index loop fordf two distinct agents classes resulting in duplicates of 
        #unique identifier i, therefore implanted a simple counter 
        counter = 0


        if og_cords:

            if path == "path_1":
                waypoints = []
                path =  time_og_data2.og_path1
                for pos in path:
                    waypoints.append((pos[0]- self.hard_dx, pos[1] - self.hard_dy))
                a = PhysicalAgent(counter, self, real_path= True, PF = self.PF, PF_N= self.PF_N,  waypoints=waypoints) 
                self.schedule.add(a)            
                self.space.place_agent(a, waypoints[0])
        
                # a.reset_waypoints(waypoints)
                counter += 1
            if path == "path_2":
               
                waypoints = []
                path =  time_og_data2.og_path2
                for pos in path:
                    waypoints.append((pos[0]- self.hard_dx, pos[1] - self.hard_dy))
                a = PhysicalAgent(counter, self, real_path= True, PF = self.PF, PF_N= False, waypoints=waypoints)
                self.schedule.add(a)            
                self.space.place_agent(a, waypoints[0])
        
                # a.reset_waypoints(waypoints)
                counter += 1
            if path == "path_3":
                waypoints = []
                path =  time_og_data2.og_path3
                for pos in path:
                    waypoints.append((pos[0]- self.hard_dx, pos[1] - self.hard_dy))
                a = PhysicalAgent(counter, self, real_path= True, PF = self.PF, PF_N= self.PF_N, waypoints = waypoints)
                self.schedule.add(a)            
                self.space.place_agent(a, waypoints[0])
        
                # a.reset_waypoints(waypoints)    
                counter += 1   
        
        if RSSI:

            df = RSSI_df
            # df['time'] = pd.to_datetime(df['time'], format='mixed')
            if path == "path_1":
                RSSI_1_df = df[(df['time'] >= time_og_data2.start_time1) & (df['time'] <= time_og_data2.end_time1)].copy()
                RSSI_1_df['x'] = RSSI_1_df['x'].round().astype(int)
                RSSI_1_df['y'] = RSSI_1_df['y'].round().astype(int)
                counter = self.fp_agent_create_rssi(RSSI_1_df, counter)
            if path == "path_2":
                RSSI_2_df = df[(df['time'] >= time_og_data2.start_time2) & (df['time'] <= time_og_data2.end_time2)].copy()
                RSSI_2_df['x'] = RSSI_2_df['x'].round().astype(int)
                RSSI_2_df['y'] = RSSI_2_df['y'].round().astype(int)
                counter = self.fp_agent_create_rssi(RSSI_2_df, counter)
            if path == "path_3":
                RSSI_3_df = df[(df['time'] >= time_og_data2.start_time3) & (df['time'] <= time_og_data2.end_time3)].copy()
                RSSI_3_df['x'] = RSSI_3_df['x'].round().astype(int)
                RSSI_3_df['y'] = RSSI_3_df['y'].round().astype(int)
                counter = self.fp_agent_create_rssi(RSSI_3_df, counter)
            # print("RSSI_3_df", RSSI_3_df)


           
        # this needs to change! Try to understand this code! 
        self.data_collector = DataCollector(
            model_reporters = {'moving_agents_num': 'moving_agents_num'}, agent_reporters={'x': lambda a: a.pos[0], 'y': lambda a: a.pos[1], "x_pf": lambda a:  a.w_PF[0] if a.PF_N is not False else None, "y_pf": lambda a:  a.w_PF[1] if a.PF_N is not False else None})
        self.moving_agents_num = 0
        self.running = True
        self.data_collector.collect(self)




    def fp_agent_create_rssi(self, df, counter):
        # this is only good for the agent that follows the waypoint

        waypoints_loc = [(int(row['x'] - self.hard_dx), int(row['y'] - self.hard_dy   )) for _, row in df.iterrows()]
        # waypoints_a = [(int(row['ax'] - self.hard_dx), int(row['ay']) - self.hard_dy) for _, row in df.iterrows()]
        # waypoints_b = [(int(row['bx'] - self.hard_dx), int(row['by']) - self.hard_dy) for _, row in df.iterrows()]
        # waypoints_c = [(int(row['cx'] - self.hard_dx), int(row['cy']) - self.hard_dy) for _, row in df.iterrows()]
        
        # waypoints_loc = []

            
        a = PhysicalAgent(counter, self, real_path= True, PF= self.PF, PF_N= self.PF_N, waypoints=waypoints_loc)
        self.schedule.add(a)            
        self.space.place_agent(a, waypoints_loc[0])
        #very unclear syntax regarding setting waypoints here...
        # This sets the waypoints as features from the agent!
            
        # a.reset_waypoints(waypoints_loc)
        counter += 1 
        
        return counter
    

    def fp_agent_create_av(self, df, counter):

        #need to tweak immediately! 
        waypoints_av = [(int(row['av_x'] - self.hard_dx), int(row['av_y']) - self.hard_dy) for _, row in df.iterrows()]
        waypoints_0 = [(int(row['dt0_x'] - self.hard_dx), int(row['dt0_y']) - self.hard_dy) for _, row in df.iterrows()]
        waypoints_1 = [(int(row['dt1_x'] - self.hard_dx), int(row['dt1_y']) - self.hard_dy) for _, row in df.iterrows()]
        waypoints_2 = [(int(row['dt2_x'] - self.hard_dx), int(row['dt2_y']) - self.hard_dy) for _, row in df.iterrows()]
        waypoints_3 = [(int(row['dt3_x'] - self.hard_dx), int(row['dt3_y']) - self.hard_dy) for _, row in df.iterrows()]
        waypoints_4 = [(int(row['dt4_x'] - self.hard_dx), int(row['dt4_y']) - self.hard_dy) for _, row in df.iterrows()]

        
            
        av = PhysicalAgent(counter, self, real_path= True, PF = self.PF, PF_N = self.PF_N, waypoints=waypoints_av)
        self.schedule.add(av)            
        self.space.place_agent(av, waypoints_av[0])          
        # av.reset_waypoints(waypoints_av)
        counter += 1 
            
        non_av = [ waypoints_0, waypoints_1, waypoints_2, waypoints_3, waypoints_4]

        # for waypoint in non_av:

        #     a = PhysicalAgent(counter, self, real_path= False)
        #     self.schedule.add(a)            
        #     self.space.place_agent(a, waypoint[0])
                   
        #     a.reset_waypoints(waypoint)
        #     counter += 1 

        return counter
    
    def run_model(self, n_step):
        """Run model for n_steps"""
        for i in range(n_step):
            self.step()
        
        

    def step(self):
        self.schedule.step()
        self.data_collector.collect(self)
        self.moving_agents_num = sum([a.is_moving for a in self.schedule.agents])
        self.running = self.moving_agents_num > 0

    def plot_explicitly(self):
        plt.imshow(self.env_map.img)
        for a in self.schedule.agents:
            plt.plot(a.pos[0], a.pos[1], 'bo')
        # plt.plot(self.target[0], self.target[1], 'r+')

