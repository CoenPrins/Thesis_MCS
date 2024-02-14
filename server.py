import os
import base64
from mesa.visualization.ModularVisualization import ModularServer, VisualizationElement
from mesa.visualization.modules import TextElement
from mesa.visualization.UserParam import UserSettableParameter

# from physical_agent import IndoorModel
from physical_agent import IndoorModel_fp
import pandas as pd 
import numpy as np

df_kp_cor = pd.read_csv('C:\\Users\\coen_\\OneDrive\\Documents\\ma_thesis\\MAIN\\data\\environment\\kp_pos_color.csv')
hard_dx = 235
hard_dy = 76


kp_cor = []
for index, row in df_kp_cor.iterrows():
    kp_cor.append([(row['x_pos'] - hard_dx, row['y_pos'] - hard_dy), row['sim']])

ap_key  = np.load('C:\\Users\\coen_\\OneDrive\\Documents\\ma_thesis\\files\\iBeacon_data\\summer_2_floor_04_08\\points_wifi_2.npy')
ap_cor = []
for cor in ap_key:
    ap_cor.append((cor[0] - hard_dx, cor[1] - hard_dy))

class IndoorVisualCanvas(VisualizationElement):
    local_includes = ["json\\simple_continuous_canvas.js"]

    def __init__(self, portrayal_method, canvas_height=500, canvas_width=500, bg_path=None):
        self.portrayal_method = portrayal_method
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        new_element = f"new Simple_Continuous_Module({self.canvas_width}, {self.canvas_height}, '{bg_path}')"
        self.js_code = "elements.push(" + new_element + ");"

    def transform_xy(self, model, pos):
        x, y = pos
        x = (x - model.space.x_min) / (model.space.x_max - model.space.x_min)
        y = (y - model.space.y_min) / (model.space.y_max - model.space.y_min)
        return x, y

    def render(self, model):
        space_state = []

        
        # for now

        # acces points
        

        for pos in ap_cor:
            # ap_portrayal = {'Shape': 'circle', 'r': 5, 'Filled': 'true', 'Color': 'purple'}
            ap_portrayal = {'Shape': 'circle', 'r': 5, 'Filled': 'true', 'Color': 'purple', 'text': "Lorem ipsum", 'text_color':'red'}
            # ap_text_portrayal = {'Shape': 'text', 'text': 'hallo meneertje',"text_size":12, 'text_color': 'black'}

        
            x, y = self.transform_xy(model, pos)
            ap_portrayal["x"], ap_portrayal["y"] = x, y
            # ap_text_portrayal["x"], ap_text_portrayal["y"] = x, y 
            space_state.append(ap_portrayal)
            # space_state.append(ap_text_portrayal)

            
    
        # key points
        # color_ls = ["020344","061553","0A2761","0D3970","114B7E","155E8D","19709B","1D82AA","2094B8","24A6C7","28B8D5"]
        color_ls = ['#28B8D5',"#24A6C7","#2094B8","#1D82AA","#19709B","#155E8D","#114B7E","#0D3970","#0A2761","#061553","#020344"]
        for item in kp_cor:
            color_strength = int(item[1])
      
            kp_portrayal = {'Shape': 'circle', 'r': 2, 'Filled': 'true', 'Color':color_ls[color_strength]}
            kp_portrayal["x"], kp_portrayal["y"] = self.transform_xy(model, item[0])
            space_state.append(kp_portrayal)
            


        for obj in model.schedule.agents:
            
            
            # targets
            tg = obj.get_points_to_show()
            tm_portrayal = {'Shape': 'circle', 'r': 2, 'Filled': 'true', 'Color': 'magenta'}
            tm_portrayal["x"], tm_portrayal["y"] = self.transform_xy(model, tg['next_target'])
            space_state.append(tm_portrayal)
            tf_portrayal = {'Shape': 'circle', 'r': 2, 'Filled': 'true', 'Color': 'brown'}
            tf_portrayal["x"], tf_portrayal["y"] = self.transform_xy(model, tg['final_target'])
            space_state.append(tf_portrayal)


            if obj.PF_N:
                print("hallo meneer")
                # prtcl_portrayal = {'Shape': 'circle', 'r': 2, 'Filled': 'true', 'Color': 'Orange'}
                # prtcl_["x"], kp_portrayal["y"] = self.transform_xy(model, item[0])
               
                
                
                for element in obj.PF_ar:

                    P_size = 1.5 *element[2] *len(obj.PF_ar)
                    if P_size > 6:
                        P_size = 6
                    
                    # P_portrayal = {'Shape': 'circle', 'r': 6, 'Filled': 'true', 'Color': 'orange'}
                    P_portrayal = {'Shape': 'circle', 'r':P_size, 'Filled': 'true', 'Color': 'orange'}
                    pos = (element[0], element[1])
                    # print("small particelpos", pos)
                    P_portrayal["x"], P_portrayal["y"] = self.transform_xy(model, pos)
                    space_state.append(P_portrayal)

                av_P_portrayal = {'Shape': 'circle', 'r': 4, 'Filled': 'true', 'Color': 'red'}
                pos = (obj.w_PF[0], obj.w_PF[1])
                av_P_portrayal["x"], av_P_portrayal["y"] = self.transform_xy(model, pos)
                print("why no weight?=", pos)
                space_state.append(av_P_portrayal)

            # agent
            portrayal = self.portrayal_method(obj)            
            portrayal["x"], portrayal["y"] = self.transform_xy(model, obj.pos)
            space_state.append(portrayal)
            # particles! 
       
        return space_state

class RunningAgentsNum(TextElement):
    def __init__(self):
        pass

    def render(self, model):
        return f'Running agents: {model.moving_agents_num}'

def agent_portrayal(a):
    if a.real_path == True:
        cl = '#00FF00' if a.is_moving else '#0000FF'
    else:
        cl = '#556B2F' if a.is_moving else '#000000'
    return {'Shape': 'circle', 'r': 4, 'Filled': 'true', 'Color': cl}

bg_path = 'data\\environment\\map_2floor_PF_bw.png'

file_path = "data\\paths\\dt_10\\raw\\lp\\ignore\\6\\gaus_normal\\path0.csv"
file_df = pd.read_csv(file_path)


file_df['time'] = pd.to_datetime(file_df['time'], format='mixed')
file_df['x'] =  file_df['x'].round().astype(int)
file_df['y'] =  file_df['y'].round().astype(int)
file_df = file_df.drop(columns=['ax', 'ay', 'bx', 'by', 'cx', 'cy'])
# file_path = "junk\\fake_og.csv"
wp_path = "data\\distance\\waypoints.pkl"
dist_path = 'data\\distance\\dist_ls.pkl'

with open(bg_path, "rb") as img_file:
    b64_string = base64.b64encode(img_file.read())

running_counter_element = RunningAgentsNum()
canvas_element = IndoorVisualCanvas(agent_portrayal, 250, 600, 'data:image/png;base64, ' + b64_string.decode('utf-8')) 


# this overides the code in physical agents designating the agents_json path! 
# Which is good actually! 

# some of the paths are the same right now! because they don't exist yet!
model_params = {
    "path": UserSettableParameter('choice', "path choice", value = "path_1", choices=["path_1", "path_2", "path_3"]),
    "og_cords" : UserSettableParameter('checkbox', "Original coords", True), 
    "RSSI": UserSettableParameter('checkbox', "RSSI", False),
    "PF": UserSettableParameter('checkbox', 'PF',False ),
    "obj_env":UserSettableParameter('checkbox', "Object_Environment", False),
    "part_env": UserSettableParameter('checkbox', "Particle_Environment", False),
    "PF_N": 300, 
    'env_map_path': bg_path,
    "RSSI_df": file_df,
    "RSSI_error": 5/0.1333,
    "dt": 5,
    "wp_path" :wp_path,
    "dist_path":dist_path
}

server = ModularServer(IndoorModel_fp, [canvas_element, running_counter_element], 'Indoor model', model_params)
# server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# def __init__(self, env_map_path, RSSI_path, path_1, path_2, path_3, og_cords, RSSI, PF_N, RSSI_error, dt):