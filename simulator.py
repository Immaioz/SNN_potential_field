from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import csv
import os
import numpy as np
import cv2
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from snnae import SAE
from utils import utils


class Simulator:
    def __init__(self, scene_path, seed, num_blocks, min_distance, min_goal_distance, stepping=True, save_path='./simulation_data', num_run=1, online=False, model = None, THR_base=0.1, comparison=False):
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        self.scene_path = scene_path
        self.num_run = num_run
        self.save_path = save_path
        self.seed = seed
        self.num_blocks = num_blocks
        self.min_distance = min_distance
        self.min_goal_distance = min_goal_distance
        self.stepping = stepping
        self.online = online
        self.model = model
        self.THR_base = THR_base
        self.comparison = comparison 

    def init_scene(self):
        self.sim.loadScene(self.scene_path)
        self.sim.startSimulation()

        self.handle = {
            'pioneer': self.sim.getObject('/PioneerP3DX'),
            'camera': self.sim.getObject('/PioneerP3DX/Vision_sensor'),
            'left_motor': self.sim.getObject("/PioneerP3DX/leftMotor"),
            'right_motor': self.sim.getObject("/PioneerP3DX/rightMotor"),
            'template': self.sim.getObject("/ConcretBlockTemplate"),
            'floor': self.sim.getObject("/ResizableFloor_5_25"),
            'goal': self.sim.getObject("/goal"),
            'sensors': []   
        }
        self.handle['sensors'] = self.init_sensors()
        if self.comparison:
            print("Initializing twin handles for comparison...")
            self.handle_twin = {
                'pioneer': self.sim.getObject('/PioneerP3DX_twin'),
                'camera': self.sim.getObject('/PioneerP3DX_twin/Vision_sensor'),
                'left_motor': self.sim.getObject("/PioneerP3DX_twin/leftMotor"),
                'right_motor': self.sim.getObject("/PioneerP3DX_twin/rightMotor"),
                'template': self.sim.getObject("/ConcretBlockTemplate_twin"),
                'floor': self.sim.getObject("/ResizableFloor_5_25_twin"),
                'goal': self.sim.getObject("/goal_twin"),
                "sensors": []
            }
            self.handle_twin['sensors'] = self.init_sensors(twin=True)


        # self.pioneer = self.handle['pioneer']
        # self.camera = self.handle['camera']
        # self.left_motor = self.handle['left_motor']
        # self.right_motor = self.handle['right_motor']
        # self.template_handle = self.handle['template']
        # self.floor_handle = self.handle['floor']
        # self.goal_handle = self.handle['goal']
        # self.sensors = self.init_sensors()
            
        


    def run(self):
        if self.comparison:
            run_results, run_results_twin = self.comparison_run(self.num_run)
            return run_results, run_results_twin
        else:
            run_results = self.single_run(self.num_run, twin=False)
            return run_results

    def single_run(self, run, twin=False):
        # self.init_scene()
        self.init_scene()
        self.sim.setStepping(self.stepping)
        handle = self.handle if not twin else self.handle_twin
        results_dict, frame_dir, log_filename, PF = self.init_run(twin=twin, run=run)

        frame_id = 0   
        distance_to_goal = float('inf')

        while distance_to_goal >.25:            
            pioneer_position = self.sim.getObjectPosition(handle["pioneer"], -1)[:-1]
            pioneer_orientation = self.sim.getObjectOrientation(handle["pioneer"], -1)[2]
            results_dict['pioneer_positions'].append(pioneer_position)

            v_l, v_r, distance_to_goal, fwd_speed = PF.p_field(pioneer_position, results_dict['goal_position'], results_dict['block_positions'], pioneer_orientation) 
            
            results_dict['speeds'].append(fwd_speed)
            self.set_velocity(handle, v_l, v_r)

            self.sim.step()
            frame_id += 1
            _, _ = self.save_frame(frame_dir, frame_id, handle)
            
            sensors_vals = self.read_proximity_sensors(handle)  
            self.update_csv(log_filename, run, frame_id, pioneer_position, results_dict['goal_position'], v_l, v_r, sensors_vals, results_dict['block_positions'])
            
            if self.online:
                X_total = self.extract_SNN_inputs(handle, sensors_vals)
                error = self.model.compute_reconstruction_error(X_total.unsqueeze(0))

                if error > self.THR_base:
                    results_dict['preds'].append(1)
                    PF.mod_thr(1) #increase THR
                else:
                    results_dict['preds'].append(0)
                    PF.mod_thr(0) #decrease THR

            results_dict['thresholds'].append(PF.THR)
            self.sim.step()
        self.sim.stopSimulation()
        time.sleep(2)
        
        results_dict['preds'] = np.array(results_dict['preds'])
        results_dict['thresholds'] = np.array(results_dict['thresholds'])
        results_dict['speeds'] = np.array(results_dict['speeds'])
        
        return results_dict

    def comparison_run(self, run):
        self.init_scene()
        self.sim.setStepping(self.stepping)
        results_dict, frame_dir, log_filename, PF = self.init_run(twin=False, run=run)
        results_dict_twin, frame_dir_t, log_filename_t, PF_twin = self.init_run(twin=True, run=run)

        frame_id = 0
        dist = float('inf')
        dist_t = float('inf')
        arrived = False
        arrived_twin = False

        while not arrived or not arrived_twin:

            # --- ROBOT A ---
            pioneer_position = (self.sim.getObjectPosition(self.handle["pioneer"], -1)[:-1])
            results_dict['pioneer_positions'].append(pioneer_position)
            yaw = self.sim.getObjectOrientation(self.handle["pioneer"], -1)[2]
            v_l, v_r, dist, fwd_speed = PF.p_field(pioneer_position, results_dict['goal_position'], results_dict['block_positions'], yaw)
            results_dict['speeds'].append(fwd_speed)

            if (not arrived) and dist <= 0.25:
                arrived = True
                arrival_frame = frame_id  
                v_l, v_r = 0.0, 0.0       

            if arrived:
                v_l, v_r = 0.0, 0.0   

            # --- ROBOT TWIN ---
            pioneer_position_twin = (self.sim.getObjectPosition(self.handle_twin["pioneer"], -1)[:-1])
            results_dict_twin["pioneer_positions"].append(pioneer_position_twin)
            yaw_t = self.sim.getObjectOrientation(self.handle_twin["pioneer"], -1)[2]
            v_l_t, v_r_t, dist_t, fwd_speed_t = PF_twin.p_field(pioneer_position_twin, results_dict_twin['goal_position'], results_dict_twin['block_positions'], yaw_t)
            results_dict_twin['speeds'].append(fwd_speed_t)


            if (not arrived_twin) and dist_t <= 0.25:
                arrived_twin = True
                arrival_frame_twin = frame_id
                v_l_t, v_r_t = 0.0, 0.0

            if arrived_twin:
                v_l_t, v_r_t = 0.0, 0.0

            self.set_velocity(self.handle, v_l, v_r)
            self.set_velocity(self.handle_twin, v_l_t, v_r_t)
            frame_id += 1

            self.save_frame(frame_dir, frame_id, self.handle)
            self.save_frame(frame_dir_t, frame_id, self.handle_twin)

            sensors_vals = self.read_proximity_sensors(self.handle)
            if not arrived:
                v_l = v_l.item()
                v_r = v_r.item()
            self.update_csv(log_filename, run, frame_id, pioneer_position, results_dict['goal_position'], v_l, v_r, sensors_vals, results_dict['block_positions'])

            sensors_vals_t = self.read_proximity_sensors(self.handle_twin)  
            if not arrived_twin:
                v_l_t = v_l_t.item()
                v_r_t = v_r_t.item()
            self.update_csv(log_filename_t, run, frame_id, pioneer_position_twin, results_dict_twin['goal_position'], v_l_t, v_r_t, sensors_vals_t, results_dict_twin['block_positions'])        
            
            if self.online:
                X_total = self.extract_SNN_inputs(self.handle, sensors_vals)
                error = self.model.compute_reconstruction_error(X_total.unsqueeze(0))

                if error > self.THR_base:
                    results_dict['preds'].append(1)
                    PF.mod_thr(1) #increase THR
                else:
                    results_dict['preds'].append(0)
                    PF.mod_thr(0) #decrease THR

            results_dict['thresholds'].append(PF.THR)
            results_dict_twin['thresholds'].append(PF_twin.THR)
            self.sim.step()
        self.sim.stopSimulation()
        time.sleep(2)
        
        results_dict['preds'] = np.array(results_dict['preds'])
        results_dict['thresholds'] = np.array(results_dict['thresholds'])
        results_dict['speeds'] = np.array(results_dict['speeds'])
        results_dict["arrival_frame"] = arrival_frame

        results_dict_twin['preds'] = np.array(results_dict_twin['preds'])
        results_dict_twin['thresholds'] = np.array(results_dict_twin['thresholds'])
        results_dict_twin['speeds'] = np.array(results_dict_twin['speeds'])
        results_dict_twin["arrival_frame"] = arrival_frame_twin

        return results_dict, results_dict_twin


    #### UTILS FUNCTIONS ####   
    def set_velocity(self, handle, v_l, v_r):
        self.sim.setJointTargetVelocity(handle["left_motor"],v_l)
        self.sim.setJointTargetVelocity(handle["right_motor"], v_r)

    def init_run(self, twin, run):
        handle = self.handle if not twin else self.handle_twin
        results_dict = {
            'pioneer_positions': [],
            'block_positions': [],
            'goal_position': None,
            'preds': [],
            'thresholds': [],
            'speeds': []
        }
        _, results_dict['goal_position'], results_dict['block_positions'] = self.place_objects(run, handle)
        if twin:
            log_filename = os.path.join(self.save_path, f"simulation_log_run{run + 1}_twin.csv")
            frame_dir = os.path.join(self.save_path, f"frames_run{run + 1}_twin")
        else:
            log_filename =  os.path.join(self.save_path, f"simulation_log_run{run + 1}.csv")
            frame_dir = os.path.join(self.save_path, f"frames_run{run + 1}")
        self.init_csv(log_filename, results_dict['block_positions'])
        os.makedirs(frame_dir, exist_ok=True)
        PF = PotentialField(K_att=5.0, K_rep=5.0, THR=0.6, KP_rot=4.0, KP_fwd=2.0)       
        return results_dict, frame_dir, log_filename, PF
    
    def extract_SNN_inputs(self, handle, sensors_vals):
        img = utils.preprocess_2828(self.get_frame(handle))
        X_img_tensor = torch.tensor(img.flatten()/255.0, dtype=torch.float32)
        X_sensor_tensor = torch.tensor(sensors_vals, dtype=torch.float32)
        X_total = torch.concat((X_img_tensor, X_sensor_tensor), axis=-1)
        return X_total

    def init_csv(self, log_filename, block_positions):
        header = ['run', 'time', 'pioneer_x', 'pioneer_y', 'goal_x', 'goal_y', 'left_velocity', 'right_velocity'] + [f'sensor_{i}' for i in range(16)] + [f"'obstacle_{i}_x" for i in range(len(block_positions))] + [f"obstacle_{i}_y" for i in range(len(block_positions))]
        with open(log_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)    

    def update_csv(self, log_filename, run, frame_id, pioneer_position, goal_position, v_l, v_r, sensors_vals, block_positions):
        black_positions_array = np.array(block_positions)
        row = [run + 1, frame_id, pioneer_position[0], pioneer_position[1], goal_position[0], goal_position[1], v_l, v_r] + sensors_vals + list(black_positions_array[:,0]) + list(black_positions_array[:,1])
        with open(log_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def init_sensors(self, twin=False):
        self.sensors = []
        for i_s in range(16):
            self.sensors.append(self.sim.getObject(f'/PioneerP3DX{"" if not twin else "_twin"}/ultrasonicSensor[{i_s}]'))
        return self.sensors

    def read_proximity_sensors(self, handle):
        sensors_vals = []
        for sensor in handle['sensors']:
            detected, distance, *_ = self.sim.readProximitySensor(sensor)
            sensors_vals.append(distance if detected else 0.0)
        return sensors_vals

    def get_frame(self, handle):
        image, resolution = self.sim.getVisionSensorImg(handle["camera"])
        img = np.array(self.sim.unpackUInt8Table(image), dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
        img = np.flipud(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def save_frame(self, frame_dir, frame_id, handle):
        image, resolution = self.sim.getVisionSensorImg(handle["camera"])
        img = np.array(self.sim.unpackUInt8Table(image), dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
        img = np.flipud(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        filename = os.path.join(frame_dir, f"frame_{frame_id:04d}.png")
        cv2.imwrite(filename, img)
        return img, filename


    def place_objects(self, run, handle):
        np.random.seed(self.seed[run])
        handle["floor"]
        floor_pos = self.sim.getObjectPosition(handle["floor"], -1)

        xmin = floor_pos[0] - 6.5
        xmax = floor_pos[0] + 6.5 
        ymin = floor_pos[1] - 6.5 
        ymax = floor_pos[1] + 6.5 
        z = floor_pos[2] + 0.5

        placed_positions = []

        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymin, ymax)
        robot_pos = (x, y)
        self.sim.setObjectPosition(handle["pioneer"], -1, [x, y, z])
        self.sim.setObjectOrientation(handle["pioneer"], -1, [0, 0, np.random.uniform(-np.pi, np.pi)])
        self.set_velocity(handle, 0, 0)
        # self.sim.setJointTargetVelocity(handle["left_motor"], 0)
        # self.sim.setJointTargetVelocity(handle["right_motor"], 0)
        placed_positions.append((x, y))
        
        attempts = 0
        max_attempts = 1000
        while attempts < max_attempts:
            x = np.random.uniform(xmin, xmax)
            y = np.random.uniform(ymin, ymax)
            dist_to_robot = np.hypot(x - robot_pos[0], y - robot_pos[1])
            if self.is_far_enough(x, y, self.min_distance, placed_positions) and dist_to_robot >= self.min_goal_distance:
                self.sim.setObjectPosition(handle["goal"], -1, [x, y, -1])
                placed_positions.append((x, y))
                break
            attempts += 1

        attempts = 0
        tot_blocks = 0
        while tot_blocks < self.num_blocks and attempts < max_attempts:
            x = np.random.uniform(xmin, xmax)
            y = np.random.uniform(ymin, ymax)
            if self.is_far_enough(x, y, self.min_distance, placed_positions):
                pos = [x, y, z - .25]
                new_block = self.sim.copyPasteObjects([handle["template"]], 0)[0]
                self.sim.setObjectPosition(new_block, -1, pos)
                self.sim.setObjectAlias(new_block, f"ConcretBlock#{tot_blocks}", True)
                self.sim.setIntProperty(new_block, 'layer', 1)
                placed_positions.append((x, y))
                tot_blocks += 1
            attempts += 1
        return placed_positions[0], placed_positions[1], placed_positions[2:]

    def is_far_enough(self, x, y, min_dist, placed_positions):
        for px, py in placed_positions:
            if np.hypot(x - px, y - py) < min_dist:
                return False
        return True
    



class PotentialField:
    def __init__(self, K_att=1.0, K_rep=100.0, THR=1.0,  KP_rot=1.0, KP_fwd=1.0):
        self.K_att = K_att
        self.K_rep = K_rep
        self.THR = THR
        self.KP_rot = KP_rot
        self.KP_fwd = KP_fwd

    def mod_thr(self, mode):
        if mode == 1:  # increase THR
            self.THR = min(self.THR* 1.03, 2.0)
            # self.THR += 0.4
        elif mode == 0:  # decrease THR
            self.THR = max(0.1, self.THR * 0.97)
        return self.THR

    def attractive_force(self, robot_x, robot_y, goal_x, goal_y):
        dx = goal_x - robot_x
        dy = goal_y - robot_y
        fx_attr = self.K_att * dx
        fy_attr = self.K_att * dy
        return fx_attr, fy_attr

    def repulsive_force(self, robot_x, robot_y, obstacle_x, obstacle_y):
        dx = robot_x - obstacle_x
        dy = robot_y - obstacle_y
        dist_to_obstacle = np.sqrt(dx**2 + dy**2)
        dist_to_obstacle -= 0.5
        if dist_to_obstacle < self.THR:
            fx_rep = self.K_rep * (1.0 / dist_to_obstacle - 1.0 / self.THR) * (1.0 / (dist_to_obstacle ** 2)) * (dx / dist_to_obstacle)
            fy_rep = self.K_rep * (1.0 / dist_to_obstacle - 1.0 / self.THR) * (1.0 / (dist_to_obstacle ** 2)) * (dy / dist_to_obstacle)
        else:
            fx_rep = 0.0
            fy_rep = 0.0
        return fx_rep, fy_rep

    def compute_angle(self, fx, fy, pioneer_orientation):
        angle = np.arctan2(fy, fx)
        current_orientation = pioneer_orientation
        angle_diff = (angle - current_orientation + np.pi) % (2 * np.pi) - np.pi
        if abs(angle_diff) < .1:
            rot_speed_raw = 0.0
        else:
            rot_speed_raw = self.KP_rot * angle_diff

        return rot_speed_raw, angle_diff

    def p_field(self, pioneer_position, goal_position, block_positions, pioneer_orientation):
        repulsion_active = False
        dx = goal_position[0] - pioneer_position[0]
        dy = goal_position[1] - pioneer_position[1]
        distance_to_goal = np.sqrt(dx**2 + dy**2)
        fx_attr, fy_attr = self.attractive_force(pioneer_position[0], pioneer_position[1], goal_position[0], goal_position[1])
        fx_rep_total = 0.0
        fy_rep_total = 0.0
        for ox, oy in block_positions :
            fx_rep, fy_rep = self.repulsive_force(pioneer_position[0], pioneer_position[1], ox, oy)
            fx_rep_total += fx_rep
            fy_rep_total += fy_rep
            if fx_rep != 0.0 or fy_rep != 0.0:
                repulsion_active = True   
        fx_total = fx_attr + fx_rep_total
        fy_total = fy_attr + fy_rep_total
        
        rot_speed_raw, angle_diff = self.compute_angle(fx_total, fy_total, pioneer_orientation)
        rot_speed = rot_speed_raw 


        # Speed avanti
        fwd_speed = self.KP_fwd * np.exp(-2 * abs(angle_diff))

        if not repulsion_active:
            fwd_speed *= 5

        L = 0.4
        v_l = fwd_speed - rot_speed * L / 2
        v_r = fwd_speed + rot_speed * L / 2
        return v_l, v_r, distance_to_goal, fwd_speed