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
    def __init__(self, scene_path, seed, num_blocks, min_distance, min_goal_distance, stepping=True, save_path='./simulation_data', num_run=1, online=False, model = None, THR_base=0.1):
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

    def init_scene(self):
        self.sim.loadScene(self.scene_path)
        self.sim.startSimulation()
        self.pioneer = self.sim.getObject('/PioneerP3DX')
        self.camera = self.sim.getObject('/PioneerP3DX/Vision_sensor')
        self.left_motor = self.sim.getObject("/PioneerP3DX/leftMotor")
        self.right_motor = self.sim.getObject("/PioneerP3DX/rightMotor")
        self.template_handle = self.sim.getObject("/ConcretBlockTemplate")
        self.floor_handle = self.sim.getObject("/ResizableFloor_5_25")
        self.goal_handle = self.sim.getObject("/goal")
        self.sensors = self.init_sensors()


    def run(self):
        for run in range(self.num_run):
            self.single_run(run)

    def single_run(self, run):
        self.init_scene()
        preds = []
        pioneer_position, goal_position, block_positions = self.place_objects(run)
        frame_id = 0
        self.sim.setJointTargetVelocity(self.left_motor, 0)
        self.sim.setJointTargetVelocity(self.right_motor, 0)
        self.sim.setStepping(self.stepping)
        time.sleep(2)
        log_filename = os.path.join(self.save_path, f"simulation_log_run{run + 1}.csv")
        self.init_csv(log_filename, block_positions)
        
        frame_dir = os.path.join(self.save_path, f"frames_run{run + 1}")
        os.makedirs(frame_dir, exist_ok=True)
      
        PF = PotentialField(K_att=1.0, K_rep=1.0, THR=0.6, KP_rot=1.0, KP_fwd=2.0)

        pioneer_positions = []
        distance_to_goal = float('inf')
        while distance_to_goal >.1:            
            pioneer_position = self.sim.getObjectPosition(self.pioneer, -1)[:-1]
            pioneer_orientation = self.sim.getObjectOrientation(self.pioneer, -1)[2]
            pioneer_positions.append(pioneer_position)

            v_l, v_r, distance_to_goal = PF.p_field(pioneer_position, goal_position, block_positions, pioneer_orientation) 
    
            self.sim.setJointTargetVelocity(self.left_motor,v_l)
            self.sim.setJointTargetVelocity(self.right_motor, v_r)
            self.sim.step()
            frame_id += 1
            img, filename = self.save_frame(frame_dir, frame_id)
            
            sensors_vals = self.read_proximity_sensors(self.sensors)  
            self.update_csv(log_filename, run, frame_id, pioneer_position, goal_position, v_l, v_r, sensors_vals, block_positions)
            if self.online:
                img = (utils.preprocess_2828(utils.acquire_image(filename)))
                X_img_tensor = torch.tensor(img.flatten()/255.0, dtype=torch.float32)
                # print(f"shape img : {X_img_tensor.shape}")
                X_sensor_tensor = torch.tensor(sensors_vals, dtype=torch.float32)
                # print(f"shape sensor : {X_sensor_tensor.shape}")
                X_total = torch.concat((X_img_tensor, X_sensor_tensor), axis=-1)
                error = self.model.compute_reconstruction_error(X_total.unsqueeze(0))
                preds.append(PF.mod_thr(self.THR_base,error))        
        self.sim.stopSimulation()
        time.sleep(2)

        # self.plot_ano(pioneer_positions, block_positions, goal_position, preds)

    def predict_model(self):
        pass


    def init_csv(self, log_filename, block_positions):
        header = ['run', 'time', 'pioneer_x', 'pioneer_y', 'goal_x', 'goal_y', 'left_velocity', 'right_velocity'] + [f'sensor_{i}' for i in range(16)] + [f"'obstacle_{i}_x" for i in range(len(block_positions))] + [f"obstacle_{i}_y" for i in range(len(block_positions))]
        with open(log_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)    

    def update_csv(self, log_filename, run, frame_id, pioneer_position, goal_position, v_l, v_r, sensors_vals, block_positions):
        black_positions_array = np.array(block_positions)
        print()
        row = [run + 1, frame_id, pioneer_position[0], pioneer_position[1], goal_position[0], goal_position[1], v_l.item(), v_r.item()] + sensors_vals + list(black_positions_array[:,0]) + list(black_positions_array[:,1])
        with open(log_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def init_sensors(self):
        self.sensors = []
        for i_s in range(16):
            self.sensors.append(self.sim.getObject(f'/PioneerP3DX/ultrasonicSensor[{i_s}]'))
        return self.sensors

    def read_proximity_sensors(self, sensors):
        sensors_vals = []
        for sensor in sensors:
            detected, distance, *_ = self.sim.readProximitySensor(sensor)
            sensors_vals.append(distance if detected else 0.0)
        return sensors_vals

    def save_frame(self, frame_dir, frame_id):
        image, resolution = self.sim.getVisionSensorImg(self.camera)
        img = np.array(self.sim.unpackUInt8Table(image), dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
        img = np.flipud(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        filename = os.path.join(frame_dir, f"frame_{frame_id:04d}.png")
        cv2.imwrite(filename, img)
        return img, filename


    def place_objects(self, run):

        np.random.seed(self.seed[run])
        floor_pos = self.sim.getObjectPosition(self.floor_handle, -1)

        xmin = floor_pos[0] - 6.5
        xmax = floor_pos[0] + 6.5 
        ymin = floor_pos[1] - 6.5 
        ymax = floor_pos[1] + 6.5 
        z = floor_pos[2] + 0.5

        placed_positions = []

        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymin, ymax)
        robot_pos = (x, y)
        self.sim.setObjectPosition(self.pioneer, -1, [x, y, z])
        placed_positions.append((x, y))
        
        attempts = 0
        max_attempts = 1000
        while attempts < max_attempts:
            x = np.random.uniform(xmin, xmax)
            y = np.random.uniform(ymin, ymax)
            dist_to_robot = np.hypot(x - robot_pos[0], y - robot_pos[1])
            if self.is_far_enough(x, y, self.min_distance, placed_positions) and dist_to_robot >= self.min_goal_distance:
                self.sim.setObjectPosition(self.goal_handle, -1, [x, y, -1])
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
                new_block = self.sim.copyPasteObjects([self.template_handle], 0)[0]
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
    

    def plot_tot(self, pioneer_positions, block_positions, goal_position):

        plt.figure(figsize=(10, 10))
        plt.title("Traiettorie Pioneer con goal e blocchi")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.grid(True)
        plt.axis('equal')
        block_size = 1.0
        plt.xticks(np.arange(8.5, -8.5, -1))
        plt.yticks(np.arange(8.5, -8.5, -1))
        plt.plot(pioneer_positions[0][0], pioneer_positions[0][1] , 'ro', markersize=12, label='Start')

        x_path = [pos[0] for pos in pioneer_positions[1:]]
        y_path = [pos[1] for pos in pioneer_positions[1:]]
        plt.plot(x_path, y_path, '-', label='Path')
        # Goal
        plt.plot(goal_position[0], goal_position[1] , 'go', markersize=12, label='Goal')

        for i in range(len(block_positions)):
                x, y = block_positions[i]
                rect = plt.Rectangle((x - block_size/2, y - block_size/2), block_size, block_size,
                                linewidth=1, edgecolor='r', facecolor='none')
                plt.gca().add_patch(rect)
                plt.plot(x, y, 'r.', markersize=5)
        plt.grid(False)
        plt.legend()
        plt.show()

    def plot_ano(self, pioneer_positions, block_positions, goal_position, preds):

            plt.figure(figsize=(10, 10))
            plt.title("Traiettorie Pioneer con goal e blocchi")
            plt.xlabel("X [m]")
            plt.ylabel("Y [m]")
            plt.grid(True)
            plt.axis('equal')
            block_size = 1.0
            plt.xticks(np.arange(8.5, -8.5, -1))
            plt.yticks(np.arange(8.5, -8.5, -1))
            plt.plot(pioneer_positions[0][0], pioneer_positions[1][0] , 'ro', markersize=12, label='Start')

            x_path = pioneer_positions[0][:]
            y_path = pioneer_positions[1][:]
            plt.plot(x_path, y_path, '-', label='Path')
            anom_idx = np.where(preds[:] == 1)[0]  # attenzione a sincronizzare con x_path/y_path
            plt.plot(x_path[anom_idx], y_path[anom_idx], 'ro', label='Anomalie')
            
            # Goal
            plt.plot(goal_position[0][0], goal_position[1][0] , 'go', markersize=12, label='Goal')
            


            for i in range(len(block_positions)):
                    x, y = block_positions[i]
                    rect = plt.Rectangle((x - block_size/2, y - block_size/2), block_size, block_size,
                                    linewidth=1, edgecolor='r', facecolor='none')
                    plt.gca().add_patch(rect)
                    plt.plot(x, y, 'r.', markersize=5)
            plt.grid(False)
            plt.legend()
            plt.show()


class PotentialField:
    def __init__(self, K_att=1.0, K_rep=100.0, THR=1.0,  KP_rot=1.0, KP_fwd=1.0):
        self.K_att = K_att
        self.K_rep = K_rep
        self.THR = THR
        self.KP_rot = KP_rot
        self.KP_fwd = KP_fwd

    def mod_thr(self, base_value, error):
        print(f"THR: {base_value}, Error: {error}")
        if error > base_value:
            return 1
        else:
            return 0

        # self.THR = base_value + error

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
        fx_total = fx_attr + fx_rep_total
        fy_total = fy_attr + fy_rep_total
        rot_speed_raw, angle_diff = self.compute_angle(fx_total, fy_total, pioneer_orientation)
        rot_speed = rot_speed_raw 

        # Speed avanti
        fwd_speed = self.KP_fwd * np.exp(-2 * abs(angle_diff))

        L = 0.4
        v_l = fwd_speed - rot_speed * L / 2
        v_r = fwd_speed + rot_speed * L / 2
        return v_l, v_r, distance_to_goal   