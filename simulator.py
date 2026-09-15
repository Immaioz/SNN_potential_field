from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import csv
import os
import numpy as np
import cv2
import torch
from utils import utils


class _RobotState:
    __slots__ = ('handle', 'results', 'PF', 'frame_dir', 'log_filename',
                 'run_inference', 'THR_base', 'arrived', 'arrival_frame')

    def __init__(self, handle, results, PF, frame_dir, log_filename,
                 run_inference=False, THR_base=0.1):
        self.handle = handle
        self.results = results
        self.PF = PF
        self.frame_dir = frame_dir
        self.log_filename = log_filename
        self.run_inference = run_inference
        self.THR_base = THR_base
        self.arrived = False
        self.arrival_frame = None


class Simulator:
    def __init__(self, scene_path, seed, num_blocks, min_distance, min_goal_distance, stepping=True, save_path='./simulation_data', num_run=1, online=False, model=None, THR_base=0.1, comparison=False, three_mode=False, model_class=None, THR_base_class=0.06012838, decay_rate=0.3):
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
        self.three_mode = three_mode
        self.model_class = model_class
        self.THR_base_class = THR_base_class
        self.decay_rate = decay_rate

    def init_scene(self):
        if self.num_run == 0:
            self.sim.stopSimulation()
            while self.sim.getSimulationState() != self.sim.simulation_stopped:
                pass
            self.sim.loadScene(self.scene_path)
        self.sim.startSimulation()

        self.handle = self._init_handle('')
        self.handle['sensors'] = self.init_sensors()
        if self.comparison or self.three_mode:
            self.handle_twin = self._init_handle('_twin')
            self.handle_twin['sensors'] = self.init_sensors('twin')
        if self.three_mode:
            self.handle_class = self._init_handle('_class')
            self.handle_class['sensors'] = self.init_sensors('class')

    def _init_handle(self, suffix):
        return {
            'pioneer': self.sim.getObject(f'/PioneerP3DX{suffix}'),
            'camera': self.sim.getObject(f'/PioneerP3DX{suffix}/Vision_sensor'),
            'left_motor': self.sim.getObject(f'/PioneerP3DX{suffix}/leftMotor'),
            'right_motor': self.sim.getObject(f'/PioneerP3DX{suffix}/rightMotor'),
            'template': self.sim.getObject(f'/ConcretBlockTemplate{suffix}'),
            'floor': self.sim.getObject(f'/ResizableFloor_5_25{suffix}'),
            'goal': self.sim.getObject(f'/goal{suffix}'),
            'sensors': []
        }

    def run(self):
        self.init_scene()
        self.sim.setStepping(self.stepping)
        robots = self._setup_robots()
        num_steps = 1 if not (self.comparison or self.three_mode) else 1
        obstacle_threshold = 0.052 if self.comparison and not self.three_mode else 0.07
        self._run_loop(robots, self.num_run, num_steps=num_steps, obstacle_threshold=obstacle_threshold)
        self.sim.stopSimulation()
        time.sleep(2)
        for r in robots:
            self._finalise_results(r.results, r.arrival_frame)
        return self._pack_results(robots)

    def _setup_robots(self):
        run = self.num_run
        robots = [self._create_robot(run, '', run_inference=self.online, THR_base=self.THR_base)]
        if self.comparison or self.three_mode:
            robots.append(self._create_robot(run, '_twin', variant='twin'))
        if self.three_mode:
            robots.append(self._create_robot(run, '_class', variant='class',
                           run_inference=True, THR_base=self.THR_base_class[0]))
        return robots

    def _create_robot(self, run, suffix, variant=None, run_inference=False, THR_base=None):
        handle = self._init_handle(suffix)
        handle['sensors'] = self.init_sensors(variant or '')

        results = {
            'pioneer_positions': [], 'block_positions': [], 'goal_position': None,
            'preds': [], 'thresholds': [], 'speeds': [], 'obstacle_hit': [], 'inference_time': []
        }
        _, results['goal_position'], results['block_positions'] = self.place_objects(run, handle)

        log_filename = os.path.join(self.save_path, f"simulation_log_run{run}{suffix}.csv")
        frame_dir = os.path.join(self.save_path, f"frames_run{run}{suffix}")
        self.init_csv(log_filename, results['block_positions'])
        os.makedirs(frame_dir, exist_ok=True)

        PF = PotentialField(K_att=10.0, K_rep=5.0, THR=0.6, KP_rot=4.0, KP_fwd=2.0, rate=self.decay_rate)
        return _RobotState(handle, results, PF, frame_dir, log_filename,
                           run_inference=run_inference, THR_base=THR_base or self.THR_base)

    def _run_loop(self, robots, run, num_steps=1, obstacle_threshold=0.07):
        frame_id = 0
        while any(not r.arrived for r in robots):
            velocities = []
            for r in robots:
                pos = self.sim.getObjectPosition(r.handle["pioneer"], -1)[:-1]
                yaw = self.sim.getObjectOrientation(r.handle["pioneer"], -1)[2]
                r.results['pioneer_positions'].append(pos)

                v_l, v_r, dist, fwd = r.PF.p_field(
                    pos, r.results['goal_position'], r.results['block_positions'], yaw)
                r.results['speeds'].append(fwd)

                if not r.arrived and dist <= 0.25:
                    r.arrived = True
                if r.arrived:
                    v_l, v_r = 0.0, 0.0

                self.set_velocity(r.handle, v_l, v_r)
                velocities.append((v_l, v_r))

            for _ in range(num_steps):
                self.sim.step()
            frame_id += 1

            for r, (v_l, v_r) in zip(robots, velocities):
                if r.arrival_frame is None and r.arrived:
                    r.arrival_frame = frame_id

                self.save_frame(r.frame_dir, frame_id, r.handle)
                sensors_vals = self.read_proximity_sensors(r.handle)
                self.update_csv(r.log_filename, run, frame_id,
                                r.results['pioneer_positions'][-1],
                                r.results['goal_position'],
                                float(v_l), float(v_r),
                                sensors_vals, r.results['block_positions'])

                if r.run_inference:
                    self._run_inference(r.results, r.handle, sensors_vals, r.PF, r.THR_base)

                r.results['thresholds'].append(r.PF.THR)
                r.results['obstacle_hit'].append(
                    1 if any(0 < v < obstacle_threshold for v in sensors_vals) else 0)

    def _run_inference(self, results, handle, sensors_vals, PF, THR_base):
        X_total = self.extract_SNN_inputs(handle, sensors_vals)
        t = time.time()
        error = self.model.compute_reconstruction_error(X_total.unsqueeze(0))
        toc = time.time() - t
        if error > THR_base:
            results['preds'].append(1)
            PF.mod_thr(1)
        else:
            results['preds'].append(0)
            PF.mod_thr(0)
        results['inference_time'].append(toc)

    def _finalise_results(self, results, arrival_frame):
        results['preds'] = np.array(results['preds']) if results['preds'] else np.zeros(arrival_frame, dtype='int64')
        results['thresholds'] = np.array(results['thresholds'])
        results['speeds'] = np.array(results['speeds'])
        results['inference_time'] = np.array(results['inference_time'])
        results['arrival_frame'] = arrival_frame

    def _pack_results(self, robots):
        if self.three_mode:
            return robots[0].results, robots[1].results, robots[2].results
        elif self.comparison:
            return robots[0].results, robots[1].results
        return robots[0].results

    #### UTILS FUNCTIONS ####
    def set_velocity(self, handle, v_l, v_r):
        self.sim.setJointTargetVelocity(handle["left_motor"], v_l)
        self.sim.setJointTargetVelocity(handle["right_motor"], v_r)

    def init_run(self, twin, run, class_mode):
        suffix = '_class' if class_mode else ('_twin' if twin else '')
        variant = 'class' if class_mode else ('twin' if twin else None)
        r = self._create_robot(run, suffix, variant=variant)
        return r.results, r.frame_dir, r.log_filename, r.PF

    def extract_SNN_inputs(self, handle, sensors_vals):
        img = utils.preprocess_2828(self._read_image(handle))
        X_img_tensor = torch.tensor(img.flatten() / 255.0, dtype=torch.float32)
        X_sensor_tensor = torch.tensor(sensors_vals, dtype=torch.float32)
        # print(f"Image tensor shape: {X_img_tensor.shape}, Sensor tensor shape: {X_sensor_tensor.shape}")
        return torch.concat((X_img_tensor, X_sensor_tensor), axis=-1)

    def init_csv(self, log_filename, block_positions):
        header = (
            ['run', 'time', 'pioneer_x', 'pioneer_y', 'goal_x', 'goal_y', 'left_velocity', 'right_velocity']
            + [f'sensor_{i}' for i in range(16)]
            + [f"obstacle_{i}_x" for i in range(len(block_positions))]
            + [f"obstacle_{i}_y" for i in range(len(block_positions))]
        )
        with open(log_filename, 'w', newline='') as f:
            csv.writer(f).writerow(header)

    def update_csv(self, log_filename, run, frame_id, pioneer_position, goal_position, v_l, v_r, sensors_vals, block_positions):
        black_positions_array = np.array(block_positions)
        row = (
            [run + 1, frame_id, pioneer_position[0], pioneer_position[1],
             goal_position[0], goal_position[1], v_l, v_r]
            + sensors_vals
            + list(black_positions_array[:, 0])
            + list(black_positions_array[:, 1])
        )
        with open(log_filename, 'a', newline='') as f:
            csv.writer(f).writerow(row)

    def init_sensors(self, mode=""):
        suffix = {"twin": "_twin", "class": "_class"}.get(mode, "")
        return [self.sim.getObject(f'/PioneerP3DX{suffix}/ultrasonicSensor[{i}]') for i in range(16)]

    def read_proximity_sensors(self, handle):
        sensors_vals = []
        for sensor in handle['sensors']:
            detected, distance, *_ = self.sim.readProximitySensor(sensor)
            sensors_vals.append(distance if detected else 0.0)
        return sensors_vals

    def _read_image(self, handle):
        image, resolution = self.sim.getVisionSensorImg(handle["camera"])
        img = np.array(self.sim.unpackUInt8Table(image), dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
        img = np.flipud(img)
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def get_frame(self, handle):
        return cv2.cvtColor(self._read_image(handle), cv2.COLOR_RGB2GRAY)

    def save_frame(self, frame_dir, frame_id, handle):
        img_bgr = cv2.cvtColor(self._read_image(handle), cv2.COLOR_RGB2BGR)
        filename = os.path.join(frame_dir, f"frame_{frame_id:04d}.png")
        cv2.imwrite(filename, img_bgr)
        return img_bgr, filename

    def place_objects(self, run, handle):
        np.random.seed(self.seed[run])
        floor_pos = self.sim.getObjectPosition(handle["floor"], -1)

        xmin = floor_pos[0] - 6.5
        xmax = floor_pos[0] + 6.5
        ymin = floor_pos[1] - 6.5
        ymax = floor_pos[1] + 6.5
        z = floor_pos[2] + 0.25

        placed_positions = []

        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymin, ymax)
        robot_pos = (x, y)
        self.sim.setObjectPosition(handle["pioneer"], -1, [x, y, z])
        self.sim.setObjectOrientation(handle["pioneer"], -1, [0, 0, np.random.uniform(-np.pi, np.pi)])
        self.set_velocity(handle, 0, 0)
        placed_positions.append((x, y))

        max_attempts = 1000
        attempts = 0
        while attempts < max_attempts:
            x = np.random.uniform(xmin, xmax)
            y = np.random.uniform(ymin, ymax)
            if (self._is_far_enough(x, y, self.min_distance, placed_positions)
                    and np.hypot(x - robot_pos[0], y - robot_pos[1]) >= self.min_goal_distance):
                self.sim.setObjectPosition(handle["goal"], -1, [x, y, -1])
                placed_positions.append((x, y))
                break
            attempts += 1

        attempts = 0
        tot_blocks = 0
        while tot_blocks < self.num_blocks and attempts < max_attempts:
            x = np.random.uniform(xmin, xmax)
            y = np.random.uniform(ymin, ymax)
            if self._is_far_enough(x, y, self.min_distance, placed_positions):
                pos = [x, y, z - 0.125]
                new_block = self.sim.copyPasteObjects([handle["template"]], 0)[0]
                self.sim.setObjectPosition(new_block, -1, pos)
                self.sim.setShapeColor(new_block, None, self.sim.colorcomponent_ambient_diffuse, [1.0, 1.0, 1.0])
                self.sim.setShapeColor(new_block, None, self.sim.colorcomponent_emission, [0.3, 0.3, 0.3])
                self.sim.setObjectAlias(new_block, f"ConcretBlock#{tot_blocks}", True)
                self.sim.setIntProperty(new_block, 'layer', 1)
                placed_positions.append((x, y))
                tot_blocks += 1
            attempts += 1
        return placed_positions[0], placed_positions[1], placed_positions[2:]

    def _is_far_enough(self, x, y, min_dist, placed_positions):
        return all(np.hypot(x - px, y - py) >= min_dist for px, py in placed_positions)


class PotentialField:
    def __init__(self, K_att=1.0, K_rep=100.0, THR=1.0, KP_rot=1.0, KP_fwd=1.0, rate=0.03):
        self.K_att = K_att
        self.K_rep = K_rep
        self.THR = THR
        self.KP_rot = KP_rot
        self.KP_fwd = KP_fwd
        self.increase_rate = 1 + rate
        self.decay_rate = 1 - rate
        self.obstacle_half_size = 0.25
        self.robot_radius = 0.2

    def mod_thr(self, mode):
        if mode == 1:
            self.THR = min(self.THR * self.increase_rate, 2.0)
        elif mode == 0:
            self.THR = max(0.2, self.THR * self.decay_rate)
        return self.THR

    def attractive_force(self, robot_x, robot_y, goal_x, goal_y):
        dx = goal_x - robot_x
        dy = goal_y - robot_y
        return self.K_att * dx, self.K_att * dy

    def repulsive_force(self, robot_x, robot_y, obstacle_x, obstacle_y):
        h = self.obstacle_half_size + self.robot_radius
        cx = np.clip(robot_x, obstacle_x - h, obstacle_x + h)
        cy = np.clip(robot_y, obstacle_y - h, obstacle_y + h)
        dx = robot_x - cx
        dy = robot_y - cy
        dist_to_obstacle = np.sqrt(dx**2 + dy**2)

        if dist_to_obstacle < self.THR:
            factor = self.K_rep * (1.0 / dist_to_obstacle - 1.0 / self.THR) * (1.0 / (dist_to_obstacle**2))
            return factor * dx / dist_to_obstacle, factor * dy / dist_to_obstacle
        return 0.0, 0.0

    def compute_angle(self, fx, fy, pioneer_orientation):
        angle = np.arctan2(fy, fx)
        angle_diff = (angle - pioneer_orientation + np.pi) % (2 * np.pi) - np.pi
        rot_speed_raw = 0.0 if abs(angle_diff) < 0.1 else self.KP_rot * angle_diff
        return rot_speed_raw, angle_diff

    def p_field(self, pioneer_position, goal_position, block_positions, pioneer_orientation):
        repulsion_active = False
        dx = goal_position[0] - pioneer_position[0]
        dy = goal_position[1] - pioneer_position[1]
        distance_to_goal = np.sqrt(dx**2 + dy**2)
        fx_attr, fy_attr = self.attractive_force(pioneer_position[0], pioneer_position[1], goal_position[0], goal_position[1])
        fx_rep_total = 0.0
        fy_rep_total = 0.0
        for ox, oy in block_positions:
            fx_rep, fy_rep = self.repulsive_force(pioneer_position[0], pioneer_position[1], ox, oy)
            fx_rep_total += fx_rep
            fy_rep_total += fy_rep
            if fx_rep != 0.0 or fy_rep != 0.0:
                repulsion_active = True
        fx_total = fx_attr + fx_rep_total
        fy_total = fy_attr + fy_rep_total

        rot_speed_raw, angle_diff = self.compute_angle(fx_total, fy_total, pioneer_orientation)

        fwd_speed = self.KP_fwd * np.exp(-2 * abs(angle_diff))
        if not repulsion_active:
            fwd_speed *= 5

        L = 0.4
        v_l = fwd_speed - rot_speed_raw * L / 2
        v_r = fwd_speed + rot_speed_raw * L / 2
        return v_l, v_r, distance_to_goal, fwd_speed
