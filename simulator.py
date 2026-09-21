from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import csv
import os
import numpy as np
import cv2
import torch
from utils import utils
from potential_field import PotentialField
from adaptive_apf_potential_field import AdaptiveAPFPotentialField
from adaptive_tangential_apf_potential_field import AdaptiveTangentialAPFPotentialField


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
    def __init__(self, scene_path, seed, num_blocks, min_distance, min_goal_distance, stepping=True, save_path='./simulation_data', num_run=1, online=False, model=None, THR_base=0.1, comparison=False, three_mode=False, model_class=None, THR_base_class=0.06012838, decay_rate=0.3, decay_factor=None, pf_class=None, class_pf_only=False, moving_obstacles=False, obstacle_step=0.05, obstacle_turn=0.15, obstacle_every=1, obstacle_goal_margin=2.5):
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
        self.decay_factor = decay_factor
        # pf_class can be: None -> PotentialField for all robots,
        # a single class -> used everywhere, or a dict
        # {'base': cls, 'twin': cls, 'class': cls} to select a different
        # PF class for each of the three elements (e.g. three_mode).
        self.pf_class = pf_class
        # class_pf_only: in three_mode the 'class' robot does not run AE
        # inference, it only uses the selected PF class (like the 'twin').
        self.class_pf_only = class_pf_only
        self.moving_obstacles = moving_obstacles
        self.obstacle_step = obstacle_step
        self.obstacle_turn = obstacle_turn
        self.obstacle_every = obstacle_every
        self.obstacle_goal_margin = obstacle_goal_margin
        self._obstacle_dirs = {}
        self._obstacle_timers = {}
        self._obstacle_moves = {}
        self._dir_change_every = 10  # hardcoded: new preferred direction every 10 moves

    def _resolve_pf_class(self, variant):
        pc = self.pf_class
        if isinstance(pc, dict):
            return pc.get(variant or 'base') or PotentialField
        return pc or PotentialField

    def init_scene(self):
        if self.num_run == 0:
            self.sim.stopSimulation()
            while self.sim.getSimulationState() != self.sim.simulation_stopped:
                pass
            self.sim.loadScene(self.scene_path)
        self.sim.setBoolProperty(self.sim.handle_scene, 'ode.quickStepEnabled', False)
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
            'sensors': [],
            'blocks': []
        }

    def run(self):
        self.init_scene()
        self.sim.setStepping(self.stepping)
        robots = self._setup_robots()
        num_steps = 1 if not (self.comparison or self.three_mode) else 1
        obstacle_threshold = 0.5 if self.comparison and not self.three_mode else 0.5
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
                           run_inference=not self.class_pf_only,
                           THR_base=self.THR_base_class[0] if not self.class_pf_only else None))
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

        PF = self._resolve_pf_class(variant)(K_att=10.0, K_rep=3.0, THR=0.6, KP_rot=4.0, KP_fwd=2.0, rate=self.decay_rate, decay_factor=self.decay_factor)
        return _RobotState(handle, results, PF, frame_dir, log_filename,
                           run_inference=run_inference, THR_base=THR_base or self.THR_base)

    def _move_obstacles(self, r):
        if not self.moving_obstacles:
            return
        floor_pos = self.sim.getObjectPosition(r.handle["floor"], -1)
        z = floor_pos[2] + 0.125
        xmin, xmax = floor_pos[0] - 13.0, floor_pos[0] + 13.0
        ymin, ymax = floor_pos[1] - 13.0, floor_pos[1] + 13.0
        for i, block in enumerate(r.handle['blocks']):
            key = id(block)
            steps = self._obstacle_timers.get(key, 0) + 1
            if steps < self.obstacle_every:
                self._obstacle_timers[key] = steps
                continue
            self._obstacle_timers[key] = 0

            goal = r.results['goal_position']
            robot_pos = self.sim.getObjectPosition(r.handle["pioneer"], -1)[:-1]
            margin = self.obstacle_goal_margin
            robot_margin = margin - 1

            def _bounce(nx, ny, d):
                if nx < xmin or nx > xmax:
                    d = np.array([-d[0], d[1]])
                    nx = bx + d[0] * self.obstacle_step
                if ny < ymin or ny > ymax:
                    d = np.array([d[0], -d[1]])
                    ny = by + d[1] * self.obstacle_step
                return d, np.clip(nx, xmin, xmax), np.clip(ny, ymin, ymax)

            # Obstacles move only every obstacle_every frames. Each block keeps
            # a persistent preferred direction (initialized once at random) and
            # drifts with small random turns, never entering the goal's margin
            # nor the safety zone around the robot (goal margin - 1).
            # Every _dir_change_every moves the preferred direction is reset.
            moves = self._obstacle_moves.get(key, 0)
            if key not in self._obstacle_dirs or (moves > 0 and moves % self._dir_change_every == 0):
                angle = np.random.uniform(0, 2 * np.pi)
                self._obstacle_dirs[key] = np.array([np.cos(angle), np.sin(angle)])

            bx, by = r.results['block_positions'][i]
            accepted = None
            for _ in range(10):
                dirn = self._obstacle_dirs[key]
                turn = np.random.uniform(-self.obstacle_turn, self.obstacle_turn)
                ct, st = np.cos(turn), np.sin(turn)
                dirn = np.array([ct * dirn[0] - st * dirn[1],
                                 st * dirn[0] + ct * dirn[1]])
                dirn /= np.hypot(*dirn)
                self._obstacle_dirs[key] = dirn

                nx = bx + dirn[0] * self.obstacle_step
                ny = by + dirn[1] * self.obstacle_step
                dirn, nx, ny = _bounce(nx, ny, dirn)
                if (np.hypot(nx - goal[0], ny - goal[1]) > margin
                        and np.hypot(nx - robot_pos[0], ny - robot_pos[1]) > robot_margin):
                    accepted = (dirn, nx, ny)
                    break
            if accepted is None:
                # Flee instead of stopping: scan candidate directions and pick
                # the one that maximizes the distance from the robot, while
                # keeping the goal margin where possible.
                angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
                best = None
                for angle in angles:
                    d = np.array([np.cos(angle), np.sin(angle)])
                    tx = bx + d[0] * self.obstacle_step
                    ty = by + d[1] * self.obstacle_step
                    d, tx, ty = _bounce(tx, ty, d)
                    g_dist = np.hypot(tx - goal[0], ty - goal[1])
                    if g_dist <= margin:
                        continue
                    if best is None:
                        best = (d, tx, ty)
                        best_r = np.hypot(tx - robot_pos[0], ty - robot_pos[1])
                    else:
                        r_dist = np.hypot(tx - robot_pos[0], ty - robot_pos[1])
                        if r_dist > best_r:
                            best = (d, tx, ty)
                            best_r = r_dist
                if best is None:
                    # Boxed near the goal: still flee the robot.
                    best = None
                    best_r = -1.0
                    for angle in angles:
                        d = np.array([np.cos(angle), np.sin(angle)])
                        tx = bx + d[0] * self.obstacle_step
                        ty = by + d[1] * self.obstacle_step
                        d, tx, ty = _bounce(tx, ty, d)
                        r_dist = np.hypot(tx - robot_pos[0], ty - robot_pos[1])
                        if r_dist > best_r:
                            best = (d, tx, ty)
                            best_r = r_dist
                if best is None:
                    continue
                accepted = (best[0], best[1], best[2])
            dirn, nx, ny = accepted
            self._obstacle_dirs[key] = dirn
            self._obstacle_moves[key] = moves + 1
            self.sim.setObjectPosition(block, -1, [nx, ny, z])
            r.results['block_positions'][i] = (float(nx), float(ny))

    def _run_loop(self, robots, run, num_steps=1, obstacle_threshold=0.07):
        frame_id = 0
        while any(not r.arrived for r in robots):
            velocities = []
            for r in robots:
                self._move_obstacles(r)
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
                det = min((v for v in sensors_vals if 0 < v < obstacle_threshold), default=None)
                hit = 0
                if det is not None and len(r.results['pioneer_positions']) >= 2:
                    disp = np.array(pos) - np.array(r.results['pioneer_positions'][-2])
                    if np.hypot(*disp) > 1e-9:
                        patch = r.PF.obstacle_half_size + r.PF.robot_radius
                        for bx, by in r.results['block_positions']:
                            to_obs = np.array([bx - pos[0], by - pos[1]])
                            if np.hypot(*to_obs) - patch <= obstacle_threshold + 0.1 and np.dot(disp, to_obs) > 0:
                                hit = 1
                                break
                r.results['obstacle_hit'].append(hit)

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

    #### UTIL FUNCTIONS ####
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
        img_bgr = self._read_image(handle)
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
                handle['blocks'].append(new_block)
                placed_positions.append((x, y))
                tot_blocks += 1
            attempts += 1
        return placed_positions[0], placed_positions[1], placed_positions[2:]

    def _is_far_enough(self, x, y, min_dist, placed_positions):
        return all(np.hypot(x - px, y - py) >= min_dist for px, py in placed_positions)
