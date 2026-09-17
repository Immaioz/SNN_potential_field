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
                continue
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


class PotentialField:
    def __init__(self, K_att=1.0, K_rep=100.0, THR=1.0, KP_rot=1.0, KP_fwd=1.0, rate=0.03, decay_factor=None):
        self.K_att = K_att
        self.K_rep = K_rep
        self.THR = THR
        self.KP_rot = KP_rot
        self.KP_fwd = KP_fwd
        self.increase_rate = 1 + rate
        self.decay_rate = 1 - rate if decay_factor is None else decay_factor
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
        dx = robot_x - obstacle_x #cx
        dy = robot_y - obstacle_y #cy
        dist_to_obstacle = np.sqrt(dx**2 + dy**2)
        dist_to_obstacle -= .35

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


class AdaptiveAPFPotentialField:
    """Comparison version: APF with adaptive coefficients (A*-APF paper).

    Implements the mechanism proposed in "Adaptive Path Planning of UAV Based
    on A* Algorithm and Artificial Potential Field Method" (Drones 2026, 10, 93):
      - normalized state variables: deg (eq. 5), demin (eq. 6),
        obstacle density rho_e (eq. 7), curvature kappa_e;
      - sigmoid map sigma(x) (eq. 8);
      - adaptive gains ka (eq. 9), kr (eq. 10), kt (eq. 11);
      - gradients: attractive (eq. 19), repulsive (eq. 22),
        trajectory adherence (eq. 23);
      - control with damping -Kd*v (eq. 25);
      - optional online self-tuning based on a cost function (eq. 12-14)
        with range projection and rate-limiting every M cycles.

    The paper kmin/kmax ranges are used as DIMENSIONLESS FACTORS that
    modulate the structural gains K_att/K_rep of the current simulation,
    so that the same dynamic regime as the current potential field is kept.
    With K_att=K_rep=1.0 the absolute values of the paper are obtained.

    Interface identical to PotentialField (p_field, mod_thr, THR,
    obstacle_half_size, robot_radius) so that it can be used as a drop-in
    in Simulator(pf_class=AdaptiveAPFPotentialField).
    """

    def __init__(self, K_att=10.0, K_rep=3.0, THR=0.6, KP_rot=4.0, KP_fwd=2.0,
                 rate=0.03, decay_factor=None,
                 ds=3.0, dsafe=0.8, gamma_rho=2.5, N_rho=6,
                 kmin_a=0.4, kmax_a=1.2,
                 kmin_r=0.6, kmax_r=3.0,
                 kmin_t=0.2, kmax_t=1.0,
                 w_ka=(1.0, 0.5, 0.3), w_kr=(1.5, 1.0, 0.5), w_kt=(1.0, 1.0),
                 Kd=0.0, vmax=1.2, dt=0.02,
                 self_tuning=False, eta=0.005, M=5,
                 wg=1.0, wo=2.0, ws=0.5, wv=0.5, kappa_max=0.5,
                 reference_path=None):
        self.K_att = K_att
        self.K_rep = K_rep
        self.THR = THR
        self.KP_rot = KP_rot
        self.KP_fwd = KP_fwd
        self.increase_rate = 1 + rate
        self.decay_rate = 1 - rate if decay_factor is None else decay_factor
        self.obstacle_half_size = 0.25
        self.robot_radius = 0.2

        self.ds = ds
        self.dsafe = dsafe
        self.gamma_rho = gamma_rho
        self.N_rho = N_rho
        self.kmin_a, self.kmax_a = kmin_a, kmax_a
        self.kmin_r, self.kmax_r = kmin_r, kmax_r
        self.kmin_t, self.kmax_t = kmin_t, kmax_t
        self.w_ka = w_ka
        self.w_kr = w_kr
        self.w_kt = w_kt
        self.Kd = Kd
        self.vmax = vmax
        self.dt = dt
        self.self_tuning = self_tuning
        self.eta = eta
        self.M = M
        self.wg, self.wo, self.ws, self.wv = wg, wo, ws, wv
        self.kappa_max = kappa_max
        self.reference_path = reference_path

        self._pos_history = []
        self._yaw_history = []
        self._gains = {'ka': None, 'kr': None, 'kt': None}
        self._frame = 0

    def mod_thr(self, mode):
        if mode == 1:
            self.THR = min(self.THR * self.increase_rate, 2.0)
        elif mode == 0:
            self.THR = max(0.2, self.THR * self.decay_rate)
        return self.THR

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _state_vars(self, robot_x, robot_y, goal_x, goal_y, block_positions):
        dg = np.hypot(goal_x - robot_x, goal_y - robot_y)
        deg = dg / (dg + self.ds)

        dists = [np.hypot(ox - robot_x, oy - robot_y) for ox, oy in block_positions]
        dmin = min(dists) if dists else self.ds
        demin = float(np.clip((self.ds - dmin) / (self.ds - self.dsafe), 0.0, 1.0))

        n_rho = sum(1 for d in dists if d <= self.gamma_rho)
        rho_e = float(np.clip(n_rho / self.N_rho, 0.0, 1.0))

        if len(self._yaw_history) >= 2:
            d_theta = abs(self._yaw_history[-1] - self._yaw_history[-2])
            if d_theta > np.pi:
                d_theta = 2 * np.pi - d_theta
        else:
            d_theta = 0.0
        kappa_e = float(np.clip(d_theta / self.kappa_max, 0.0, 1.0))
        kappa = d_theta / self.dt

        return deg, demin, rho_e, kappa_e, kappa, dg, dmin, dists

    def _adaptive_gains(self, deg, demin, rho_e, kappa_e):
        w1, w2, w3 = self.w_ka
        ka = self.kmin_a + (self.kmax_a - self.kmin_a) * self._sigmoid(w1 * deg - w2 * rho_e - w3 * kappa_e)

        b1, b2, b3 = self.w_kr
        kr = self.kmin_r + (self.kmax_r - self.kmin_r) * self._sigmoid(b1 * demin + b2 * rho_e + b3 * kappa_e)

        g1, g2 = self.w_kt
        kt = self.kmin_t + (self.kmax_t - self.kmin_t) * self._sigmoid(g1 * rho_e + g2 * demin)
        return ka, kr, kt

    def _forces(self, robot_x, robot_y, goal_x, goal_y, block_positions, ka, kr, kt,
                vx=0.0, vy=0.0):
        ka = self.K_att * ka
        kr = self.K_rep * kr
        fx = ka * (goal_x - robot_x)
        fy = ka * (goal_y - robot_y)

        repulsion_active = False
        for ox, oy in block_positions:
            dx = robot_x - ox
            dy = robot_y - oy
            de = max(np.hypot(dx, dy) - 0.35, 1e-3)
            if de < self.ds:
                factor = kr * (1.0 / de - 1.0 / self.ds) / (de ** 3)
                fx += factor * dx
                fy += factor * dy
                repulsion_active = True

        if self.reference_path is not None:
            rx, ry = self._nearest_on_path(robot_x, robot_y)
            fx += kt * (rx - robot_x)
            fy += kt * (ry - robot_y)

        fx -= self.Kd * vx
        fy -= self.Kd * vy
        return fx, fy, repulsion_active

    def _nearest_on_path(self, x, y):
        pts = np.asarray(self.reference_path, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2:
            return x, y
        best = (pts[0], (x - pts[0][0]) ** 2 + (y - pts[0][1]) ** 2)
        for i in range(len(pts) - 1):
            p1, p2 = pts[i], pts[i + 1]
            seg = p2 - p1
            t = np.clip(np.dot([x, y] - p1, seg) / max(np.dot(seg, seg), 1e-9), 0.0, 1.0)
            q = p1 + t * seg
            d2 = (x - q[0]) ** 2 + (y - q[1]) ** 2
            if d2 < best[1]:
                best = (q, d2)
        return best[0]

    def _self_tune(self, robot_x, robot_y, goal_x, goal_y, block_positions, deg,
                   rho_e, kappa_e, vx, vy, ka, kr, kt):
        def cost(x, y, vvx, vvy):
            dg = np.hypot(goal_x - x, goal_y - y)
            ob = 0.0
            for ox, oy in block_positions:
                de = max(np.hypot(x - ox, y - oy) - 0.35, 1e-3)
                ob += max(1.0 / de - 1.0 / self.ds, 0.0) ** 2
            return (self.wg * dg ** 2 + self.wo * ob + self.ws * (kappa_e * self.kappa_max) ** 2
                    + self.wv * (vvx ** 2 + vvy ** 2))

        def step(params):
            kxa, kxr, kxt = params
            fxx, fyy, _ = self._forces(robot_x, robot_y, goal_x, goal_y, block_positions,
                                       kxa, kxr, kxt, vx, vy)
            vnx = vx + fxx * self.dt
            vny = vy + fyy * self.dt
            xnp = robot_x + vnx * self.dt
            ynp = robot_y + vny * self.dt
            return cost(xnp, ynp, vnx, vny)

        lo = [self.kmin_a, self.kmin_r, self.kmin_t]
        hi = [self.kmax_a, self.kmax_r, self.kmax_t]
        new_gains = [ka, kr, kt]
        for i in range(3):
            delta = 0.01 * max(new_gains[i], 1e-3)
            p_plus = list(new_gains); p_plus[i] += delta
            p_minus = list(new_gains); p_minus[i] -= delta
            grad = (step(p_plus) - step(p_minus)) / (2 * delta)
            new_gains[i] = float(np.clip(new_gains[i] - self.eta * grad, lo[i], hi[i]))
        return new_gains

    def compute_angle(self, fx, fy, pioneer_orientation):
        angle = np.arctan2(fy, fx)
        angle_diff = (angle - pioneer_orientation + np.pi) % (2 * np.pi) - np.pi
        rot_speed_raw = 0.0 if abs(angle_diff) < 0.1 else self.KP_rot * angle_diff
        return rot_speed_raw, angle_diff

    def p_field(self, pioneer_position, goal_position, block_positions, pioneer_orientation):
        self._frame += 1
        x, y = float(pioneer_position[0]), float(pioneer_position[1])
        gx, gy = float(goal_position[0]), float(goal_position[1])

        if self._pos_history:
            dt = self.dt
            vx = (x - self._pos_history[-1][0]) / dt
            vy = (y - self._pos_history[-1][1]) / dt
        else:
            vx = vy = 0.0
        self._pos_history.append((x, y))
        self._pos_history = self._pos_history[-10:]
        self._yaw_history.append(pioneer_orientation)
        self._yaw_history = self._yaw_history[-20:]

        deg, demin, rho_e, kappa_e, kappa, dg, dmin, dists = self._state_vars(
            x, y, gx, gy, block_positions)

        self._gains['ka'], self._gains['kr'], self._gains['kt'] = self._adaptive_gains(
            deg, demin, rho_e, kappa_e)

        if self.self_tuning and self._frame % self.M == 0:
            self._gains['ka'], self._gains['kr'], self._gains['kt'] = self._self_tune(
                x, y, gx, gy, block_positions, deg, rho_e, kappa_e, vx, vy,
                self._gains['ka'], self._gains['kr'], self._gains['kt'])

        ka, kr, kt = self._gains['ka'], self._gains['kr'], self._gains['kt']
        fx_total, fy_total, repulsion_active = self._forces(
            x, y, gx, gy, block_positions, ka, kr, kt, vx, vy)

        rot_speed_raw, angle_diff = self.compute_angle(fx_total, fy_total, pioneer_orientation)

        fwd_speed = self.KP_fwd * np.exp(-2 * abs(angle_diff))
        if not repulsion_active:
            fwd_speed *= 5
        fwd_speed = min(fwd_speed, self.vmax)

        L = 0.4
        v_l = fwd_speed - rot_speed_raw * L / 2
        v_r = fwd_speed + rot_speed_raw * L / 2
        return v_l, v_r, dg, fwd_speed


class AdaptiveTangentialAPFPotentialField:
    """Comparison version: APF with tangential/inertial forces and
    dynamic coefficients (paper "Adaptive artificial potential field
    method for small autonomous vehicles", Kilic et al., Robotics and
    Autonomous Systems 198 (2026) 105364).

    Implements (formulas and logic verified against the paper code,
    papers/new/APFproject-paper1/pathplanner.py):
      - attractive force capped by the lookahead ldist (eq. 2, capped after k_attr);
      - repulsive force with effective distance d_eff = d - robot.size - obs.size
        and profile k_rep*(1/d_eff - 1/k_dist)*(1/d_eff**2)*(d_eff/d)*R (eq. 3-4,
        code line 2070);
      - tangential force, -90°/(+90°) rotation of the repulsive one depending
        on whether the obstacle is on the right/left of the heading (obs_angle < 0 / >= 0,
        eq. 5-6, code lines 2077-2082);
      - inertial force, average of previous velocities * k_inert, scaled by
        d_goal/ldist within ldist (eq. 7, code lines 2132-2135);
      - velocity = total force capped at vmax (eq. 8);
      - local minima detection (Alg. 3, code lines 1829-1866): crossing on a
        point of the first half of the window (time/dt poses) or window average
        + sum of the heading deltas;
      - buffered virtual obstacles: added as pending at detection and
        activated when the recent_lm window (2 s) expires, as in the code
        (plan(), lines 2421-2432);
      - dynamic coefficients k_rep/k_attr/k_inert/k_dist (Alg. 4, lines
        2294-2313) active for the whole recent_lm window, with gradual
        recovery and k_rep reduction near the goal (GNRON).

    K_att/K_rep are structural scale factors: K_att=K_rep=1.0 uses the
    absolute values of the paper (paper defaults: ldist=4, dt=0.1). The
    same forces feed heading (compute_angle) and velocity
    (min(||F_tot||, vmax)) with the differential kinematics of the P3DX.
    Interface identical to PotentialField for drop-in use.
    """

    def __init__(self, K_att=1.0, K_rep=1.0, THR=0.6, KP_rot=4.0, KP_fwd=1.0,
                 rate=0.03, decay_factor=None,
                 ldist=4.0, vmax=1.2, dt=0.1,
                 k_rep_min=1.0, k_rep_max=4.0, k_rep_start=1.0,
                 k_attr_min=0.2, k_attr_max=1.0, k_attr_start=1.0,
                 k_dist_min=2.0, k_dist_max=4.0, k_dist_start=2.0,
                 k_inert_min=0.0, k_inert_max=1.0, k_inert_start=0.5,
                 ro=0.5, rn=0.5, ao=0.1, an=0.5, do=0.1, dn=1.0, io=0.02, inr=0.02,
                 lm_prec=0.1, lm_range=2.0, lm_turning=2 * np.pi,
                 lm_time=6.0, lm_window=2.0, vo_lifetime=None,
                 n_inert=10, d_safe_floor=0.2, enabled=('attr', 'rep', 'tan', 'iner', 'dyn')):
        self.K_att = K_att
        self.K_rep = K_rep
        self.THR = THR
        self.KP_rot = KP_rot
        self.KP_fwd = KP_fwd
        self.increase_rate = 1 + rate
        self.decay_rate = 1 - rate if decay_factor is None else decay_factor
        self.obstacle_half_size = 0.25
        self.robot_radius = 0.2

        self.ldist = ldist
        self.vmax = vmax
        self.dt = dt
        self.k_rep_min, self.k_rep_max, self.k_rep = k_rep_min, k_rep_max, k_rep_start
        self.k_attr_min, self.k_attr_max, self.k_attr = k_attr_min, k_attr_max, k_attr_start
        self.k_dist_min, self.k_dist_max, self.k_dist = k_dist_min, k_dist_max, k_dist_start
        self.k_inert_min, self.k_inert_max, self.k_inert = k_inert_min, k_inert_max, k_inert_start
        self.ro, self.rn = ro, rn
        self.ao, self.an = ao, an
        self.do, self.dn = do, dn
        self.io, self.inr = io, inr
        self.lm_prec = lm_prec
        self.lm_range = lm_range
        self.lm_turning = lm_turning
        self.lm_time = lm_time          # in seconds, as in the code (time=6., dt)
        self.lm_window = lm_window      # recent_lm window in seconds (code: 2.)
        self.vo_lifetime = vo_lifetime  # None = persistent (as in the code); int = n. frames
        self._lm_frames = max(2, int(lm_time / dt))
        self.n_inert = n_inert
        self.d_safe_floor = d_safe_floor
        self.enabled = set(enabled)
        self.robot_size = self.robot_radius
        self.obstacle_size = self.obstacle_half_size

        self._pos_history = []
        self._yaw_history = []
        self._vel_history = []
        self._virtual = []          # (x, y, rem) active virtual obstacles
        self._vobs_buffer = []      # (x, y) pending, activated at the end of the recent_lm window
        self._recent_lm = 0.0       # LM window timer in seconds (code: recent_lm)
        self._frame = 0

    def mod_thr(self, mode):
        if mode == 1:
            self.THR = min(self.THR * self.increase_rate, 2.0)
        elif mode == 0:
            self.THR = max(0.2, self.THR * self.decay_rate)
        return self.THR

    def _active_obstacles(self, block_positions):
        keep = []
        active = [(bx, by, self.obstacle_size) for bx, by in block_positions]
        for vx, vy, rem in self._virtual:
            if rem is None:
                keep.append((vx, vy, None))
                active.append((vx, vy, 0.0))
            elif rem > 0:
                keep.append((vx, vy, rem - 1))
                active.append((vx, vy, 0.0))
        self._virtual = keep
        return active

    def _lm_detection(self, x, y):
        """Mirror of the code's local_minima_detection (pathplanner.py,
        lines 1829-1866): crossing on a point of the first half of the
        window or window average + sum of the heading deltas."""
        h = self._pos_history
        yw = self._yaw_history
        n = self._lm_frames
        if len(h) <= n + 1:
            return False, None
        pos_start = len(h) - n - 1
        pos_end = len(h) - 2
        half = pos_end / 2.0
        sumt = 0.0
        mx = my = 0.0
        for i in range(pos_start, pos_end):
            if i < half and np.hypot(x - h[i][0], y - h[i][1]) < self.lm_prec:
                return True, (float(h[i][0]), float(h[i][1]))
            dth = (((yw[i + 1] - yw[i]) + np.pi) % (2 * np.pi)) - np.pi
            sumt += abs(dth)
            mx += h[i + 1][0]
            my += h[i + 1][1]
        mx /= n
        my /= n
        if np.hypot(x - mx, y - my) < self.lm_range and sumt > self.lm_turning:
            return True, (float(mx), float(my))
        return False, None

    def _update_coefficients(self, in_lm, d_goal):
        if in_lm:
            self.k_rep = min(self.k_rep_max, self.k_rep * (1 + self.dt * self.ro))
            self.k_attr = max(self.k_attr_min, self.k_attr * (1 - self.dt * self.ao))
            self.k_inert = max(self.k_inert_min, self.k_inert * (1 - self.dt * self.io))
            self.k_dist = min(self.k_dist_max, self.k_dist * (1 + self.dt * self.do))
        else:
            self.k_rep = max(self.k_rep_min, self.k_rep * (1 - self.dt * self.rn))
            self.k_attr = min(self.k_attr_max, self.k_attr * (1 + self.dt * self.an))
            self.k_inert = min(self.k_inert_max, self.k_inert * (1 + self.dt * self.inr))
            self.k_dist = max(self.k_dist_min, self.k_dist * (1 - self.dt * self.dn))
        if d_goal < self.k_dist:
            self.k_rep *= (d_goal / self.k_dist) ** 2

    def _passive_forces(self, x, y, gx, gy, obstacles, heading_x, heading_y):
        ka = self.K_att * self.k_attr
        ax, ay = gx - x, gy - y
        if 'attr' in self.enabled:
            m = np.hypot(ka * ax, ka * ay)
            if m > 0 and m > self.ldist:
                fa_x, fa_y = ka * ax * self.ldist / m, ka * ay * self.ldist / m
            else:
                fa_x, fa_y = ka * ax, ka * ay
        else:
            fa_x = fa_y = 0.0

        fr_x, fr_y = 0.0, 0.0
        ft_x, ft_y = 0.0, 0.0
        kd = self.k_dist
        kr = self.K_rep * self.k_rep
        for ox, oy, osize in obstacles:
            rx, ry = x - ox, y - oy
            d = np.hypot(rx, ry)
            d_eff = max(d - self.robot_size - osize, self.d_safe_floor)
            if d_eff >= kd:
                continue
            f = kr * (1.0 / d_eff - 1.0 / kd) * (1.0 / d_eff ** 2) * (d_eff / d)
            if 'rep' in self.enabled:
                fr_x += f * rx
                fr_y += f * ry
            if 'tan' in self.enabled:
                obs_angle = np.arctan2(heading_x * (oy - y) - heading_y * (ox - x),
                                       heading_x * (ox - x) + heading_y * (oy - y))
                if obs_angle < 0:
                    ft_x += f * ry
                    ft_y += f * (-rx)
                else:
                    ft_x += f * (-ry)
                    ft_y += f * rx
        return fa_x, fa_y, fr_x, fr_y, ft_x, ft_y

    def compute_angle(self, fx, fy, pioneer_orientation):
        angle = np.arctan2(fy, fx)
        angle_diff = (angle - pioneer_orientation + np.pi) % (2 * np.pi) - np.pi
        rot_speed_raw = 0.0 if abs(angle_diff) < 0.1 else self.KP_rot * angle_diff
        return rot_speed_raw, angle_diff

    def p_field(self, pioneer_position, goal_position, block_positions, pioneer_orientation):
        self._frame += 1
        x, y = float(pioneer_position[0]), float(pioneer_position[1])
        gx, gy = float(goal_position[0]), float(goal_position[1])

        self._pos_history.append((x, y))
        keep = max(self._lm_frames + 2, self.n_inert + 2)
        self._pos_history = self._pos_history[-keep:]
        self._yaw_history.append(pioneer_orientation)
        self._yaw_history = self._yaw_history[-keep:]
        if len(self._pos_history) >= 2:
            vx = (self._pos_history[-1][0] - self._pos_history[-2][0]) / self.dt
            vy = (self._pos_history[-1][1] - self._pos_history[-2][1]) / self.dt
        else:
            vx = vy = 0.0
        self._vel_history.append((vx, vy))
        self._vel_history = self._vel_history[-self.n_inert:]

        d_goal = np.hypot(gx - x, gy - y)

        if abs(self._recent_lm) < self.dt:
            while self._vobs_buffer:
                vx0, vy0 = self._vobs_buffer.pop()
                self._virtual.append((vx0, vy0, self.vo_lifetime))
            in_lm, lm_pos = self._lm_detection(x, y)
            if in_lm:
                self._recent_lm = self.lm_window
                self._vobs_buffer.append((lm_pos[0], lm_pos[1]))
        else:
            self._recent_lm -= self.dt
        if 'dyn' in self.enabled:
            self._update_coefficients(self.dt < self._recent_lm <= self.lm_window, d_goal)

        obstacles = self._active_obstacles(block_positions)
        heading_x = np.cos(pioneer_orientation)
        heading_y = np.sin(pioneer_orientation)
        fa_x, fa_y, fr_x, fr_y, ft_x, ft_y = self._passive_forces(
            x, y, gx, gy, obstacles, heading_x, heading_y)

        fi_x = fi_y = 0.0
        if 'iner' in self.enabled and self._vel_history:
            mvx = np.mean([v[0] for v in self._vel_history])
            mvy = np.mean([v[1] for v in self._vel_history])
            if d_goal < self.ldist:
                s = d_goal / self.ldist
                mvx *= s
                mvy *= s
            fi_x = self.k_inert * mvx
            fi_y = self.k_inert * mvy

        fx_total = fa_x + fr_x + ft_x + fi_x
        fy_total = fa_y + fr_y + ft_y + fi_y

        rot_speed_raw, angle_diff = self.compute_angle(fx_total, fy_total, pioneer_orientation)

        fwd_speed = min(np.hypot(fx_total, fy_total), self.vmax)
        fwd_speed = min(self.KP_fwd * fwd_speed, self.vmax)

        L = 0.4
        v_l = fwd_speed - rot_speed_raw * L / 2
        v_r = fwd_speed + rot_speed_raw * L / 2
        return v_l, v_r, d_goal, fwd_speed
