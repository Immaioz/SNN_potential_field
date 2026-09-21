import numpy as np


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
