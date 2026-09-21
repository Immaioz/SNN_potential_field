import numpy as np


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
