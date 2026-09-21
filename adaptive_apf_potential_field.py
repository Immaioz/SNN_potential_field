import numpy as np


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
