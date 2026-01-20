
import time
import torch
import numpy as np
from scipy import integrate
import casadi, mpctools as mpc
import matplotlib.pyplot as plt
import os
import random

from paths.path_set import PathSampler, curriculum_path_pool, all_paths

# Define parameters
pi = np.pi
rho_aero = 1.225
mveh = 1500
Cd = 0.389
radwhl = 0.2159
Af = 4
Lxf = 1.2
Lxr = 1.4
mu = 1
Iw = 3.8782
I = 4192
Calpha = -0.08 * 180 / pi
sa_max = 10 / 180 * pi
g = 9.8

# TODO: any difference?
param_mpc = [rho_aero, mveh, Cd, radwhl, Af, Lxf, Lxr, mu, Iw, I, Calpha, sa_max, g]
param_plant = [rho_aero, mveh * 0.9, Cd, radwhl, Af, Lxf + 0.1, Lxr - 0.1, mu * 0.95, Iw, I * 1.1, Calpha * 1.1, sa_max,
               g]


def veh_rhs(x0, t, u0, param=param_plant, vx=10):
    rho_aero = param[0]
    mveh = param[1]
    Cd = param[2]
    radwhl = param[3]
    Af = param[4]
    Lxf = param[5]
    Lxr = param[6]
    mu = param[7]
    # Iw = param[8]
    I = param[9]
    Calpha = param[10]
    # sa_max = param[11]    
    g = param[12]
    sigma = 0.0

    # initialization
    vx = x0[1]
    vy = x0[3]
    psi = x0[4]
    r = x0[5]

    Tf = u0[0] ###36.0  # u0[0]
    Tr = 0.0  # u0[1]
    dmf = 0.0
    dmr = 0.0
    betaf = u0[1]
    betar = 0.0  # u0[3]

    # Define rotation matrix
    rotM_00 = np.cos(betaf)
    rotM_01 = -np.sin(betaf)
    rotM_10 = np.sin(betaf)
    rotM_11 = np.cos(betaf)
    rotM_20 = np.cos(betar)
    rotM_21 = -np.sin(betar)
    rotM_30 = np.sin(betar)
    rotM_31 = np.cos(betar)

    # Calculating corner and wheel velocity
    v_c_00 = vx
    v_c_01 = vy + Lxf * r
    v_c_10 = vx
    v_c_11 = vy - Lxr * r

    v_w_00 = v_c_00 * rotM_00 + v_c_01 * rotM_10
    v_w_01 = v_c_00 * rotM_01 + v_c_01 * rotM_11
    v_w_10 = v_c_10 * rotM_20 + v_c_11 * rotM_30
    v_w_11 = v_c_10 * rotM_21 + v_c_11 * rotM_31

    # Calculating tire force at each wheel
    Fz_0 = Lxr * mveh * g / 2 / (Lxf + Lxr)
    Fz_1 = Lxf * mveh * g / 2 / (Lxf + Lxr)

    F_w00 = Tf / 2 / radwhl
    F_w10 = Tr / 2 / radwhl

    sa0 = np.arctan(v_w_01 / v_w_00)
    sa1 = np.arctan(v_w_11 / v_w_10)

    F_w01 = Calpha * mu * sa0 * Fz_0
    F_w11 = Calpha * mu * sa1 * Fz_1

    F_c_00 = F_w00*rotM_00+F_w01*rotM_01
    F_c_01 = F_w00 * rotM_10 + F_w01 * rotM_11
    F_c_10 = F_w10*rotM_20+F_w11*rotM_21
    F_c_11 = F_w10 * rotM_30 + F_w11 * rotM_31

    # calculating the xdot
    # xdot = np.zeros(6)
    xdot0 = vx * np.cos(psi) - vy * np.sin(psi)
    xdot1 = vy*r+2*(F_c_00+F_c_10)/mveh-g*np.sin(sigma)-0.5*rho_aero*Cd*Af*vx*vx/mveh
    xdot2 = vx * np.sin(psi) + vy * np.cos(psi)
    xdot3 = -vx * r + 2 * (F_c_01 + F_c_11) / mveh
    xdot4 = r
    xdot5 = (2 * F_c_01 * Lxf - 2 * F_c_11 * Lxr + dmf + dmr) / I  # TODO: difference with Eqn21-f
    xdot = (xdot0, xdot1, xdot2, xdot3, xdot4, xdot5)
    # return xdot = [x, vx, y, vy, \phi, r]
    return xdot

######test the nonlinear dynamics of the system
# aux = veh_rhs((1., 2., 3., 4., 5., 6.), 0., [ 9.,0])
# print(aux)

# System parameters.
DEFAULT_P = 5
Nx = 6
Nu = 2
dt = 0.2


def ode(x, u):
    """ODE right-hand side."""
    # u0 = np.zeros(4)
    # u0[0]=36
    # u0[2]=u
    dxdt = veh_rhs(x, 0., u, param=param_mpc)
    return np.array(dxdt)


f = mpc.getCasadiFunc(ode, [Nx, Nu], ["x", "u"], rk4=True, Delta=dt, M=4)

# Initial condition, bounds, etc.
x0 = np.array([0, 10, 0, -0.0691, 0.2343, -0.0123])
# lower and upper bounds for states and actions
lb = dict(x=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]), u=np.array([-50, -0.54105]))
ub = dict(x=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]), u=np.array([50, 0.54105]))
udiscrete = np.array([False, False])


def make_guess(horizon):
    x = np.zeros((horizon + 1, Nx))
    u = np.zeros((horizon, Nu))
    x[0, :] = x0
    for t in range(horizon):
        x[t + 1, :] = np.squeeze(f(x[t, :], u[t, :]))
    return dict(x=x, u=u)


def make_stagecost(ref_func):
    def stagecost(x, u):
        """Quadratic stage cost."""
        return 2.0 * (x[2] - ref_func(x[0])) ** 2 + 1e-6 * u[0] ** 2 + 0.001 * u[1] ** 2
    return stagecost


def make_controller(horizon, ref_func):
    l = mpc.getCasadiFunc(make_stagecost(ref_func), [Nx, Nu], ["x", "u"])
    N = dict(x=Nx, u=Nu, t=horizon)
    guess = make_guess(horizon)
    return mpc.nmpc(f, l, N, x0, lb, ub, guess, udiscrete=udiscrete)


class vehEnv:
    def __init__(
        self,
        T=20,
        rho=0,
        x0=None,
        sph=0,
        np_eco=DEFAULT_P,
        np_pro=DEFAULT_P,
        path_set="single",
        curriculum=0,
        domain_rand=0,
        seed=0,
        difficulty=1.0,
        total_episodes=500,
    ):
        self.T = T
        self.rho = rho
        self.dt = dt
        self.x0 = x0 if x0 is not None else [0, 10, 0, -0.0691, 0.2343, -0.0123]
        self.np_eco = np_eco
        self.np_pro = np_pro
        self.sph = sph
        self.path_set = path_set
        self.curriculum = curriculum
        self.domain_rand = domain_rand
        self.difficulty = difficulty
        self.total_episodes = max(1, total_episodes)
        self.path_sampler = PathSampler(seed=seed)
        self.rng = np.random.RandomState(seed)
        self.action_space = 3 if sph else 2
        self.obs_space = 12
        """change the code stop threshold"""
        self.threshold_error = 100
        self.max_horizon = max(self.np_eco, self.np_pro, DEFAULT_P)
        self.current_horizon = DEFAULT_P
        self.ref_path = None

    def connect(self):
        pass

    def loadInitState(self, iniStateName):
        pass

    def reset(self, episode_idx=0):
        self._select_path(episode_idx)
        self.x = np.array(self.x0, dtype=np.float64)
        if self.domain_rand:
            noise = self.rng.normal(scale=0.05 * self.difficulty, size=len(self.x))
            self.x = self.x + noise
        self.k = 0
        self.useq = np.zeros([self.max_horizon, Nu])
        self.xseq = np.zeros([self.max_horizon, Nx])
        self.t = 0.
        self.current_horizon = DEFAULT_P
        return self.getFeat()

    def getFeat(self):
        return np.concatenate((self.x, self.xseq[self.k]))  #, axis=1

    def step(self, action):
        solve_time_ms = 0.0
        solve_ok = True
        if self.sph:
            solve_action = action in (1, 2)
            horizon = self.np_eco if action == 1 else self.np_pro
        else:
            solve_action = action > 0
            horizon = DEFAULT_P

        if solve_action:
            self.current_horizon = horizon
            start_time = time.perf_counter()
            try:
                cont = make_controller(horizon, self.ref_path.ref)
                cont.solve()
                solve_ok = True
                solve_time_ms = (time.perf_counter() - start_time) * 1000.0
                useq = cont.vardict["u"]
                xseq = cont.vardict["x"]
            except Exception:
                solve_ok = False
                solve_time_ms = (time.perf_counter() - start_time) * 1000.0
                useq = None
                xseq = None

            if useq is not None:
                self.useq[:horizon] = useq
                self.xseq[:horizon + 1] = xseq
                if horizon < self.max_horizon:
                    self.useq[horizon:] = useq[-1]
                    self.xseq[horizon + 1:] = xseq[-1]
                self.k = 0
        else:
            self.k += 1
            if self.k > self.current_horizon - 1:
                self.k = self.current_horizon - 1

        u = self.useq[self.k]
        t = np.linspace(self.t, self.t + dt, 10)
        aux = integrate.odeint(veh_rhs, self.x, t, args=(u,))
        self.x = aux[-1]
        self.t += self.dt

        # Reward should be calculated using current state as forward Euler is using.
        reward = self._stagecost(self.x, u) * self.dt * (-1) - self.rho * action
        jmpc = self._stagecost(self.x, u) * self.dt * (-1)
        next_state = self.getFeat()
        current_time = torch.tensor(self.t)

        # TODO: are there any other terminal conditions
        tracking_error = (next_state[2] - self.ref_path.ref(next_state[0])) ** 2
        if self.t >= self.T:
            done = True
        elif tracking_error > self.threshold_error:
            done = True
            reward -= 10
        else:
            done = False

        info = {
            "time": current_time,
            "jmpc": jmpc,
            "solve_time_ms": solve_time_ms,
            "solve_ok": solve_ok,
        }
        return next_state, reward, done, info

    def close(self,):
        pass

    def _stagecost(self, x, u):
        return 2.0 * (x[2] - self.ref_path.ref(x[0])) ** 2 + 1e-6 * u[0] ** 2 + 0.001 * u[1] ** 2

    def _select_path(self, episode_idx):
        if self.path_set == "single":
            pool = ["dang_single"]
        else:
            if self.curriculum:
                progress = min(1.0, max(0.0, episode_idx / max(1, self.total_episodes - 1)))
                pool = curriculum_path_pool(progress)
            else:
                pool = all_paths()
        name = pool[self.rng.randint(len(pool))]
        self.ref_path = self.path_sampler.sample(name, domain_rand=bool(self.domain_rand))

    def reference(self, x):
        return self.ref_path.ref(x)


if __name__ == "__main__":
    seed = 66
    np.random.seed(seed)
    random.seed(seed)

    if not os.path.exists('runs' + '/Veh/'):
        os.mkdir('runs' + '/Veh/')

    vehEnv = vehEnv()
    # test the cloud mpc
    vehEnv.reset()
    done = False
    cloud_return = 0
    cloud_obs, cloud_a, cloud_all_u = [], [], []
    while not done:
        state, reward, done, info = vehEnv.step(1)
        cloud_return += reward
        cloud_obs.append(state)
        cloud_a.append(1)
        cloud_all_u.append(np.array(info["jmpc"]))

    print("Return for MPC:", cloud_return)
    np.save('runs' + '/Veh/ob_history', cloud_obs)
    np.save('runs' + '/Veh/actions', cloud_a)
    np.save('runs' + '/Veh/numpy_u', cloud_all_u)

    aaa = np.load('runs/Veh/numpy_u.npy')
    uuu = np.load('runs/Veh/ob_history.npy')
    plt.figure()
    plt.plot(uuu[:, 2], label='gt')
    plt.plot(4 * np.sin(2 * np.pi / 50 * np.array(uuu[:, 0])), label='pred')
    plt.legend()
    plt.show()
