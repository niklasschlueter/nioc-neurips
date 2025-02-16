from typing import NamedTuple

import jax.numpy as jnp

from nioc.envs.base import Env
from jax.numpy import sin, cos, tan


class QuadrotorParams(NamedTuple):
    action_cost: float = 1e-4
    #velocity_cost: float = 1e-2
    #motor_noise: float = 1e-1
    #obs_noise: float = 1.


class Quadrotor(Env):
    def __init__(self, dt=0.01, x0=jnp.zeros(10,)):
        """ Non-linear reaching task from Weiwei Li's PhD thesis

        Args:
            dt (float): time step duration
            target (jnp.array): target position (x, y)
            x0 (jnp.array): initial state (theta1, theta2, theta1_dot, theta2_dot)
            indep_noise (float): state-independent noise on dynamics
            upper_arm_length (float): upper arm length in meters
            forearm_length (float): forearm length in meters
            I1, I2 (float): moments of inertia of the joints (kg / m**2)
        """
        self.target = jnp.zeros(10,)
        self.target.at[2].set(1.0)
        self.dt = dt

        
        #self.target = target
        self.x0 = x0

        super().__init__(state_shape=(10,), action_shape=(4,), observation_shape=(10,))

    def dyn(self, state, action, noise=0.0, params=0.0):
        GRAVITY = 9.81
        params_pitch_rate = [
                -6.003842038081178,
                6.213752925707588]
        params_roll_rate = [-3.960889336015948,
                4.078293254657104]
        params_yaw_rate = [-0.005347588299390372, 0.0]
        params_acc = [20.907574256269616, 3.653687545690674]

        (px,
        py,
        pz,
        vx,
        vy,
        vz,
        r,
        p,
        y,
        f_collective,
        ) = state.squeeze()

        f_collective_cmd, r_cmd, p_cmd, y_cmd = action

        # Define nonlinear system dynamics
        f = jnp.array([
            vx,
            vy,
            vz,
            (params_acc[0] * f_collective + params_acc[1])
            * (cos(r) * sin(p) * cos(y) + sin(r) * sin(y)),
            (params_acc[0] * f_collective + params_acc[1])
            * (cos(r) * sin(p) * sin(y) - sin(r) * cos(y)),
            (params_acc[0] * f_collective + params_acc[1]) * cos(r) * cos(p) - GRAVITY,
            params_roll_rate[0] * r + params_roll_rate[1] * r_cmd,
            params_pitch_rate[0] * p + params_pitch_rate[1] * p_cmd,
            params_yaw_rate[0] * y + params_yaw_rate[1] * y_cmd,
            10.0 * (f_collective_cmd - f_collective),
        ])
        return f

    def _dynamics(self, state, action, noise=0.0, params=0.0):
        f = self.dyn(state, action, noise, params)
        f = self.dt * f
        #w = jnp.sqrt(self.dt) * (G @ (params.motor_noise * jnp.diag(action) @ noise[2:]) + H @ (self.v * noise[:2]))
        return state + f #+ w

    def _observation(self, state, noise, params):
        #return state + self.dt * params.obs_noise * jnp.eye(self.observation_shape[0]) @ noise
        return state 

    def _cost(self, state, action, params):
        return 0.5 * params.action_cost * jnp.sum(action ** 2)

    def _final_cost(self, state, params):
        return jnp.sum((state - self.target) ** 2)

    def _reset(self, noise, params):
        x0 = self.x0
        return x0

    @staticmethod
    def get_params_type():
        return QuadrotorParams

    @staticmethod
    def get_params_bounds():
        lo = QuadrotorParams(action_cost=1e-5,)# velocity_cost=1e-3, motor_noise=1e-2, obs_noise=1e-1)
        hi = QuadrotorParams(action_cost=1e-1,)# velocity_cost=1e-1, motor_noise=1., obs_noise=100.)
        return lo, hi
