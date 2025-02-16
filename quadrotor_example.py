from jax import jit, random, vmap, numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import jax

from nioc.envs import NonlinearReaching
from nioc.envs import Quadrotor 
from nioc.control import gilqr, lqr
from nioc.control.policy import create_lqg_policy
from nioc.envs.wrappers import EKFWrapper
from nioc.infer import FixedLinearizationInverseGILQG, FixedInverseMaxEntBaseline
from nioc.infer.utils import compute_mle
from pathlib import Path


from numpy import array, float32
from mpcc.control.controller_single import Controller
from utils import load_config, load_controller, additional_config_processing

#logging.basicConfig(level=logging.INFO)


#@jit
def simulate_trajectories(key, params):
    # solve control problem
    T = 10
    print(f"env action shape: {env.action_shape}")
    print(f"env state shape: {env.state_shape}")
    gains, xbar, ubar = gilqr.solve(p=env,
                                    x0=x0, U_init=jnp.zeros(shape=(T, env.action_shape[0])),
                                    params=params, max_iter=10)


    # create a policy and belief dynamics
    policy = create_lqg_policy(gains, xbar, ubar)
    ekf = EKFWrapper(Quadrotor)(b0=b0)

    # simulate some trajectories
    xs, *_ = ekf.simulate(key=key, steps=T, trials=20, policy=policy, params=params)

    # compute positions from angles
    #pos = vmap(vmap(env.e))(xs)

    return xs#, pos

#@jit
#def simulate_trajectories_acados(key, params, policy):
#    # solve control problem
#    T = 10
#
#
#    # create a policy and belief dynamics
#    ekf = EKFWrapper(Quadrotor)(b0=b0)
#
#    # simulate some trajectories
#    xs, *_ = ekf.simulate(key=key, steps=T, trials=20, policy=policy, params=params)
#    #create_and_run_sim(Quadrotor, policy, x0, dt, T)
#
#
#    return xs

# create a simulation function 
def create_and_run_sim(env, policy, x0, dt, T):

    # Closed-loop dynamics with controller
    def create_closed_loop_dynamics(env, policy):
        """Dynamics function for closed-loop system"""
        # Calculate control input (u = u_eq - K(x - x_eq))
        return lambda x, it: env.dyn(x, policy(it, x))

    closed_loop_dynamics = create_closed_loop_dynamics(env, policy)

    # Simulation function
    # TODO: Make this JITable if i have time
    #@jax.jit
    #@partial(jax.jit, static_argnames=['dt', 'T'])
    def simulate(x0, dt=0.01, T=5.0):
        """
        Simulate closed-loop system
        Args:
            K: LQR gain matrix
            x0: Initial state [theta, theta_dot]
            x_eq: Equilibrium state
            u_eq: Equilibrium control
            dt: Time step
            T: Total simulation time
        Returns:
            t: Time vector
            X: State trajectory
            U: Control inputs
        """
        num_steps = jnp.array(T/dt, dtype=int)
        t = jnp.linspace(0, T, num_steps)
        
        def step(carry, _):
            x, it = carry
            # Use RK4 integration
            k1 = dt * closed_loop_dynamics(x, it) #0, K, x_eq, u_eq)
            k2 = dt * closed_loop_dynamics(x + k1/2, it)#, 0, K, x_eq, u_eq)
            k3 = dt * closed_loop_dynamics(x + k2/2, it)#, 0, K, x_eq, u_eq)
            k4 = dt * closed_loop_dynamics(x + k3, it)#, 0, K, x_eq, u_eq)
            x_next = x + (k1 + 2*k2 + 2*k3 + k4)/6
            u = policy(it, x)

            it += 1
            carry = (x_next, it)
            return carry, (x_next, u)
        
        #_, (X, U) = jax.lax.scan(step, (x0, 0), None, length=num_steps)

        carry = (x0, 0)
        X = np.zeros((*np.shape(x0), num_steps))
        U = np.zeros((*np.shape(policy(0, x0)), num_steps))
        for i in range(num_steps):
            carry, (x_next, u) = step(carry, None)
            X[..., i] = x_next
            U[..., i] = u 
            print(x_next)
        return t, X, U

    return simulate(x0, dt, T)


class ControllerHandler():
    def __init__(self):
        config_path = "/home/niklas/repos/master_thesis/cflib_deployment/config/circle_track.toml"
        config = load_config(Path(config_path))
        config = additional_config_processing(config)

        self.initial_info = {
            "collisions": [],
            "symbolic_model": None,
            "sim_freq": 500,
            "low_level_ctrl_freq": 500,
            "drone_mass": 0.03454,
            "env_freq": 100,  # 200,
            "sim": False,
            "id": None,  # To be changed for each drone.
        }
        self.initial_info["id"] = 0
    
        gates_pos = np.array([gate.pos for gate in config.env.track.gates])
    
        gates_rpy = np.array([gate.rpy for gate in config.env.track.gates])
        obstacles_pos = np.array(
            [obstacle.pos for obstacle in config.env.track.obstacles]
            )
    
        initial_obs = {
                # raw data for comparison plotting later.
                "pos_raw": np.zeros(3),
                "rpy_raw": np.zeros(3),
                # Filtered info
                "pos": np.zeros(3),
                "rpy": np.zeros(3),
                "vel": np.zeros(3),
                "ang_vel": np.array([0.0, 0.0, 0.0], dtype=float32),
                "target_gate": 0,
                "gates_pos": gates_pos,
                "gates_rpy": gates_rpy,
                "gates_in_range": np.array([False] * len(config.env.track.gates)),
                "obstacles_pos": obstacles_pos,
                "obstacles_in_range": np.array([False] * len(config.env.track.obstacles)),
                "opponent": {
                    "pos": np.array([-1.0, 1.0, 0.05], dtype=float32),
                    "rpy": np.array([0.0, -0.0, 0.0], dtype=float32),
                    "vel": np.array([0.0, 0.0, 0.0], dtype=float32),
                    "ang_vel": np.array([0.0, 0.0, 0.0], dtype=float32),
                },
            }
        self.ctrl = Controller(initial_obs, self.initial_info, config)


    def compute_control(self, t, state, noise=0):
        obs = {}
        obs["pos"] = np.array(state[:3])
        obs["vel"] = np.array(state[3:6])
        obs["rpy"] = np.array(state[6:9])
        obs["ang_vel"] = np.zeros(3)
        obs["gates_pos"] = self.ctrl.initial_obs["gates_pos"]
        obs["gates_rpy"] = self.ctrl.initial_obs["gates_rpy"]
        obs["obstacles_pos"] = self.ctrl.initial_obs["obstacles_pos"]

        action = self.ctrl.compute_control(obs, self.initial_info)
        return jnp.array(action)



if __name__ == '__main__':
    control = ControllerHandler()
    # get parameter type and setup default parameter values
    QuadrotorParams = Quadrotor.get_params_type()
    params = QuadrotorParams()
    print(f"Simulating data with: {params}")

    # create an environment, setup initial state and belief
    env = Quadrotor()
    x0 = env._reset(None, params)
    b0 = (x0, jnp.eye(x0.shape[0]))
    print(f"x0: {x0}")
    control.compute_control(0.0, np.array(x0))

    # setup random seed
    key = random.PRNGKey(1)

    # simulate some trajectories given ground truth parameters
    key, subkey = random.split(key)

    dt = 0.01
    T = 0.3
    horizon = int(T/dt)
    num_samples =1 
    X = np.zeros((num_samples, horizon, len(x0)))
    U = np.zeros((num_samples, horizon, 4))
    for i in range(num_samples):
        t, xs, us = create_and_run_sim(env, control.compute_control, x0, dt, T)
        print(f"shape xs: {xs}")
        X[i, :, :] = xs.T
        U[i, :, :] = us.T
    print(f"shape X: {np.shape(X)}")
    print(f"shape U: {np.shape(U)}")

    plt.figure()
    [plt.plot(X[i, ...]) for i in range(num_samples)]
    plt.legend([f"x{i}" for i in range(jnp.shape(X)[1])])
    plt.savefig("x_traj.png", dpi=400)
    plt.figure()
    #plt.plot(U)
    [plt.plot(U[i, ...]) for i in range(num_samples)]
    plt.legend([f"u{i}" for i in range(jnp.shape(U)[1])])
    plt.savefig("u_traj.png", dpi=400)
    #plt.show()

    X = jnp.array(X)
    U = jnp.array(U)




    #xs, pos = simulate_trajectories(subkey, params)
    #xs = simulate_trajectories_acados(subkey, params, control.compute_control)
    #print(f"xs: {xs}")
    #print(f"shape: {jnp.shape(xs)}")
    #plt.figure()
    #for i in range(1):
    #    print(f"x0: {xs[0, : ,0]}")
    #    plt.plot(xs[i, :, 0], color="r", label="x0")
    #    plt.plot(xs[i, :, 1], color="g", label="x1")
    #    plt.plot(xs[i, :, 2], color="b", label="x2")
    #    plt.plot(xs[i, :, 3], color="y", label="x3")
    #plt.legend()
    #plt.show()

    # visualize trajectories
    #f, ax = plt.subplots()
    #ax.plot(pos[..., 0].T, pos[..., 1].T, color="C0", alpha=0.8, linewidth=1,
    #        label="Ground truth")
    #ax.scatter(env.target[0], env.target[1], color="k", marker="x", zorder=2, linewidth=1)

    # setup inverse optimal control
    ioc = FixedLinearizationInverseGILQG(env, b0=b0)

    # run maximization of the IOC likelihood
    print("Running inverse ILQG...")
    key, subkey = random.split(key)
    result = compute_mle(X, ioc, subkey, restarts=10,
                         bounds=env.get_params_bounds(), optim="L-BFGS-B")
    print(f"Estimated with inverse ILQG: {result.params}")

    # simulate some trajectories given the maximum likelihood estimate parameters
    key, subkey = random.split(key)
    #xs_sim, pos_sim = simulate_trajectories(subkey, result.params)
    xs_sim = simulate_trajectories(subkey, result.params)

    # plot trajectories simulated using MLE parameters
    #ax.plot(pos_sim[..., 0].T, pos_sim[..., 1].T, color="C1", alpha=0.8, linewidth=1,
    #label="Ours")

    # setup baseline method
    baseline = FixedInverseMaxEntBaseline(env)

    # run maxent baseline
    print("Running MaxEnt baseline...")
    key, subkey = random.split(key)
    baseline_result = compute_mle(X, baseline, subkey, restarts=10,
                                  bounds=env.get_params_bounds(), optim="L-BFGS-B")
    print(f"Estimated with MaxEnt baseline: {baseline_result.params}")

    # simulate some trajectories given the parameters estimated using the maxent baseline
    key, subkey = random.split(key)
    xs_baseline = simulate_trajectories(subkey, baseline_result.params)

    # plot trajectories simulated using MLE parameters
    #ax.plot(pos_baseline[..., 0].T, pos_baseline[..., 1].T, color="C2", alpha=0.8, linewidth=1,
    #        label="Baseline")

    # make the plot pretty (get unique labels, remove spines, set labels etc.)
    #handles, labels = ax.get_legend_handles_labels()
    #labels, ids = np.unique(labels, return_index=True)
    #handles = [handles[i] for i in ids]
    #ax.legend(handles, labels, frameon=False)
    #ax.spines[['right', 'top']].set_visible(False)
    #ax.set_xlabel("x [m]")
    #ax.set_ylabel("y [m]")
    #f.suptitle("Reaching trajectories")
    #f.show()
