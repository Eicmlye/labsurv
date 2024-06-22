# CartPoleEnv is reconstructed from gym into mmcv format

import math
from functools import partial
from typing import Any, Dict, Optional

import numpy as np
from gym import spaces
from labsurv.builders import ENVIRONMENTS
from labsurv.models.envs import BaseEnv


@ENVIRONMENTS.register_module()
class CartPoleEnv(BaseEnv):
    def __init__(self):
        """
        Description

            This environment corresponds to the version of the cart-pole problem described
            by Barto, Sutton, and Anderson in Neuronlike Adaptive Elements That Can Solve
            Difficult Learning Control Problem [[1]](https://ieeexplore.ieee.org/document/6313077).

            A pole is attached by an un-actuated joint to a cart, which moves along a
            frictionless track. The pendulum is placed upright on the cart and the goal is
            to balance the pole by applying forces in the left and right direction on the
            cart.

        Action Space

            The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}`
            indicating the direction of the fixed force the cart is pushed with.

            | Num | Action                 |
            |-----|------------------------|
            | 0   | Push cart to the left  |
            | 1   | Push cart to the right |

            NOTE: The velocity that is reduced or increased by the applied force is not
            fixed and it depends on the angle the pole is pointing. The center of gravity
            of the pole varies the amount of energy needed to move the cart underneath it.

        Observation Space

            The observation is a `ndarray` with shape `(4,)` with the values corresponding
            to the following positions and velocities:

            | Num | Observation           | Min                 | Max               |
            |-----|-----------------------|---------------------|-------------------|
            | 0   | Cart Position         | -4.8                | 4.8               |
            | 1   | Cart Velocity         | -Inf                | Inf               |
            | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
            | 3   | Pole Angular Velocity | -Inf                | Inf               |

            NOTE: While the ranges above denote the possible values for observation
                space of each element, it is not reflective of the allowed values of the
                state space in an unterminated episode. Particularly:
            -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but
            the episode terminates if the cart leaves the `(-2.4, 2.4)` range.
            -  The pole angle can be observed between  `(-.418, .418)` radians (or ±24°),
            but the episode terminates if the pole angle is not in the range
            `(-.2095, .2095)` (or ±12°)

        Rewards

            Since the goal is to keep the pole upright for as long as possible, a reward of
            `+1` for every step taken, including the termination step, is allotted. The
            threshold for rewards is 475 for v1.

        Starting State

            All observations are assigned a uniformly random value in `(-0.05, 0.05)`

        Episode End

            The episode ends if any one of the following occurs:

            1. Termination: Pole Angle is greater than ±12°
            2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches
            the edge of the display)
            3. Truncation: Episode length is greater than 500 (200 for v0)
        """

        super().__init__()

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        # should clarify the init observation distribution in convenience of `reset()`
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.init_observ = partial(self._np_random.uniform, size=(4,))

    def step(self, observation, action) -> Dict[str, Any]:
        """
        Run one timestep of the environment's dynamics.
        """
        assert self.action_space.contains(action), "Unknown action."
        assert self.observation_space.contains(
            observation
        ), "Unknown input observation."

        x, x_dot, theta, theta_dot = observation
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        new_observation = (x, x_dot, theta, theta_dot)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        reward = 1.0 if not terminated else 0.0

        transition = dict(
            next_observation=new_observation,
            reward=reward,
            terminated=terminated,
        )

        assert transition[
            "next_observation"
        ] is None or self.observation_space.contains(
            transition["next_observation"]
        ), "Unknown output observation."

        return transition

    def reset(self, seed: Optional[int] = None):
        """
        Resets the environment to an initial state and returns the initial observation.
        """

        # do env init works
        super().reset(seed=seed)
        # return init observation according to observation distribution
        init_observation = self.init_observ(low=-0.05, high=0.05)

        return np.array(init_observation, dtype=np.float32)
