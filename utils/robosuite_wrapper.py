"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like
interface.

adapt from: https://github.com/StanfordVL/robosuite/blob/master/robosuite/wrappers/gym_wrapper.py

"""

import numpy as np
from gym import spaces
from gym.utils import seeding
from robosuite.wrappers import Wrapper


class GymWrapper(Wrapper):
    def __init__(self, env, keys=None):
        """
        Initializes the Gym wrapper.

        Args:
            env (MujocoEnv instance): The environment to wrap.
            keys (list of strings): If provided, each observation will
                consist of concatenated keys from the wrapped environment's
                observation dictionary. Defaults to robot-state and object-state.
        """
        Wrapper.__init__(self, env)
        self.np_random = None

        if keys is None:
            assert self.env.use_object_obs, "Object observations need to be enabled."
            keys = ["robot-state", "object-state"]
        self.keys = keys

        # set up observation and action spaces
        flat_ob = self._flatten_obs(self.env.reset(), verbose=True)
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high)

        # set up reward range and metadata
        self.reward_range = (-np.inf, np.inf)
        self.metadata = {}
        # self.metadata = {
        #     'render.modes': ['human', 'rgb_array', 'depth_array'],
        #     'video.frames_per_second': int(np.round(1.0 / self.dt))
        # }

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict: ordered dictionary of observations
        """
        ob_lst = []
        for key in obs_dict:
            if key in self.keys:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(obs_dict[key])
        return np.concatenate(ob_lst)

    def reset(self):
        ob_dict = self.env.reset()
        return self._flatten_obs(ob_dict)

    def step(self, action):
        ob_dict, reward, done, info = self.env.step(action)
        return self._flatten_obs(ob_dict), reward, done, info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
