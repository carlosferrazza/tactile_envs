"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like
interface.

Supporting gym==0.21.0
"""

import numpy as np
import gym
from gym import spaces, Env

from robosuite.wrappers import Wrapper


def convert_observation_to_space(observation):

    space = spaces.Dict(spaces={})
    for key in observation.keys():
        if key == 'image':
            space.spaces[key] = spaces.Box(low = 0, high = 1, shape = observation[key].shape, dtype = np.float32)
        elif key == 'tactile':
            space.spaces[key] = spaces.Box(low = -float('inf'), high = float('inf'), shape = observation[key].shape, dtype = np.float32)
        elif key == 'reward':
            space.spaces[key] = spaces.Box(low = -float('inf'), high = float('inf'), shape = (1,), dtype = np.float32)
        elif key == 'id':
            space.spaces[key] = spaces.Box(low= -float('inf'), high = float('inf'), shape = (1,), dtype = np.float32)

    return space

class OldGymWrapper(Wrapper, gym.Env):
    metadata = None
    render_mode = None
    """
    Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module

    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env, keys=None, env_id=-1, state_type="pixels_and_tactile"):
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)
        self.state_type = state_type

        if keys is None:
            keys = []
            # Add object obs if requested
            if self.env.use_object_obs:
                keys += ["object-state"]
            # Add image obs if requested
            if self.env.use_camera_obs:
                keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
            # Iterate over all robots to add to state
            for idx in range(len(self.env.robots)):
                keys += ["robot{}_proprio-state".format(idx)]
        self.keys = keys

        self.id = env_id

        # Gym specific attributes
        self.env.spec = None

        # set up observation and action spaces
        obs = self.reset()
        self.observation_space = convert_observation_to_space(obs)
        self.action_space = spaces.Box(*self.env.action_spec)


    def slim_obs(self, obs):

        new_obs = {}
        new_obs['image'] = obs['agentview_image'][::-1].astype(np.float32)/255
        if "robot1_tactile_left" in obs:
            new_obs['tactile'] = np.concatenate((obs['robot0_tactile_left'], obs['robot0_tactile_right'], obs['robot1_tactile_left'], obs['robot1_tactile_right']),axis=0)
        else:
            new_obs['tactile'] = np.concatenate((obs['robot0_tactile_left'], obs['robot0_tactile_right']),axis=0)

        new_obs['reward'] = np.array([0])
        new_obs['id'] = np.array([self.id])

        if self.state_type == "pixels_and_tactile":
            pass
        elif self.state_type == "pixels_dict":
            del new_obs['tactile']
        elif self.state_type == "tactile_dict":
            del new_obs['image']

        return new_obs

    def reset(self):
        obs = self.env.reset()
        return self.slim_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        obs = self.slim_obs(obs)
        info = obs.copy()
        if "reward" not in info:
            info["reward"] = reward

        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()
