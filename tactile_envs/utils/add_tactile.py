"""Wrapper for resizing observations."""
from __future__ import annotations

import numpy as np

import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from gymnasium.spaces import Box, Dict


class AddTactile(gym.ObservationWrapper):
   
    def __init__(self, env: gym.Env) -> None:
   
        gym.ObservationWrapper.__init__(self, env)

        self.obs_shape = (3, 32, 32)

        self.observation_space['tactile'] = Box(
            low=-np.inf,
            high=np.inf,
            shape=self.obs_shape,
            dtype=np.float32,
        )

        self.mj_data = self.env.unwrapped.data

    def observation(self, observation):
        
        rh_palm_touch = self.mj_data.sensor('rh_palm_touch').data.reshape((3, 3, 3)) # 3 x nx x ny
        rh_palm_touch = rh_palm_touch[[1, 2, 0]] # zxy -> xyz

        rh_ffproximal_touch = self.mj_data.sensor('rh_ffproximal_touch').data.reshape((3, 3, 3))
        rh_ffproximal_touch = rh_ffproximal_touch[[1, 2, 0]]

        rh_ffmiddle_touch = self.mj_data.sensor('rh_ffmiddle_touch').data.reshape((3, 3, 3))
        rh_ffmiddle_touch = rh_ffmiddle_touch[[1, 2, 0]]

        rh_ffdistal_touch = self.mj_data.sensor('rh_ffdistal_touch').data.reshape((3, 3, 3))
        rh_ffdistal_touch = rh_ffdistal_touch[[1, 2, 0]]

        rh_mfproximal_touch = self.mj_data.sensor('rh_mfproximal_touch').data.reshape((3, 3, 3))
        rh_mfproximal_touch = rh_mfproximal_touch[[1, 2, 0]]

        rh_mfmiddle_touch = self.mj_data.sensor('rh_mfmiddle_touch').data.reshape((3, 3, 3))
        rh_mfmiddle_touch = rh_mfmiddle_touch[[1, 2, 0]]

        rh_mfdistal_touch = self.mj_data.sensor('rh_mfdistal_touch').data.reshape((3, 3, 3))
        rh_mfdistal_touch = rh_mfdistal_touch[[1, 2, 0]]

        rh_rfproximal_touch = self.mj_data.sensor('rh_rfproximal_touch').data.reshape((3, 3, 3))
        rh_rfproximal_touch = rh_rfproximal_touch[[1, 2, 0]]

        rh_rfmiddle_touch = self.mj_data.sensor('rh_rfmiddle_touch').data.reshape((3, 3, 3))
        rh_rfmiddle_touch = rh_rfmiddle_touch[[1, 2, 0]]

        rh_rfdistal_touch = self.mj_data.sensor('rh_rfdistal_touch').data.reshape((3, 3, 3))
        rh_rfdistal_touch = rh_rfdistal_touch[[1, 2, 0]]

        rh_lfmetacarpal_touch = self.mj_data.sensor('rh_lfmetacarpal_touch').data.reshape((3, 3, 3))
        rh_lfmetacarpal_touch = rh_lfmetacarpal_touch[[1, 2, 0]]

        rh_lfproximal_touch = self.mj_data.sensor('rh_lfproximal_touch').data.reshape((3, 3, 3))
        rh_lfproximal_touch = rh_lfproximal_touch[[1, 2, 0]]

        rh_lfmiddle_touch = self.mj_data.sensor('rh_lfmiddle_touch').data.reshape((3, 3, 3))
        rh_lfmiddle_touch = rh_lfmiddle_touch[[1, 2, 0]]
        
        rh_lfdistal_touch = self.mj_data.sensor('rh_lfdistal_touch').data.reshape((3, 3, 3))
        rh_lfdistal_touch = rh_lfdistal_touch[[1, 2, 0]]

        rh_thproximal_touch = self.mj_data.sensor('rh_thproximal_touch').data.reshape((3, 3, 3))
        rh_thproximal_touch = rh_thproximal_touch[[1, 2, 0]]

        rh_thmiddle_touch = self.mj_data.sensor('rh_thmiddle_touch').data.reshape((3, 3, 3))
        rh_thmiddle_touch = rh_thmiddle_touch[[1, 2, 0]]

        rh_thdistal_touch = self.mj_data.sensor('rh_thdistal_touch').data.reshape((3, 3, 3))
        rh_thdistal_touch = rh_thdistal_touch[[1, 2, 0]]

        block_1 = np.zeros((3, 8, 32))
        
        block_2 = np.concatenate((np.zeros((3, 3, 2)), rh_lfdistal_touch, np.zeros((3, 3, 3)), rh_rfdistal_touch, np.zeros((3, 3, 3)), rh_mfdistal_touch, np.zeros((3, 3, 3)), rh_ffdistal_touch, np.zeros((3, 3, 4)), rh_thdistal_touch, np.zeros((3, 3, 2))), axis=2)
        
        block_3 = np.concatenate((np.zeros((3, 3, 2)), rh_lfmiddle_touch, np.zeros((3, 3, 3)), rh_rfmiddle_touch, np.zeros((3, 3, 3)), rh_mfmiddle_touch, np.zeros((3, 3, 3)), rh_ffmiddle_touch, np.zeros((3, 3, 4)), rh_thmiddle_touch, np.zeros((3, 3, 2))), axis=2)

        block_4 = np.concatenate((np.zeros((3, 3, 2)), rh_lfproximal_touch, np.zeros((3, 3, 3)), rh_rfproximal_touch, np.zeros((3, 3, 3)), rh_mfproximal_touch, np.zeros((3, 3, 3)), rh_ffproximal_touch, np.zeros((3, 3, 4)), rh_thproximal_touch, np.zeros((3, 3, 2))), axis=2)

        block_5 = np.concatenate((np.zeros((3, 3, 2)), rh_lfmetacarpal_touch, np.zeros((3, 3, 27))), axis=2)

        block_6 = np.concatenate((np.zeros((3, 3, 14)), rh_palm_touch, np.zeros((3, 3, 15))), axis=2)

        block_7 = np.zeros((3, 9, 32))

        tactiles = np.concatenate((block_1, block_2, block_3, block_4, block_5, block_6, block_7), axis=1)

        tactiles = np.sign(tactiles) * np.log(1 + np.abs(tactiles))
        observation['tactile'] = tactiles
        
        return observation
