# import gym
import gymnasium as gym
import tactile_envs
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2 
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper
from tactile_envs.utils.add_tactile import AddTactile
from tactile_envs.utils.resize_dict import ResizeDict
import robosuite as suite
from robosuite.wrappers.tactile_wrapper import TactileWrapper
from robosuite import load_controller_config


if __name__ == "__main__":

    n_episodes = 10
    n_steps = 300

    robots = ["PandaTactile"]
    placement_initializer = None
    init_qpos = [-0.073, 0.016, -0.392, -2.502, 0.240, 2.676, 0.189]
    env_config = {}
    env_config["robot_configs"] = [{"initial_qpos": init_qpos}]
    env_config["initialization_noise"] = None

    config = load_controller_config(default_controller="OSC_POSE")
    
    env_name = "Door"
    env = TactileWrapper(
                suite.make(
                    env_name,
                    robots=robots,  # use PandaTactile robot
                    use_camera_obs=True,  # use pixel observations
                    use_object_obs=False,
                    has_offscreen_renderer=True,  # needed for pixel obs
                    has_renderer=False,  # not needed due to offscreen rendering
                    reward_shaping=True,  # use dense rewards
                    camera_names="agentview",
                    horizon=300,
                    controller_configs=config,
                    placement_initializer=placement_initializer,
                    camera_heights=480,
                    camera_widths=480,
                    **env_config,
                ),
                env_id=0,
                state_type='vision_and_touch',
            )
    
    for j in range(n_episodes):
        seed = np.random.randint(0,1000)
        print("seed: ", seed)
        env.reset(**{'seed': seed})
        tic = time.time()

        for i in range(n_steps):
            # Take a random action
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            cv2.imshow('img', obs['image'][:,:,::-1])
            cv2.waitKey(1)
            
            if done == True:
                print("Done")
                break

        print("Number of steps: ", i)
        print("frequency: ", i/(time.time()-tic))
        
        