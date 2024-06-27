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

if __name__ == "__main__":

    n_episodes = 10
    n_steps = 300
    
    env_name = "HandManipulateBlockRotateZFixed-v1" # can also be HandManipulateEggRotateFixed-v1 or HandManipulatePenRotateFixed-v1

    env = gym.make(env_name, render_mode="rgb_array", reward_type='dense')
    env = PixelObservationWrapper(env, pixel_keys=('image',))
    # env = ResizeDict(env, 64, pixel_key='image')
    env = AddTactile(env)
    
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
        
        