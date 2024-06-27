"""
This script shows how to adapt an environment to be compatible
with the old Gym API. This is useful when using
learning pipelines that require supporting these APIs.
"""

import robosuite as suite
from robosuite.wrappers.old_gym_wrapper import OldGymWrapper
from robosuite import load_controller_config
from robosuite.utils.placement_samplers import UniformRandomSampler

import cv2
import numpy as np

import sys 

def show_tactile(tactile, size=(400,400), max_shear=0.1, max_pressure=0.2, name='tactile'):

    nx = tactile.shape[2]
    ny = tactile.shape[1]

    loc_x = np.linspace(0,size[1],nx)
    loc_y = np.linspace(size[0],0,ny)

    img = np.zeros((size[0],size[1],3))

    for i in range(len(loc_x)):
        for j in range(len(loc_y)):
            dir_x = np.clip(tactile[1,j,i]/max_shear,-1,1) * 7
            dir_y = np.clip(tactile[2,j,i]/max_shear,-1,1) * 7

            color = np.clip(tactile[0,j,i]/max_pressure,0,1)
            r = color
            g = 1-color

            cv2.arrowedLine(img, (int(loc_x[i]),int(loc_y[j])), (int(loc_x[i]+dir_x),int(loc_y[j]-dir_y)), (0,g,r), 1, tipLength=0.9)

    cv2.imshow(name, img)


if __name__ == "__main__":

    config = load_controller_config(default_controller='OSC_POSE')

    placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=None,
                x_range=[-0.0, 0.0],
                y_range=[-0.0, 0.0],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=[0,0,0.8],
                rotation=(np.pi, np.pi),
            )

    # Notice how the environment is wrapped by the wrapper
    env = OldGymWrapper(
        suite.make(
            "TwoArmLift",
            robots=["PandaTactile","PandaTactile"],  # use Sawyer robot
            use_camera_obs=True,  # do not use pixel observations
            has_offscreen_renderer=True,  # not needed since not using pixel obs
            has_renderer=False,  # make sure we can render to the screen
            reward_shaping=True,  # use dense rewards
            camera_names= "agentview",
            control_freq=20,  # control should happen fast enough so that simulation looks smooth
            controller_configs=config,
            placement_initializer=placement_initializer,
            initialization_noise=None,
            camera_heights = 64,
            camera_widths=64
        ), keys=["frontview_image"]
    )

    obs = env.reset()
    # print(obs.keys())
    # print(env.observation_space)
    # sys.exit()


    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            # env.render()
            action = env.action_space.sample()*0
            if t < 10:
                # action[5] = 0.3
                action[0] = -0.2
                
                action[7] = -0.2
            if t >= 10 and t < 20:
                action[2] = -0.2
                action[9] = -0.2
            if t >=20 and t < 30:
                action[6] = 0.2
                action[13] = 0.2
            if t >= 30:
                action[0] = 0.2
                action[7] = 0.2
            # if t >= 20 and t < 30:
                # action[2] = -1
                # action[9] = -1
            # if t > 30:
                # action[6] = 0.5
                # action[13] = 0.5
            observation, reward, terminated, info = env.step(action)
            
            robot0_tactile_left, robot0_tactile_right, robot1_tactile_left, robot1_tactile_right = np.split(observation['tactile'], 4)

            show_tactile(robot0_tactile_right, name="tactile_right")
            show_tactile(robot0_tactile_left, name="tactile_left")
            cv2.waitKey(1)

            cv2.imshow('image',observation['image'])
            cv2.waitKey(1)
    
            if terminated:
                print("Episode finished after {} timesteps".format(t + 1))
                observation, info = env.reset()
                env.close()
                break
            
