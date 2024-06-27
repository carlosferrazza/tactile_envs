"""
This script shows how to adapt an environment to be compatible
with the old Gym API. This is useful when using
learning pipelines that require supporting these APIs.
"""

import robosuite as suite
from robosuite.wrappers.old_gym_wrapper import OldGymWrapper

import cv2
import numpy as np


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

    # Notice how the environment is wrapped by the wrapper
    env = OldGymWrapper(
        suite.make(
            "TwoArmLift",
            robots=["PandaTactile","PandaTactile"],  # use Sawyer robot
            use_camera_obs=False,  # do not use pixel observations
            has_offscreen_renderer=False,  # not needed since not using pixel obs
            has_renderer=True,  # make sure we can render to the screen
            reward_shaping=True,  # use dense rewards
            control_freq=20,  # control should happen fast enough so that simulation looks smooth
        )
    )

    env.reset()

    for i_episode in range(20):
        observation = env.reset()
        for t in range(500):
            env.render()
            action = env.action_space.sample()
            observation, reward, terminated, info = env.step(action)

            show_tactile(observation["robot0_tactile_right"], name="tactile_right")
            show_tactile(observation["robot0_tactile_left"], name="tactile_left")
            cv2.waitKey(1)

            if terminated:
                print("Episode finished after {} timesteps".format(t + 1))
                observation, info = env.reset()
                env.close()
                break
