from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *

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

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # Choose environment and add it to options
    options["env_name"] = choose_environment()

    # If a multi-arm environment has been chosen, choose configuration and appropriate robot(s)
    if "TwoArm" in options["env_name"]:
        # Choose env config and add it to options
        options["env_configuration"] = choose_multi_arm_config()

        # If chosen configuration was bimanual, the corresponding robot must be Baxter. Else, have user choose robots
        if options["env_configuration"] == "bimanual":
            options["robots"] = "Baxter"
        else:
            options["robots"] = []

            # Have user choose two robots
            print("A multiple single-arm configuration was chosen.\n")

            for i in range(2):
                print("Please choose Robot {}...\n".format(i))
                options["robots"].append(choose_robots(exclude_bimanual=True))

    # Else, we simply choose a single (single-armed) robot to instantiate in the environment
    else:
        options["robots"] = choose_robots(exclude_bimanual=True)

    # Choose controller
    controller_name = choose_controller()

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    env.reset()
    env.viewer.set_camera(camera_id=0)

    # Get action limits
    low, high = env.action_spec

    # do visualization
    for i in range(10000):
        action = np.random.uniform(low, high)*0
        action[3] = -1
        action[7] = -1
        obs, reward, done, _ = env.step(action)
        env.render()
        if "robot0_tactile_right" in obs:
            show_tactile(obs["robot0_tactile_right"], name="robot0_tactile_right")
            show_tactile(obs["robot0_tactile_left"], name="robot0_tactile_left")
            cv2.waitKey(1)
        if "robot1_tactile_left" in obs:
            show_tactile(obs["robot1_tactile_right"], name="robot1_tactile_right")
            show_tactile(obs["robot1_tactile_left"], name="robot1_tactile_left")
            cv2.waitKey(1)
