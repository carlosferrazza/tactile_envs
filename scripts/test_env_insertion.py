# import gym
import gymnasium as gym
import tactile_envs
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2 

def show_tactile(tactile, size=(480,480), max_shear=0.05, max_pressure=0.1, name='tactile'): # Note: default params work well for 16x16 or 32x32 tactile sensors, adjust for other sizes
    nx = tactile.shape[2]
    ny = tactile.shape[1]

    loc_x = np.linspace(0,size[1],nx)
    loc_y = np.linspace(size[0],0,ny)

    img = np.zeros((size[0],size[1],3))

    for i in range(0,len(loc_x),1):
        for j in range(0,len(loc_y),1):
            
            dir_x = np.clip(tactile[0,j,i]/max_shear,-1,1) * 20
            dir_y = np.clip(tactile[1,j,i]/max_shear,-1,1) * 20

            color = np.clip(tactile[2,j,i]/max_pressure,0,1)
            r = color
            g = 1-color

            cv2.arrowedLine(img, (int(loc_x[i]),int(loc_y[j])), (int(loc_x[i]+dir_x),int(loc_y[j]-dir_y)), (0,g,r), 4, tipLength=0.5)

    cv2.imshow(name, img)

    return img

if __name__ == "__main__":

    show_highres = False # Set to True to show highres images (slow!)
    
    n_episodes = 10
    n_steps = 300
    
    env = gym.make("tactile_envs/Insertion-v0", state_type='vision_and_touch', multiccd=False, im_size=64, no_gripping=True, no_rotation=True, tactile_shape=(32,32), max_delta=None)
    
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

            img_tactile1 = show_tactile(obs['tactile'][:3], name='tactile1')
            img_tactile2 = show_tactile(obs['tactile'][3:], name='tactile2')

            if show_highres:
                img = env.unwrapped.render(highres=True)
                cv2.imshow('img', img[:,:,::-1])
            else:
                cv2.imshow('img', obs['image'][:,:,::-1])
            cv2.waitKey(1)
            
            if done == True:
                print("Done")
                break

        print("Number of steps: ", i)
        print("frequency: ", i/(time.time()-tic))
        
        