import os
import cv2 

import gymnasium as gym

import mujoco

import numpy as np
from gymnasium import spaces
import cv2

from pathlib import Path

def convert_observation_to_space(observation):
    
    space = spaces.Dict(spaces={})
    for key in observation.keys():
        if key == 'image':
            space.spaces[key] = spaces.Box(low = 0, high = 1, shape = observation[key].shape, dtype = np.float64)
        elif key == 'tactile' or key == 'state':
            space.spaces[key] = spaces.Box(low = -float('inf'), high = float('inf'), shape = observation[key].shape, dtype = np.float64)
        
    return space

class InsertionEnv(gym.Env):

    def __init__(self, no_rotation=True, 
        no_gripping=True, state_type='vision_and_touch', camera_idx=0, symlog_tactile=True, 
        env_id = -1, im_size=64, tactile_shape=(32,32), skip_frame=10, max_delta=None, multiccd=False,
        objects = ["square", "triangle", "horizontal", "vertical", "trapezoidal", "rhombus"],
        holders = ["holder1", "holder2", "holder3"]):

        """
        'no_rotation': if True, the robot will not be able to rotate its wrist
        'no_gripping': if True, the robot will keep the gripper opening at a fixed value
        'state_type': choose from 'privileged', 'vision', 'touch', 'vision_and_touch'
        'camera_idx': index of the camera to use
        'symlog_tactile': if True, the tactile values will be squashed using the symlog function
        'env_id': environment id
        'im_size': side of the square image
        'tactile_shape': shape of the tactile sensor (rows, cols)
        'skip_frame': number of frames to skip between actions
        'max_delta': maximum change allowed in the x, y, z position
        'multiccd': if True, the multiccd flag will be enabled (makes tactile sensing more accurate but slower)
        'objects': list of objects to insert (list from "square", "triangle", "horizontal", "vertical", "trapezoidal", "rhombus")
        'holders': list of holders to insert the objects (list from "holder1", "holder2", "holder3")
        """

        super(InsertionEnv, self).__init__()

        self.id = env_id

        self.skip_frame = skip_frame
        
        asset_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets')

        self.model_path = os.path.join(asset_folder, 'insertion/scene.xml')
        self.current_dir = os.path.join(Path(__file__).parent.parent.absolute(), 'assets/insertion')
        with open(self.model_path,"r") as f:
            self.xml_content = f.read()
            self.update_include_path()
            self.xml_content_reference = self.xml_content

        self.multiccd = multiccd

        self.fixed_gripping = 200

        self.max_delta = max_delta

        self.symlog_tactile = symlog_tactile # used to squash tactile values and avoid large spikes

        self.tactile_rows = tactile_shape[0]
        self.tactile_cols = tactile_shape[1]
        self.tactile_comps = 3

        self.im_size = im_size

        self.state_type = state_type

        self.holders = holders
        self.objects = objects

        print("state_type: ", self.state_type)

        if self.state_type == 'privileged':
            self.curr_obs = {'state': np.zeros(8)}
        elif self.state_type == 'vision':
            self.curr_obs = {'image': np.zeros((self.im_size, self.im_size, 3))}
        elif self.state_type == 'touch':
            self.curr_obs = {'tactile': np.zeros((2 * self.tactile_comps, self.tactile_rows, self.tactile_cols))}
        elif self.state_type == 'vision_and_touch':
            self.curr_obs = {'image': np.zeros((self.im_size, self.im_size, 3)), 'tactile': np.zeros((2 * self.tactile_comps, self.tactile_rows, self.tactile_cols))}
        else:
            raise ValueError("Invalid state type")
        
        self.sim = mujoco.MjModel.from_xml_string(self.xml_content)
        self.mj_data = mujoco.MjData(self.sim)
        if self.multiccd:
            self.sim.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD

        

        self.init_z = self.mj_data.qpos[-5]

        self.adaptive_gripping = not no_gripping
        self.with_rotation = not no_rotation

        self.camera_idx = camera_idx        
        
        obs_tmp = self._get_obs()
        self.observation_space = convert_observation_to_space(obs_tmp)
        
        self.ndof_u = 5
        if no_rotation:
            self.ndof_u -= 1
        if no_gripping:
            self.ndof_u -= 1

        print("ndof_u: ", self.ndof_u)
        
        self.action_space = spaces.Box(low = np.full(self.ndof_u, -1.), high = np.full(self.ndof_u, 1.), dtype = np.float32)
        self.action_scale = np.array([[-0.2,0.2],[-0.2,0.2],[-0.12,0.3],[-np.pi,np.pi],[0,255]])

        self.action_mask = np.ones(5, dtype=bool)
        if no_rotation:
            self.action_mask[3] = False
        if no_gripping:
            self.action_mask[4] = False
        self.action_scale = self.action_scale[self.action_mask]
        
        self.renderer = mujoco.Renderer(self.sim, height=self.im_size, width=self.im_size)

    def update_include_path(self):
        
        file_idx = self.xml_content.find('<include file="', 0)
        while file_idx != -1:
            file_start_idx = file_idx + len('<include file="')
            self.xml_content = self.xml_content[:file_start_idx] + self.current_dir + '/' + self.xml_content[file_start_idx:]

            file_idx = self.xml_content.find('<include file="', file_start_idx + len(self.current_dir))

        file_idx = self.xml_content.find('meshdir="', 0)
        file_start_idx = file_idx + len('meshdir="')
        self.xml_content = self.xml_content[:file_start_idx] + self.current_dir + '/' + self.xml_content[file_start_idx:]


    def edit_xml(self):
        
        holders = self.holders
        objects = self.objects

        self.xml_content = self.xml_content_reference

        def edit_attribute(attribute, offset_x, offset_y, offset_yaw, holder, object):
            box_idx = self.xml_content.find('<body name="' + attribute + '"')
            if box_idx == -1:
                 print("ERROR: Could not find joint name: " + attribute)
                 return False
            
            pos_key = 'pos="'
            pos_idx = box_idx + self.xml_content[box_idx:].find(pos_key)
            pos_start_idx = pos_idx + len(pos_key)
            pos_end_idx = pos_start_idx + self.xml_content[pos_start_idx:].find('"')

            pos = self.xml_content[pos_start_idx:pos_end_idx].split(" ")
            correction_rot = np.array([float(pos[0]), float(pos[1])])
            rotMatrix = np.array([[np.cos(offset_yaw), -np.sin(offset_yaw)], 
                         [np.sin(offset_yaw),  np.cos(offset_yaw)]])
            correction_rot = rotMatrix.dot(correction_rot)
            
            new_pos = [str(offset_x + correction_rot[0]), str(offset_y + correction_rot[1]), str(float(pos[2]))]
            new_pos_str = " ".join(new_pos)
            
            self.xml_content = self.xml_content[:pos_start_idx] + new_pos_str + self.xml_content[pos_end_idx:]

            euler_key = 'axisangle="'
            euler_idx = box_idx + self.xml_content[box_idx:].find(euler_key)
            euler_start_idx = euler_idx + len(euler_key)
            euler_end_idx = euler_start_idx + self.xml_content[euler_start_idx:].find('"')

            euler = self.xml_content[euler_start_idx:euler_end_idx].split(" ")
            new_euler = [str(float(euler[0])), str(float(euler[1])), str(float(euler[2])), str(float(euler[3]) + offset_yaw)]
            new_euler_str = " ".join(new_euler)
            
            self.xml_content = self.xml_content[:euler_start_idx] + new_euler_str + self.xml_content[euler_end_idx:]
            
            if attribute == 'object':
                for key in ['peg_visual', 'peg_collision']:
                    key_idx = euler_end_idx + self.xml_content[euler_end_idx:].find('name="' + key + '"')
                    key_end_idx = key_idx + len('name="' + key + '"')
            
                    mesh_idx = key_end_idx + self.xml_content[key_end_idx:].find('mesh="')
                    mesh_start_idx = mesh_idx + len('mesh="')
                    mesh_end_idx = mesh_start_idx + self.xml_content[mesh_start_idx:].find('"')

                    self.xml_content = self.xml_content[:mesh_start_idx] + object + self.xml_content[mesh_end_idx:]

                for key in ['holder_visual', 'holder_collision']:
                    key_idx = euler_end_idx + self.xml_content[euler_end_idx:].find('name="' + key + '"')
                    key_end_idx = key_idx + len('name="' + key + '"')
                    
                    mesh_idx = key_end_idx + self.xml_content[key_end_idx:].find('mesh="')
                    mesh_start_idx = mesh_idx + len('mesh="')
                    mesh_end_idx = mesh_start_idx + self.xml_content[mesh_start_idx:].find('"')

                    self.xml_content = self.xml_content[:mesh_start_idx] + holder + self.xml_content[mesh_end_idx:]
                    
            else:
                for i in range(1,5):
                    for key in ['wall{}_visual'.format(i), 'wall{}_collision'.format(i)]:
                        key_idx = euler_end_idx + self.xml_content[euler_end_idx:].find('name="' + key + '"')
                        key_end_idx = key_idx + len('name="' + key + '"')
                    
                        mesh_idx = key_end_idx + self.xml_content[key_end_idx:].find('mesh="')
                        mesh_start_idx = mesh_idx + len('mesh="')
                        mesh_end_idx = mesh_start_idx + self.xml_content[mesh_start_idx:].find('"')

                        self.xml_content = self.xml_content[:mesh_start_idx] + object + '_wall' + str(i) + self.xml_content[mesh_end_idx:]
                
            return True
            
        offset_x = 0.05*np.random.rand()
        offset_y = 0.05*np.random.rand()

        if self.with_rotation:
            offset_yaw = 2*np.pi*np.random.rand()-np.pi
        else:
            offset_yaw = 0.

        holder = np.random.choice(holders)
        object = np.random.choice(objects)

        edit_attribute("object", offset_x, offset_y, offset_yaw, holder, object)
        edit_attribute("walls", offset_x, offset_y, offset_yaw, holder, object)

        self.offset_x = offset_x
        self.offset_y = offset_y
        self.target_quat = np.array([np.cos(offset_yaw/2), 0, 0, np.sin(offset_yaw/2)])

    def generate_initial_pose(self, show_full=False):
        
        cruise_height = 0.
        gripping_height = -0.11
        
        mujoco.mj_resetData(self.sim, self.mj_data)

        rand_x = np.random.rand()*0.2 - 0.1
        rand_y = np.random.rand()*0.2 - 0.1
        if self.with_rotation:
            rand_yaw = np.random.rand()*2*np.pi - np.pi
        else:
            rand_yaw = 0

        steps_per_phase = 60

        for i in range(steps_per_phase): # go on top of object
            self.mj_data.ctrl[:3] = [self.offset_x, self.offset_y, cruise_height]
            mujoco.mj_step(self.sim, self.mj_data, self.skip_frame+1)
            if show_full:
                self.renderer.update_scene(self.mj_data, camera=0)
                img = cv2.cvtColor(self.renderer.render(), cv2.COLOR_BGR2RGB)
                cv2.imshow('img', img)
                cv2.waitKey(1)

        for i in range(steps_per_phase): # rotate wrist
            self.mj_data.ctrl[3] = -np.arcsin(self.target_quat[-1])*2
            mujoco.mj_step(self.sim, self.mj_data, self.skip_frame+1)
            if show_full:
                self.renderer.update_scene(self.mj_data, camera=0)
                img = cv2.cvtColor(self.renderer.render(), cv2.COLOR_BGR2RGB)
                cv2.imshow('img', img)
                cv2.waitKey(1)
            
        for i in range(steps_per_phase): # move around object
            self.mj_data.ctrl[:3] = [self.offset_x, self.offset_y, gripping_height]
            mujoco.mj_step(self.sim, self.mj_data, self.skip_frame+1)
            if show_full:
                self.renderer.update_scene(self.mj_data, camera=0)
                img = cv2.cvtColor(self.renderer.render(), cv2.COLOR_BGR2RGB)
                cv2.imshow('img', img)
                cv2.waitKey(1)
            
        for i in range(steps_per_phase): # close gripper
            self.mj_data.ctrl[-1] = self.fixed_gripping
            mujoco.mj_step(self.sim, self.mj_data, self.skip_frame+1)
            if show_full:
                self.renderer.update_scene(self.mj_data, camera=0)
                img = cv2.cvtColor(self.renderer.render(), cv2.COLOR_BGR2RGB)
                cv2.imshow('img', img)
                cv2.waitKey(1)
            
        for i in range(steps_per_phase): # lift object
            self.mj_data.ctrl[:3] = [self.offset_x, self.offset_y, cruise_height]
            mujoco.mj_step(self.sim, self.mj_data, self.skip_frame+1)
            if show_full:
                self.renderer.update_scene(self.mj_data, camera=0)
                img = cv2.cvtColor(self.renderer.render(), cv2.COLOR_BGR2RGB)
                cv2.imshow('img', img)
                cv2.waitKey(1)

        
        for i in range(steps_per_phase): # rotate in place
            self.mj_data.ctrl[3] = -rand_yaw
            mujoco.mj_step(self.sim, self.mj_data, self.skip_frame+1)
            if show_full:
                self.renderer.update_scene(self.mj_data, camera=0)
                img = cv2.cvtColor(self.renderer.render(), cv2.COLOR_BGR2RGB)
                cv2.imshow('img', img)
                cv2.waitKey(1)
            
        for i in range(steps_per_phase): # move to random position
            self.mj_data.ctrl[:3] = [rand_x, rand_y, cruise_height]
            mujoco.mj_step(self.sim, self.mj_data, self.skip_frame+1)
            if show_full:
                self.renderer.update_scene(self.mj_data, camera=0)
                img = cv2.cvtColor(self.renderer.render(), cv2.COLOR_BGR2RGB)
                cv2.imshow('img', img)
                cv2.waitKey(1)

        self.prev_action_xyz = np.array([rand_x, rand_y, cruise_height])

        pos = self.mj_data.qpos[-7:-4]
        
        if pos[2] < (cruise_height - gripping_height)/2:
            print('Failed to grasp')

    def _get_obs(self):
        return self.curr_obs
    
    def get_privileged(self):
        idxs = [0,1,2,12,13,14]
        return np.append(self.mj_data.qpos[idxs].copy(),[self.offset_x,self.offset_y])
    
    def seed(self, seed):
        np.random.seed(seed)
    
    def reset(self, seed=None, options=None):

        if seed is not None:
            np.random.seed(seed)
        
        # Reload XML (and update robot)
        self.edit_xml()
        self.sim = mujoco.MjModel.from_xml_string(self.xml_content)
        
        self.mj_data = mujoco.MjData(self.sim)
        if self.multiccd:
            self.sim.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD 
        
        del self.renderer
        self.renderer = mujoco.Renderer(self.sim, height=self.im_size, width=self.im_size)
        
        self.generate_initial_pose()

        if self.state_type == 'vision_and_touch': 
            tactiles_right = self.mj_data.sensor('touch_right').data.reshape((3, self.tactile_rows, self.tactile_cols))
            tactiles_right = tactiles_right[[1, 2, 0]] # zxy -> xyz
            tactiles_left = self.mj_data.sensor('touch_left').data.reshape((3, self.tactile_rows, self.tactile_cols))
            tactiles_left = tactiles_left[[1, 2, 0]] # zxy -> xyz
            tactiles = np.concatenate((tactiles_right, tactiles_left), axis=0)
            if self.symlog_tactile:
                tactiles = np.sign(tactiles) * np.log(1 + np.abs(tactiles))
            img = self.render()
            self.curr_obs = {'image': img, 'tactile': tactiles}
        elif self.state_type == 'vision':
            img = self.render()
            self.curr_obs = {'image': img}
        elif self.state_type == 'touch':
            tactiles_right = self.mj_data.sensor('touch_right').data.reshape((3, self.tactile_rows, self.tactile_cols))
            tactiles_right = tactiles_right[[1, 2, 0]] # zxy -> xyz
            tactiles_left = self.mj_data.sensor('touch_left').data.reshape((3, self.tactile_rows, self.tactile_cols))
            tactiles_left = tactiles_left[[1, 2, 0]] # zxy -> xyz
            tactiles = np.concatenate((tactiles_right, tactiles_left), axis=0)
            if self.symlog_tactile:
                tactiles = np.sign(tactiles) * np.log(1 + np.abs(tactiles))
            self.curr_obs = {'tactile': tactiles}
        elif self.state_type == 'privileged':
            self.curr_obs = np.append(self.mj_data.qpos.copy(),[self.offset_x,self.offset_y])
        
        info = {'id': np.array([self.id])}

        return self._get_obs(), info


    def render(self, highres = False):
        
        if highres:
            del self.renderer
            self.renderer = mujoco.Renderer(self.sim, height=480, width=480)
            self.renderer.update_scene(self.mj_data, camera=self.camera_idx)
            img = self.renderer.render()/255
            del self.renderer
            self.renderer = mujoco.Renderer(self.sim, height=self.im_size, width=self.im_size)
        else:
            self.renderer.update_scene(self.mj_data, camera=self.camera_idx)
            img = self.renderer.render()/255

        return img

    def step(self, u):

        action = u
        action = np.clip(u, -1., 1.)
        
        action_unnorm = (action + 1)/2 * (self.action_scale[:,1]-self.action_scale[:,0]) + self.action_scale[:,0]

        if self.max_delta is not None:
            action_unnorm = np.clip(action_unnorm[:3], self.prev_action_xyz - self.max_delta, self.prev_action_xyz + self.max_delta)
        
        self.prev_action_xyz = action_unnorm

        if self.with_rotation:
            self.mj_data.ctrl[3] = -action_unnorm[3]
        else:
            self.mj_data.ctrl[3] = 0
        if not self.adaptive_gripping:
            self.mj_data.ctrl[-1] = self.fixed_gripping
        else:
            self.mj_data.ctrl[-1] = action_unnorm[-1]
    
        self.mj_data.ctrl[:3] = action_unnorm[:3]

        mujoco.mj_step(self.sim, self.mj_data, self.skip_frame+1)

        pos = self.mj_data.qpos[-7:-4]
        quat = self.mj_data.qpos[-4:]
        
        delta_x = pos[0] - self.offset_x
        delta_y = pos[1] - self.offset_y
        delta_z = pos[2] - self.init_z
        delta_quat = np.linalg.norm(quat - self.target_quat)

        reward = - np.log(100*np.sqrt(delta_x**2 + delta_y**2 + delta_z**2 + int(self.with_rotation)*delta_quat**2) + 1)
        
        if self.state_type == 'vision_and_touch': 
            tactiles_right = self.mj_data.sensor('touch_right').data.reshape((3, self.tactile_rows, self.tactile_cols))
            tactiles_right = tactiles_right[[1, 2, 0]] # zxy -> xyz
            tactiles_left = self.mj_data.sensor('touch_left').data.reshape((3, self.tactile_rows, self.tactile_cols))
            tactiles_left = tactiles_left[[1, 2, 0]] # zxy -> xyz
            tactiles = np.concatenate((tactiles_right, tactiles_left), axis=0)
            if self.symlog_tactile:
                tactiles = np.sign(tactiles) * np.log(1 + np.abs(tactiles))
            img = self.render()
            self.curr_obs = {'image': img, 'tactile': tactiles}
            info = {'id': np.array([self.id])}
        elif self.state_type == 'vision':
            img = self.render()
            self.curr_obs = {'image': img}
            info = {'id': np.array([self.id])}
        elif self.state_type == 'touch':
            tactiles_right = self.mj_data.sensor('touch_right').data.reshape((3, self.tactile_rows, self.tactile_cols))
            tactiles_right = tactiles_right[[1, 2, 0]]
            tactiles_left = self.mj_data.sensor('touch_left').data.reshape((3, self.tactile_rows, self.tactile_cols))
            tactiles_left = tactiles_left[[1, 2, 0]]
            tactiles = np.concatenate((tactiles_right, tactiles_left), axis=0)
            if self.symlog_tactile:
                tactiles = np.sign(tactiles) * np.log(1 + np.abs(tactiles))
            self.curr_obs = {'tactile': tactiles}
            info = {'id': np.array([self.id])}
        elif self.state_type == 'privileged':
            self.curr_obs = np.append(self.mj_data.qpos.copy(),[self.offset_x,self.offset_y])
            info = {'id': np.array([self.id])}

        done = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2) < 4e-3
        info['is_success'] = done

        if done:
            reward = 1000

        obs = self._get_obs()

        return obs, reward, done, False, info
        
