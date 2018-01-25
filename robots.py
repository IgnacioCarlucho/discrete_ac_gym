import numpy as np
import gym
from gym.envs.registration import register, spec
import time 

# down = 1
# left = 0 
# up = 3
# rigth = 2

class gym_environment(object):

    def __init__(self,name, is_slippery = False, render=False, temp=False):

        self.MY_ENV_NAME = name
        self.render = render
        self.temp = temp
        try:
            self.spec = spec(self.MY_ENV_NAME)
        except:
            print('Environment not found, using default Environment: frozen lake')
            self.register = register(
                id=self.MY_ENV_NAME,
                entry_point='gym.envs.toy_text:FrozenLakeEnv',
                kwargs={'map_name': '4x4', 'is_slippery': False},
                timestep_limit=100,
                reward_threshold=0.8196, # optimum = .8196
            )
        
        self.env = gym.make(self.MY_ENV_NAME)
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.n
        self.uncodedstate = self.env.reset()
        self.state = self.to_one_hot(self.uncodedstate)
        self.reward = -1.
        self.info = []
        self.action = []
        self.done = False
        self.step = 0
        self.goal = 0


    def update(self, action):
        
        

        self.action = action
        self.uncodedstate, self.reward, self.done, self.info = self.env.step(self.action)
        self.state = self.to_one_hot(self.uncodedstate)
        self.step = self.step + 1
        if self.uncodedstate == (self.state_dim-1):
            # I reached the goal
            self.goal = self.goal + 1
            #self.reward = 10.

        self.development() # this is just in case you want to render, or slow down the execution
        return self.state, self.reward, self.done, self.step

    def reset(self):
        self.uncodedstate = self.env.reset()
        self.state = self.to_one_hot(self.uncodedstate)
        self.reward = -1.
        self.info = []
        self.action = []
        self.done = False
        self.step = 0
        return self.state, self.done, self.step 

    def to_one_hot(self, number):
        one_hot = np.zeros(self.state_dim)
        one_hot[number] = 1.
        return one_hot
    
    def development(self):
        if self.render: 
            self.env.render()
        if self.temp:
            if (self.step %10 == 0):
                time.sleep(1) 