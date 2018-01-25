import numpy as np
import tensorflow as tf
from actor import ActorNetwork
from critic import CriticNetwork
from robots import gym_environment
import gym
from gym.envs.registration import register, spec
robot = gym_environment('FrozenLakeNonskid8x8-v0', True)
import time

# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE =  0.001
# Gym environment


BUFFER_SIZE = 10000
RANDOM_SEED = 1234
EXPLORE = 150

# REMBER ACTOR CRITIC IS ON POLICY ALGORITHM

def actor_critic(epochs=1000, GAMMA = 0.99, train_indicator=True, render=False, temp = False):
    with tf.Session() as sess:
        
        
        # define objects
        # the gym environment is wrapped in a class. this way of working allows portability with other robots in the lab & makes the main very clear
        robot = gym_environment('FrozenLakeNonskid8x8-v0', False, render, temp) 
        actor = ActorNetwork(sess, robot.state_dim, robot.action_dim, ACTOR_LEARNING_RATE)
        critic = CriticNetwork(sess, robot.state_dim, CRITIC_LEARNING_RATE, actor.get_num_trainable_vars())
        # starting tensorflow
        sess.run(tf.global_variables_initializer())
        
        
        
        for i in range(epochs):
            # Reset the environment and pick the fisrst action
            state, done, step = robot.reset()
            ep_reward = 0
            

            
            while (not done):
                
                
                action_prob = actor.predict(np.reshape(state,(1,robot.state_dim)))
                action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
                next_state, reward, done, step = robot.update(action)
                
                #print(step,'action', action, 'state', robot.uncodedstate,'r', round(reward,3), 'prob', action_prob)
                
                V_minib = critic.predict(np.reshape(state, (1, robot.state_dim)))
                V_minib_next = critic.predict(np.reshape(next_state, (1, robot.state_dim)))
                # 5.2 get y_t where: 
                td_target = []
                td_error = []
                if done:
                    td_target = reward
                    td_error = reward - V_minib # not - V_minib[k] ?
                else:
                    td_target = reward + GAMMA*V_minib_next
                    td_error = reward + GAMMA*V_minib_next - V_minib
                
                critic.train(np.reshape(state, (1, robot.state_dim)), np.reshape(td_target, (1, 1)))
                             
                actor.train(np.reshape(state,(1,robot.state_dim)), np.reshape(action, (1, 1)), np.reshape(td_error, (1, 1)) )
              
                state = next_state
                ep_reward = ep_reward + reward
                       
            #print(step,'action', action, 'state', robot.uncodedstate,'r', round(reward,3), 'prob', action_prob)
            print("episode" i+1,'Steps', step,'Reward:',ep_reward,'goal achieved:', robot.goal, 'Efficiency', round(100.*((robot.goal)/(i+1.)),0), '%' )
            
            
            
               

               
        print('*************************')
        print('now we save the model')
        critic.save_critic()
        actor.save_actor()
        print('model saved succesfuly')
        print('*************************')




if __name__ == '__main__':
    actor_critic(epochs=2000)       
        