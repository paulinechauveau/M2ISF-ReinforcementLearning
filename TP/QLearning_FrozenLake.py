# -*- coding: utf-8 -*-
"""
@author: pauline

Q-Learning for Frozen Lake
"""

# !pip install gym
# !pip install pygame
# !pip install gym[toy_text]

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm


# Load environnement

env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='rgb_array')
state, info = env.reset()

screen = env.render()
plt.imshow(screen)

# Set hyperparameters

learning_rate = 0.02 #alpha
discount_factor = 0.95 #gamma
epsilon = 0.1
n_episodes = 100000
n_steps = 99

# Initialize qtable

action_size = env.action_space.n 
state_size = env.observation_space.n 
qtable = np.zeros((state_size, action_size))

# Q-learning algorithm

rewards = []    

for episode in tqdm(range(n_episodes)):
    
    state, info = env.reset() #initialize S
    terminated = False
    truncated = False
    total_reward = 0
    
    for step in range(n_steps):
        
        # e-greedy strategy to choose action
        x = np.random.rand()
        if(x<epsilon):      #exploration
            action = np.random.choice(action_size) 
        else:               #exploitation
            action = np.argmax(qtable[state,:]) 
    
        new_state, reward, terminated, truncated, info = env.step(action)
        
        qtable[state,action] = qtable[state,action] + learning_rate * (reward + discount_factor*np.max(qtable[new_state,:]) - qtable[state,action])
        total_reward += reward
        
        state = new_state
        
        # screen = env.render()
        # plt.imshow(screen)
        # plt.show()
        
        if terminated or truncated :
            break

        
    rewards.append(total_reward)
     
env.close()


# visualization Q table for FrozenLake 

def plotQ(q_table,env):
  MAP=env.desc
  map_size=MAP.shape[0]
  best_value = np.max(q_table, axis = 1).reshape((map_size,map_size))
  best_policy = np.argmax(q_table, axis = 1).reshape((map_size,map_size))
    
  fig, ax = plt.subplots()
  im = ax.imshow(best_value,cmap=plt.cm.Wistia)#Pastel1, spring, autumn
  arrow_list=['<','v','>','^']
  for i in range(best_value.shape[0]):
      for j in range(best_value.shape[1]):
          if MAP[i][j].decode('utf-8') in 'GH':#terminal states
            arrow = MAP[i][j].decode('utf-8')
          else :
            arrow=arrow_list[best_policy[i, j]]
          if MAP[i][j].decode('utf-8') in 'S':
              arrow = 'S ' + arrow
          ax.text(j, i, arrow, ha = "center", va = "center",
                         color = "black")
          ax.text(j, i+0.2, str(np.round(best_value[i,j],2)), 
                          ha = "center", va = "center",color = "blue")
          
            
  ax.figure.colorbar(im, ax = ax)
  plt.axis('off')  
  fig.tight_layout()
  plt.show() 
plotQ(qtable,env)


# Use Q-table to play Frozen Lake

rewards = []
n_test_episodes = 5

for episode in range(n_test_episodes):
    state, info = env.reset()
    done = False
    total_rewards = 0

    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(n_steps):
        
        action = np.argmax(qtable[state,:])
        new_state, reward, terminated ,truncated, info = env.step(action)
        
        total_rewards +=reward   
        
        screen = env.render()
        plt.imshow(screen)
        plt.show()
        
        if terminated or truncated:
 
            screen = env.render()
            plt.imshow(screen)
            plt.show()

            print("Number of steps", step)
            break
        state = new_state
    rewards.append(total_rewards)

print ("Score over time: " +  str(sum(rewards)/n_test_episodes))

env.close()



