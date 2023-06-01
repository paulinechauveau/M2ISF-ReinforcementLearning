# -*- coding: utf-8 -*-
"""
@author: pauline

Playing games in gym

"""

# !pip install gymnasium
# !pip install pygame

import matplotlib.pyplot as plt
import gymnasium as gym


# Frozen Lake

env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='rgb_array')

state, info = env.reset()

screen = env.render()
if len(screen>0):
    plt.imshow(screen)
    
    
## Random   
 
action = env.action_space.sample()
new_state, reward, terminated, truncated, info = env.step(action)
rewards = []    
steps = [state, new_state]
actions = [action]

for _ in range(1000):
   action = env.action_space.sample()
   new_state, reward, terminated, truncated, info = env.step(action)
   
   rewards.append(reward)
   steps.append(new_state)
   actions.append(action)
   
   screen = env.render()
   plt.imshow(screen)
   plt.show()

   if terminated or truncated:
      observation, info = env.reset()
      if reward == 1:
          break
      steps = []
      action = []
      rewards = []  
env.close()

## Manually

state, info = env.reset()
screen = env.render()
plt.imshow(screen)
commands = {"L":0,"R":3,"U":1,"D":2}
for _ in range(100):
    prompt = input('where the little man has to go ? (L/R/U/D): ')
    prompt = commands[prompt]
    print('command=',prompt)
    new_state, reward, terminated, truncated, info = env.step(prompt)
    screen = env.render()
    plt.imshow(screen)
    plt.show()
    if terminated or truncated or reward==1:
        break
env.close()



# MountainCar

env = gym.make('MountainCar-v0', render_mode='rgb_array')

state, info = env.reset()

screen = env.render()
if len(screen>0):
    plt.imshow(screen)
    
## Random   
 
action = env.action_space.sample()
new_state, reward, terminated, truncated, info = env.step(action)
rewards = []    
steps = [state, new_state]
actions = [action]

for _ in range(1000):
   action = env.action_space.sample()
   new_state, reward, terminated, truncated, info = env.step(action)
   
   rewards.append(reward)
   steps.append(new_state)
   actions.append(action)
   
   screen = env.render()
   plt.imshow(screen)
   plt.show()

   if terminated or truncated:
      break
      # observation, info = env.reset()
      # steps = []
      # action = []
      # rewards = []  
env.close()
   

## Manually

state, info = env.reset()
screen = env.render()
plt.imshow(screen)
states = [state]
commands = {"L":0,"R":2,"S":1}
for _ in range(100):
    prompt = input('where the car has to go ? (L/R/S): ')
    prompt = commands[prompt]
    print('command=',prompt)
    new_state, reward, terminated, truncated, info = env.step(prompt)
    states.append(new_state)
    screen = env.render()
    plt.imshow(screen)
    plt.show()
    if terminated or truncated:
        break
env.close()