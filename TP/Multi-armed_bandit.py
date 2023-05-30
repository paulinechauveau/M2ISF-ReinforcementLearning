# -*- coding: utf-8 -*-
"""
@author: Pauline Chauveau

Implementation of the multi-armed bandit

"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Initialization

k = 10  # number of arms
T = 1000  # time steps
M = 2000  # realizations
epsilon = 0.1  # epsilon for e-greedy strategy

rewards = np.zeros((T, M))
q_star = np.zeros((k, M))

# MAB Algorithm

for m in tqdm(range(M)):

    qstar = np.random.randn(k)
    q_star[:, m] = qstar
    N = np.zeros(k)
    Q = np.zeros(k)

    for t in range(T):

        x = np.random.rand()

        if (x<epsilon):
            action = np.random.choice(k)
        else:
            action = np.argmax(Q)

        N[action] += 1
        rewards[t,m] = qstar[action] + np.random.randn()
        Q[action] = Q[action] + (rewards[t,m]-Q[action])/N[action]


# Plot mean of reward for each step 
     
mean_reward=np.mean(rewards,axis=1)

mean_maxq= np.mean(np.max(q_star,axis=0))
plt.plot(range(T),mean_reward,"-r",[0,T],[mean_maxq,mean_maxq],"-b" )
plt.legend([r'$\epsilon$=' + str(epsilon), r'mean max $q_*$'])
plt.show()


# Compare results with different initial value for Q 

rewards = np.zeros((T, M))
q_star = np.zeros((k, M))
Q0 = [0, 5, -5]
mean_reward = np.zeros((len(Q0),T))
mean_maxq= np.zeros(len(Q0))


for q in range(len(Q0)):
    
    for m in tqdm(range(M)):
    
        qstar = np.random.randn(k)
        q_star[:, m] = qstar
        N = np.zeros(k)
        Q = np.zeros(k) + Q0[q]
    
        for t in range(T):
    
            x = np.random.rand()
    
            if (x<epsilon):
                action = np.random.choice(k)
            else:
                action = np.argmax(Q)
    
            N[action] += 1
            rewards[t,m] = qstar[action] + np.random.randn()
            Q[action] = Q[action] + (rewards[t,m]-Q[action])/N[action]
    
    mean_reward[q,:] = np.mean(rewards,axis=1)
    mean_maxq[q]= np.mean(np.max(q_star,axis=0))

mean_maxq = np.mean(mean_maxq)

for q in range(len(Q0)):
    plt.plot(range(T), mean_reward[q], label=r'Q0=' + str(Q0[q]) +r' $\epsilon$=' + str(epsilon))
plt.plot([0, T], [mean_maxq, mean_maxq], label=r'mean max $q_*$')
plt.legend()
plt.show()

# UCB Method

c = 2  # (Hoeffding inequality)

rewards = np.zeros((T, M))
q_star = np.zeros((k, M))
Q0 = [0, 5, -5]
mean_reward = np.zeros((len(Q0),T))
mean_maxq= np.zeros(len(Q0))


for q in range(len(Q0)):
    
    for m in tqdm(range(M)):
    
        qstar = np.random.randn(k)
        q_star[:, m] = qstar
        N = np.zeros(k)
        Q = np.zeros(k) + Q0[q]
    
        for t in range(T):
    
            x = np.random.rand()
    
            if (x<epsilon):
                action = np.random.choice(k)
            else:
                action = np.argmax(Q - c*np.sqrt(np.log(t)/(0.001+N)))
    
            N[action] += 1
            rewards[t,m] = qstar[action] + np.random.randn()
            Q[action] = Q[action] + (rewards[t,m]-Q[action])/N[action]
    
    mean_reward[q,:] = np.mean(rewards,axis=1)
    mean_maxq[q]= np.mean(np.max(q_star,axis=0))

mean_maxq = np.mean(mean_maxq)

for q in range(len(Q0)):
    plt.plot(range(T), mean_reward[q], label=r'Q0=' + str(Q0[q]) +r' $\epsilon$=' + str(epsilon))
plt.plot([0, T], [mean_maxq, mean_maxq], label=r'mean max $q_*$')
plt.legend()
plt.show()


# UCB vs e-greedy
rewards = np.zeros((T, M))
q_star = np.zeros((k, M))

for m in tqdm(range(M)):

    qstar = np.random.randn(k)
    q_star[:, m] = qstar
    N = np.zeros(k)
    Q = np.zeros(k)

    for t in range(T):

        x = np.random.rand()

        if (x<epsilon):
            action = np.random.choice(k)
        else:
            action = np.argmax(Q)

        N[action] += 1
        rewards[t,m] = qstar[action] + np.random.randn()
        Q[action] = Q[action] + (rewards[t,m]-Q[action])/N[action]


mean_reward=np.mean(rewards,axis=1)
plt.plot(range(T), mean_reward, label=r' $\epsilon$=' + str(epsilon))

rewards = np.zeros((T, M))
q_star = np.zeros((k, M))

for m in tqdm(range(M)):

    qstar = np.random.randn(k)
    q_star[:, m] = qstar
    N = np.zeros(k)
    Q = np.zeros(k)

    for t in range(T):

        x = np.random.rand()

        if (x<epsilon):
            action = np.random.choice(k)
        else:
            action = np.argmax(Q - c*np.sqrt(np.log(t)/(0.001+N)))

        N[action] += 1
        rewards[t,m] = qstar[action] + np.random.randn()
        Q[action] = Q[action] + (rewards[t,m]-Q[action])/N[action]


mean_reward=np.mean(rewards,axis=1)
plt.plot(range(T), mean_reward, label=r'c=' + str(c) + r' $\epsilon$='+ str(epsilon))
mean_maxq= np.mean(np.max(q_star,axis=0))
plt.plot([0, T], [mean_maxq, mean_maxq], label=r'mean max $q_*$')
plt.legend()
plt.show()

