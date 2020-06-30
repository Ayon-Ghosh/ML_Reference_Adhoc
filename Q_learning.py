# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:51:53 2019

@author: 140524
"""

# Q Learning - Bellman's equation - Markov chain

#PPT: Reinforcement_Epsilon_greedy_Markov

import numpy as np

#initialize parameters

gamma = 0.5
alpha = 0.99
#cost = 0.04

#definiting the states:

location_to_state = {
        'L1':0,
        'L2':1,
        'L3':2,
        'L4':3,
        'L5':4,
        'L6':5,
        'L7':6,
        'L8':7}

# Defining actions

actions = {0,1,2,3,4,5}

# defining reward matrix

rewards = np.array(              
        [[0,0,-1,-1,0,0,0,0],
         [0,0,0,0,0,0,0,100],
         [0,-1,0,-1,0,0,0,0],
         [0,-1,0,-1,0,0,0,100],
         [0,-1,0,-1,0,0,0,0],
         [0,0,0,0,0,0,0,100],
         [0,0,-1,0,0,0,0,0],
         [0,0,-1,0,0,0,-1,-1]]              
        )
rewards
 
#mapping indices to locations

state_to_location = dict((state, location) for location, state in location_to_state.items())
state_to_location

def optimal_route(start_location,end_location):
#update reward matrix
             reward_new =np.copy(rewards)
             end_state = location_to_state[end_location]
             #reward_new[end_state,end_state] = 999
#initializing Q matrix
             Q = np.zeros((8,8))
# building the Q learning algo
             for i in range(1000):
                 current_state = np.random.randint(0,8)
                 action_list = []
#iteraing through the reward matrix and get actions>=0
                 for j in range(8):
                     if (reward_new[current_state,j])>=0:
                         action_list.append(j)
                 next_state = np.random.choice(action_list)
#temporal_differencer
             TD = reward_new[current_state,next_state] + gamma*Q[next_state,np.argmax(Q[next_state,])] - Q[current_state,next_state]
             Q[current_state,next_state] = Q[current_state,next_state] + alpha*TD
#initializing the route with start_location
             route = [start_location]
             next_location = start_location
             while (next_location!=end_location):
                 starting_state = location_to_state[start_location]
                 next_state = np.argmax(Q[starting_state,])
                 next_location = state_to_location[next_state]
                 route.append(next_location)
                 start_location = next_location
             return route
 
print(optimal_route('L4','L6'))        
                 
                             
             
     
