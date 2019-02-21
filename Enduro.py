#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 16:32:55 2019

@author: esteban
"""

import gym
import os
import keras
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Conv2D, Flatten,Dense
from keras.optimizers import RMSprop
from keras.preprocessing.image import img_to_array

#%% Define class with all the methods
class Agent:
    def __init__(self,input_size,action_size):
        self.input_shape=input_shape
        self.action_size=action_size
        self.memory=deque(maxlen=50000)
        self.modelA=self.networkA()
        self.modelB=self.networkB()
        self.epsilon=1.0
        self.epsilon_min=0.001
        self.decay=0.999
        self.gamma=0.95

    
#    def double_sarsa(self):
        
#    def double_expected_sarsa(self):
#    
    def storage(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
        
    def networkA(self):
        model=Sequential()
        model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=self.input_shape))
        model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
        model.add(Flatten())
        model.add(Dense(self.action_size,activation='softmax'))
        model.compile(loss='mse',optimizer=RMSprop(lr=0.01))
        return model
    
    def networkB(self):
        model=Sequential()
        model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=self.input_shape))
        model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
        model.add(Flatten())
        model.add(Dense(self.action_size,activation='softmax'))
        model.compile(loss='mse',optimizer=RMSprop(lr=0.01))
        return model
    
    def move(self,state):
        action = 0 
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
#            action = random.choice([2,3,4])
            
        else:
            actionA=self.modelA.predict(state)
            actionB=self.modelB.predict(state)
            action=np.argmax(np.mean([actionA[0],actionB[0]],axis=0))
        
        return action
    
        # Restricting the actions with recursive function
#        if action in [2,3,4]:
#            return action
#        else:
#            self.move(state)
            
        
        
        
    
    def train(self,batch_size,mod):
        minibatch=random.sample(self.memory,batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target=reward
            if mod=='A':
                if not done:
                    target=reward+self.gamma*self.modelB.predict(next_state)[0]\
                    -self.modelA.predict(state)
                    self.modelA.fit(state,target, epochs=10,verbose=0)
            else:
                if not done:
                    target=reward+self.gamma*self.modelA.predict(next_state)[0]\
                    -self.modelB.predict(state)
                    self.modelB.fit(state,target, epochs=10,verbose=0)
#            target_f=self.modelA.predict(state)
#            target=np.array(target)
#            target_f[0][action]=target
            
        
        if self.epsilon>self.epsilon_min:
            self.epsilon=self.epsilon*self.decay
        
    def load(self, fileA,fileB):
         self.modelA.load_weights(fileA)
         self.modelB.load_weights(fileB)
    def save(self,fileA,fileB):
         self.modelA.save_weights(fileA)
         self.modelB.save_weights(fileB)

        
#%% Execute game loop
        
env = gym.make('Enduro-v0')
action_size=env.action_space.n
input_shape=env.observation_space.shape
CNN=Agent(input_shape,action_size)

batch_size=32
episode=5

def reshape_img(state):
    state_array=img_to_array(state)
#    state_array=np.swapaxes(state_array, 0,2)
#    state_array=np.swapaxes(state_array, 1,2)
    state_array=state_array.reshape((1,)+state_array.shape)
    return state_array

#def reward_reduction():

fig = plt.figure(figsize=(16,10))
ax1 = fig.add_subplot(1,1,1)

def plot_(file_name, y_name):
    pullData = open(file_name,"r").read()
    xar = []
    yar = []
    myData  =  pullData.split("\n")
    for data in myData:
        try:
            splitted_val  = data.split(",")
            y_variable  =  splitted_val[1]
            iteration = splitted_val[0]

            
            xar.append(float(iteration))
            yar.append(float(y_variable))
            
            
        except:
            pass 
    
    img_name=y_name+".png"
    
    plt.plot(xar,yar)
    plt.xlabel("Episodes")
    plt.ylabel(y_name)
    plt.title(y_name+" vs Episode")
    plt.savefig(img_name, bbox_inches="tight")
    plt.close()
    

count = 0

for episode in range(episode):
#    print(episode)
    state=env.reset()
    state=reshape_img(state)
    action=CNN.move(state)    
    time=0  
    
    while True:
        env.render()
        
        next_state, reward, done, _ = env.step(action)
        if  not done:
            count +=reward
        reward  = count 
        time+=1
#        print(time)
        next_state=reshape_img(next_state)
        action_=CNN.move(next_state)
        
        
        CNN.storage(state,action_,reward, next_state,done)
        if len(CNN.memory)>batch_size:
            if np.random.random_sample() <= 0.5:
                mod='A'
            else:
                mod='B'
            CNN.train(batch_size,mod)
        state=next_state
        action=action_
        if done:
#            print(time,episode,CNN.epsilon)
            break
    if episode%10==0:
        CNN.save('modelA_weights.h5','modelB_weights.h5')
    
  
    rewards=open("rewards.txt","a")
    rewards.write(str(episode)+","+str(reward))
    rewards.write("\n")
    plot_("rewards.txt","Rewards")
   
    steps=open('steps.txt','a')
    steps.write(str(episode)+','+str(time))
    steps.write("\n")
    plot_("steps.txt","Steps")
    
env.close() 
#rewards.close()
#steps.close()   
        
    
            
#        plt.imshow(next_state)
#        print(next_state.shape)