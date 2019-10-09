import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from gym import wrappers
from time import time 

# CREATE ENVIRONMENT
env = gym.make('CartPole-v0')
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]
print('Action space size: ', n_actions)
print('State space size: ', n_states)
print('states high value:', env.observation_space.high)
print('states low value:', env.observation_space.low)

# HYPERPARAMETERS
n_train_episodes = 500
n_test_episodes = 10        
n_steps = 200               
gamma = 0.95                # discount factor
epsilon = 1                 # exploration threshold
epsilon_min = 0.01
epsilon_decay = 0.995
alpha = 0.01                # learning rate
alpha_decay = 0.001
batch_size = 64
memory = deque(maxlen=100000)

# DEFINE NEURAL NETWORK
model = Sequential()
model.add(Dense(24, input_dim=n_states, activation='tanh'))
model.add(Dense(48, activation='tanh'))
model.add(Dense(n_actions, activation='linear'))
model.compile(loss='mse', optimizer=Adam(lr=alpha, decay=alpha_decay))


def epsilon_policy(state, epsilon):
    ''' return an action using the epsilon policy '''
    if np.random.random() <= epsilon:       # exploration
        action = env.action_space.sample()  
    else:                                   # exploitation
        action = np.argmax(model.predict(state))
    return action

def greedy_policy(state):
    ''' return an action using the greedy policy '''
    return np.argmax(model.predict(state))

def update_epsilon(epsilon):
    ''' decrease the exploration rate at the end of each episode '''
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    return epsilon

def preprocess_state(state):
    ''' reshape the state so that it can be read by the NN'''
    return np.reshape(state, [1, n_states])

def replay(batch_size, epsilon):
    ''' 
    Trains the network on a smaller sample selection of the runs in memory 
    '''
    x_batch, y_batch = [], []
    
    # Sample batch randomly from the memory
    batch = random.sample(memory, min(len(memory), batch_size))

    # Extract informations from each batch
    for state, action, reward, next_state, done in batch:
        y_target = model.predict(state) # size: (1 x 2)
        # update the Q value for this state
        if done:                # the target is the reward
            y_target[0][action] = reward  
        else:                   # predict the future discounted reward
            y_target[0][action] = reward + gamma * np.max(model.predict(next_state)[0])

        x_batch.append(state[0])
        y_batch.append(y_target[0])
    
    X = np.array(x_batch)  # size: (64 x 4)
    y = np.array(y_batch)  # size: (64 x 2)
    model.fit(X, y, batch_size=len(x_batch), verbose=0)


# TRAINING PHASE
rewards = [] 

for episode in range(n_train_episodes):
    current_state = env.reset()
    current_state = preprocess_state(current_state)
    episode_rewards = 0

    for t in range(n_steps):
        # env.render()
        action = epsilon_policy(current_state, epsilon)
        next_state, reward, done, _ = env.step(action)
        next_state = preprocess_state(next_state)
        memory.append((current_state, action, reward, next_state, done))
        current_state = next_state
        episode_rewards += reward
        replay(batch_size, epsilon)

        if done:
            print('Episode: {:d}/{:d} | cumulative reward: {:.0f} | epsilon: {:.2f}'.format(episode, n_train_episodes, episode_rewards, epsilon))
            break

    rewards.append(episode_rewards)
    epsilon = update_epsilon(epsilon)

# PLOT RESULTS
x = range(n_train_episodes)
plt.plot(x, rewards)
plt.xlabel('Episode number')
plt.ylabel('Training cumulative reward')
plt.savefig('DQN_CartPole.png', dpi=300)
plt.show()

# TEST PHASE
env = wrappers.Monitor(env, './videos/' + str(time()) + '/')
for episode in range(n_test_episodes):
    current_state = env.reset()
    current_state = preprocess_state(current_state)
    episode_rewards = 0

    for t in range(n_steps):
        env.render()
        action = greedy_policy(current_state)
        next_state, reward, done, _ = env.step(action)
        next_state = preprocess_state(next_state)
        memory.append((current_state, action, reward, next_state, done))
        current_state = next_state
        episode_rewards += reward 

        if done:
            print('Episode: {:d}/{:d} | cumulative reward: {:.0f}'.format(episode, n_test_episodes, episode_rewards))
            break

env.close()
