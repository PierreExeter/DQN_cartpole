import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# CREATE ENVIRONMENT
ENV_NAME = 'CartPole-v0'
env = gym.make(ENV_NAME)
nb_actions = env.action_space.n
nb_states = env.observation_space.shape

# NETWORK ARCHITECTURE
model = Sequential()
model.add(Flatten(input_shape=(1,) + nb_states))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# DEFINE AGENT
policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# TRAIN AGENT
train = True

if train:
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
    dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
else:       # enjoy pre-trained agent
    dqn.load_weights('dqn_{}_weights.h5f'.format(ENV_NAME))

# TEST AGENT
dqn.test(env, nb_episodes=5, visualize=True)
