import gym
import gym_anytrading
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
#pip install protobuf==3.20.*
#pip install "gym==0.19.0" 

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



def prep_get_data_for_environment():
    df = pd.read_csv(
        'C:/Users/SUPREME/Documents/GitHub/reinforcement-learning/VIX-weekly.csv',
        parse_dates=['Date'])
    df.set_index(
        'Date',
        inplace=True)
    return df

def random_actions_in_trading_environment_no_learning(df,training_points):

    env = gym.make(
        'stocks-v0',
        df=df,
        frame_bound=(training_points,len(df)),
        window_size=training_points)

    print(f'signal features: \n{env.signal_features}')

    observation = env.reset()
    while True:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)    
        if done:
            print("info:", info)
            break

    plt.cla()
    env.render_all()
    plt.show()

def train_get_reinforcement_learning_model(df,training_points,test_points):
    env_maker = lambda: gym.make(
        'stocks-v0',
        df=df,
        frame_bound=(training_points,len(df)-test_points),
        window_size=training_points)

    env_ = DummyVecEnv([env_maker])

    model = A2C(
        'MlpLstmPolicy',
        env_,
        verbose=1)
    print('modeling starting to learn')
    model.learn(total_timesteps=10000)
    return model

def evaluate_reinforcement_model_on_unseen_points(df,model,training_points,test_points):
    env__ = gym.make(
        'stocks-v0',
        df=df,
        frame_bound=(len(df)-test_points,len(df)),
        window_size=training_points)

    obs = env__.reset()
    while True:
        obs = obs[np.newaxis,...]
        action, _states = model.predict(obs)
        obs, rewards, done, info = env__.step(action)
        if done:
            print(f'info: {info}')
            break

    plt.cla()
    env__.render_all()
    plt.show()  

training_points = 1000
test_points = 50

df = prep_get_data_for_environment()

random_actions_in_trading_environment_no_learning(
    df,
    training_points)

model = train_get_reinforcement_learning_model(
    df,
    training_points,
    test_points)

evaluate_reinforcement_model_on_unseen_points(
    df,
    model,
    training_points,
    test_points)
