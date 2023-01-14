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

def random_buy_sell_in_trading_environment_no_learning(df):

    env = gym.make(
        'stocks-v0',
        df=df,
        frame_bound=(1000,len(df)),
        window_size=1000)

    #print(f'signal features: \n{env.signal_features}')

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

def train_get_reinforcement_learning_model(df):
    env_maker = lambda: gym.make(
        'stocks-v0',
        df=df,
        frame_bound=(1000,len(df)-50),
        window_size=1000)

    env_ = DummyVecEnv([env_maker])

    model = A2C(
        'MlpLstmPolicy',
        env_,
        verbose=1)
    print('modeling starting to learn')
    model.learn(total_timesteps=100000)
    return model

def evaluate_reinforcement_model(model):
    env__ = gym.make(
        'stocks-v0',
        df=df,
        frame_bound=(len(df)-50,len(df)),
        window_size=1000)

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


df = prep_get_data_for_environment()
print(f'length of df: {len(df)}')
random_buy_sell_in_trading_environment_no_learning(df)
model = train_get_reinforcement_learning_model(df)
evaluate_reinforcement_model(model)
x=0