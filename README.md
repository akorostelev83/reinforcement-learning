# Stock Market Volatility Reinforcement-Learning

### This repo contains a reinforcement learning trader for stock market volatility index VIX(weekly)
--This model is uses MlpLstmPolicy in A2C algorithm to train the agent

##### this method takes random actions in VIX: 
--random_actions_in_trading_environment_no_learning()
##### this method trains the agent: 
--train_get_reinforcement_learning_model()
##### this method evaluates agent on unseen points: 
--evaluate_reinforcement_model_on_unseen_points()

##### Input features for this model are the closing prices and the difference with previous closing price

[Click here to view RL trader code for VIX](https://github.com/akorostelev83/reinforcement-learning/blob/main/stock-market-volatility-trader-reinforcement-learning.py)
