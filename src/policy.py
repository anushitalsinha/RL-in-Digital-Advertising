import pandas as pd
import numpy as np
import pymc3 as pm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Beta

class Policy():
    """Generalized class representing a policy
    """
    def __init__(self, model_type, n_creatives, n_websites, n_features):
        """Initialization function for policy object

        :param model_type: model type, one of 'ts', 'blr', 'nn', 'random'
        :type model_type: str
        :param n_creatives: the number of creatives being considered
        :type n_creatives: int
        :param n_websites: the number of websites being considered
        :type n_websites: int
        :param n_features: the number of features in the observation
        :type n_features: int
        """
        super(Policy, self).__init__()
        valid_models = ['ts', 'blr', 'nn', 'random']
        assert model_type in valid_models, f"Incorrect model input: {model_type}"
        self.model_type = model_type
        if self.model_type == 'random':
            self.model = RandomModel(n_creatives, n_websites, n_features)
        elif self.model_type == 'ts':
            self.model = ThompsonSamplingModel(n_creatives, n_websites, n_features)
        elif self.model_type == 'blr':
            self.model = BayesianLinearRegressionModel(n_creatives, n_websites, n_features)
        elif self.model_type == 'nn':
            self.model = NeuralNetworkModel(n_creatives, n_websites, n_features)

    def predict(self, obs):
        return self.model.predict(obs)

    def update(self, obs, action, reward):
        self.model.update(obs, action, reward)


class RandomModel():
    """Random model, returns a random action, used for baseline predictions
    """
    def __init__(self, n_creatives, n_websites, n_features):
        super(RandomModel, self).__init__()
        self.n_creatives = n_creatives
        self.n_websites = n_websites
    
    def predict(self, obs):
        return (np.random.randint(self.n_creatives), 
            np.random.randint(self.n_websites))
    
    def update(self, obs, action, reward):
        return


class ThompsonSamplingModel():
    """Thompson sampling algorithm for continuous rewards"""
    def __init__(self, n_creatives, n_websites, n_features, alpha = 2, beta = 1, n_samples = 10, 
        baseline = 0):
        """Initialization function for Thompson Sampling Model

        :param n_creatives: number of creatives
        :type n_creatives: int
        :param n_websites: number of websites
        :type n_websites: int
        :param alpha: starting alpha parameter, defaults to 2
        :type alpha: int, optional
        :param beta: starting beta parameter, defaults to 1
        :type beta: int, optional
        :param n_samples: number of samples to take when choosing action, defaults to 10
        :type n_samples: int, optional
        :param baseline: baseline value for maximum reward, defaults to 0
        :type baseline: int, optional
        """
        super(ThompsonSamplingModel, self).__init__()
        self.n_creatives = n_creatives
        self.n_websites = n_websites
        self.action_list = [(x,y) for x in range(n_creatives) for y in range(n_websites)]
        self.n_actions = len(self.action_list)
        self.alpha = alpha
        self.beta = beta
        self.n_samples = n_samples
        self.shape_params = [[alpha, beta] for _ in range(self.n_actions)]
        self.max_reward = baseline
        self.min_reward = np.inf

    def predict(self, obs):
        reward_samples = [ # Shape: (n_actions, n_samples)
            [np.random.beta(*self.shape_params[j]) 
                for _ in range(self.n_samples)]
            for j in range(self.n_actions)]
        action = np.argmax(np.mean(reward_samples, axis = 1))
        return self.action_list[action]

    def update(self, obs, raw_action, reward):
        action = self.action_list.index(raw_action)
        if reward > self.max_reward:
            self.max_reward = reward
        if reward < self.min_reward:
            self.min_reward = reward
        # Scale reward between 0 and 1
        if self.min_reward != self.max_reward:
            scaled_reward = (reward - self.min_reward) / (self.max_reward - self.min_reward)
        else:
            scaled_reward = 0.5
        # Sample from binomial if 0 then beta++ if 1 then alpha++
        sampled_b = np.random.binomial(1, scaled_reward)
        # Update alpha or beta of distribution
        self.shape_params[action][1 - sampled_b] += 1


class ReplayMemory(object):
    def __init__(self, columns):
        self.columns = columns
        self.memory = {c : [] for c in columns}

    def push(self, datum):
        """Save a transition"""
        assert all([c in datum for c in self.columns])
        for col in self.columns:
            self.memory[col].append(datum[col])

    def __len__(self):
        return len(pd.DataFrame(self.memory))
    
    def get_data(self):
        return pd.DataFrame(self.memory)


class BayesianLinearRegressionModel():
    """Bayesian Linear Regression model 
    """
    def __init__(self, n_creatives, n_websites, n_features):
        self.n_creatives = n_creatives
        self.n_websites = n_websites
        self.action_list = [(x,y) for x in range(n_creatives) for y in range(n_websites)]
        self.models = [BayesianLinearRegression(n_features) for _ in range(len(self.action_list))]
        self.memories = [ReplayMemory(['is_holiday', 'day_of_week', 'week_of_year', 'ctr']) 
            for _ in range(len(self.action_list))]
    
    def predict(self, obs):
        if any([x.model is None for x in self.models]):
            best_action = np.random.choice([i for i in range(len(self.models)) 
                if self.models[i].model is None])
        else:
            best_action = np.argmax(
                [
                    m.predict(pd.DataFrame(
                            {
                                'is_holiday' : [obs[0]], 
                                'day_of_week' : [obs[1]], 
                                'week_of_year' : [obs[2]],
                            }))
                        for m in self.models
                ]
            )
        return self.action_list[best_action]
    
    def update(self, obs, action, reward):
        idx = self.action_list.index(action)
        self.memories[idx].push({'is_holiday' : obs[0], 'day_of_week' : obs[1], 
            'week_of_year' : obs[2], 'ctr' : reward})
        self.models[idx].update(self.memories[idx].get_data())


class BayesianLinearRegression():
    """Bayesian linear regression algorithm"""
    def __init__(self, n_features):
        super(BayesianLinearRegression, self).__init__()
        self.n_features = n_features
        self.model = None
        self.trace = None

    def predict(self, datum):
        with self.model:
            pm.set_data({'x': datum})
            posterior_predictive = pm.sample_posterior_predictive(self.trace)
        return posterior_predictive['ctr'].mean()

    def update(self, train_data):
        with pm.Model() as model:
            # Define data
            x = pm.Data('x', train_data[[x for x in train_data.columns if x != 'ctr']])
            y = pm.Data('y', train_data['ctr'])
            # Define priors
            sigma = pm.HalfCauchy('sigma', beta=10)
            intercept = pm.Normal('intercept', 0, 20)
            coeff = pm.Normal('coeff', 0, 20, shape = 3)

            # Define likelihood
            ctr = pm.LogNormal('ctr', mu = intercept + pm.math.dot(x, coeff), sigma = sigma, observed = y)
            trace = pm.sample(return_inferencedata = True, draws = 100, tune = 100, cores = 1)
        self.model = model
        self.trace = trace


class NeuralNetworkModel():
    """Neural Network model, with probabilistic outputs
    """
    def __init__(self, n_creatives, n_websites, n_features):
        self.n_creatives = n_creatives
        self.n_websites = n_websites
        self.action_list = [(x,y) for x in range(n_creatives) for y in range(n_websites)]
        self.n_features = self._format_input([0] * n_features).shape[0]
        self.models = [NeuralNet(self.n_features, [8]).float() 
            for _ in range(len(self.action_list))]
        self.optimizers = [optim.AdamW(self.models[i].parameters()) 
            for i in range(len(self.models))]
    
    def _format_input(self, obs):
        is_holiday = nn.functional.one_hot(torch.tensor(obs[0]), num_classes = 2)
        day_of_week = nn.functional.one_hot(torch.tensor(obs[1]), num_classes = 7)
        week_of_year = nn.functional.one_hot(torch.tensor(obs[2]), num_classes = 53)
        formatted_input = torch.cat([is_holiday, day_of_week, week_of_year])
        return formatted_input.float()
    
    def predict(self, obs):
        self.action_predictions = [m(self._format_input(obs)).sample() for m in self.models]
        return self.action_list[np.argmax(self.action_predictions)]
    
    def update(self, obs, action, reward):
        idx = self.action_list.index(action)
        self.optimizers[idx].zero_grad()
        loss = -self.models[idx](self._format_input(obs)) \
            .log_prob(self.action_predictions[idx]) * reward
        loss.backward()
        self.optimizers[idx].step()


class NeuralNet(nn.Module):
    """Neural network that predicts using a LogNormal distribution
    Parameters:
    - n_features (int): count of the number of features in the input
    - hidden_layers (list of int) <OPTIONAL>: a list of the number of nodes in each
        hidden layer
    """
    def __init__(self, n_features, hidden_layers = None):
        super(NeuralNet, self).__init__()
        hidden_layers = [] if hidden_layers is None else hidden_layers
        self.layers = nn.ModuleList()
        input_size = n_features
        for nodes in hidden_layers:
            self.layers.append(nn.Linear(input_size, nodes))
            input_size = nodes
        self.output = nn.Linear(input_size, 2)
        self.dist = None

    def forward(self, x):
        for l in self.layers:
            x = torch.relu(l(x))
        params = self.output(x)
        return Beta(
            torch.abs(params[0]) + 0.01, 
            torch.abs(params[1]) + 0.01,
        )