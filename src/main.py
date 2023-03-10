import os
from datetime import datetime

import numpy as np
from tqdm import tqdm

from policy import Policy
from environment import DigitalAdvertisingEnv

output_dir = './output'
now_str = str(datetime.now()).replace(':', '-').replace(' ', '_').split('.')[0]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
creatives = [(1,2), (0.5,0.5), (2,1)]
websites = [(1,1), (3,2), (3,4)]
modifiers = {'mon' : (0.5,1), 'tue' : (1,1), 'wed' : (2,2), 
    'thu' : (2,2), 'fri' : (1,10), 'sat' : (3,2), 'sun' : (3,2),
    'w50' : (5,3), 'holiday' : (3,2)}

env = DigitalAdvertisingEnv(creatives, websites, modifiers)
obs = env.reset()
random_policy = Policy(
    model_type = 'random',
    n_creatives = len(creatives),
    n_websites = len(websites),
    n_features = len(obs),
)
ts_policy = Policy(
    model_type = 'ts',
    n_creatives = len(creatives),
    n_websites = len(websites),
    n_features = len(obs),
)
blr_policy = Policy(
    model_type = 'blr',
    n_creatives = len(creatives),
    n_websites = len(websites),
    n_features = len(obs),
)
nn_policy = Policy(
    model_type = 'nn',
    n_creatives = len(creatives),
    n_websites = len(websites),
    n_features = len(obs),
)

n_trials = 100_000

# Bayesian Linear Regression Policy
print("--- Bayesian Linear Regression Policy ---")
blr_trials = 5000
blr_regret = []
for i in tqdm(range(blr_trials)):
    action = blr_policy.predict(obs)
    new_obs, reward, done, info = env.step(action)
    blr_regret.append(info['total_regret'])
    blr_policy.update(obs, action, reward)
    obs = new_obs

print(f"Cumulative Regret: {np.sum(blr_regret)}")
print(f"Mean Regret: {np.mean(blr_regret)}")
print(f"Std Dev Regret: {np.std(blr_regret)}")
with open(os.path.join(output_dir, f'{now_str}_blr.csv'), 'w') as f:
    for i in range(blr_trials):
        f.write(f"{blr_regret[i]}\n")

# Neural Network Policy
print("--- Neural Network Policy ---")
nn_regret = []
for i in tqdm(range(n_trials)):
    action = nn_policy.predict(obs)
    new_obs, reward, done, info = env.step(action)
    nn_regret.append(info['total_regret'])
    nn_policy.update(obs, action, reward)
    obs = new_obs

print(f"Cumulative Regret: {np.sum(nn_regret)}")
print(f"Mean Regret: {np.mean(nn_regret)}")
print(f"Std Dev Regret: {np.std(nn_regret)}")
with open(os.path.join(output_dir, f'{now_str}_nn.csv'), 'w') as f:
    for i in range(n_trials):
        f.write(f"{nn_regret[i]}\n")

# Random Policy
print("--- Random Policy ---")
random_regret = []
for i in tqdm(range(n_trials)):
    action = random_policy.predict(obs)
    new_obs, reward, done, info = env.step(action)
    random_regret.append(info['total_regret'])
    random_policy.update(obs, action, reward)
    obs = new_obs

print(f"Cumulative Regret: {np.sum(random_regret)}")
print(f"Mean Regret: {np.mean(random_regret)}")
print(f"Std Dev Regret: {np.std(random_regret)}")
with open(os.path.join(output_dir, f'{now_str}_random.csv'), 'w') as f:
    for i in range(n_trials):
        f.write(f"{random_regret[i]}\n")

# Thompson Sampling Policy
print("--- Thompson Sampling Policy ---")
ts_regret = []
for i in tqdm(range(n_trials)):
    action = ts_policy.predict(obs)
    new_obs, reward, done, info = env.step(action)
    ts_regret.append(info['total_regret'])
    ts_policy.update(obs, action, reward)
    obs = new_obs

print(f"Cumulative Regret: {np.sum(ts_regret)}")
print(f"Mean Regret: {np.mean(ts_regret)}")
print(f"Std Dev Regret: {np.std(ts_regret)}")
with open(os.path.join(output_dir, f'{now_str}_ts.csv'), 'w') as f:
    for i in range(n_trials):
        f.write(f"{ts_regret[i]}\n")

