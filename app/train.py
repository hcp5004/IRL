import sys
import gym
import pylab
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from app import *
from plot import plot_reward

n_states = 400 # position - 20, velocity - 20
n_actions = 3
one_feature = 20 # number of state per one feature of observation
feature_num = 4 # theta is vector of R^4
q_table = np.zeros((n_states, n_actions))  # (400, 3)

gamma = 0.99
q_learning_rate = 0.03

epsilon = 0.01

'''
This code simply represents continuous states (position x velocity x action) with 20x20x3 grid.

For Reward function, R = theta^T phi(s), it is Gaussian radial basis function which has 1 as derivation

For RL step, it used Q learning.
'''
def idx_state(env, state):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_distance = (env_high - env_low) / one_feature
    positioone_feature = int((state[0] - env_low[0]) / env_distance[0])
    velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
    state_idx = positioone_feature + velocity_idx * one_feature
    return state_idx

def update_q_table(state, action, reward, next_state):
    q_1 = q_table[state][action]
    q_2 = reward + gamma * max(q_table[next_state])
    q_table[state][action] += q_learning_rate * (q_2 - q_1)

'''
Feature expectation is computed by MonteCarlo
it doesn't need to reset the q table since it just use random q value first.
'''

def main():
    #global q_table
    env = gym.make('MountainCar-v0')
    demonstrations = np.load(file="expert_demo/expert_demo.npy")
    
    feature_estimate = FeatureEstimate(feature_num, env)
    
    learner = calc_feature_expectation(feature_num, gamma, q_table, demonstrations, env)
    learner = np.matrix([learner])
    
    expert = expert_feature_expectation(feature_num, gamma, demonstrations, env)
    expert = np.matrix([expert])
    
    w, status = QP_optimizer(feature_num, learner, expert)
    
    
    episodes, scores = [], []
    
    for episode in range(60_000):
        state = env.reset()
        score = 0

        while True:
            state_idx = idx_state(env, state)
            action = np.argmax(q_table[state_idx])
            next_state, reward, done, _ = env.step(action)
            
            features = feature_estimate.get_features(state)
            irl_reward = np.dot(w, features)
            
            next_state_idx = idx_state(env, next_state)
            update_q_table(state_idx, action, irl_reward, next_state_idx)

            score += reward
            state = next_state

            if done:
                scores.append(score)
                episodes.append(episode)
                break

        # if episode % 1000 == 0:
        #     score_avg = np.mean(scores)
        #     print('{} episode score is {:.2f}'.format(episode, score_avg))
        #     pylab.plot(episodes, scores, 'b')
        #     pylab.savefig("./learning_curves/app_eps_60000.png")
        #     np.save("./results/app_q_table", arr=q_table)

        '''
        RL(q learning) is done with 5000 epsides

        if the QP solver can't handle the problem, we removed the first learner expectation and do it again.
        '''
        if episode % 5000 == 0:
            # optimize weight per 5000 episode
            status = "infeasible"
            temp_learner = calc_feature_expectation(feature_num, gamma, q_table, demonstrations, env)
            learner = add_feature_expectation(learner, temp_learner)

            while status=="infeasible":
                w, status = QP_optimizer(feature_num, learner, expert)
                if status=="infeasible":
                    learner = subtract_feature_expectation(learner)
            
            if -np.log(np.linalg.norm(w)) <= epsilon or episode == 60_000 - 5000 : #scale the margin into [0,inf)
                print("results are saved")
                np.save("./results/reward", arr=w)
                np.save("./results/app_q_table", arr=q_table)
                plot_reward(env, one_feature, w, feature_estimate)
                break
            
            #q_table = np.zeros((n_states, n_actions)) # reset q values

if __name__ == '__main__':
    main()