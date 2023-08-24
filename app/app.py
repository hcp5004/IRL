import numpy as np
import cvxpy as cp
from train import idx_state

class FeatureEstimate:
    def __init__(self, feature_num, env):
        self.env = env
        self.feature_num = feature_num
        self.feature = np.ones(self.feature_num)

    def gaussian_function(self, x, mu):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(1., 2.)))

    def get_features(self, state):
        env_low = self.env.observation_space.low
        env_high = self.env.observation_space.high
        env_distance = (env_high - env_low) / (self.feature_num - 1)

        for i in range(int(self.feature_num/2)):
            # position
            self.feature[i] = self.gaussian_function(state[0], 
                                    env_low[0] + i * env_distance[0])
            # velocity
            self.feature[i+int(self.feature_num/2)] = self.gaussian_function(state[1], 
                                                        env_low[1] + i * env_distance[1])

        return self.feature


def calc_feature_expectation(feature_num, gamma, q_table, demonstrations, env):
    feature_estimate = FeatureEstimate(feature_num, env)
    feature_expectations = np.zeros(feature_num)
    demo_num = len(demonstrations)
    
    for _ in range(demo_num):
        state = env.reset()
        demo_length = 0
        done = False
        
        while not done:
            demo_length += 1

            state_idx = idx_state(env, state)
            action = np.argmax(q_table[state_idx]) # the policy is greedy over q value. (q learning)
            next_state, reward, done, _ = env.step(action)
            
            features = feature_estimate.get_features(next_state)
            feature_expectations += (gamma**(demo_length)) * np.array(features)

            state = next_state
    
    feature_expectations = feature_expectations/ demo_num

    return feature_expectations

def expert_feature_expectation(feature_num, gamma, demonstrations, env):
    feature_estimate = FeatureEstimate(feature_num, env)
    feature_expectations = np.zeros(feature_num)
    
    for demo_num in range(len(demonstrations)):
        for demo_length in range(len(demonstrations[0])):
            state = demonstrations[demo_num][demo_length]
            features = feature_estimate.get_features(state)
            feature_expectations += (gamma**(demo_length)) * np.array(features)
    
    feature_expectations = feature_expectations / len(demonstrations)
    
    return feature_expectations

'''
SVM problem: 
it is labeling expert feature expectation as 1, theta^T * mu_E + b = 1
And Learner's feature expectations as -1, theta^T * mu_i + b <= -1
Then the constraints are theta^T * mu_i - theta^T * mu_E <= -2
Since we labeled 1 and -1, the margin t is 2/norm(theta)
'''

def QP_optimizer(feature_num, learner, expert):
    w = cp.Variable(feature_num)
    
    obj_func = cp.Minimize(cp.norm(w))
    constraints = [(expert-learner) @ w >= 2] 

    prob = cp.Problem(obj_func, constraints)
    prob.solve()

    if prob.status == "optimal":
        print("status:", prob.status)
        print("optimal value", prob.value)
    
        weights = np.squeeze(np.asarray(w.value))
        return weights, prob.status
    else:
        print("status:", prob.status)
            
        weights = np.zeros(feature_num)
        return weights, prob.status


def add_feature_expectation(learner, temp_learner):
    # save new feature expectation to list after RL step
    learner = np.vstack([learner, temp_learner])
    return learner

def subtract_feature_expectation(learner):
    # if status is infeasible, subtract first feature expectation
    learner = learner[1:][:]
    return learner

'''
In paper, author said the designer manually picks best one over i = 1, ..., n with accpetable performance 
or alternatively we can find the closest point by solving QP.

Then the new policy get to be mixing policy where the probability of picking policy_i is given by lambda_i.
'''

def QP_policy(learner, expert):
    lamdba = cp.Variable(learner.shape[0])
    
    obj_func = cp.Minimize(cp.norm(expert - learner.T @ lamdba))
    constraints = [lamdba >= 0, np.ones(learner.shape).T @ lamdba == 1] 

    prob = cp.Problem(obj_func, constraints)
    prob.solve()

    if prob.status == "optimal":
        print("status:", prob.status)
        print("optimal value", prob.value)
    
        weights = np.squeeze(np.asarray(lamdba.value))
        return weights, prob.status
    else:
        print("status:", prob.status)
        
        weights = np.zeros(learner.shape[0])
        return weights, prob.status