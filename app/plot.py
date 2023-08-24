import matplotlib.pyplot as plt
import numpy as np
from app import *

def irl_reward(w, feature_estimate, state):
    features = feature_estimate.get_features(state)
    irl_reward = np.dot(w, features)
    return irl_reward

def plot_reward(env, one_feature, w, feature_estimate):

    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_distance = (env_high - env_low) / one_feature

    position, velocity = np.meshgrid(
        np.array([x_n*env_distance[0] + env_low[0] for x_n in range(0, one_feature)]),
        np.array([xdot_n*env_distance[1] + env_low[1] for xdot_n in range(0, one_feature)]),
    )

    reward = np.apply_along_axis(
            lambda obs: irl_reward(w, feature_estimate,[obs[0], obs[1]]),
            axis=2,
            arr=np.dstack([position, velocity]),
        )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    
    ax.plot_surface(
        position,
        velocity,
        reward,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )

    #plt.show()
    plt.savefig("./results/reward.png")