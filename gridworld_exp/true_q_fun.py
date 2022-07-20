from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from grid import EnvFromGrid, generate_u_grid
from plot_grids import count_state_freqs, plot_heatmap


def get_back_graph(env):
    n_rows, n_cols = env.grid.shape
    n_actions = len(env.action_space)
    back_graph = np.full((n_rows, n_cols, n_actions), fill_value=None, dtype=object)
    forw_rewards = np.zeros((n_rows, n_cols, n_actions), dtype=float)
    # Build forward graph
    for i in range(n_rows):
        for j in range(n_cols):
            pos = np.array([i, j])
            if env._is_position_legal(pos):
                for aid in range(n_actions):
                    action = env.action_space[aid]
                    if env._is_action_legal(pos, action):
                        env.assign_position(pos)
                        next_s, r, _ = env.step(action)
                        if next_s is not None:
                            back_graph[next_s[0], next_s[1], aid] = pos
                        forw_rewards[pos[0], pos[1], aid] = r

    return back_graph, forw_rewards


def get_q(env, gamma=0.9):
    Q = np.full((env.grid.shape[0], env.grid.shape[1], len(env.action_space)), fill_value=np.nan, dtype=float)
    b_graph, f_rews = get_back_graph(env)

    dones = np.zeros_like(f_rews, dtype=bool)
    to_expand = deque()
    maxes = np.transpose(np.where(f_rews == np.max(f_rews)))
    assert np.all(maxes[:, :2] == maxes[0, :2]), "All states of high reward must be the same"

    # Assigning the reward to maxes, and adding them to expand
    for max_i, max_j, max_a in maxes:
        Q[max_i, max_j, max_a] = f_rews[max_i, max_j, max_a]
        dones[max_i, max_j, max_a] = True
        to_expand.append((max_i, max_j, max_a))

    while len(to_expand) > 0:
        # Get next state to study in the backward graph
        next0, next1, act = to_expand.popleft()
        # s = G(s', a), because G is a backward graph
        pos = b_graph[next0, next1, act]
        if pos is not None:
            if not dones[pos[0], pos[1], act]:
                # Update Q: Q(s, a) = r + gamma * max(Q(s', a))
                assert np.any(dones[next0, next1, :])
                Q[pos[0], pos[1], act] = f_rews[pos[0], pos[1], act] + gamma * np.nanmax(Q[next0, next1, :])
                for aid in range(len(env.action_space)):
                    to_expand.append((pos[0], pos[1], aid))
            dones[pos[0], pos[1], act] = True

    return Q


def get_optimal_policy(env, gamma=0.9):
    Q = get_q(env, gamma)
    policy = lambda s: np.nanargmax(Q[s[0], s[1], :])
    return policy


def generate_trajectories(env, n_steps, gamma=0.9):
    policy = get_optimal_policy(env, gamma)
    pos = env.reset()
    states = np.empty((n_steps, 2), dtype=int)
    actions = np.empty((n_steps,), dtype=int)
    rewards = np.empty((n_steps,), dtype=float)
    for i in range(n_steps):
        states[i, :] = pos
        act_id = policy(pos)
        act = env.action_space[act_id]
        actions[i] = act_id
        pos, r, done = env.step(act)
        if done:
            pos = env.reset()
        rewards[i] = r
    return states, actions, rewards


def get_optimal_state_dist(env, n_steps=1000, gamma=0.9):
    states, actions, rewards = generate_trajectories(env, n_steps, gamma)
    size = env.grid.shape
    state_freqs = count_state_freqs(states, size)
    # Normalize
    state_freqs /= np.sum(state_freqs)
    return state_freqs


def get_optimal_state_action_dist(env, n_steps=1000, gamma=0.9):
    states, actions, rewards = generate_trajectories(env, n_steps, gamma)
    state_acts = np.concatenate((states, actions[:, np.newaxis]), axis=1)
    assert state_acts.shape == (n_steps, 3)
    size = env.grid.shape + (len(env.action_space),)
    state_act_freqs = count_state_freqs(state_acts, size)
    # Normalize
    state_act_freqs /= np.sum(state_act_freqs)
    return state_act_freqs


def main():
    n = 15
    grid = generate_u_grid(n, n)
    env = EnvFromGrid(grid, goal=np.array([n // 2, n // 2]))
    # Testing everything here
    q = get_q(env, gamma=1)
    fig, ax = plt.subplots()
    plot_heatmap(np.nanmax(q, axis=-1), ax=ax)
    fig, ax = plt.subplots()
    plot_heatmap(get_optimal_state_dist(env, n_steps=10000), ax=ax)
    state_act_dist = get_optimal_state_action_dist(env, n_steps=10000)
    plt.show()


if __name__ == "__main__":
    main()
