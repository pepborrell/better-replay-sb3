from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from grid import EnvFromGrid, generate_u_grid
from plot_grids import plot_heatmap


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
                        if not ((pos[0] == next_s[0]) and (pos[1] == next_s[1])):
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
                if pos[0] == 0:
                    print(Q[pos[0], pos[1], act])
                for aid in range(len(env.action_space)):
                    to_expand.append((pos[0], pos[1], aid))
            dones[pos[0], pos[1], act] = True

    return Q


def main():
    n = 15
    grid = generate_u_grid(n, n)
    env = EnvFromGrid(grid, goal=np.array([n // 2, n // 2]))
    q = get_q(env)
    print(np.nanmax(q, axis=-1))
    plot_heatmap(np.nanmax(q, axis=-1))
    plt.show()


if __name__ == "__main__":
    main()