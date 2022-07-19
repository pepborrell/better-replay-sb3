from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# from plotting.sampling_heatmaps.grid_u_shaped import generate_grid
# from plotting.sampling_heatmaps.gridworld_random_agent import GridworldRandomAgent
# from plotting.sampling_heatmaps.sampling_strategies import get_UER_samples, get_USAR_samples, get_USR_samples

sns.set()
plt.rcParams["font.family"] = "Source Sans Pro"


def count_state_freqs(trajs: list, size: Iterable[int]) -> np.ndarray:
    freqs = np.zeros(size)
    for state in trajs:
        freqs[tuple(state)] += 1
    return freqs


def plot_heatmap(hm, ax=None, title=None):
    if ax:
        sns.heatmap(hm, ax=ax)
    else:
        sns.heatmap(hm)
    if title:
        ax.set_title(title)
    plt.tight_layout()


# def plot_surface(hist):
#     nr, nc = hist.shape
#     x = np.arange(nr)
#     y = np.arange(nc)
#     x, y = np.meshgrid(x, y)

#     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#     surf = ax.plot_surface(x, y, hist, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
#     plt.axis("off")


# def plot_state_histogram(freqs):
#     sorted_freqs = np.sort(freqs.flatten())[::-1]
#     norm_freqs = sorted_freqs / np.sum(sorted_freqs)
#     x = np.arange(len(norm_freqs))
#     fig, ax = plt.subplots(figsize=(6, 3))
#     sns.lineplot(x=x, y=norm_freqs, ax=ax)
#     ax.fill_between(x, norm_freqs, alpha=0.3)
#     ax.set_xlabel("States sorted by sampling probability")
#     ax.set_ylabel("Sampling probability")
#     fig.tight_layout()


# def fill_replay_buffer(grid, n_trajs=50, n_steps=1300):
#     # starting_pos = np.array([[0, i + n_cols // 4] for i in range(n_cols // 2)])
#     starting_pos = np.array([[3, 10]])
#     agent = GridworldRandomAgent(grid, starting_pos=starting_pos)
#     all_trajs = []
#     for _ in range(n_trajs):
#         trajectory = agent.get_trajectory(n_steps=n_steps)
#         all_trajs.extend(trajectory)
#     return all_trajs


# def main():
#     n_cols = 20
#     n_trajs = 50
#     n_steps = 1300
#     grid = generate_grid(n_rows=n_cols, n_cols=n_cols)
#     all_trajs = fill_replay_buffer(grid, n_trajs=n_trajs, n_steps=n_steps)
#     print("Got replay buffer")
#     sampling_strategies = [get_USR_samples, get_USAR_samples, get_UER_samples]
#     sampling_strategy_names = ["USR", "USAR", "UER"]
#     for strat, name in zip(sampling_strategies, sampling_strategy_names):
#         fig, ax = plt.subplots(figsize=(8, 7))
#         samples = strat(all_trajs, n_samples=int(0.3 * n_trajs * n_steps))
#         freqs = count_state_freqs(samples, grid.shape)
#         plot_heatmap(freqs, ax=ax, title=f"State coverage obtained with {name}")
#
#     plt.show()


# if __name__ == "__main__":
#     main()
