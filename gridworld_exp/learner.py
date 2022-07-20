import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import wandb

from grid import EnvFromGrid, generate_u_grid
from grid_agent import GridworldAgent
from plot_grids import count_state_freqs, plot_heatmap
from sampling_strategies import OptimalStrategyReplay, SmallUER, SmallUSAR, SmallUSR
from true_q_fun import get_optimal_state_action_dist


class AgentFromQ(GridworldAgent):
    def __init__(self, env, Q: npt.NDArray[np.float_]) -> None:
        super().__init__(env)
        self.Q = Q

    def _get_action(
        self, pos: npt.NDArray[np.int_], possible_actions: npt.NDArray[np.int_], rng: np.random.Generator = None
    ) -> npt.NDArray[np.int_]:
        a, _ = choose_action(pos, self.env, self.Q, epsilon=0.0)
        return a


class PerfectReplayBuffer:
    def __init__(self, env) -> None:
        self.env = env
        self.n_rows, self.n_cols = env.grid.shape
        self.n_actions = env.action_space.shape[0]

    def sample(self, n=1):
        state = np.array([-1, -1])
        while not self.env._is_position_legal(state):
            row = np.random.randint(self.n_rows)
            col = np.random.randint(self.n_cols)
            state = np.array([row, col])
        self.env.pos = state
        action_id = np.random.randint(self.n_actions)
        action = self.env.action_space[action_id]
        while not self.env._is_action_legal(action):
            action_id = np.random.randint(self.n_actions)
            action = self.env.action_space[action_id]
        next_state, reward, done = self.env.step(action)
        return state, action, next_state, reward, done


def choose_action(state, env, Q, epsilon):
    possible_actions = env.action_space
    legal_actions = [ind for ind, act in enumerate(possible_actions) if env._is_action_legal(state, act)]
    Q_s = Q[state[0], state[1], legal_actions]
    assert len(legal_actions) == len(Q_s)
    if not np.all(np.isnan(Q_s)):
        a_id = np.random.choice(np.flatnonzero(Q_s == np.nanmax(Q_s)))  # argmax with random tie-breaking
    else:
        a_id = np.random.randint(len(legal_actions))
    if np.random.random() < epsilon:  # epsilon-greedy
        a_id = np.random.randint(len(legal_actions))
    return env.action_space[legal_actions[a_id], :], legal_actions[a_id]


def q_learn_er(
    env,
    Q: npt.NDArray[np.int_],
    n_steps: int,
    alpha: float = 0.2,
    gamma: float = 0.9,
    epsilon: float = 0.25,
    first_exp_steps: int = 1000,
    repbuf_cls=SmallUER,
):
    """Q-Learning algorithm with experience replay.

    repbuf_cls: class of replay buffer user for replay
    """
    repbuf = repbuf_cls([])
    s = env.reset()
    for i in range(n_steps + first_exp_steps):
        if i % 10000 == 0:
            print("Iteration", i)
        # Choosing action
        a, a_id = choose_action(s, env, Q, epsilon)
        # Taking action and getting next state
        next_s, r, done = env.step(a)
        repbuf.append({"state": s, "action": a_id, "next_state": next_s, "reward": r, "done": done})
        s = next_s
        if done:
            s = env.reset()

        # Learning from experience replay
        if i > first_exp_steps:
            for t in repbuf.sample(n=1):
                if np.isnan(Q[t["state"][0], t["state"][1], t["action"]]):
                    Q[t["state"][0], t["state"][1], t["action"]] = 0.0
                Q[t["state"][0], t["state"][1], t["action"]] += alpha * (
                    t["reward"]
                    + gamma
                    * (
                        0.0
                        if t["next_state"] is None or np.all(np.isnan(Q[t["next_state"][0], t["next_state"][1], :]))
                        else np.nanmax(Q[t["next_state"][0], t["next_state"][1], :])
                    )
                    - Q[t["state"][0], t["state"][1], t["action"]]
                )
    return Q


def q_learn(env, Q, n_steps, alpha=0.2, gamma=0.95, epsilon=0.25):
    s = env.reset()
    for i in range(n_steps):
        if i % 10000 == 0:
            print("Iteration", i)
        # Choosing action
        a, a_id = choose_action(s, env, Q, epsilon)
        # Taking action and getting next state
        next_s, r, done = env.step(a)
        Q[s[0], s[1], a_id] += alpha * (
            r + gamma * (np.nanmax(Q[next_s[0], next_s[1], :]) if next_s is not None else 0.0) - Q[s[0], s[1], a_id]
        )
        s = next_s
        if done:
            s = env.reset()
    return Q


def fill_replay_buffer(env, Q, n_trajs=50, n_steps=100):
    agent = AgentFromQ(env, Q=Q)
    all_trajs = []
    for _ in range(n_trajs):
        trajectory = agent.get_trajectory(n_steps=n_steps)
        all_trajs.extend(trajectory)
    return all_trajs


def main():
    n = 10
    grid = generate_u_grid(n, n)
    goal = np.array([n // 2, n // 2])
    env = EnvFromGrid(grid, goal)
    print(env.grid.shape)

    # Q = np.random.randn(n, n, 4)
    Q = np.full((n, n, 4), fill_value=np.nan, dtype=np.float_)
    Q = q_learn_er(
        env,
        Q.copy(),
        100000,
        repbuf_cls=lambda ts: OptimalStrategyReplay(ts, dist=get_optimal_state_action_dist(env, n_steps=10000)),
    )
    # Q = q_learn(env, Q.copy(), 1000000)
    plot_heatmap(np.nanmax(Q, axis=-1))
    plt.show()

    all_trajs = fill_replay_buffer(env, Q, n_trajs=1000, n_steps=100)
    all_states = [t["state"] for t in all_trajs]
    freqs = count_state_freqs(all_states, grid.shape)
    fig, ax = plt.subplots()
    plot_heatmap(freqs, ax=ax, title=f"State coverage obtained with")
    plt.show()


if __name__ == "__main__":
    main()
