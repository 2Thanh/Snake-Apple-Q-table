"""
agent.py - Q-learning agent with an 11-bit state.

The state has only 2^11 = 2048 possible values, which keeps the Q-table small
and fast to train. No discretization is needed because the state is binary.
"""

import numpy as np
import pickle
import os


class QLearningAgent:

    def __init__(
        self,
        alpha         = 0.1,     # learning rate
        gamma         = 0.9,     # discount factor
        epsilon       = 1.0,     # initial exploration rate
        epsilon_min   = 0.01,
        epsilon_decay = 0.995,
    ):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: key = state tuple (11 bit) -> [Q(0), Q(1), Q(2)]
        # Actions: 0=straight, 1=right, 2=left
        self.q_table = {}

    # Q-table helpers
    def _get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]
        return self.q_table[state]

    # Action selection
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(3)
        return int(np.argmax(self._get_q(state)))

    # Q-table update (Bellman)
    def update(self, state, action, reward, next_state, done):
        q      = self._get_q(state)
        q_next = self._get_q(next_state)

        target = reward if done else reward + self.gamma * max(q_next)
        q[action] += self.alpha * (target - q[action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # Save / load
    def save(self, path="q_table.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"q_table": self.q_table, "epsilon": self.epsilon}, f)
        print(f"[Agent] Saved -> {path}  ({len(self.q_table)} states)")

    def load(self, path="q_table.pkl"):
        if not os.path.exists(path):
            print(f"[Agent] Could not find {path}; starting from scratch.")
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.epsilon = data["epsilon"]
        print(f"[Agent] Loaded <- {path}  ({len(self.q_table)} states, eps={self.epsilon:.4f})")
