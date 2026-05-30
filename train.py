"""
train.py - Snake training loop.

Usage:
    python train.py              # train with graphics every 20 episodes
    python train.py --no-render  # train faster without graphics
    python train.py --load       # continue from a saved Q-table
"""

import sys
import os
from game  import SnakeEnv
from agent import QLearningAgent

# Configuration
TOTAL_EPISODES = 5000
RENDER_EVERY   = 20     # render every N episodes
SAVE_EVERY     = 500
QTABLE_PATH    = "models/q_table.pkl"
LOG_PATH       = "models/train_log.csv"

NO_RENDER = "--no-render" in sys.argv
LOAD      = "--load"      in sys.argv

# Setup
agent = QLearningAgent(
    alpha         = 0.1,
    gamma         = 0.9,
    epsilon       = 1.0,
    epsilon_min   = 0.01,
    epsilon_decay = 0.995,
)
if LOAD:
    agent.load(QTABLE_PATH)

if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w") as f:
        f.write("episode,score,epsilon,q_states\n")

# Statistics
best_score   = 0
score_window = []   # last 100 episodes

print("=" * 58)
print("   Snake - Q-Learning (11-bit state)")
print("=" * 58)
print(f"  Episodes : {TOTAL_EPISODES}  |  Render every: {RENDER_EVERY} ep")
print(f"  Load prev: {LOAD}")
print("=" * 58)

# Main loop
for ep in range(1, TOTAL_EPISODES + 1):

    should_render = (not NO_RENDER) and (ep % RENDER_EVERY == 0)
    env   = SnakeEnv(render=should_render)
    state = env.reset()
    done  = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state

        if should_render:
            avg = sum(score_window[-100:]) / max(len(score_window[-100:]), 1)
            env.render(episode=ep, epsilon=agent.epsilon,
                       best=best_score, avg=avg)

    env.close()

    score = env.score
    agent.decay_epsilon()
    score_window.append(score)
    if len(score_window) > 100:
        score_window.pop(0)
    avg100 = sum(score_window) / len(score_window)

    if score > best_score:
        best_score = score
        agent.save("q_table_best.pkl")

    with open(LOG_PATH, "a") as f:
        f.write(f"{ep},{score},{agent.epsilon:.5f},{len(agent.q_table)}\n")

    if ep % 20 == 0:
        bar = "#" * min(int(avg100), 25)
        print(
            f"  Ep {ep:>5}/{TOTAL_EPISODES}"
            f"  Score:{score:>4}"
            f"  Avg100:{avg100:>5.1f}"
            f"  Best:{best_score:>4}"
            f"  eps:{agent.epsilon:.3f}"
            f"  {bar}"
        )

    if ep % SAVE_EVERY == 0:
        agent.save(QTABLE_PATH)

# Finish
agent.save(QTABLE_PATH)
print()
print("=" * 58)
print(f"  Training complete!  Best: {best_score}  |  Q-states: {len(agent.q_table)}")
print("=" * 58)
