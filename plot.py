"""
plot.py - Plot Snake training progress.
Run after training: python plot.py
"""

import csv, os

def plot(log_path="scores.csv"):
    if not os.path.exists(log_path):
        print("No data found. Run train.py first.")
        return

    episodes, scores, epsilons = [], [], []
    with open(log_path) as f:
        for row in csv.DictReader(f):
            episodes.append(int(row["episode"]))
            scores.append(int(row["score"]))
            epsilons.append(float(row["epsilon"]))

    window = 100
    avg_scores = []
    for i in range(len(scores)):
        lo = max(0, i - window + 1)
        avg_scores.append(sum(scores[lo:i+1]) / (i - lo + 1))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        fig.suptitle("Snake - Q-Learning (11-bit state) Training Progress",
                     fontsize=13, fontweight="bold")

        ax1.plot(episodes, scores,     color="#6ec6e8", alpha=0.3, lw=0.7, label="Score per ep")
        ax1.plot(episodes, avg_scores, color="#1a7abf", lw=2,              label=f"Avg {window} ep")
        ax1.set_ylabel("Score")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.25)
        ax1.set_ylim(bottom=0)

        ax2.plot(episodes, epsilons, color="#e07b39", lw=1.5)
        ax2.set_ylabel("Epsilon")
        ax2.set_xlabel("Episode")
        ax2.grid(True, alpha=0.25)
        ax2.set_ylim(0, 1.05)

        plt.tight_layout()
        out = "training_progress.png"
        plt.savefig(out, dpi=130)
        print(f"[Plot] Saved -> {out}")
        plt.close()

    except ImportError:
        print("(matplotlib is not installed - showing ASCII output)\n")
        step = max(1, len(episodes) // 40)
        for i in range(0, len(episodes), step):
            bar = "#" * min(int(avg_scores[i]), 30)
            print(f"  Ep {episodes[i]:>5}: {bar} ({avg_scores[i]:.1f})")

    print(f"\n-- Summary --")
    print(f"  Episodes      : {len(episodes)}")
    print(f"  Best score    : {max(scores)}")
    print(f"  Final avg     : {avg_scores[-1]:.1f}")
    print(f"  Current epsilon: {epsilons[-1]:.4f}")

if __name__ == "__main__":
    plot()
