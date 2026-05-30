"""
watch.py - Watch the Snake agent play live.

Usage:
    python watch.py              # watch the trained agent
    python watch.py --slow       # slow down playback
    python watch.py --random     # watch an untrained random agent
    python watch.py --episodes 5 # watch only 5 games
"""

import sys
import time
import os
import pygame
from game  import SnakeEnv, GRID, CELL, WIDTH, HEIGHT, FPS
from game  import BG, GRID_COL, SNAKE_H, SNAKE_B, FOOD_COL, WHITE, YELLOW, TEXT_COL
from agent import QLearningAgent

# Args
SLOW   = "--slow"   in sys.argv
RANDOM = "--random" in sys.argv
QTABLE = "models/q_table_best.pkl" if os.path.exists("models/q_table_best.pkl") else "models/q_table.pkl"
SPEED  = FPS // 3 if SLOW else FPS

N_EP = 999999
for i, arg in enumerate(sys.argv):
    if arg == "--episodes" and i + 1 < len(sys.argv):
        N_EP = int(sys.argv[i + 1])

# Agent
agent = QLearningAgent(epsilon=0.0)
if not RANDOM:
    agent.load(QTABLE)
    agent.epsilon = 0.0

# Pygame setup
pygame.init()

HUD_W  = 200
screen = pygame.display.set_mode((WIDTH + HUD_W, HEIGHT))
pygame.display.set_caption("Snake - Watching AI")
clock  = pygame.time.Clock()

font_big  = pygame.font.SysFont("Arial", 26, bold=True)
font_med  = pygame.font.SysFont("Arial", 17, bold=True)
font_sm   = pygame.font.SysFont("Arial", 13)

PANEL_BG  = (20, 22, 28)
GRAY      = (130, 130, 130)
GREEN_C   = ( 80, 210,  90)
RED_C     = (210,  70,  70)

# Draw right-side HUD
def draw_hud(env, ep, history, state, paused):
    panel = pygame.Rect(WIDTH, 0, HUD_W, HEIGHT)
    pygame.draw.rect(screen, PANEL_BG, panel)
    pygame.draw.line(screen, (40, 44, 52), (WIDTH, 0), (WIDTH, HEIGHT), 2)

    x0 = WIDTH + 10
    y  = 12

    def label(text, color=GRAY, size="sm"):
        f = font_sm if size == "sm" else font_med
        s = f.render(text, True, color)
        screen.blit(s, (x0, y))
        return s.get_height() + 4

    def row(lbl, val, vc=WHITE):
        nonlocal y
        ls = font_sm.render(lbl, True, GRAY)
        vs = font_sm.render(str(val), True, vc)
        screen.blit(ls, (x0, y))
        screen.blit(vs, (x0 + HUD_W - vs.get_width() - 14, y))
        y += ls.get_height() + 5

    # Title
    t = font_med.render("SNAKE RL", True, YELLOW)
    screen.blit(t, (x0, y)); y += t.get_height() + 10
    pygame.draw.line(screen, (40,44,52), (x0, y), (x0 + HUD_W - 14, y)); y += 8

    # Episode statistics
    row("Episode",  ep)
    row("Score",    env.score,  GREEN_C)
    row("Best",     max(history) if history else 0, YELLOW)
    avg = sum(history[-50:]) / max(len(history[-50:]), 1)
    row("Avg 50",   f"{avg:.1f}")
    row("Length",   len(env.snake))
    row("eps",      f"{agent.epsilon:.3f}")
    row("Mode",     "SLOW" if SLOW else "NORMAL")
    row("Agent",    "RANDOM" if RANDOM else "Q-TABLE")

    y += 6
    pygame.draw.line(screen, (40,44,52), (x0, y), (x0 + HUD_W - 14, y)); y += 8

    # State 11-bit
    t2 = font_sm.render("STATE (11-bit)", True, GRAY)
    screen.blit(t2, (x0, y)); y += t2.get_height() + 4

    state_labels = [
        ("danger straight", state[0]),
        ("danger right",    state[1]),
        ("danger left",     state[2]),
        ("dir up",          state[3]),
        ("dir down",        state[4]),
        ("dir left",        state[5]),
        ("dir right",       state[6]),
        ("food up",         state[7]),
        ("food down",       state[8]),
        ("food left",       state[9]),
        ("food right",      state[10]),
    ]
    for lbl, val in state_labels:
        color = GREEN_C if val == 1 else (50, 55, 65)
        ls = font_sm.render(lbl, True, GRAY)
        vs = font_sm.render("1" if val else "0", True, color)
        screen.blit(ls, (x0, y))
        screen.blit(vs, (x0 + HUD_W - vs.get_width() - 14, y))
        y += ls.get_height() + 3

    y += 6
    pygame.draw.line(screen, (40,44,52), (x0, y), (x0 + HUD_W - 14, y)); y += 8

    # Q-values
    t3 = font_sm.render("Q-VALUES", True, GRAY)
    screen.blit(t3, (x0, y)); y += t3.get_height() + 4

    qvals  = agent._get_q(state)
    labels = ["straight", "right", "left"]
    best_a = int(qvals.index(max(qvals)))
    for i, (lbl, q) in enumerate(zip(labels, qvals)):
        color = GREEN_C if i == best_a else RED_C
        marker = ">" if i == best_a else " "
        s = font_sm.render(f"{marker}{lbl}: {q:.2f}", True, color)
        screen.blit(s, (x0, y)); y += s.get_height() + 4

    # Pause overlay
    if paused:
        ov = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 120))
        screen.blit(ov, (0, 0))
        ps = font_big.render("PAUSED", True, YELLOW)
        screen.blit(ps, (WIDTH//2 - ps.get_width()//2, HEIGHT//2 - 20))
        hint = font_sm.render("SPACE to resume", True, WHITE)
        screen.blit(hint, (WIDTH//2 - hint.get_width()//2, HEIGHT//2 + 16))

    # Bottom shortcuts
    hy = HEIGHT - 36
    for txt in ["SPACE: pause", "S: slow/normal", "Q: quit"]:
        hs = font_sm.render(txt, True, (60, 64, 72))
        screen.blit(hs, (x0, hy)); hy += 13


# Draw game board
def draw_game(env):
    # Background
    game_surf = pygame.Rect(0, 0, WIDTH, HEIGHT)
    pygame.draw.rect(screen, BG, game_surf)

    # Grid
    for gx in range(GRID):
        for gy in range(GRID):
            pygame.draw.rect(screen, GRID_COL,
                             (gx*CELL+1, gy*CELL+1, CELL-2, CELL-2))

    # Food
    fx, fy = env.food
    pygame.draw.rect(screen, FOOD_COL,
                     (fx*CELL+4, fy*CELL+4, CELL-8, CELL-8),
                     border_radius=5)

    # Snake
    for i, (sx, sy) in enumerate(env.snake):
        color = SNAKE_H if i == 0 else SNAKE_B
        pygame.draw.rect(screen, color,
                         (sx*CELL+2, sy*CELL+2, CELL-4, CELL-4),
                         border_radius=3 if i == 0 else 4)


# Main loop
history = []
ep      = 0
paused  = False
running = True

while running and ep < N_EP:
    ep += 1

    env   = SnakeEnv(render=False)
    state = env.reset()
    done  = False

    while not done and running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:     running = False
                if event.key == pygame.K_SPACE: paused  = not paused
                if event.key == pygame.K_s:
                    SLOW  = not SLOW
                    SPEED = FPS // 3 if SLOW else FPS

        if paused:
            draw_game(env)
            draw_hud(env, ep, history, state, paused=True)
            pygame.display.flip()
            clock.tick(10)
            continue

        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        state = next_state

        draw_game(env)
        draw_hud(env, ep, history, state, paused=False)
        pygame.display.flip()
        clock.tick(SPEED)

    history.append(env.score)

    if running and not paused:
        # Short game-over screen
        go = font_big.render(f"Game {ep} - Score: {env.score}", True, RED_C)
        screen.blit(go, (WIDTH//2 - go.get_width()//2, HEIGHT//2 - 16))
        pygame.display.flip()
        time.sleep(0.6)

pygame.quit()

print(f"\n-- Results for {ep} games --")
if history:
    print(f"  Best   : {max(history)}")
    print(f"  Average: {sum(history)/len(history):.1f}")
