"""
game.py - Snake game environment.
Pure game logic, separated from the AI agent.
"""

import pygame
import random
from collections import deque

# Constants
GRID   = 20          # cells per side
CELL   = 28          # pixels per cell
WIDTH  = GRID * CELL # 560px
HEIGHT = GRID * CELL # 560px
FPS    = 15

# Movement directions (dx, dy)
UP    = ( 0, -1)
DOWN  = ( 0,  1)
LEFT  = (-1,  0)
RIGHT = ( 1,  0)
DIRS  = [UP, DOWN, LEFT, RIGHT]

# Colors
BG       = ( 15,  17,  20)
GRID_COL = ( 25,  28,  33)
SNAKE_H  = ( 80, 220, 100)   # snake head
SNAKE_B  = ( 50, 160,  70)   # snake body
FOOD_COL = (220,  70,  70)
TEXT_COL = (200, 200, 200)
WHITE    = (255, 255, 255)
YELLOW   = (255, 210,  50)


class SnakeEnv:
    """
    Snake environment for agent interaction.

    action: 0=straight, 1=turn right, 2=turn left
    Actions are relative to the current direction.

    Each step:
        state, reward, done = env.step(action)
    """

    def __init__(self, render=True):
        self.render_mode = render
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Snake - RL Agent")
            self.clock = pygame.time.Clock()
            self.font  = pygame.font.SysFont("Arial", 20, bold=True)
            self.small = pygame.font.SysFont("Arial", 14)
        self.reset()

    # Reset
    def reset(self):
        cx, cy = GRID // 2, GRID // 2
        self.snake = deque([(cx, cy), (cx-1, cy), (cx-2, cy)])
        self.direction = RIGHT
        self.score = 0
        self.steps = 0          # steps since the last food
        self._place_food()
        return self.get_state()

    def _place_food(self):
        while True:
            pos = (random.randint(0, GRID-1), random.randint(0, GRID-1))
            if pos not in self.snake:
                self.food = pos
                break

    # 11-bit state
    def get_state(self):
        """
        Return an 11-value tuple of 0/1 values:

        [0-2]  danger: straight, right, left
        [3-6]  current direction: up, down, left, right
        [7-10] food position: above, below, left, right
        """
        head   = self.snake[0]
        dir_   = self.direction

        # Relative directions
        straight = dir_
        right_   = self._turn_right(dir_)
        left_    = self._turn_left(dir_)

        state = (
            # Danger
            int(self._is_danger(head, straight)),
            int(self._is_danger(head, right_)),
            int(self._is_danger(head, left_)),

            # Current direction
            int(dir_ == UP),
            int(dir_ == DOWN),
            int(dir_ == LEFT),
            int(dir_ == RIGHT),

            # Relative food position
            int(self.food[1] < head[1]),   # food above
            int(self.food[1] > head[1]),   # food below
            int(self.food[0] < head[0]),   # food left
            int(self.food[0] > head[0]),   # food right
        )
        return state

    def _is_danger(self, head, direction):
        nx = head[0] + direction[0]
        ny = head[1] + direction[1]
        # Wall collision
        if nx < 0 or nx >= GRID or ny < 0 or ny >= GRID:
            return True
        # Body collision, excluding the tail because it will move
        if (nx, ny) in list(self.snake)[:-1]:
            return True
        return False

    @staticmethod
    def _turn_right(d):
        return {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}[d]

    @staticmethod
    def _turn_left(d):
        return {UP: LEFT, LEFT: DOWN, DOWN: RIGHT, RIGHT: UP}[d]

    # Step
    def step(self, action):
        """
        action: 0=straight, 1=turn right, 2=turn left
        """
        self.steps += 1

        # Update direction
        if action == 1:
            self.direction = self._turn_right(self.direction)
        elif action == 2:
            self.direction = self._turn_left(self.direction)
        # action == 0 keeps the current direction

        # Move
        head = self.snake[0]
        new_head = (head[0] + self.direction[0],
                    head[1] + self.direction[1])

        # Check death
        hit_wall = (new_head[0] < 0 or new_head[0] >= GRID or
                    new_head[1] < 0 or new_head[1] >= GRID)
        hit_self = new_head in list(self.snake)[:-1]

        if hit_wall or hit_self:
            return self.get_state(), -10, True

        # Timeout: allow only GRID * GRID steps between food pickups
        if self.steps > GRID * GRID:
            return self.get_state(), -10, True

        self.snake.appendleft(new_head)

        # Eat food
        if new_head == self.food:
            self.score += 1
            self.steps = 0          # reset step count after eating
            reward = 10
            self._place_food()
        else:
            self.snake.pop()        # no food: remove the tail

            # Distance reward: small bonus when moving closer to food,
            # small penalty when moving farther away.
            dist_before = abs(head[0]     - self.food[0]) + abs(head[1]     - self.food[1])
            dist_after  = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            reward = 0.1 if dist_after < dist_before else -0.1

        return self.get_state(), reward, False

    # Render
    def render(self, episode=0, epsilon=0.0, best=0, avg=0.0):
        if not self.render_mode:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    raise SystemExit

        self.screen.fill(BG)

        # Draw grid
        for x in range(GRID):
            for y in range(GRID):
                pygame.draw.rect(self.screen, GRID_COL,
                                 (x*CELL+1, y*CELL+1, CELL-2, CELL-2))

        # Draw food
        fx, fy = self.food
        pygame.draw.rect(self.screen, FOOD_COL,
                         (fx*CELL+3, fy*CELL+3, CELL-6, CELL-6),
                         border_radius=4)

        # Draw snake
        for i, (sx, sy) in enumerate(self.snake):
            color = SNAKE_H if i == 0 else SNAKE_B
            r = 3 if i == 0 else 4
            pygame.draw.rect(self.screen, color,
                             (sx*CELL+2, sy*CELL+2, CELL-4, CELL-4),
                             border_radius=r)

        # HUD
        y0 = 4
        items = [
            (f"Ep: {episode}", YELLOW),
            (f"Score: {self.score}", WHITE),
            (f"Best: {best}", (100, 220, 100)),
            (f"Avg: {avg:.1f}", TEXT_COL),
            (f"eps: {epsilon:.3f}", TEXT_COL),
        ]
        for text, color in items:
            surf = self.small.render(text, True, color)
            self.screen.blit(surf, (4, y0))
            y0 += 17

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        if self.render_mode:
            pygame.quit()
