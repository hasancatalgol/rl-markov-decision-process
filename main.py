from __future__ import annotations
from typing import Tuple, Dict, Optional, Iterable, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import imageio.v2 as imageio

import pygame
from pygame import gfxdraw

import gymnasium as gym
from gymnasium import spaces


# ----------------------------
# Maze Environment (Gymnasium)
# ----------------------------
class Maze(gym.Env):
    """
    A simple 5x5 deterministic maze with walls. 
    The agent starts at (0, 0) unless exploring_starts=True. 
    The goal is at (size-1, size-1).

    Rewards:
      - shaped_rewards=False: -1 per step until goal (sparse reward)
      - shaped_rewards=True: -normalized distance to goal (dense shaping)
    """

    # Metadata for rendering options
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 20}

    def __init__(
        self,
        exploring_starts: bool = False,  # Whether to randomize the starting state
        shaped_rewards: bool = False,    # Use dense or sparse reward function
        size: int = 5,                   # Maze grid size (must be 5x5 for this example)
        render_mode: Optional[str] = None, # Render mode ("human" or "rgb_array")
        max_steps: Optional[int] = None, # Maximum steps before truncation
    ) -> None:
        super().__init__()
        assert size == 5, "This example maze is coded for size=5."  # Hardcoded maze walls

        # Maze configuration
        self.size = size
        self.exploring_starts = exploring_starts
        self.shaped_rewards = shaped_rewards
        self.goal = (size - 1, size - 1)  # Bottom-right corner
        self.maze = self._create_maze(size=size)  # Adjacency dict defining valid moves
        self.distances = self._compute_distances(self.goal, self.maze)  # Shortest-path distances to goal

        # Define Gymnasium spaces
        self.action_space = spaces.Discrete(4)  # Four discrete actions (up, right, down, left)
        self.action_meanings = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}  # For readability
        self.observation_space = spaces.MultiDiscrete([size, size])  # State is (row, col)

        # Rendering attributes
        self.render_mode = render_mode
        self.screen = None  
        self.clock = None  

        # Agent state
        self.state: Tuple[int, int] = (0, 0)  # Initial position
        self.step_count = 0
        self.max_steps = max_steps  # Optional step limit

    # -------------
    # Gymnasium API
    # -------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment and return the initial state."""
        super().reset(seed=seed)

        if self.exploring_starts:
            # Randomize start state (cannot be the goal state)
            s = tuple(self.observation_space.sample())
            while s == self.goal:
                s = tuple(self.observation_space.sample())
            self.state = s
        else:
            self.state = (0, 0)  # Fixed start

        self.step_count = 0

        # Ensure rendering display is created
        if self.render_mode == "human":
            self._ensure_display()

        return self.state, {"dist_to_goal": self.distances[self.state]}  # Return state + info

    def step(self, action: int):
        """Take one step in the environment given an action."""
        assert self.action_space.contains(action), "Invalid action"

        # Compute reward before transitioning
        reward = self.compute_reward(self.state, action)

        # Move to the next state (if valid)
        self.state = self._get_next_state(self.state, action)
        self.step_count += 1

        # Episode termination conditions
        terminated = self.state == self.goal
        truncated = False
        if self.max_steps is not None and self.step_count >= self.max_steps and not terminated:
            truncated = True

        info = {}

        # Render if needed
        if self.render_mode is not None:
            self.render()

        return self.state, reward, terminated, truncated, info

    def render(self):
        """Render the maze and agent using pygame."""
        assert self.render_mode in (None, "rgb_array", "human")
        screen_size = 600
        scale = screen_size / 5  # Scale cells to fit screen

        # Initialize rendering
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((screen_size, screen_size))
                pygame.display.set_caption("Maze")
                self.clock = pygame.time.Clock()
            else:
                self.screen = pygame.Surface((screen_size, screen_size))  # Offscreen surface

        # Intermediate surface (so we can flip Y later)
        surf = pygame.Surface((screen_size, screen_size))
        surf.fill((22, 36, 71))  # Background color

        # Draw maze walls
        for row in range(5):
            for col in range(5):
                state = (row, col)
                for next_state in [(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]:
                    if next_state not in self.maze[state]:
                        # Draw wall between state and next_state
                        row_diff, col_diff = np.subtract(next_state, state)
                        left = (col + (col_diff > 0)) * scale - 2 * (col_diff != 0)
                        right = ((col + 1) - (col_diff < 0)) * scale + 2 * (col_diff != 0)
                        top = (5 - (row + (row_diff > 0))) * scale - 2 * (row_diff != 0)
                        bottom = (5 - ((row + 1) - (row_diff < 0))) * scale + 2 * (row_diff != 0)
                        gfxdraw.filled_polygon(
                            surf,
                            [(left, bottom), (left, top), (right, top), (right, bottom)],
                            (255, 255, 255),
                        )

        # Draw goal square
        left, right, top, bottom = scale * 4 + 10, scale * 5 - 10, scale - 10, 10
        gfxdraw.filled_polygon(
            surf, [(left, bottom), (left, top), (right, top), (right, bottom)], (40, 199, 172)
        )

        # Draw agent as a circle
        agent_row = int(screen_size - scale * (self.state[0] + 0.5))
        agent_col = int(scale * (self.state[1] + 0.5))
        gfxdraw.filled_circle(surf, agent_col, agent_row, int(scale * 0.6 / 2), (228, 63, 90))

        # Flip vertically so (0,0) is bottom-left
        surf = pygame.transform.flip(surf, False, True)

        # Blit & update
        if self.render_mode == "human":
            self.screen.blit(surf, (0, 0))
            pygame.display.flip()
            if self.clock is not None:
                self.clock.tick(self.metadata["render_fps"])
            return None

        # Return numpy array for rgb_array mode
        self.screen.blit(surf, (0, 0))
        arr = np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
        return arr

    def close(self):
        """Close the rendering window and cleanup pygame."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    # -----------------
    # Helper functions
    # -----------------
    def compute_reward(self, state: Tuple[int, int], action: int) -> float:
        """Compute reward for taking an action in the current state."""
        next_state = self._get_next_state(state, action)
        if self.shaped_rewards:
            # Dense shaping: negative normalized distance to goal
            return -float(self.distances[next_state] / self.distances.max())
        # Sparse reward: -1 per step unless already at goal
        return -float(state != self.goal)

    def simulate_step(self, state: Tuple[int, int], action: int):
        """Simulate a step without changing environment state (for planning)."""
        reward = self.compute_reward(state, action)
        next_state = self._get_next_state(state, action)
        terminated = next_state == self.goal
        truncated = False
        info = {}
        return next_state, reward, terminated, truncated, info

    def _get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Return the next state after applying an action, if valid."""
        if action == 0:      # UP
            cand = (state[0] - 1, state[1])
        elif action == 1:    # RIGHT
            cand = (state[0], state[1] + 1)
        elif action == 2:    # DOWN
            cand = (state[0] + 1, state[1])
        elif action == 3:    # LEFT
            cand = (state[0], state[1] - 1)
        else:
            raise ValueError(f"Action not supported: {action}")

        # Only move if the candidate is connected in the maze
        return cand if cand in self.maze[state] else state

    @staticmethod
    def _create_maze(size: int) -> Dict[Tuple[int, int], Iterable[Tuple[int, int]]]:
        """Generate a hardcoded 5x5 maze layout with walls and obstacles."""
        # Initialize adjacency dict (all 4 neighbors per cell)
        maze = {(row, col): [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
                for row in range(size) for col in range(size)}

        # Define outer boundaries
        left_edges = [[(row, 0), (row, -1)] for row in range(size)]
        right_edges = [[(row, size - 1), (row, size)] for row in range(size)]
        upper_edges = [[(0, col), (-1, col)] for col in range(size)]
        lower_edges = [[(size - 1, col), (size, col)] for col in range(size)]

        # Internal walls (hardcoded for 5x5)
        walls = [
            [(1, 0), (1, 1)], [(2, 0), (2, 1)], [(3, 0), (3, 1)],
            [(1, 1), (1, 2)], [(2, 1), (2, 2)], [(3, 1), (3, 2)],
            [(3, 1), (4, 1)], [(0, 2), (1, 2)], [(1, 2), (1, 3)],
            [(2, 2), (3, 2)], [(2, 3), (3, 3)], [(2, 4), (3, 4)],
            [(4, 2), (4, 3)], [(1, 3), (1, 4)], [(2, 3), (2, 4)],
        ]

        # Remove blocked edges from adjacency dict
        obstacles = upper_edges + lower_edges + left_edges + right_edges + walls
        for src, dst in obstacles:
            maze[src].remove(dst)
            if dst in maze:
                maze[dst].remove(src)
        return maze

    @staticmethod
    def _compute_distances(
        goal: Tuple[int, int],
        maze: Dict[Tuple[int, int], Iterable[Tuple[int, int]]]
    ) -> np.ndarray:
        """Compute shortest-path distances from all states to goal using Dijkstra."""
        distances = np.full((5, 5), np.inf)
        visited = set()
        distances[goal] = 0.0

        # Dijkstra’s algorithm
        while visited != set(maze):
            # Find closest unvisited node
            sorted_dst = [(v // 5, v % 5) for v in distances.argsort(axis=None)]
            closest = next(x for x in sorted_dst if x not in visited)
            visited.add(closest)

            # Relax edges
            for neighbour in maze[closest]:
                distances[neighbour] = min(distances[neighbour], distances[closest] + 1)
        return distances

# ------------------------
# Utilities (VS Code-safe)
# ------------------------
def save_video_gif(frames: List[np.ndarray], path: str = "maze_run.gif", fps: int = 20) -> str:
    # imageio expects HxWxC arrays (uint8)
    imageio.mimsave(path, frames, duration=1.0 / fps)
    return path

def save_video_mp4(frames: List[np.ndarray], path: str = "maze_run.mp4", fps: int = 20) -> str:
    # MP4 via matplotlib animation (no IPython required)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_axis_off()
    im = ax.imshow(frames[0])

    def _update(frame):
        im.set_data(frame)
        return [im]

    anim_obj = animation.FuncAnimation(fig=fig, func=_update, frames=frames,
                                       interval=1000 / fps, blit=True, repeat=False)
    anim_obj.save(path, fps=fps)
    plt.close(fig)
    return path


# ------------------------
# Example usage / walkthrough
# ------------------------
def random_policy(_state: Tuple[int, int]) -> np.ndarray:
    return np.array([0.25, 0.25, 0.25, 0.25], dtype=float)

def test_agent(env: Maze, policy, record: bool = True, fps: int = 20) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    state, _info = env.reset()
    terminated = False
    truncated = False

    if record:
        img = env.render()  # returns array if render_mode="rgb_array"
        if isinstance(img, np.ndarray):
            frames.append(img)

    while not (terminated or truncated):
        action_probs = policy(state)
        action = int(np.random.choice(range(4), p=action_probs))
        state, reward, terminated, truncated, info = env.step(action)
        if record:
            img = env.render()
            if isinstance(img, np.ndarray):
                frames.append(img)

    return frames


if __name__ == "__main__":
    # Quick demo: RGB recording (no on-screen window)
    env = Maze(render_mode="rgb_array", shaped_rewards=False, exploring_starts=False, max_steps=200)
    frames = test_agent(env, random_policy, record=True, fps=env.metadata["render_fps"])
    env.close()

    # Save as GIF (easy to view in VS Code/Explorer)
    out_gif = save_video_gif(frames, "maze_run.gif", fps=env.metadata["render_fps"])
    print(f"Saved rollout to {out_gif}")

    # Also save as MP4 if you prefer a video file:
    # out_mp4 = save_video_mp4(frames, "maze_run.mp4", fps=env.metadata['render_fps'])
    # print(f"Saved rollout to {out_mp4}")

    # Basics: state/action space & a quick trajectory
    env = Maze()
    (initial_state, _info) = env.reset()
    print(f"The new episode will start in state: {initial_state}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space} (meanings: {env.action_meanings})")

    # Step once
    action = 2  # DOWN
    next_state, reward, terminated, truncated, _ = env.step(action)
    print(f"After moving down 1 row, the agent is in state: {next_state}")
    print(f"Reward: {reward}, terminated={terminated}, truncated={truncated}")
    env.close()
