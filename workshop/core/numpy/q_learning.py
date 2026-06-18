"""Q-Learning GridWorld Maze Navigation from scratch using NumPy."""

import matplotlib.pyplot as plt
import numpy as np

# Action mappings
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTION_LABELS = {UP: "↑", DOWN: "↓", LEFT: "←", RIGHT: "→"}


class GridWorld:
    def __init__(self, size=6):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        # Define obstacles
        self.obstacles = {(1, 1), (2, 2), (3, 3), (4, 4), (1, 3), (3, 1), (4, 2)}
        self.reset()

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        r, c = self.state
        if action == UP:
            next_state = (max(0, r - 1), c)
        elif action == DOWN:
            next_state = (min(self.size - 1, r + 1), c)
        elif action == LEFT:
            next_state = (r, max(0, c - 1))
        elif action == RIGHT:
            next_state = (r, min(self.size - 1, c + 1))
        else:
            next_state = self.state

        # Check if next state is an obstacle
        if next_state in self.obstacles:
            # Hit obstacle: bounce back and get negative reward
            reward = -1.0
            done = False
        elif next_state == self.goal:
            self.state = next_state
            reward = 10.0
            done = True
        else:
            self.state = next_state
            reward = -0.1  # Small negative step cost
            done = False

        return self.state, reward, done


def main(hook=None, config=None):
    from workshop.utils.hooks import NoOpProgressHook

    config = config or {}
    hook = hook or NoOpProgressHook()

    epochs = int(config.get("epochs", 200))
    learning_rate = float(config.get("learning_rate", 0.1))
    discount_factor = float(config.get("discount_factor", 0.9))
    epsilon = float(config.get("epsilon", 0.9))
    epsilon_decay = float(config.get("epsilon_decay", 0.98))

    print("Q-Learning GridWorld Maze Navigation")
    print("=" * 45)
    print(f"Episodes: {epochs}")
    print(f"Alpha: {learning_rate}, Gamma: {discount_factor}")
    print(f"Initial Epsilon: {epsilon}, Decay: {epsilon_decay}")
    print()

    # Initialize environment
    env = GridWorld(size=6)

    # Initialize Q-table: State space is size x size, action space is 4
    q_table = np.zeros((env.size, env.size, len(ACTIONS)))

    if hook.is_cancelled():
        return
    hook.update_stage("Environment Setup", 10)
    print("Environment setup complete. Maze size: 6x6.")

    if hook.is_cancelled():
        return
    hook.update_stage("Agent Training", 20)

    # Training loop
    epsilon_curr = epsilon
    reporting_interval = max(1, epochs // 10)

    for ep in range(epochs):
        if hook.is_cancelled():
            return

        state = env.reset()
        done = False
        steps = 0
        total_reward = 0.0

        # Max 50 steps to prevent infinite loop early in training
        while not done and steps < 50:
            r, c = state

            # Epsilon-greedy action selection
            if np.random.rand() < epsilon_curr:
                action = np.random.choice(ACTIONS)
            else:
                action = np.argmax(q_table[r, c])

            next_state, reward, done = env.step(action)
            nr, nc = next_state

            # Q-learning Update formula
            best_next_q = np.max(q_table[nr, nc])
            q_table[r, c, action] += learning_rate * (reward + discount_factor * best_next_q - q_table[r, c, action])

            state = next_state
            total_reward += reward
            steps += 1

        # Decay exploration rate
        epsilon_curr = max(0.01, epsilon_curr * epsilon_decay)

        # Update metrics
        if ep % reporting_interval == 0 or ep == epochs - 1:
            progress = 20 + int(60 * (ep / epochs))
            hook.update_stage("Agent Training", progress)
            hook.update_metrics({"epoch": ep + 1, "steps_taken": steps, "total_reward": total_reward, "epsilon": epsilon_curr})
            print(f"Episode {ep + 1:3d}/{epochs}: Steps={steps}, Reward={total_reward:6.2f}, Epsilon={epsilon_curr:.4f}")

    if hook.is_cancelled():
        return
    hook.update_stage("Policy Evaluation", 80)
    print("\nTraining complete. Evaluating optimal path...")

    # Run evaluation episode
    state = env.reset()
    path = [state]
    done = False
    eval_steps = 0
    while not done and eval_steps < 30:
        r, c = state
        action = np.argmax(q_table[r, c])
        next_state, _, done = env.step(action)
        state = next_state
        path.append(state)
        eval_steps += 1

    print(f"Optimal Path found in {eval_steps} steps: {path}")

    if hook.is_cancelled():
        return
    hook.update_stage("Visualization", 90)

    # 1. Grid Path Plot
    plt.figure(figsize=(6, 6))
    grid = np.zeros((env.size, env.size))
    for r in range(env.size):
        for c in range(env.size):
            if (r, c) in env.obstacles:
                grid[r, c] = 0.5  # Dark block for obstacles
            elif (r, c) == env.goal:
                grid[r, c] = 0.8  # Goal color

    plt.imshow(grid, cmap="gray_r", origin="upper")

    # Plot start and goal
    plt.text(env.start[1], env.start[0], "START", ha="center", va="center", color="green", fontweight="bold")
    plt.text(env.goal[1], env.goal[0], "GOAL", ha="center", va="center", color="white", fontweight="bold")

    # Plot obstacles
    for obs in env.obstacles:
        plt.text(obs[1], obs[0], "█", ha="center", va="center", color="black", fontsize=20)

    # Plot path taken
    path_y, path_x = zip(*path)
    plt.plot(path_x, path_y, color="red", marker="o", linewidth=2.5, markersize=6, label="Agent Path")

    plt.title("Q-Learning GridWorld Final Route")
    plt.legend(loc="upper left")
    plt.grid(True, which="both", color="gray", linestyle="-", linewidth=0.5)
    plt.xticks(range(env.size))
    plt.yticks(range(env.size))
    hook.save_plot("q_learning_grid_path.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Q-Value Policy Heatmap
    plt.figure(figsize=(6, 6))
    max_q = np.max(q_table, axis=2)
    plt.imshow(max_q, cmap="YlGn", origin="upper")

    # Draw arrows on grid cells
    for r in range(env.size):
        for c in range(env.size):
            if (r, c) in env.obstacles:
                plt.text(c, r, "█", ha="center", va="center", color="black", fontsize=20)
                continue
            if (r, c) == env.goal:
                plt.text(c, r, "★", ha="center", va="center", color="red", fontsize=15, fontweight="bold")
                continue

            best_action = np.argmax(q_table[r, c])
            arrow = ACTION_LABELS[best_action]
            plt.text(c, r, arrow, ha="center", va="center", color="blue", fontsize=16, fontweight="bold")

    plt.colorbar(label="Max Q-value")
    plt.title("Learned Policy Map (Arrows show best actions)")
    plt.xticks(range(env.size))
    plt.yticks(range(env.size))
    hook.save_plot("q_learning_policy_map.png", dpi=150, bbox_inches="tight")
    plt.close()

    hook.update_stage("Complete", 100)
