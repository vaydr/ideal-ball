import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from objects import Player, Ball, Court, Basket
import math
import random

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

# Define the MCTS node
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == 3  # Assuming 3 possible actions: Dribble, Pass, Shoot

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self):
        action = len(self.children)
        new_state = self.state.copy()
        if action == 0:  # Dribble
            direction = np.random.uniform(-1, 1, 2)  # Random direction
            new_state = self.dribble(new_state, direction)
        elif action == 1:  # Pass
            new_state, _ = self.pass_ball(new_state)
        elif action == 2:  # Shoot
            new_state = self.shoot(new_state)
        child_node = MCTSNode(new_state, parent=self)
        self.children.append(child_node)
        return child_node

    def dribble(self, state, direction):
        new_state = state + direction
        return new_state

    def pass_ball(self, state):
        target = np.random.randint(0, 5)
        if np.random.rand() < 0.8:  # 80% success rate
            success = 1
        else:
            success = 0
        return state[target], success

    def shoot(self, state):
        if np.random.rand() < 0.5:  # 50% success rate
            success = 1
        else:
            success = 0
        return success

# Define the reinforcement learning agent with MCTS
class ReinforcementLearningAgent(Player):
    def __init__(self, size, speed, fg_dict, input_size, hidden_size, output_size, learning_rate=0.01):
        super().__init__(size, speed, fg_dict)
        self.policy_network = PolicyNetwork(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.gamma = 0.99
        self.saved_log_probs = []
        self.rewards = []
        self.root = None

    def select_action(self, state):
        if self.root is None or not np.array_equal(self.root.state, state):
            self.root = MCTSNode(state)
        for _ in range(100):  # Number of MCTS simulations
            node = self.root
            while node.is_fully_expanded():
                node = node.best_child()
            if node.visits > 0:
                node = node.expand()
            reward = self.simulate(node.state)
            self.backpropagate(node, reward)
        best_child = self.root.best_child(c_param=0)
        action = self.root.children.index(best_child)
        self.root = best_child
        return action

    def simulate(self, state):
        total_reward = 0
        for _ in range(10):  # Number of simulation steps
            action = random.choice([0, 1, 2])
            if action == 0:  # Dribble
                direction = np.random.uniform(-1, 1, 2)  # Random direction
                state = self.dribble(state, direction)
            elif action == 1:  # Pass
                state, success = self.pass_ball(state)
                total_reward += success
            elif action == 2:  # Shoot
                success = self.shoot(state)
                total_reward += success
        return total_reward

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def update_policy(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.saved_log_probs = []
        self.rewards = []

# Define the reinforcement learning team
class RLTeam:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.agents = [ReinforcementLearningAgent(10, 5, {}, input_size, hidden_size, output_size, learning_rate) for _ in range(5)]

    def select_actions(self, state):
        return [agent.select_action(state.flatten()) for agent in self.agents]

    def update_policies(self):
        for agent in self.agents:
            agent.update_policy()

# Define the basketball environment
class BasketballEnv:
    def __init__(self):
        self.court = Court(28, 15)
        self.basket = Basket(0.45)
        self.team1 = RLTeam(22, 128, 3)  # Adjusted input size for more state info
        self.team2 = RLTeam(22, 128, 3)  # Adjusted input size for more state info
        self.state = self.reset()

    def reset(self):
        # Reset player positions (10 players) and ball position
        self.state = np.zeros((11, 2))  # 10 players + 1 ball, x and y positions
        return self.state

    def step(self, actions):
        rewards = [0, 0]  # Team 1 and Team 2 rewards
        new_state = self.state.copy()

        for i in range(5):
            if actions[i] == 0:  # Dribble
                direction = np.random.uniform(-1, 1, 2)  # Random direction
                new_state[i] = self.dribble(self.state[i], direction)
            elif actions[i] == 1:  # Pass
                new_state[i], success = self.pass_ball(self.state[i], self.state)
                rewards[0] += success
            elif actions[i] == 2:  # Shoot
                success = self.shoot(self.state[i])
                rewards[0] += success

        for i in range(5, 10):
            if actions[i] == 0:  # Dribble
                direction = np.random.uniform(-1, 1, 2)  # Random direction
                new_state[i] = self.dribble(self.state[i], direction)
            elif actions[i] == 1:  # Pass
                new_state[i], success = self.pass_ball(self.state[i], self.state)
                rewards[1] += success
            elif actions[i] == 2:  # Shoot
                success = self.shoot(self.state[i])
                rewards[1] += success

        self.state = new_state
        return self.state, rewards

    def dribble(self, position, direction):
        # Dribble logic: move in the chosen direction
        new_position = position + direction
        return new_position

    def pass_ball(self, position, state):
        # Pass logic: random pass
        target = np.random.randint(0, 5)
        if np.random.rand() < 0.8:  # 80% success rate
            success = 1
        else:
            success = 0
        return state[target], success

    def shoot(self, position):
        # Shoot logic: random chance
        if np.random.rand() < 0.5:  # 50% success rate
            success = 1
        else:
            success = 0
        return success

# Train the agents
env = BasketballEnv()
num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    for t in range(100):  # Each episode lasts for 100 time steps
        actions = env.team1.select_actions(state) + env.team2.select_actions(state)
        new_state, rewards = env.step(actions)
        for agent in env.team1.agents:
            agent.rewards.append(rewards[0])
        for agent in env.team2.agents:
            agent.rewards.append(rewards[1])
        state = new_state

    env.team1.update_policies()
    env.team2.update_policies()

print("Training completed.")
