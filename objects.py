import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import random

def shooting_probability(distance_feet, enemy_players, player_position):
    AMPLITUDE = 0.9
    SPREAD = 0.07
    base_probability = AMPLITUDE / (1 + SPREAD * distance_feet)
    
    for enemy in enemy_players:
        enemy_distance = np.linalg.norm(np.array(enemy.position) - np.array(player_position))
        if enemy_distance <= enemy.size:
            base_probability *= (1 - 0.1 * enemy.size)
    
    return base_probability

class Player:
    def __init__(self, size, speed, shooting_probability, team, position):
        self.size = size
        self.speed = speed
        self.shooting_probability = shooting_probability
        self.team = team  # 'home' or 'away'
        self.position = position  # (x, y) on the court

    def shoot(self, distance, enemy_players):
        return 1 if np.random.rand() < shooting_probability(distance, enemy_players, self.position) else 0

class Ball:
    def __init__(self, position):
        self.position = position

class Court:
    def __init__(self, length=94, width=50):
        self.length = length
        self.width = width

class Basket:
    def __init__(self, size, position):
        self.size = size
        self.position = position

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == 3

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self):
        action = len(self.children)
        new_state = self.state.copy()
        if action == 0:
            direction = np.random.uniform(-1, 1, 2)
            new_state = self.parent.agent.dribble(new_state, direction)
        elif action == 1:
            target = np.random.randint(0, len(new_state))  # Random target for pass
            new_state, _ = self.parent.agent.pass_ball(new_state, target)
        elif action == 2:
            new_state = self.parent.agent.shoot(new_state)
        child_node = MCTSNode(new_state, parent=self)
        self.children.append(child_node)
        return child_node

class ReinforcementLearningAgent(Player):
    def __init__(self, size, speed, shooting_probability, team, input_size, hidden_size, output_size, learning_rate=0.01):
        super().__init__(size, speed, shooting_probability, team, (0, 0))
        self.policy_network = PolicyNetwork(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.gamma = 0.99
        self.saved_log_probs = []
        self.rewards = []
        self.root = None

    def select_action(self, state):
        if self.root is None or not np.array_equal(self.root.state, state):
            self.root = MCTSNode(state)
            self.root.agent = self
        for _ in range(100):
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
        for _ in range(10):
            action = random.choice([0, 1, 2])
            if action == 0:
                direction = np.random.uniform(-1, 1, (state.shape[0], 2))
                state = self.dribble(state, direction)
            elif action == 1:
                target = np.random.randint(0, len(state))  # Random target for pass
                state, success = self.pass_ball(state, target)
                total_reward += success
            elif action == 2:
                success, points = self.shoot(state)  # Use the shoot method directly
                total_reward += points
        return total_reward

    def backpropagate(self, node, reward):
        while node:
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

    def dribble(self, state, direction):
        return state + direction

    def pass_ball(self, state, target):
        success = 1 if np.random.rand() < 0.8 else 0
        if success:
            return state[target], success
        else:
            # Ball lands on a random player
            ball_landing_position = state[target] + np.random.uniform(-1, 1, 2)
            closest_player = np.argmin(np.linalg.norm(state - ball_landing_position, axis=1))
            return state[closest_player], success

    def shoot(self, state):
        distance = np.linalg.norm(self.position - state[-1])  # Distance to the basket
        success = 1 if np.random.rand() < shooting_probability(distance, [], self.position) else 0
        points = 3 if distance > 22 else 2  # 3 points if beyond 22 feet, else 2 points
        return success, points if success else 0

class RLTeam:
    def __init__(self, team, input_size, hidden_size, output_size, learning_rate=0.01):
        self.agents = [ReinforcementLearningAgent(10, 5, 0.5, team, input_size, hidden_size, output_size, learning_rate) for _ in range(5)]

    def select_actions(self, state):
        return [agent.select_action(state) for agent in self.agents]

    def update_policies(self):
        for agent in self.agents:
            agent.update_policy()

class BasketballEnv:
    def __init__(self):
        self.court = Court()
        self.basket1 = Basket(0.45, np.array([0, 0]))
        self.basket2 = Basket(0.45, np.array([94, 0]))
        self.team1 = RLTeam('home', 22, 128, 3)  # Adjusted input size for more state info
        self.team2 = RLTeam('away', 22, 128, 3)  # Adjusted input size for more state info
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
                new_state[i] = self.team1.agents[i].dribble(self.state[i], direction)
            elif actions[i] == 1:  # Pass
                target = np.random.randint(0, 5)  # Choose a random teammate to pass to
                new_state[i], success = self.team1.agents[i].pass_ball(self.state, target)
                rewards[0] += success
            elif actions[i] == 2 and self.state[i][1] <= self.court.length / 2:  # Shoot only if in their half
                success, points = self.team1.agents[i].shoot(np.linalg.norm(self.state[i] - self.basket2.position), self.team2.agents)
                rewards[0] += points

        for i in range(5, 10):
            if actions[i] == 0:  # Dribble
                direction = np.random.uniform(-1, 1, 2)  # Random direction
                new_state[i] = self.team2.agents[i-5].dribble(self.state[i], direction)
            elif actions[i] == 1:  # Pass
                target = np.random.randint(0, 5)  # Choose a random teammate to pass to
                new_state[i], success = self.team2.agents[i-5].pass_ball(self.state, target)
                rewards[1] += success
            elif actions[i] == 2 and self.state[i][1] >= self.court.length / 2:  # Shoot only if in their half
                success, points = self.team2.agents[i-5].shoot(np.linalg.norm(self.state[i] - self.basket1.position), self.team1.agents)
                rewards[1] += points

        self.state = new_state
        return self.state, rewards

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
